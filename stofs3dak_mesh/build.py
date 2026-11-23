"""Build the mesh with OCSMesh (Geom from the domain polygon, Hfun from the
refinements, MeshDriver), wrap it to 0-360 and write .2dm, .gr3 and a
summary JSON.

How the antimeridian is handled: OCSMesh's ``utils.estimate_bounds_utm``
returns None for longitudes > 180, which makes size-function distances run
in degrees. With ``force_metric_crs`` every metric-CRS lookup returns
``metric_crs`` (EPSG:3338, Alaska Albers, which PROJ evaluates correctly for
0-360 input). OCSMesh branch ``ak/lon360-patched`` also reads
OCSMESH_METRIC_CRS, which worker processes inherit. The mesh is written in
EPSG:4326 with lon % 360 (connectivity untouched).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from . import __version__
from .inputs import (check_0_360, ensure_dem_360, load_shape, read_polygons,
                     refinement_parts)

_logger = logging.getLogger("akmesh")
OUTPUT_CRS = "EPSG:4326"


def _git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%h %s"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def force_metric_crs(metric_crs: str) -> None:
    from ocsmesh import utils
    from pyproj import CRS
    metric = CRS.from_user_input(metric_crs)
    os.environ["OCSMESH_METRIC_CRS"] = metric.to_string()

    # same signature as utils.estimate_bounds_utm (callers pass crs=...)
    def _metric(bounds, crs="EPSG:4326"):  # noqa: ARG001
        return metric if CRS.from_user_input(crs).is_geographic else None

    utils.estimate_bounds_utm = _metric
    _logger.info("Metric CRS for all calculations forced to %s", metric.to_string())


def wrap_lon_360(mesh) -> None:
    from pyproj import CRS
    md = mesh.meshdata
    if md.crs is None or not CRS.from_user_input(md.crs).is_geographic:
        raise ValueError(f"expected a geographic CRS to wrap, got {md.crs}")
    coords = md.coords.copy()
    coords[:, 0] = np.mod(coords[:, 0], 360.0)
    md.coords = coords


def mesh_summary(mesh) -> dict:
    md = mesh.meshdata
    lon, lat = md.coords[:, 0], md.coords[:, 1]
    tri = md.tria
    span = np.ptp(lon[tri], axis=1) if len(tri) else np.array([0.0])
    return {"nodes": int(md.num_nodes), "triangles": int(len(tri)),
            "lon_min": float(lon.min()), "lon_max": float(lon.max()),
            "lat_min": float(lat.min()), "lat_max": float(lat.max()),
            "max_element_lon_span_deg": float(span.max()),
            "crs": str(md.crs)}


def load_refinements(cfg):
    """{name: [(shape, size, rate), ...]} with every shape checked 0-360."""
    out = {}
    for ref in cfg.refinements:
        parts = refinement_parts(ref, read_polygons(ref.shp))
        for shape, _, _ in parts:
            x0, _, x1, _ = check_0_360(ref.name, shape)
        _logger.info("  %-18s %-6s %d part(s), lon %.3f .. %.3f", ref.name,
                     ref.kind, len(parts), x0, x1)
        out[ref.name] = parts
    return out


def apply_refinements(hfun, cfg, parts, shape_crs) -> None:
    for ref in cfg.refinements:
        for shape, size, rate in parts[ref.name]:
            _logger.info("  %s: %s size %g rate %s", ref.name, ref.kind, size, rate)
            if ref.kind == "patch":
                hfun.add_patch(shape=shape, expansion_rate=rate, target_size=size)
            else:
                kw = dict(value=size, shape=shape, crs=shape_crs,
                          value_type=ref.value_type)
                if rate is not None:
                    kw["rate"] = rate
                hfun.add_region_constraint(**kw)


def build(cfg, out_dir: Path, nprocs: int = 1, force_raster: bool = False) -> dict:
    import geopandas as gpd
    import ocsmesh
    from ocsmesh import Geom, Hfun, MeshDriver, Raster

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0, timings = time.time(), {}

    ocsmesh_dir = Path(ocsmesh.__file__).resolve().parent
    ocsmesh_commit = _git_commit(ocsmesh_dir)
    pkg_dir = Path(__file__).resolve().parent
    _logger.info("stofs3dak_mesh %s (%s)", __version__, _git_commit(pkg_dir))
    _logger.info("OCSMesh: %s (%s)", ocsmesh_dir, ocsmesh_commit)
    _logger.info("Mesh %s: method=%s engine=%s hmin=%g hmax=%g bnd_factor=%g "
                 "nprocs=%d", cfg.name, cfg.method, cfg.engine, cfg.hmin,
                 cfg.hmax, cfg.bnd_factor, nprocs)
    if cfg.force_metric_crs:
        force_metric_crs(cfg.metric_crs)

    _logger.info("Step 1: shapes")
    t = time.time()
    domain, domain_crs = load_shape(cfg.domain)
    check_0_360("domain", domain)
    parts = load_refinements(cfg)
    shp_dir = out / "shapes_360"
    shp_dir.mkdir(exist_ok=True)
    gpd.GeoSeries([domain], crs=domain_crs).to_file(shp_dir / "domain_360.shp")
    for name, plist in parts.items():
        gpd.GeoDataFrame({"size": [p[1] for p in plist],
                          "rate": [p[2] if p[2] is not None else 0.0 for p in plist]},
                         geometry=[p[0] for p in plist], crs=domain_crs
                         ).to_file(shp_dir / f"{name}_360.shp")
    timings["load_shapes_s"] = time.time() - t

    _logger.info("Step 2: Geom")
    t = time.time()
    # No rasters: the domain polygon alone defines the geometry, which avoids
    # raster boundary extraction across the antimeridian.
    geom = Geom([], base_shape=domain, base_shape_crs=domain_crs,
                zmin=-10000.0, zmax=99999.0)
    timings["geom_s"] = time.time() - t

    _logger.info("Step 3: Hfun")
    t = time.time()
    rast = Raster(ensure_dem_360(cfg.dem, force_raster))
    _logger.info("  DEM %s bbox %s", rast.path, rast.get_bbox().bounds)
    hfun = Hfun([rast], hmin=cfg.hmin, hmax=cfg.hmax, nprocs=nprocs,
                base_shape=domain, base_shape_crs=domain_crs, method=cfg.method)
    if nprocs > 1:
        hfun.execution_mode = "parallel"
    apply_refinements(hfun, cfg, parts, domain_crs)
    timings["hfun_setup_s"] = time.time() - t

    _logger.info("Step 4: MeshDriver")
    t = time.time()
    mesh = MeshDriver(geom, hfun, crs=OUTPUT_CRS, engine_name=cfg.engine,
                      bnd_representation="adapt",
                      bnd_factor=cfg.bnd_factor).run()
    timings["mesh_s"] = time.time() - t

    _logger.info("Step 5: wrap to 0-360 and write")
    wrap_lon_360(mesh)
    summary = mesh_summary(mesh)
    _logger.info("Mesh: %s", summary)
    if summary["lon_min"] < 0 or summary["lon_max"] >= 360:
        raise RuntimeError("output longitudes are not within [0, 360)")
    if summary["max_element_lon_span_deg"] > 90:
        raise RuntimeError("an element spans > 90 deg of longitude: wrap error")
    mesh.write(out / f"{cfg.name}.2dm", format="2dm", overwrite=True)
    mesh.write(out / f"{cfg.name}.gr3", format="grd", overwrite=True)

    timings["total_s"] = time.time() - t0
    result = {"name": cfg.name, "config": cfg.raw,
              "config_file": str(cfg.source) if cfg.source else None,
              "data_dir": str(cfg.data_dir), "nprocs": nprocs,
              "stofs3dak_mesh": {"version": __version__,
                                 "commit": _git_commit(pkg_dir)},
              "ocsmesh": {"path": str(ocsmesh_dir), "commit": ocsmesh_commit},
              "timings": timings, "mesh": summary}
    with open(out / f"{cfg.name}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _logger.info("Done in %.1f min: %s", timings["total_s"] / 60, out)
    return result
