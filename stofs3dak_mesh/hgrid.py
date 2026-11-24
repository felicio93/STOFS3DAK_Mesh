"""From the OCSMesh mesh (.2dm) to the SCHISM hgrid: order, depths,
elevation limits, boundaries.

Steps (recipe configs/hgrid_*.yaml, command ``akmesh hgrid``):

1. open the mesh (EPSG:4326, lon 0-360) and orient every element
   counterclockwise (``ocsmesh.utils.order_mesh``);
2. interpolate the DEM (GEBCO, 0-360) to the nodes;
3. elevation limits, all as maxima on the elevation (positive up):
   - ``max_elevation`` (default -3) everywhere: at least 3 m deep;
   - ``max_elevation_areas``: polygons with their own maximum (0 at or below
     MSL, 5 m above MSL for the floodplain) in place of the default; where
     these polygons overlap, the highest maximum applies;
   - ``force_channels``: polygons whose attribute (``Depth``) gives a
     minimum depth; the node is at least that deep (|Depth| m below MSL),
     whatever the area limit. The sign of the attribute is ignored.
4. boundaries: all land (``auto_generate`` with a threshold below any
   elevation), then the mesh edges crossing the ``open_boundaries`` shapes
   become open boundaries; each boundary type is written largest first;
5. write ``<name>.gr3`` (SCHISM depth positive down, with boundaries),
   ``<name>.2dm``, ``<name>.ll`` (= the .gr3: the mesh is in lon/lat) and
   ``<name>_summary.json``.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import __version__
from .build import _git_commit
from .inputs import ensure_dem_360, read_polygons, shift_to_360

_logger = logging.getLogger("akmesh")


def _nodes_in(geoms, x, y):
    """Boolean mask of the nodes inside any of ``geoms``."""
    import shapely
    from shapely.ops import unary_union
    shape = unary_union(list(geoms))
    if shape.is_empty:
        return np.zeros(len(x), bool)
    shapely.prepare(shape)
    x0, y0, x1, y1 = shape.bounds
    box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    out = np.zeros(len(x), bool)
    out[box] = shapely.contains_xy(shape, x[box], y[box])
    return out


def _read_lines_or_polygons(shp: Path):
    """Union of all geometries of ``shp`` (any type), EPSG:4326, 0-360."""
    import geopandas as gpd
    from shapely.ops import unary_union
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()]
    if gdf.empty:
        raise ValueError(f"no geometry in {shp}")
    return unary_union(list(shift_to_360(gdf.geometry.values)))


def elevation_limits(x, y, z, hc) -> tuple:
    """New elevations and a report of what each limit changed."""
    z = np.asarray(z, dtype="f8").copy()
    cap = np.full(len(z), float(hc["max_elevation"]))
    report = {"max_elevation": float(hc["max_elevation"]), "areas": [],
              "channels": []}

    in_any = np.zeros(len(z), bool)
    area_caps = np.full(len(z), -np.inf)
    for a in hc.get("max_elevation_areas") or []:
        gdf = read_polygons(a["shp"])
        m = _nodes_in(gdf.geometry, x, y)
        overlap = int((m & in_any).sum())
        area_caps[m] = np.maximum(area_caps[m], float(a["max_elevation"]))
        in_any |= m
        report["areas"].append({"shp": Path(a["shp"]).name,
                                "max_elevation": float(a["max_elevation"]),
                                "nodes": int(m.sum()),
                                "nodes_overlapping_earlier_areas": overlap})
    cap[in_any] = area_caps[in_any]

    fc = hc.get("force_channels")
    if fc:
        gdf = read_polygons(fc["shp"])
        fld = fc.get("depth_field", "Depth")
        if fld not in gdf.columns:
            raise ValueError(f"{Path(fc['shp']).name}: no attribute {fld!r} "
                             f"(columns: {list(gdf.columns)})")
        for i, row in gdf.iterrows():
            d = row[fld]
            try:
                d = abs(float(d))
            except (TypeError, ValueError):
                d = np.nan
            m = _nodes_in([row.geometry], x, y)
            entry = {"feature": int(i), "depth": None if np.isnan(d) else d,
                     "nodes": int(m.sum())}
            if np.isnan(d):
                entry["skipped"] = f"no {fld} value"
                _logger.warning("  force_channels feature %s has no %s: skipped "
                                "(%d nodes)", i, fld, m.sum())
            else:
                entry["nodes_deepened"] = int((m & (np.minimum(z, cap) > -d)).sum())
                cap[m] = np.minimum(cap[m], -d)
            report["channels"].append(entry)

    over = z > cap
    report["nodes_lowered"] = int(over.sum())
    report["nodes_lowered_by_default"] = int((over & ~in_any).sum())
    z[over] = cap[over]
    return z, report


def sort_boundaries(mesh) -> dict:
    """Each boundary type largest first (number of nodes). Returns the
    segment sizes per type."""
    data = deepcopy(mesh.boundaries.data)
    sizes = {}
    for key, segs in data.items():
        if not segs:
            continue
        ordered = sorted(segs.values(), key=lambda b: len(b["indexes"]),
                         reverse=True)
        data[key] = {i: s for i, s in enumerate(ordered)}
        sizes[str(key)] = [len(s["indexes"]) for s in ordered]
    mesh.boundaries._refresh_boundaries(data)
    return sizes


def make_hgrid(hc, mesh_path: Path, out_dir: Path) -> dict:
    import ocsmesh
    from ocsmesh import Mesh, Raster

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = hc["name"]
    _logger.info("hgrid %s from %s", name, mesh_path)

    _logger.info("Step 1: open and order the mesh")
    mesh = Mesh.open(str(mesh_path), crs=4326)
    md = mesh.meshdata if hasattr(mesh, "meshdata") else mesh.msh_t
    mesh = Mesh(ocsmesh.utils.order_mesh(md))
    md = mesh.meshdata if hasattr(mesh, "meshdata") else mesh.msh_t
    x, y = md.coords[:, 0], md.coords[:, 1]
    if x.min() < 0:
        raise ValueError(f"mesh longitudes {x.min():.2f}..{x.max():.2f}: "
                         f"expected the 0-360 mesh of akmesh build")
    _logger.info("  %d nodes, %d triangles", len(x), len(md.tria))

    _logger.info("Step 2: interpolate %s (%s)", Path(hc["dem"]).name,
                 hc.get("interp_method", "linear"))
    mesh.interpolate([Raster(ensure_dem_360(hc["dem"]))],
                     method=hc.get("interp_method", "linear"))
    z = np.asarray(mesh.value, dtype="f8").reshape(-1)
    n_nan = int(np.isnan(z).sum())
    if n_nan:
        raise RuntimeError(f"{n_nan} nodes got no DEM value (outside the raster?)")
    z_dem = z.copy()

    _logger.info("Step 3: elevation limits")
    z, limits = elevation_limits(x, y, z, hc)
    mesh.value[:] = z.reshape(np.shape(mesh.value))
    _logger.info("  %s", json.dumps(limits))

    _logger.info("Step 4: boundaries")
    mesh.boundaries.auto_generate(threshold=-999999)
    n_open_shapes = 0
    if hc.get("open_boundaries"):
        region = _read_lines_or_polygons(hc["open_boundaries"])
        n_open_shapes = len(getattr(region, "geoms", [region]))
        mesh.boundaries.set_open(region, merge=True)
    sizes = sort_boundaries(mesh)
    n_open = len(sizes.get("None", []))
    _logger.info("  open %s, land %d, island %d", sizes.get("None", []),
                 len(sizes.get("0", [])), len(sizes.get("1", [])))
    if hc.get("open_boundaries") and n_open == 0:
        raise RuntimeError("no open boundary: the open_boundaries shapes do "
                           "not cross the mesh edge")

    _logger.info("Step 5: write")
    gr3 = out / f"{name}.gr3"
    mesh.write(str(gr3), format="grd", overwrite=True)
    mesh.write(str(out / f"{name}.2dm"), format="2dm", overwrite=True)
    shutil.copyfile(gr3, out / f"{name}.ll")

    depth = -z
    summary = {
        "name": name, "mesh": str(mesh_path), "config": hc.get("_raw"),
        "stofs3dak_mesh": {"version": __version__,
                           "commit": _git_commit(Path(__file__).parent)},
        "ocsmesh": _git_commit(Path(ocsmesh.__file__).parent),
        "nodes": int(len(z)), "triangles": int(len(md.tria)),
        "dem_elevation": {"min": float(z_dem.min()), "max": float(z_dem.max())},
        "depth": {"min": float(depth.min()), "max": float(depth.max())},
        "limits": limits,
        "boundaries": {"open_shapes": n_open_shapes,
                       "open_nodes": sizes.get("None", []),
                       "land": len(sizes.get("0", [])),
                       "island": len(sizes.get("1", []))},
        "time_s": time.time() - t0,
    }
    with open(out / f"{name}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _logger.info("Done in %.1f min: %s", summary["time_s"] / 60, out)
    return summary


def load_recipe(path, data_dir=None) -> dict:
    """The hgrid recipe with input paths resolved (relative data_dir from
    the recipe folder, relative files from data_dir)."""
    import os
    import yaml
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    for k in ("name", "dem", "max_elevation"):
        if k not in raw:
            raise ValueError(f"{path.name}: '{k}' is required")
    known = {"name", "description", "data_dir", "dem", "interp_method",
             "max_elevation", "max_elevation_areas", "force_channels",
             "open_boundaries", "machine", "job"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path.name}: unknown key(s) {sorted(unknown)}")
    if data_dir is None:
        data_dir = Path(os.path.expanduser(str(raw.get("data_dir", "."))))
        if not data_dir.is_absolute():
            data_dir = path.resolve().parent / data_dir
    data_dir = Path(os.path.normpath(os.path.abspath(data_dir)))

    def p(v):
        v = Path(os.path.expanduser(str(v)))
        return v if v.is_absolute() else data_dir / v

    hc = dict(raw)
    hc["_raw"] = raw
    hc["data_dir"] = data_dir
    hc["dem"] = p(raw["dem"])
    hc["max_elevation_areas"] = [
        {"shp": p(a["shp"]), "max_elevation": float(a["max_elevation"])}
        for a in raw.get("max_elevation_areas") or []]
    if raw.get("force_channels"):
        hc["force_channels"] = {**raw["force_channels"],
                                "shp": p(raw["force_channels"]["shp"])}
    if raw.get("open_boundaries"):
        hc["open_boundaries"] = p(raw["open_boundaries"])
    if hc.get("interp_method", "linear") not in ("linear", "nearest", "spline"):
        raise ValueError("interp_method must be linear, nearest or spline")
    return hc


def inputs_of(hc) -> list:
    files = [hc["dem"]] + [a["shp"] for a in hc["max_elevation_areas"]]
    if hc.get("force_channels"):
        files.append(hc["force_channels"]["shp"])
    if hc.get("open_boundaries"):
        files.append(hc["open_boundaries"])
    return files
