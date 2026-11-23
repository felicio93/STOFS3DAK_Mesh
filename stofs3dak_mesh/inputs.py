"""Shapefiles and DEM in the continuous 0-360 longitude convention.

The Alaska domain crosses the antimeridian in the Bering Sea. Inputs may be
-180..180 or 0-360: shapefile vertices with lon < 0 get +360 in memory on
every run (so edited polygons just work), and a global -180..180 GeoTIFF is
rolled once into ``<stem>_360.tif`` next to it (a pure column permutation,
reused while newer than the original). A raster that is already 0-360 is
used as is.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

_logger = logging.getLogger("akmesh")


def shift_to_360(geoms):
    """Add 360 to every vertex longitude < 0 (0-360 input is unchanged)."""
    import shapely

    def _f(c):
        c = c.copy()
        c[c[:, 0] < 0, 0] += 360.0
        return c
    return shapely.transform(np.asarray(geoms, dtype=object), _f)


def read_polygons(shp: Path):
    """GeoDataFrame of the polygon features of ``shp``, EPSG:4326, 0-360,
    each geometry made valid."""
    import geopandas as gpd
    if not Path(shp).is_file():
        raise FileNotFoundError(shp)
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise ValueError(f"no polygon geometry in {shp}")
    gdf["geometry"] = shift_to_360(gdf.geometry.values)
    gdf["geometry"] = gdf.geometry.make_valid().buffer(0)
    return gdf


def union(geoms):
    """(Multi)Polygon union of ``geoms``, non-polygon parts dropped."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union
    shape = unary_union(list(geoms)).buffer(0)
    if shape.geom_type == "GeometryCollection":
        shape = unary_union([g for g in shape.geoms
                             if isinstance(g, (Polygon, MultiPolygon))])
    return shape


def load_shape(shp: Path):
    """(Polygon|MultiPolygon, crs) of all polygons of ``shp``, 0-360."""
    gdf = read_polygons(shp)
    return union(gdf.geometry), gdf.crs


def refinement_parts(ref, gdf=None):
    """[(shape, size, rate), ...] for one refinement: one entry, or one per
    distinct (size, rate) when the refinement has size_field/rate_field."""
    if gdf is None:
        gdf = read_polygons(ref.shp)
    size_field = ref.size_field if ref.size_field in gdf.columns else None
    rate_field = ref.rate_field if ref.rate_field in gdf.columns else None
    for f, used in ((ref.size_field, size_field), (ref.rate_field, rate_field)):
        if f and not used:
            _logger.info("  %s: no attribute %r in %s; all polygons take the "
                         "default", ref.name, f, Path(ref.shp).name)
    if not size_field and not rate_field:
        if ref.size is None:
            raise ValueError(f"refinement '{ref.name}': no {ref.size_field} "
                             f"attribute and no default 'size'")
        return [(union(gdf.geometry), ref.size, ref.rate)]

    def _val(row, fld, default):
        if not fld:
            return default
        v = row[fld]
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return default if np.isnan(v) or v <= 0 else v

    groups = {}
    for _, row in gdf.iterrows():
        size = _val(row, size_field, ref.size)
        rate = _val(row, rate_field, ref.rate)
        if size is None:
            raise ValueError(f"refinement '{ref.name}': a polygon has no "
                             f"{size_field} and no default 'size' is set")
        groups.setdefault((size, rate), []).append(row.geometry)
    return [(union(g), s, r) for (s, r), g in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0))]


def check_0_360(name: str, shape) -> tuple:
    x0, y0, x1, y1 = shape.bounds
    if x0 < 0 or x1 > 360 or y0 < -90 or y1 > 90:
        raise ValueError(f"{name}: not in 0-360 lon/lat degrees "
                         f"(bounds {x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})")
    return x0, y0, x1, y1


def ensure_dem_360(src_path: Path, force: bool = False) -> Path:
    """A 0-360 version of ``src_path`` (the file itself if already 0-360)."""
    import rasterio
    from rasterio.windows import Window
    src_path = Path(src_path)
    dst = src_path.with_name(f"{src_path.stem}_360.tif")
    if not src_path.is_file():
        if dst.is_file():
            _logger.warning("%s missing; using %s", src_path.name, dst.name)
            return dst
        raise FileNotFoundError(src_path)
    with rasterio.open(src_path) as src:
        b = src.bounds
        if b.left >= 0:
            _logger.info("  %s already 0-360; used as is", src_path.name)
            return src_path
        if (dst.is_file() and not force
                and dst.stat().st_mtime >= src_path.stat().st_mtime):
            _logger.info("  reusing %s (newer than %s)", dst.name, src_path.name)
            return dst
        res = src.transform.a
        n_neg = int(round(-b.left / res))
        if (abs(b.left + 180) > res or abs(b.right - 180) > res
                or abs(src.width * res - 360) > res or n_neg >= src.width):
            raise ValueError(f"{src_path.name} is neither 0-360 nor a global "
                             f"-180..180 raster (bounds {tuple(b)})")
        _logger.info("  rolling %s -> %s (%d x %d)", src_path.name, dst.name,
                     src.width, src.height)
        prof = src.profile.copy()
        prof.update(transform=rasterio.Affine(res, 0, b.left + n_neg * res,
                                              0, src.transform.e, b.top),
                    tiled=True, blockxsize=512, blockysize=512,
                    compress="deflate", BIGTIFF="IF_SAFER")
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        n_pos = src.width - n_neg
        try:
            with rasterio.open(tmp, "w", **prof) as out:
                for r0 in range(0, src.height, 512):
                    h = min(512, src.height - r0)
                    for c0, c1, d0 in ((n_neg, src.width, 0), (0, n_neg, n_pos)):
                        out.write(src.read(window=Window(c0, r0, c1 - c0, h)),
                                  window=Window(d0, r0, c1 - c0, h))
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
    return dst
