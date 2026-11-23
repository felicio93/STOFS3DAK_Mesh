"""Preflight check and size-field preview, without OCSMesh (seconds on a
laptop).

Checks every input (exists, polygons, 0-360 after the shift), then evaluates
the recipe on a regular grid in the metric CRS (default 2 km, EPSG:3338 is
equal-area) with the same rules OCSMesh uses:

  start at hmax; patches: h = min(h, size*(1 + rate*d)); then regions
  (min: h = max(h, size*(1 - rate*d)), max: h = min(h, size*(1 + rate*d)));
  clip to [hmin, hmax].

From it: the node count estimate (equilateral triangles, 2/(sqrt(3) h^2)
nodes per m^2 inside the domain), the area per size class, warnings (a min
region raising a finer patch, sizes outside [hmin, hmax], shapes outside the
domain), a PNG map and a GeoTIFF of the size field.

The estimate ignores the Gmsh boundary treatment and the DEM-based size
limits that OCSMesh may add, so it is a guide for comparing recipes: check it
once against a built mesh (configs/m5_no_center.yaml: 1,701,388 nodes).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .inputs import check_0_360, load_shape, read_polygons, refinement_parts

_logger = logging.getLogger("akmesh")
NODES_PER_M2 = 2.0 / np.sqrt(3.0)


def _grid(bounds, res):
    x0, y0, x1, y1 = bounds
    x0, y0 = np.floor(x0 / res) * res - res, np.floor(y0 / res) * res - res
    x1, y1 = np.ceil(x1 / res) * res + res, np.ceil(y1 / res) * res + res
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    from rasterio.transform import from_origin
    return from_origin(x0, y1, res, res), (ny, nx)


def _mask(geom, transform, shape):
    from rasterio.features import rasterize
    if geom.is_empty:
        return np.zeros(shape, bool)
    return rasterize([(geom, 1)], out_shape=shape, transform=transform,
                     fill=0, all_touched=False, dtype="uint8").astype(bool)


def _dist(mask, res):
    """Distance (m) from each cell to the nearest cell of ``mask``."""
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(~mask) * res


def _to_metric(geom, crs):
    import geopandas as gpd
    return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(crs).iloc[0]


def size_field(cfg, res=2000.0):
    """(h[ny, nx] in m with NaN outside the domain, transform, info) where
    info holds the per-part masks and messages."""
    msgs, parts_m = [], []
    domain, _ = load_shape(cfg.domain)
    check_0_360("domain", domain)
    dom_m = _to_metric(domain, cfg.metric_crs)
    transform, shape = _grid(dom_m.bounds, res)
    dom_mask = _mask(dom_m, transform, shape)

    for ref in cfg.refinements:
        gdf = read_polygons(ref.shp)
        for geom, size, rate in refinement_parts(ref, gdf):
            check_0_360(ref.name, geom)
            gm = _to_metric(geom, cfg.metric_crs)
            m = _mask(gm, transform, shape)
            inside = (m & dom_mask).sum() * res * res / 1e6
            if not gm.intersects(dom_m):
                msgs.append(f"WARNING {ref.name}: does not overlap the domain")
            if ref.kind == "patch" and size < cfg.hmin:
                msgs.append(f"WARNING {ref.name}: size {size:g} < hmin "
                            f"{cfg.hmin:g}; OCSMesh clips it to hmin")
            if ref.kind == "patch" and size >= cfg.hmax:
                msgs.append(f"WARNING {ref.name}: size {size:g} >= hmax "
                            f"{cfg.hmax:g}; the patch has no effect")
            if m.sum() == 0 and gm.area > 0:
                msgs.append(f"WARNING {ref.name}: thinner than the {res:g} m "
                            f"preview grid (use --res smaller)")
            parts_m.append(dict(ref=ref, geom=gm, size=size, rate=rate,
                                mask=m, area_km2=inside))

    h = np.full(shape, cfg.hmax, dtype="f8")
    patch_h = {}
    for p in parts_m:
        if p["ref"].kind != "patch":
            continue
        if p["rate"]:
            s = p["size"] * (1.0 + p["rate"] * _dist(p["mask"], res))
        else:
            s = np.where(p["mask"], p["size"], np.inf)
        patch_h[id(p)] = s
        h = np.minimum(h, s)
    h_patches = h.copy()
    for p in parts_m:
        if p["ref"].kind != "region":
            continue
        d = _dist(p["mask"], res) if p["rate"] else None
        if p["ref"].value_type == "min":
            b = (p["size"] * (1.0 - p["rate"] * d) if p["rate"]
                 else np.where(p["mask"], p["size"], -np.inf))
            raised = (b > h) & p["mask"] & dom_mask
            for q in parts_m:
                if q["ref"].kind == "patch":
                    hit = raised & q["mask"] & (q["size"] < p["size"])
                    if hit.any():
                        msgs.append(
                            f"NOTE region {p['ref'].name} (no finer than "
                            f"{p['size']:g}) raises patch {q['ref'].name} "
                            f"({q['size']:g}) over {hit.sum()*res*res/1e6:,.0f} km2")
            h = np.maximum(h, b)
        else:
            b = (p["size"] * (1.0 + p["rate"] * d) if p["rate"]
                 else np.where(p["mask"], p["size"], np.inf))
            h = np.minimum(h, b)
    h = np.clip(h, cfg.hmin, cfg.hmax)
    h[~dom_mask] = np.nan
    return h, transform, dict(parts=parts_m, messages=msgs, res=res,
                              domain=dom_m, h_patches=h_patches)


def estimate_nodes(h, res) -> int:
    ok = np.isfinite(h)
    return int(np.sum(NODES_PER_M2 * res * res / h[ok] ** 2))


def size_classes(h, res, edges=(0, 600, 1000, 1600, 2600, 3600, 1e9)):
    ok = np.isfinite(h)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = ok & (h >= a) & (h < b)
        if m.any():
            out.append((a, b, m.sum() * res * res / 1e6,
                        int(np.sum(NODES_PER_M2 * res * res / h[m] ** 2))))
    return out


def report(cfg, h, info) -> str:
    res = info["res"]
    lines = [f"Mesh {cfg.name}: hmin {cfg.hmin:g} m, hmax {cfg.hmax:g} m, "
             f"method {cfg.method}, preview grid {res:g} m ({cfg.metric_crs})",
             "", f"{'refinement':22s} {'kind':10s} {'size':>7s} {'rate':>9s} "
             f"{'area km2':>11s}  transition"]
    for p in info["parts"]:
        r, size, rate = p["ref"], p["size"], p["rate"]
        kind = r.kind if r.kind == "patch" else f"{r.kind}:{r.value_type}"
        if r.kind == "patch" and rate:
            tr = f"{size:g} -> {cfg.hmax:g} m over {(cfg.hmax/size-1)/rate/1e3:,.0f} km"
        elif r.kind == "region" and rate:
            tr = f"bound fades to 0 over {1/rate/1e3:,.0f} km"
        else:
            tr = "inside only"
        lines.append(f"{r.name:22s} {kind:10s} {size:7g} "
                     f"{(f'{rate:.2e}' if rate else '-'):>9s} "
                     f"{p['area_km2']:11,.0f}  {tr}")
    lines += ["", f"{'size class (m)':>16s} {'area km2':>12s} {'nodes':>12s}"]
    for a, b, area, n in size_classes(h, res):
        lab = f"{a:g}-{b:g}" if b < 1e8 else f">= {a:g}"
        lines.append(f"{lab:>16s} {area:12,.0f} {n:12,d}")
    lines += ["", f"Estimated nodes: {estimate_nodes(h, res):,d}   "
              f"(rough: no boundary or DEM limits; compare recipes, not "
              f"absolute counts)"]
    if info["messages"]:
        lines += [""] + info["messages"]
    return "\n".join(lines)


def write_preview(cfg, h, transform, info, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import rasterio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"{cfg.name}_hfun_preview.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=h.shape[0],
                       width=h.shape[1], count=1, dtype="float32",
                       crs=cfg.metric_crs, transform=transform,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(h.astype("float32"), 1)

    ny, nx = h.shape
    x0, y1 = transform.c, transform.f
    ext = (x0, x0 + nx * transform.a, y1 + ny * transform.e, y1)
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=130)
    im = ax.imshow(h, extent=ext, origin="upper", cmap="viridis",
                   norm=LogNorm(vmin=cfg.hmin, vmax=cfg.hmax))
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("target size (m)")
    ticks = sorted({cfg.hmin, cfg.hmax, *[p["size"] for p in info["parts"]
                                          if cfg.hmin <= p["size"] <= cfg.hmax]})
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    seen = {}
    for p in info["parts"]:
        name = p["ref"].name
        c = seen.setdefault(name, colors[len(seen) % len(colors)])
        g = p["geom"]
        for poly in getattr(g, "geoms", [g]):
            if poly.geom_type != "Polygon":
                continue
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color=c, lw=0.7,
                    ls="-" if p["ref"].kind == "patch" else "--",
                    label=name if name not in [l.get_label() for l in ax.lines] else None)
    for poly in getattr(info["domain"], "geoms", [info["domain"]]):
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color="k", lw=0.6)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    ax.set_title(f"{cfg.name}: estimated {estimate_nodes(h, info['res']):,d} "
                 f"nodes (preview, {info['res']:g} m grid; solid = patch, "
                 f"dashed = region)", fontsize=9)
    png = out_dir / f"{cfg.name}_hfun_preview.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return [png, tif]
