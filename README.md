# STOFS3DAK_Mesh

Mesh generation for the STOFS-3D-Alaska SCHISM domain with
[OCSMesh](https://github.com/noaa-ocs-modeling/OCSMesh). The domain crosses
the antimeridian in the Bering Sea and is meshed in continuous 0-360
longitudes (0 = Greenwich, 180 = Bering Sea).

One YAML **recipe** per mesh: domain polygon, DEM, `hmin`/`hmax`, method,
and a list of refinements, each a shapefile with a target size. The
`akmesh` command checks a recipe on a laptop in seconds, then writes and
submits the batch job that builds the mesh on the HPC.

| recipe | mesh |
|---|---|
| `configs/m5_no_center.yaml` | the mesh of the M70-M74 runs (2026-11-19, `fast`, 1,701,388 nodes) |
| `configs/m6_hires.yaml` | m5 + 750 m areas from `500m_areas.shp` (to 1500 m over ~10 km), `bnd_factor` 1.2 |

`legacy/AK_test/` is the script that built the M70 mesh, kept as it was,
with its logs and summary.

## Install

```bash
conda activate <env with ocsmesh and gmsh>     # on Gaea: ocsmesh_mpi_test
pip install -e .                               # optional: job.sh puts this checkout on PYTHONPATH
akmesh --help
```
Without `pip install`, use `PYTHONPATH=<checkout> python -m stofs3dak_mesh.cli`
in place of `akmesh`. `akmesh check` needs geopandas, rasterio, scipy and
matplotlib, but not OCSMesh.

## Workflow

```bash
akmesh check  -c configs/m6_hires.yaml                    # inputs, warnings, preview, node estimate
akmesh submit -c configs/m6_hires.yaml --dry-run          # look at job.sh first
akmesh submit -c configs/m6_hires.yaml                    # sbatch
```
`--data-dir` overrides the recipe's `data_dir`, e.g. to check a recipe on a
laptop against a local copy of the inputs.

**check** reads every input, shifts it to 0-360 and evaluates the recipe on a
2 km grid (`--res`) in EPSG:3338 with the OCSMesh rules. It prints each
refinement (area, transition distance), the area and node count per size
class, an estimated node count, and notes where a region constraint raises a
finer patch. It writes `preview/<name>/<name>_hfun_preview.png` and a GeoTIFF
of the size field to open in QGIS. Calibration: for `m5_no_center` it
estimates 1,624,212 nodes against 1,701,388 in the built mesh (-4.5 %). The
estimate leaves out the boundary treatment, so it is low for very coarse
meshes; use it to compare recipes.

**submit** makes `<out_root>/<name>_<method>_<YYYYmmdd_HHMM>/` with a copy of
the recipe (`config.yaml`) and `job.sh`, and submits it. The job runs
`akmesh build` there and leaves:
- `<name>.2dm`, `<name>.gr3`: the mesh (EPSG:4326, lon 0-360);
- `<name>_summary.json`: recipe, package and OCSMesh commits, timings, node count;
- `shapes_360/`: the shifted inputs, one shapefile per refinement with its size and rate;
- `akmesh_<jobid>.out/.err`: the SLURM logs.

## Recipe

```yaml
name: AK_m6_hires
data_dir: /gpfs/.../AK_LargeDomain_Mesh/2026_10_15   # relative paths below are taken from here
out_root: /gpfs/.../results                          # optional; default from the machine file
domain: AKdomain_20261014.shp
dem: GEBCO_2026.tif                                  # 0-360, or global -180..180 (rolled once to <stem>_360.tif)
hmin: 750
hmax: 3500
method: fast                                         # fast | exact
engine: gmsh                                         # gmsh | triangle
bnd_factor: 1.2                                      # Gmsh 'adapt' boundary spacing = size x this
metric_crs: EPSG:3338
force_metric_crs: true                               # the antimeridian workaround (see below)
machine: gaea_c6                                     # stofs3dak_mesh/machines/<name>.yaml, or a path
job: {time: "08:00:00"}                              # optional sbatch overrides
refinements:
  - {name: shelf_150, kind: patch, shp: 150m_contour.shp, size: 1500, rate: 1.48e-5}
  - {name: below_200, kind: region, shp: below_200m_contour.shp, size: 2500, rate: 5.0e-5}
  - {name: hires, kind: patch, shp: 500m_areas.shp, size: 750, rate: 1.0e-4, size_field: size_m}
```

Sizes in m, rates in 1/m, `d` = distance in m from the polygon:

| kind | effect |
|---|---|
| `patch` | `size` inside; `size * (1 + rate*d)` outside (none without a rate). Patches combine with min(): the finest wins. |
| `region` (`value_type: min`, default) | "no finer than `size`": the bound is `size * (1 - rate*d)` outside. |
| `region` (`value_type: max`) | "no coarser than `size`": the bound is `size * (1 + rate*d)` outside. |

OCSMesh applies all patches first, then the regions, then clips to
`[hmin, hmax]`. A `min` region therefore raises any finer patch inside it:
a high-resolution area inside `below_200` (deep water) or `coarse_*` stays at
2500 or 3500 m. `akmesh check` reports this.

`size_field` names a numeric shapefile attribute with a size per polygon
(polygons without a value take `size`); `rate_field` does the same for the
rate. Shapefiles can be in any CRS and in -180..180 or 0-360.

With `method: fast`, OCSMesh evaluates the size function on a raster of
`hmin/2` over the whole domain (750 m for the M70 mesh), so going from
`hmin` 1500 to 750 m makes that raster about 4 times larger.

## High-resolution areas

1. Draw the polygons in QGIS into one shapefile (`500m_areas.shp`), with an
   optional numeric field `size_m`.
2. Copy it into `data_dir` (or give `shp` an absolute path).
3. `akmesh check -c configs/m6_hires.yaml`: look at the preview and the notes,
   then adjust sizes and rates.
4. `akmesh submit -c configs/m6_hires.yaml`.

## Machines

`stofs3dak_mesh/machines/gaea_c6.yaml` holds the `#SBATCH` options, the shell
setup (modules, conda env), the OCSMesh checkout to put first on
`PYTHONPATH`, the default `out_root` and the launcher. Another HPC is another
file of the same form.

## Antimeridian

OCSMesh's `utils.estimate_bounds_utm` returns `None` for longitudes above 180,
which makes size-function distances run in degrees. With
`force_metric_crs: true`, every metric-CRS lookup returns `metric_crs`
(EPSG:3338, Alaska Albers), and `OCSMESH_METRIC_CRS` is exported for worker
processes (read by the OCSMesh branch `ak/lon360-patched`, which the Gaea
machine file uses). The mesh is written in EPSG:4326 and wrapped with
`lon % 360`; the build stops if an element spans more than 90 degrees of
longitude. Set `force_metric_crs: false` for an OCSMesh with antimeridian
support.

## Tests

```bash
pip install -e .[test]; pytest -q
```
