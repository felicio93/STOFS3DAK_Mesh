"""Mesh recipe: one YAML file per mesh.

Example (configs/m5_no_center.yaml reproduces the M70 mesh)::

    name: AK_LARGEDOMAIN_m5_no_center
    data_dir: /gpfs/.../AK_LargeDomain_Mesh/2026_10_15
    out_root: /gpfs/.../AK_LargeDomain_Mesh/results
    domain: AKdomain_20261014.shp
    dem: GEBCO_2026.tif
    hmin: 1500
    hmax: 3500
    method: fast            # fast | exact
    engine: gmsh            # gmsh | triangle
    bnd_factor: 1.5
    metric_crs: EPSG:3338
    force_metric_crs: true
    machine: gaea_c6
    job: {time: "08:00:00"}  # optional sbatch overrides
    refinements:
      - {name: shelf_150, kind: patch, shp: 150m_contour.shp, size: 1500, rate: 1.48e-5}
      - {name: below_200, kind: region, shp: below_200m_contour.shp, size: 2500, rate: 5.0e-5}

Refinement kinds (sizes in m, rates in 1/m, distance d in m from the polygon):

  patch   size(d) = size * (1 + rate*d) outside, ``size`` inside; without a
          rate only the inside. Patches combine with min(): the finest wins.
  region  value_type min (default): "no finer than size", bound(d) =
          size * (1 - rate*d) outside; value_type max: "no coarser than
          size", bound(d) = size * (1 + rate*d). Applied after all patches,
          so a min region raises any finer patch inside it.

``size_field`` (optional) names a shapefile attribute holding a size per
polygon; polygons with no value take ``size``. ``rate_field`` does the same
for the rate. Relative ``shp``/``domain``/``dem`` paths are taken from
``data_dir``.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

KINDS = ("patch", "region")
METHODS = ("fast", "exact")
ENGINES = ("gmsh", "triangle")
MACHINES_DIR = Path(__file__).parent / "machines"


class ConfigError(ValueError):
    pass


@dataclass
class Refinement:
    name: str
    kind: str
    shp: Path
    size: Optional[float] = None
    rate: Optional[float] = None
    value_type: str = "min"
    size_field: Optional[str] = None
    rate_field: Optional[str] = None


@dataclass
class Config:
    name: str
    data_dir: Path
    domain: Path
    dem: Path
    hmin: float
    hmax: float
    refinements: list
    out_root: Optional[Path] = None
    method: str = "fast"
    engine: str = "gmsh"
    bnd_factor: float = 1.0
    metric_crs: str = "EPSG:3338"
    force_metric_crs: bool = True
    machine: str = "gaea_c6"
    job: dict = field(default_factory=dict)
    source: Optional[Path] = None
    raw: dict = field(default_factory=dict)


def _path(value, base: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() else base / p


def _float(d: dict, key: str, where: str, required=True):
    v = d.get(key)
    if v is None:
        if required:
            raise ConfigError(f"{where}: '{key}' is required")
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: '{key}' must be a number, got {v!r}")


def parse(raw: dict, source: Optional[Path] = None,
          data_dir: Optional[Path] = None) -> Config:
    """Config from a dict (the YAML). ``data_dir`` overrides the file's."""
    raw = copy.deepcopy(raw or {})
    for key in ("name", "domain", "dem", "hmin", "hmax", "refinements"):
        if key not in raw:
            raise ConfigError(f"'{key}' is required")
    if data_dir is None:
        if "data_dir" not in raw:
            raise ConfigError("'data_dir' is required")
        data_dir = Path(os.path.expandvars(os.path.expanduser(
            str(raw["data_dir"]))))
    data_dir = Path(data_dir)

    hmin, hmax = _float(raw, "hmin", "config"), _float(raw, "hmax", "config")
    if not 0 < hmin <= hmax:
        raise ConfigError(f"need 0 < hmin <= hmax, got {hmin}, {hmax}")

    refs, names = [], set()
    if not isinstance(raw["refinements"], list):
        raise ConfigError("'refinements' must be a list")
    for i, r in enumerate(raw["refinements"]):
        where = f"refinements[{i}]"
        if not isinstance(r, dict):
            raise ConfigError(f"{where}: must be a mapping")
        name = str(r.get("name") or "")
        if not name:
            raise ConfigError(f"{where}: 'name' is required")
        where = f"refinement '{name}'"
        if name in names:
            raise ConfigError(f"{where}: duplicate name")
        names.add(name)
        kind = r.get("kind")
        if kind not in KINDS:
            raise ConfigError(f"{where}: kind must be one of {KINDS}, got {kind!r}")
        if "shp" not in r:
            raise ConfigError(f"{where}: 'shp' is required")
        size_field = r.get("size_field")
        size = _float(r, "size", where, required=size_field is None)
        rate = _float(r, "rate", where, required=False)
        if size is not None and size <= 0:
            raise ConfigError(f"{where}: size must be > 0")
        if rate is not None and rate < 0:
            raise ConfigError(f"{where}: rate must be >= 0")
        vt = r.get("value_type", "min")
        if kind == "region" and vt not in ("min", "max"):
            raise ConfigError(f"{where}: value_type must be min or max")
        unknown = set(r) - {"name", "kind", "shp", "size", "rate", "value_type",
                            "size_field", "rate_field"}
        if unknown:
            raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
        refs.append(Refinement(name=name, kind=kind,
                               shp=_path(r["shp"], data_dir), size=size,
                               rate=rate or None, value_type=vt,
                               size_field=size_field,
                               rate_field=r.get("rate_field")))

    method = raw.get("method", "fast")
    engine = raw.get("engine", "gmsh")
    if method not in METHODS:
        raise ConfigError(f"method must be one of {METHODS}, got {method!r}")
    if engine not in ENGINES:
        raise ConfigError(f"engine must be one of {ENGINES}, got {engine!r}")

    known = {"name", "data_dir", "out_root", "domain", "dem", "hmin", "hmax",
             "method", "engine", "bnd_factor", "metric_crs",
             "force_metric_crs", "machine", "job", "refinements",
             "description"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)}")

    out_root = raw.get("out_root")
    return Config(
        name=str(raw["name"]), data_dir=data_dir,
        domain=_path(raw["domain"], data_dir), dem=_path(raw["dem"], data_dir),
        hmin=hmin, hmax=hmax, refinements=refs,
        out_root=_path(out_root, data_dir) if out_root else None,
        method=method, engine=engine,
        bnd_factor=float(raw.get("bnd_factor", 1.0)),
        metric_crs=str(raw.get("metric_crs", "EPSG:3338")),
        force_metric_crs=bool(raw.get("force_metric_crs", True)),
        machine=str(raw.get("machine", "gaea_c6")),
        job=dict(raw.get("job") or {}), source=source, raw=raw)


def load(path, data_dir: Optional[Path] = None) -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return parse(raw, source=path.resolve(), data_dir=data_dir)


def load_machine(name_or_path: str) -> dict:
    """A machine file by name (stofs3dak_mesh/machines/<name>.yaml) or path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = MACHINES_DIR / f"{name_or_path}.yaml"
    if not p.is_file():
        avail = sorted(f.stem for f in MACHINES_DIR.glob("*.yaml"))
        raise ConfigError(f"machine {name_or_path!r} not found (available: {avail})")
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
