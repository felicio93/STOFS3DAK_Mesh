"""akmesh: STOFS-3D-Alaska mesh generation.

  akmesh check  -c configs/m6_hires.yaml [--res 2000] [--out preview]
      inputs, warnings, size-field preview (PNG + GeoTIFF), node estimate;
      no OCSMesh needed
  akmesh submit -c configs/m6_hires.yaml [--dry-run]
      results folder + job.sh for the recipe's machine, then sbatch
  akmesh build  -c config.yaml --out-dir DIR [--nprocs N]
      the mesh itself (what job.sh runs); needs OCSMesh
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigError, load


def _cfg(a):
    return load(a.config, data_dir=a.data_dir)


def cmd_check(a) -> int:
    from .check import report, size_field, write_preview
    cfg = _cfg(a)
    for label, p in [("domain", cfg.domain), ("dem", cfg.dem)] + [
            (r.name, r.shp) for r in cfg.refinements]:
        if not Path(p).is_file():
            print(f"ERROR {label}: {p} not found")
            return 1
    h, transform, info = size_field(cfg, res=a.res)
    print(report(cfg, h, info))
    if not a.no_plot:
        out = Path(a.out) if a.out else Path("preview") / cfg.name
        for f in write_preview(cfg, h, transform, info, out):
            print(f"wrote {f}")
    return 0


def cmd_submit(a) -> int:
    from .slurm import prepare, submit
    cfg = _cfg(a)
    missing = [str(p) for p in [cfg.domain, cfg.dem] + [r.shp for r in cfg.refinements]
               if not Path(p).is_file()]
    if missing:
        print("ERROR: input(s) not found:\n  " + "\n  ".join(missing))
        return 1
    job = prepare(cfg, Path(a.out_dir) if a.out_dir else None)
    print(f"results folder: {job.parent}\njob script:     {job}")
    if a.dry_run:
        print("dry run: not submitted")
        return 0
    print(submit(job))
    return 0


def cmd_build(a) -> int:
    from .build import build
    cfg = _cfg(a)
    build(cfg, Path(a.out_dir), nprocs=a.nprocs, force_raster=a.force_raster)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(prog="akmesh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-c", "--config", required=True, help="mesh recipe (YAML)")
        p.add_argument("--data-dir", type=Path, default=None,
                       help="override the recipe's data_dir")

    p = sub.add_parser("check", help="inputs, warnings, size-field preview")
    common(p)
    p.add_argument("--res", type=float, default=2000.0,
                   help="preview grid spacing in m (default 2000)")
    p.add_argument("--out", default=None, help="preview folder (default preview/<name>)")
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("submit", help="results folder + job script, sbatch")
    common(p)
    p.add_argument("--out-dir", default=None,
                   help="results folder (default <out_root>/<name>_<method>_<time>)")
    p.add_argument("--dry-run", action="store_true", help="write job.sh, do not submit")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("build", help="build the mesh (needs OCSMesh)")
    common(p)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--nprocs", type=int, default=1)
    p.add_argument("--force-raster", action="store_true",
                   help="re-roll the DEM to 0-360 even if the copy looks current")
    p.set_defaults(func=cmd_build)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except (ConfigError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
