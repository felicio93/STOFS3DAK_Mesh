"""Write and submit the batch job of one mesh.

The job runs in its own results folder ``<out_root>/<name>_<method>_<time>``,
which gets a copy of the recipe (config.yaml), the job script (job.sh), the
SLURM logs, the mesh, shapes_360/ and the summary JSON. The package is put
on PYTHONPATH from this checkout, so it does not have to be installed in the
job's conda env (OCSMesh does).
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from .config import ConfigError, load_machine

TEMPLATE = Path(__file__).parent / "templates" / "job.sh"
PKG_ROOT = Path(__file__).resolve().parent.parent


def _sbatch_lines(opts: dict) -> str:
    lines = []
    for k, v in opts.items():
        if v is None or v is False or v == "":
            continue
        lines.append(f"#SBATCH --{k}" if v is True else f"#SBATCH --{k}={v}")
    return "\n".join(lines)


def render_job(name: str, out_dir: Path, machine: dict, job: dict,
               command: str, log: str = "akmesh") -> str:
    """Job script running ``python -m stofs3dak_mesh.cli <command>`` (with
    {nprocs} in ``command`` replaced by cpus-per-task)."""
    sb = dict(machine.get("sbatch") or {})
    sb.update(job or {})
    cpus = int(sb.get("cpus-per-task", 1))
    paths = [str(PKG_ROOT)]
    if machine.get("ocsmesh_path"):
        paths.insert(0, str(machine["ocsmesh_path"]))
    launcher = str(machine.get("launcher", "")).format(cpus=cpus)
    return TEMPLATE.read_text().format(
        created=datetime.now().strftime("%Y-%m-%d %H:%M"), name=name,
        sbatch_lines=_sbatch_lines(sb), out_dir=out_dir, log=log,
        setup=str(machine.get("setup", "")).rstrip(),
        pythonpath=":".join(paths), launcher=launcher,
        command=command.replace("{nprocs}", str(cpus)))


def render(cfg, out_dir: Path, machine: dict) -> str:
    return render_job(cfg.name, out_dir, machine, cfg.job,
                      f'build -c "{out_dir}/config.yaml" --out-dir "{out_dir}" '
                      f'--nprocs {{nprocs}}')


def prepare(cfg, out_dir: Path = None) -> Path:
    """Results folder with config.yaml and job.sh; returns job.sh."""
    machine = load_machine(cfg.machine)
    if out_dir is None:
        root = cfg.out_root or machine.get("out_root")
        if not root:
            raise ConfigError("no out_root in the recipe or the machine file")
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = Path(root) / f"{cfg.name}_{cfg.method}_{stamp}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    raw = dict(cfg.raw)
    raw["data_dir"] = str(cfg.data_dir)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as fh:
        fh.write(f"# copy of {cfg.source}\n")
        yaml.safe_dump(raw, fh, sort_keys=False)
    job = out_dir / "job.sh"
    job.write_text(render(cfg, out_dir, machine))
    job.chmod(0o755)
    return job


def submit(job: Path) -> str:
    r = subprocess.run(["sbatch", str(job)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()
