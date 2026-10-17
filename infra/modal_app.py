"""The Modal app: an image that builds the engine, a results Volume, one job function.

PLAN.md §14.1, Phase 3.3. Driven by `ttr cloud …` (`ticket_to_ride/cli/cmd_cloud.py`), which
runs this app ephemerally -- nothing is deployed -- and needs the `cloud` dependency group
(`uv sync --group cloud`). Modal reads its credentials from `~/.modal.toml` on the client;
nothing here touches them, and the image holds only the repository's own sources.

## The image

Debian slim with Modal's Python 3.14, a pinned Rust toolchain and a pinned uv. The project
goes in as two layers, so a Python-only edit does not rebuild the engine:

1. everything the extension is built from -- the Cargo workspace, the crates, the Python
   workspace manifest and lockfile -- then `uv sync --no-install-project`, which builds
   `ttr_rust` in the release profile (fat LTO) exactly as a local `uv sync` does;
2. the `ticket_to_ride` package, then `uv sync`, which finds `ttr_rust` current (its cache
   keys are unchanged) and installs only the package.

Modal keys each layer on its inputs, so a layer rebuilds only when something it copies
changes.

The project lives in its own venv (`/opt/ttr/.venv`), not the interpreter the function runs
on: `uv sync` is exact, and syncing into Modal's interpreter would uninstall Modal's client.

## Jobs

`run(argv, job)` runs `ttr <argv>` in `/runs/<job>/` on the `ttr-runs` Volume, so every
job writes its **own** files -- an arena its own `results.db` -- and `ttr cloud pull` merges
them into the local store. SQLite on a shared volume with concurrent writers is a
corruption risk, not a shortcut (PLAN.md §14.1). Next to the command's output each job
records where it ran (`job.json`): the CPUs the container saw, its CPU quota, and the
thread count the engine chose, which PLAN.md asks to be checked inside a container. A job
gets its cores as request *and* limit, and sets `TTR_THREADS` to use all of them.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
PROJECT = "/opt/ttr"
VENV = f"{PROJECT}/.venv"
RUNS = "/runs"

RUST_TOOLCHAIN = "1.97.1"
UV_VERSION = "0.11.17"

#: The cores a job gets unless `ttr cloud … --cpu` says otherwise, as request *and* limit.
#: A bare `cpu=` is only a request: the container may burst above it and shows the whole
#: host slice (24 CPUs, measured), which makes both cost and the engine's thread count
#: depend on the neighbours. Capped, gVisor shows the container exactly its limit.
DEFAULT_CPU = 8.0
DEFAULT_MEMORY_MIB = 4096
TIMEOUT_SECONDS = 2 * 60 * 60


def _build_outputs(path: Path) -> bool:
    """Excluded from the image: build products and caches, never sources."""
    return any(part in {"target", "__pycache__", ".venv"} for part in path.parts) or (
        path.suffix in {".so", ".pyc"}
    )


image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("build-essential", "curl", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
        f" --profile minimal --default-toolchain {RUST_TOOLCHAIN}",
        f"pip install uv=={UV_VERSION}",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "UV_PROJECT_ENVIRONMENT": VENV,
            "UV_PYTHON": "python3.14",
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_LINK_MODE": "copy",
            "CARGO_TERM_COLOR": "never",
        }
    )
    # Layer 1: the engine.
    .add_local_file(REPO / "pyproject.toml", f"{PROJECT}/pyproject.toml", copy=True)
    .add_local_file(REPO / "uv.lock", f"{PROJECT}/uv.lock", copy=True)
    .add_local_file(REPO / "Cargo.toml", f"{PROJECT}/Cargo.toml", copy=True)
    .add_local_file(REPO / "Cargo.lock", f"{PROJECT}/Cargo.lock", copy=True)
    .add_local_dir(REPO / "crates", f"{PROJECT}/crates", copy=True, ignore=_build_outputs)
    .run_commands(f"cd {PROJECT} && uv sync --frozen --no-dev --no-install-project")
    # Layer 2: the Python package.
    .add_local_file(REPO / "README.md", f"{PROJECT}/README.md", copy=True)
    .add_local_dir(
        REPO / "ticket_to_ride", f"{PROJECT}/ticket_to_ride", copy=True, ignore=_build_outputs
    )
    .run_commands(f"cd {PROJECT} && uv sync --frozen --no-dev")
)

app = modal.App("ttr", image=image)
runs = modal.Volume.from_name("ttr-runs", create_if_missing=True)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _job_env() -> dict[str, str]:
    """The job owns its container, so the engine uses every CPU the container shows -- with
    request = limit, exactly the cores reserved -- instead of its desktop default (all but
    two)."""
    return {**os.environ, "TTR_THREADS": str(os.process_cpu_count() or 1)}


def _where() -> dict[str, object]:
    """The machine as the job saw it -- for the record, and for PLAN.md's thread check."""
    threads = subprocess.run(
        [f"{VENV}/bin/python", "-c", "import ttr_rust; print(ttr_rust.performance_threads())"],
        capture_output=True,
        text=True,
        check=True,
        env=_job_env(),
    )
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in (_read("/proc/cpuinfo") or "").splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        "os_cpu_count": os.cpu_count(),
        "process_cpu_count": os.process_cpu_count(),
        "cgroup_cpu_max": _read("/sys/fs/cgroup/cpu.max"),
        "performance_threads": int(threads.stdout),
    }


@app.function(
    volumes={RUNS: runs},
    cpu=(DEFAULT_CPU, DEFAULT_CPU),
    memory=DEFAULT_MEMORY_MIB,
    timeout=TIMEOUT_SECONDS,
)
def run(argv: list[str], job: str) -> dict[str, object]:
    """`ttr <argv>` in the job's own directory on the Volume; output and provenance saved."""
    jobdir = Path(RUNS) / job
    jobdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    proc = subprocess.run(
        [f"{VENV}/bin/ttr", *argv],
        cwd=jobdir,
        capture_output=True,
        text=True,
        check=False,  # a failing command is a result to record, not an exception
        env=_job_env(),
    )
    record: dict[str, object] = {
        "job": job,
        "argv": argv,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 3),
        "started_unix": round(started, 3),
        "where": _where(),
    }
    (jobdir / "stdout.txt").write_text(proc.stdout)
    (jobdir / "stderr.txt").write_text(proc.stderr)
    (jobdir / "job.json").write_text(json.dumps(record, indent=2) + "\n")
    runs.commit()
    return {**record, "stdout": proc.stdout, "stderr": proc.stderr}
