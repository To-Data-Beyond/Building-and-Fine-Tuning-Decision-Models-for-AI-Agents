"""Run Kev and Laya locally, each in its own virtual environment.

Kev needs transformers >= 5 and Laya needs transformers < 5, so they cannot share
one Python environment. Each model runs in a worker process (dmw/local_worker.py)
inside its own venv; the notebook talks to it with one JSON line per request.

    setup("laya")                       # once: create the venv and install the model package
    with LocalModel("laya") as m:       # starts the worker and loads the model
        m.predict(text, labels, instructions)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VENVS = Path(os.environ.get("DMW_VENVS", Path.home() / ".dmw_venvs"))
WORKER = Path(__file__).with_name("local_worker.py")

# Pinned exactly as in the Jev-OSS-Comparison repo.
REQUIREMENTS = {
    "laya": [
        "laya @ git+https://github.com/NandhaKishorM/laya.git@a4a8921afebfd852bba0000475cfb6ab737a124c",
        "torch>=2.6,<2.9", "transformers>=4.48,<5",
    ],
    "kev-4b": [
        "kev @ git+https://github.com/jaredpalmer/kev.git@fe64b1274ea7f80d4095866df90666abb03e9cf6",
        "torch>=2.6,<2.9", "transformers>=5.17,<6",
    ],
}
REQUIREMENTS["kev-0.8b"] = REQUIREMENTS["kev-4b"]
MLX_EXTRA = "kev[mlx] @ git+https://github.com/jaredpalmer/kev.git@fe64b1274ea7f80d4095866df90666abb03e9cf6"


# torch < 2.9 (pinned above) has wheels for Python 3.9-3.13 only, so the model venvs use
# their own Python, independent of the notebook's Python (which may be 3.14).
MODEL_PYTHON = os.environ.get("DMW_MODEL_PYTHON", "3.12")
SUPPORTED = {(3, 10), (3, 11), (3, 12), (3, 13)}
DONE = ".dmw_installed"   # written only after a successful install


def _python(key: str) -> Path:
    return VENVS / key / "bin" / "python"


def _version(py) -> tuple[int, int] | None:
    try:
        out = subprocess.run([str(py), "-c", "import sys; print(*sys.version_info[:2])"],
                             capture_output=True, text=True, check=True).stdout.split()
        return int(out[0]), int(out[1])
    except Exception:
        return None


def is_ready(key: str) -> bool:
    """True only when the venv finished installing and uses a supported Python."""
    return (VENVS / key / DONE).exists() and _version(_python(key)) in SUPPORTED


def _base_python(python: str | None) -> list[str]:
    """Arguments that create the venv with a Python that torch < 2.9 supports."""
    uv = shutil.which("uv")
    if uv:   # uv downloads the requested Python if it is not installed
        return [uv, "venv", "--python", python or MODEL_PYTHON]
    for cand in ([python] if python else []) + [sys.executable, "python3.12", "python3.13", "python3.11", "python3.10"]:
        exe = shutil.which(cand) or cand
        if _version(exe) in SUPPORTED:
            return [exe, "-m", "venv"]
    raise RuntimeError("The local models need Python 3.10-3.13 (torch < 2.9 has no wheels for newer Python). "
                       "Install uv (`pip install uv`) and rerun: it downloads Python 3.12 for these venvs only.")


def setup(key: str, python: str | None = None, quiet: bool = True) -> Path:
    """Create the venv for `key` and install its pinned packages. Safe to re-run."""
    if key not in REQUIREMENTS:
        raise ValueError(f"Unknown local model {key!r}; choose from {list(REQUIREMENTS)}")
    venv, py = VENVS / key, _python(key)
    if is_ready(key):
        return py
    if venv.exists():
        # A half-finished or incompatible venv from an earlier attempt: rebuild it.
        print(f"Rebuilding {venv} (earlier install was incomplete or used Python {_version(py)}).")
        shutil.rmtree(venv)
    VENVS.mkdir(parents=True, exist_ok=True)
    subprocess.run(_base_python(python) + [str(venv)], check=True)
    print("Model venv Python:", ".".join(map(str, _version(py) or ("?",))))
    reqs = list(REQUIREMENTS[key])
    if key.startswith("kev") and sys.platform == "darwin" and os.uname().machine == "arm64":
        reqs[0] = MLX_EXTRA
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "install", "--python", str(py)] if uv else [str(py), "-m", "pip", "install"]) + (["-q"] if quiet else []) + reqs
    print("Installing:", " ".join(reqs))
    subprocess.run(cmd, check=True)
    (venv / DONE).write_text("ok\n")
    return py


class LocalModel:
    """A Kev or Laya model running in its own process."""

    def __init__(self, key: str, device: str = "auto", load_timeout: float = 900, request_timeout: float = 300):
        if not is_ready(key):
            raise RuntimeError(f"Run dmw.local.setup({key!r}) first.")
        self.key, self.device = key, device
        self.load_timeout, self.request_timeout = load_timeout, request_timeout
        self.proc = None
        self.load_s = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        self.proc = subprocess.Popen([str(_python(self.key)), "-u", str(WORKER), "--model", self.key, "--device", self.device],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
        ready = self._read(self.load_timeout)
        if not ready.get("ok"):
            raise RuntimeError(f"{self.key} failed to load: {ready.get('error')}")
        self.load_s = ready["load_s"]
        self.info = ready
        return ready

    def _read(self, timeout: float) -> dict:
        t0 = time.time()
        while True:
            line = self.proc.stdout.readline()
            if line.startswith("{"):
                return json.loads(line)
            if not line and self.proc.poll() is not None:
                raise RuntimeError(f"{self.key} worker exited with code {self.proc.returncode}")
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{self.key} did not answer within {timeout}s")

    def predict(self, text: str, labels: dict[str, str], instructions: str) -> dict:
        """Return {label, probabilities, latency_ms} or {error, latency_ms} for over-length input."""
        self.proc.stdin.write(json.dumps({"text": text, "labels": labels, "instructions": instructions}) + "\n")
        self.proc.stdin.flush()
        return self._read(self.request_timeout)

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
