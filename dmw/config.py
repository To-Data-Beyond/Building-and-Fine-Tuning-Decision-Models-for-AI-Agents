"""Paths, keys and run mode shared by every notebook."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPLAY = DATA / "replay"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

JEV_MODEL = "jev-1.13.0"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
LLM_MODEL = "gpt-4.1-mini-2025-04-14"
LLM_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def load_env(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from .env without overriding variables already set.

    In Google Colab, keys stored under the key icon (Secrets) are read too.
    """
    env = Path(path) if path else ROOT / ".env"
    if env.is_file():
        for raw in env.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)
    try:  # Colab secrets
        from google.colab import userdata  # type: ignore

        for key in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "HF_USERNAME"):
            if not os.environ.get(key):
                try:
                    value = userdata.get(key)
                except Exception:
                    value = None
                if value:
                    os.environ[key] = value
    except Exception:
        pass


def key(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value or None


def mode(requested: str = "auto") -> str:
    """Return "live" when a Jev key is present (or requested), otherwise "replay"."""
    load_env()
    if requested == "replay":
        return "replay"
    if requested == "live":
        if not key("TYPESAFE_API_KEY"):
            raise RuntimeError("MODE='live' needs TYPESAFE_API_KEY in .env or Colab secrets.")
        return "live"
    return "live" if key("TYPESAFE_API_KEY") else "replay"


def device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False
