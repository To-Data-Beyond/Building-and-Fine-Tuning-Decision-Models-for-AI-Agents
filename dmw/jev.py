"""A small Jev client: live calls with an on-disk cache, plus replay of the saved runs.

The request shape is the one used throughout the workshop::

    {"model": "jev-1.13.0",
     "state": <string | list | object>,
     "questions": {<question_id>: {"type": "choice" | "score" | "noul",
                                   "instructions": "...",
                                   "criteria": {...} or [...]}}}
"""
from __future__ import annotations

import hashlib
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import config

CACHE = config.OUTPUTS / "jev_cache.jsonl"


# --------------------------------------------------------------------------- requests
def question(type_: str, instructions: str, criteria: Any) -> dict:
    if type_ not in ("choice", "score", "noul"):
        raise ValueError("type must be choice, score or noul")
    return {"type": type_, "instructions": instructions, "criteria": criteria}


def payload(state: Any, questions: dict, model: str = config.JEV_MODEL) -> dict:
    return {"model": model, "state": state, "questions": questions}


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _cache_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cache_get(k: str) -> dict | None:
    if not CACHE.exists():
        return None
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["key"] == k:
            return row
    return None


def call(body: dict, api_key: str | None = None, use_cache: bool = True, timeout: float = 60) -> dict:
    """POST one request. Returns {"response": ..., "latency_ms": ..., "cached": bool}.

    Responses are cached in outputs/jev_cache.jsonl, so re-running a cell does not
    pay twice. The cached latency is the latency of the original request.
    """
    k = _cache_key(body)
    if use_cache and (hit := _cache_get(k)):
        return {"response": hit["response"], "latency_ms": hit["latency_ms"], "cached": True}
    config.load_env()
    api_key = api_key or config.key("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError("No TYPESAFE_API_KEY. Use replay mode or add the key to .env / Colab secrets.")
    req = urllib.request.Request(
        config.JEV_ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            response = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Jev returned HTTP {e.code}: {detail}") from None
    ms = round((time.perf_counter() - t0) * 1000, 1)
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": k, "response": response, "latency_ms": ms}, ensure_ascii=False) + "\n")
    return {"response": response, "latency_ms": ms, "cached": False}


def ask(state: Any, questions: dict, **kw) -> dict:
    """Convenience wrapper: build the payload, call Jev, return the parsed answers."""
    out = call(payload(state, questions), **kw)
    out["answers"] = {qid: parse(a) for qid, a in out["response"]["answers"].items()}
    return out


# --------------------------------------------------------------------------- parsing
def choice_confidence(probabilities: dict[str, float]) -> float:
    """(p_max - 1/n) / (1 - 1/n): 0 for a uniform distribution, 1 for a certain one."""
    n = len(probabilities)
    p = max(probabilities.values())
    return (p - 1 / n) / (1 - 1 / n)


def expected_score(probabilities: dict[str, float]) -> float:
    """Probability-weighted level for a Score answer."""
    return sum(int(level) * p for level, p in probabilities.items())


def parse(answer: dict) -> dict:
    """Normalize one answer object into {type, value, probabilities, confidence}."""
    t = answer.get("type")
    if t == "choice":
        probs = answer["probabilities"]
        if answer["choice"] not in probs or abs(sum(probs.values()) - 1) > 0.01:
            raise ValueError(f"Invalid choice answer: {answer}")
        return {"type": t, "value": answer["choice"], "probabilities": probs, "confidence": answer.get("confidence")}
    if t == "score":
        return {"type": t, "value": float(answer["score"]), "probabilities": answer.get("probabilities"),
                "confidence": answer.get("confidence"), "legend": answer.get("legend")}
    if t == "noul":
        p = float(answer["noul"])
        if not 0 <= p <= 1:
            raise ValueError(f"Invalid noul: {p}")
        return {"type": t, "value": p, "probabilities": {"true": p, "false": 1 - p}, "confidence": None}
    raise ValueError(f"Unknown answer type: {t}")


# --------------------------------------------------------------------------- replay
def replay(use_case: str):
    """Saved runs from the article (Jev and GPT-4.1 mini, 3 repeats each) as a DataFrame."""
    import pandas as pd

    path = config.REPLAY / f"{use_case}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(rows)


def replay_answer(use_case: str, case: str, repeat: int = 1, **match) -> dict:
    """The raw Jev `answers` object for one saved case, exactly as the API returned it."""
    path = config.REPLAY / f"{use_case}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["provider"] != "jev" or r.get("repeat") != repeat:
            continue
        if r.get("case") == case and all(r.get(k) == v for k, v in match.items()):
            return {"answers": r["jev_answers"], "latency_ms": r["latency_ms"], "model": r["model"]}
    raise KeyError(f"No saved Jev answer for {use_case}/{case} repeat {repeat} {match}")
