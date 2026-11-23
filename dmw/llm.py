"""Structured-output LLM baseline (OpenAI Chat Completions with a strict JSON schema).

Optional: only used in live mode when OPENAI_API_KEY is set. The prompt mirrors the
article scripts: the same instructions and criteria, the state as JSON, and a
schema that restricts the answer to the allowed values. No explanation is requested.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import config
from .jev import _ssl_context


def _post(body: dict, api_key: str, timeout: float = 60) -> tuple[dict, float]:
    req = urllib.request.Request(
        config.LLM_ENDPOINT, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            raw = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:
            detail = detail[:300]
        hint = " Check OPENAI_API_KEY (an older key set in your shell overrides .env)." if e.code == 401 else ""
        raise RuntimeError(f"OpenAI returned HTTP {e.code}: {detail}{hint}") from None
    return raw, round((time.perf_counter() - t0) * 1000, 1)


def decide(state, field: str, instructions: str, criteria, kind: str = "choice", model: str = config.LLM_MODEL) -> dict:
    """Return {"value": ..., "latency_ms": ...} from the LLM baseline."""
    config.load_env()
    api_key = config.key("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY: the LLM baseline is only available in live mode with a key.")
    if kind == "choice":
        prop = {"type": "string", "enum": list(criteria)}
    elif kind == "noul":
        prop = {"type": "boolean"}
    elif kind == "score":
        prop = {"type": "integer", "enum": list(range(len(criteria)))}
    else:
        raise ValueError(kind)
    crit = json.dumps(dict(enumerate(criteria)) if isinstance(criteria, list) else criteria)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions + "\nCriteria:\n" + crit},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": {
            "type": "object", "properties": {field: prop}, "required": [field], "additionalProperties": False}}},
        "max_completion_tokens": 128, "store": False,
    }
    raw, ms = _post(body, api_key)
    value = json.loads(raw["choices"][0]["message"]["content"])[field]
    return {"value": value, "latency_ms": ms}
