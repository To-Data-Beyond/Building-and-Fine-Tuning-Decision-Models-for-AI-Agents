"""Worker process for one local decision model (runs inside that model's own venv).

Protocol: prints one JSON line when loaded ({"ok": true, "load_s": ...}), then for each
request line {"text", "labels", "instructions"} prints one JSON line with
{"label", "probabilities", "latency_ms"} or {"error", "latency_ms"}.

Adapters follow jev_compare/worker.py from the Jev-OSS-Comparison repo: the full
input is sent, and over-length input is rejected instead of truncated.
"""
import argparse
import json
import math
import sys
import time

SPECS = {
    "kev-4b": ("jaredpalmer/kev-4b", "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"),
    "kev-0.8b": ("jaredpalmer/kev-0.8b", "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"),
    "laya": ("convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"),
}


class TooLong(Exception):
    pass


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(key: str, device: str):
    import torch

    checkpoint, revision = SPECS[key]
    if key.startswith("kev-"):
        from kev.checkpoint import Checkpoint, LoadOptions

        ckpt = Checkpoint(f"{checkpoint}@{revision}")
        backend = "mlx" if device == "mlx" else "torch"
        if device == "cpu":
            dtype = torch.float32
        elif device == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = torch.float16  # e.g. Colab T4: untested upstream, falls back to fp16
        else:
            dtype = torch.bfloat16
        tok, model = ckpt.load("mps" if device == "mlx" else device,
                               LoadOptions(backend=backend, dtype=dtype, attn="sdpa", fused=False, cuda_graphs=False))
        return {"tok": tok, "model": model}
    if key == "laya":
        from laya import Agent

        return {"model": Agent(checkpoint, revision=revision, device="cpu", backend="eager")}
    raise ValueError(key)


def distribution(raw: dict, labels: dict) -> dict:
    probs = {k: float(raw[k]) for k in labels}
    total = sum(probs.values())
    if not all(math.isfinite(p) for p in probs.values()) or not 0.98 <= total <= 1.02:
        raise ValueError(f"Not a single-label distribution (sum={total})")
    return {k: v / total for k, v in probs.items()}


def predict(key: str, m: dict, text: str, labels: dict, instructions: str) -> dict:
    import torch

    questions = {"decision": {"type": "choice", "instructions": instructions, "criteria": labels}}
    with torch.inference_mode():
        if key.startswith("kev-"):
            from kev.api import SystemOneRequest, to_record
            from kev.model import ContextOverflow, admit

            record, _ = to_record(SystemOneRequest(state=text, questions=questions))
            try:
                encoded = admit(m["model"], m["tok"], record, truncate=False)
            except ContextOverflow as exc:
                raise TooLong(str(exc)) from exc
            scores = m["model"].probs(encoded)[0].tolist()
            raw = dict(zip(labels, scores))
        else:
            response = m["model"].predict(text, questions)
            usage = response.get("usage", {})
            if usage.get("truncated") or usage.get("options"):
                raise TooLong("Laya would have truncated the input (512-token limit); rejected instead.")
            raw = response["answers"]["decision"]["probabilities"]
    probs = distribution(raw, labels)
    return {"label": max(probs, key=probs.get), "probabilities": probs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    try:
        device = "cpu" if args.model == "laya" else pick_device(args.device)
        t0 = time.perf_counter()
        m = load(args.model, device)
        # one unscored warm-up request
        predict(args.model, m, "Warm-up.", {"a": "first option", "b": "second option"}, "Pick one.")
        print(json.dumps({"ok": True, "model": args.model, "device": device,
                          "load_s": round(time.perf_counter() - t0, 1)}), flush=True)
    except Exception as exc:  # report and exit
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        t0 = time.perf_counter()
        try:
            out = predict(args.model, m, req["text"], req["labels"], req["instructions"])
        except TooLong as exc:
            out = {"label": None, "error": f"too_long: {exc}"}
        except Exception as exc:
            out = {"label": None, "error": f"{type(exc).__name__}: {exc}"}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
