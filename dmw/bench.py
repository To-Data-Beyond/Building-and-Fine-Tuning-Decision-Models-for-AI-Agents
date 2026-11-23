"""Saved predictions from the Jev vs. local models benchmark (Jev-OSS-Comparison, Oct 2026).

data/replay/benchmark_local_models.json holds, for SST-2, AG News, BoolQ and the long
IMDb reviews, the gold labels and the predictions of Kev-4B, Laya and Jev on the same
examples, with per-request latency. Example texts are not redistributed: call
attach_texts() to download them from the pinned dataset revisions when you are online.
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import config, metrics

FIELD = {"sst2": "sentence", "ag_news": "text", "imdb_long": "text"}


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads((config.REPLAY / "benchmark_local_models.json").read_text(encoding="utf-8"))


def datasets() -> list[str]:
    return list(load())


def frame(dataset: str):
    """One row per example: id, gold, words, and label / p_top / latency / error per model."""
    import pandas as pd

    d = load()[dataset]
    rows = []
    for e in d["examples"]:
        row = {"id": e["id"], "gold": e["gold"], "words": e["words"]}
        for model, preds in d["predictions"].items():
            p = preds.get(e["id"])
            if p is None:
                continue
            row[f"{model}"] = p["label"]
            row[f"{model}_p"] = max(p["probabilities"].values()) if p.get("probabilities") else None
            row[f"{model}_ms"] = p["latency_ms"]
            row[f"{model}_error"] = (p.get("error") or "")[:80] or None
        rows.append(row)
    return pd.DataFrame(rows)


def table(dataset: str, models=("jev", "kev-4b", "laya")):
    """Accuracy, macro-F1, failures and latency per model on the examples all models share."""
    import pandas as pd

    df = frame(dataset)
    labels = sorted(set(df["gold"]))
    out = []
    for m in models:
        if m not in df:
            continue
        pred = [p if isinstance(p, str) else None for p in df[m].tolist()]
        out.append({"model": m, "n": len(df), "accuracy": round(metrics.accuracy(df["gold"].tolist(), pred), 3),
                    "macro_f1": round(metrics.macro_f1(df["gold"].tolist(), pred, labels), 3),
                    "failures": sum(p is None for p in pred), **metrics.latency(df[f"{m}_ms"].dropna().tolist())})
    return pd.DataFrame(out).set_index("model")


def labels(dataset: str) -> dict:
    return load()[dataset]["examples"][0]["labels"]


def attach_texts(dataset: str, df):
    """Add a `text` column by downloading the pinned dataset revision (needs internet + `datasets`)."""
    from datasets import load_dataset

    src = load()[dataset]["text_source"]
    ds = load_dataset(src["repo"], split=src["split"], revision=src["revision"])
    def text(i):
        row = ds[int(i)]
        if dataset == "boolq":
            return f"Passage: {row['passage']}\n\nQuestion: {row['question']}"
        return row[FIELD[dataset]]
    df = df.copy()
    df["text"] = df["id"].map(text)
    return df
