"""Metrics used in the labs and the benchmark notebooks. Plain Python + pandas."""
from __future__ import annotations

import statistics


def accuracy(gold: list, pred: list) -> float:
    """Correct / total. A missing prediction (None) counts as wrong."""
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else float("nan")


def macro_f1(gold: list, pred: list, labels: list | None = None) -> float:
    """F1 per label, averaged with equal weight. Labels default to the gold labels seen."""
    labels = labels or sorted(set(gold))
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def confusion(gold: list, pred: list, labels: list | None = None):
    """Rows = gold label, columns = predicted label (failures shown as '(failed)')."""
    import pandas as pd

    pred = ["(failed)" if p is None else p for p in pred]
    labels = labels or sorted(set(gold))
    cols = labels + sorted(set(pred) - set(labels))
    return pd.crosstab(pd.Series(gold, name="gold"), pd.Series(pred, name="predicted")).reindex(
        index=labels, columns=cols, fill_value=0)


def latency(ms: list[float]) -> dict:
    xs = sorted(x for x in ms if x is not None)
    if not xs:
        return {"p50": None, "p90": None, "p95": None}
    q = lambda f: xs[min(len(xs) - 1, int(round(f * (len(xs) - 1))))]
    return {"p50": round(statistics.median(xs), 1), "p90": round(q(0.9), 1), "p95": round(q(0.95), 1)}


def coverage_curve(confidence: list[float], correct: list[bool], thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)):
    """For each threshold: how many answers would be automatic, and how many of those are wrong."""
    import pandas as pd

    rows = []
    n = len(confidence)
    for t in thresholds:
        auto = [c for conf, c in zip(confidence, correct) if conf is not None and conf >= t]
        rows.append({"threshold": t, "automatic": len(auto), "review": n - len(auto),
                     "coverage": len(auto) / n if n else float("nan"),
                     "automatic_errors": sum(not c for c in auto),
                     "accuracy_when_automatic": (sum(auto) / len(auto)) if auto else float("nan")})
    return pd.DataFrame(rows)


def summarize(rows: list[dict], group_key: str = "group") -> dict:
    """Per-group and overall accuracy, macro-F1 and review rate for scorer outputs."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get(group_key) or "all", []).append(r)
    out = {}
    for g, rs in list(groups.items()) + [("overall", rows)]:
        gold = [r["expected"] for r in rs]
        pred = [r["choice"] for r in rs]
        out[g] = {"n": len(rs), "accuracy": accuracy(gold, pred), "macro_f1": macro_f1(gold, pred),
                  "review_rate": sum(r.get("review", False) for r in rs) / len(rs),
                  "accuracy_not_flagged": accuracy([r["expected"] for r in rs if not r.get("review")],
                                                   [r["choice"] for r in rs if not r.get("review")])
                  if any(not r.get("review") for r in rs) else float("nan"),
                  "errors_not_flagged": sum(1 for r in rs if not r.get("review") and r["choice"] != r["expected"])}
    return out


def summary_table(results: dict):
    import pandas as pd

    df = pd.DataFrame(results).T
    for c in ("accuracy", "macro_f1", "review_rate", "accuracy_not_flagged"):
        if c in df:
            df[c] = df[c].astype(float).round(3)
    return df
