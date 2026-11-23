"""Automatic checkers for the five exercises. Each prints PASS/FAIL lines and returns True/False."""
from __future__ import annotations

import math
import re

from . import config, fixtures, jev


def _report(results: list[tuple[bool, str]]) -> bool:
    for ok, msg in results:
        print(("PASS  " if ok else "FAIL  ") + msg)
    ok = all(r[0] for r in results)
    print("\nAll checks passed." if ok else "\nFix the FAIL lines above and run the checker again.")
    return ok


# ------------------------------------------------------------------ Exercise 1
EXPECT = {
    "guardrails": {"type": "noul", "fields": {"source", "content"}},
    "gating": {"type": "choice", "fields": {"user_request", "proposed_call", "trusted_permissions"}},
    "reranking": {"type": "score", "fields": {"question", "passage"}},
}
DATA_RULE = re.compile(r"\b(data|evidence|not (obey|follow|execute)|do not (obey|follow|execute)|ignore (any )?instructions|treat .{0,40} as)\b", re.I)
PRECEDENCE = re.compile(r"\b(precedence|takes priority|priority|if (more than one|several|multiple)|otherwise|in order)\b", re.I)
FALLBACK = {"deny", "review", "other", "unclear", "unknown", "needs_review", "ask"}


def _structure(task: str, d: dict) -> list[tuple[bool, str]]:
    e, out = EXPECT[task], []
    hint = "" if d.get("type") == e["type"] else "  (is the answer a category, an ordered degree, or the truth of one condition?)"
    out.append((d.get("type") == e["type"], f"{task}: question type fits the decision (got {d.get('type')!r}){hint}"))
    words = len(str(d.get("instructions", "")).split())
    out.append((words >= 15, f"{task}: instructions say what to judge in at least 15 words (got {words})"))
    out.append((bool(DATA_RULE.search(str(d.get("instructions", "")))), f"{task}: instructions say that text in the state is data, not instructions"))
    fields = set(d.get("state_fields", []))
    missing = e["fields"] - fields
    out.append((not missing, f"{task}: state has the evidence the decision needs {sorted(e['fields'])}" + (f", missing {sorted(missing)}" if missing else "")))
    crit = d.get("criteria")
    if e["type"] == "noul":
        ok = isinstance(crit, dict) and set(crit) == {"true", "false"} and all(len(str(v).split()) >= 6 for v in crit.values())
        out.append((ok, f"{task}: criteria define both 'true' and 'false' in at least 6 words each"))
    elif e["type"] == "choice":
        ok = isinstance(crit, dict) and len(crit) >= 3 and all(len(str(v).split()) >= 4 for v in crit.values())
        out.append((ok, f"{task}: at least 3 options, each described in at least 4 words"))
        out.append((isinstance(crit, dict) and bool(FALLBACK & {k.lower() for k in crit}), f"{task}: has a fallback option (one of {sorted(FALLBACK)})"))
        out.append((bool(PRECEDENCE.search(str(d.get("instructions", "")))), f"{task}: instructions state precedence when two options apply"))
    else:
        ok = isinstance(crit, list) and len(crit) >= 3 and all(len(str(v).split()) >= 4 for v in crit)
        out.append((ok, f"{task}: at least 3 ordered levels, lowest first, each described in at least 4 words"))
    return out


def _live_cases(task: str) -> list[tuple[dict, object]]:
    if task == "reranking":
        pairs = fixtures.reranking_pairs()[:8]
        return [({"question": p["question"], "passage": p["text"]}, p["grade"]) for p in pairs]
    spec = fixtures.get(task)
    return [(c["state"], c["expected"]) for c in spec["cases"][:8]]


def check_design(designs: dict, live: bool | None = None) -> bool:
    """Structural checks always; with a Jev key (or live=True) also 8 labeled cases per task."""
    results = []
    for task in EXPECT:
        if task not in designs:
            results.append((False, f"{task}: no design found"))
            continue
        results += _structure(task, designs[task])
    live = config.mode() == "live" if live is None else live
    if live:
        for task, d in designs.items():
            if task not in EXPECT:
                continue
            q = {"decision": {"type": d["type"], "instructions": d["instructions"], "criteria": d["criteria"]}}
            cases = _live_cases(task)
            if d["type"] == "choice" and not all(want in d["criteria"] for _, want in cases):
                results.append((True, f"{task}: live check skipped: your option names differ from the reference labels "
                                      f"{sorted({w for _, w in cases})}, so answers cannot be compared"))
                continue
            hits = 0
            for state, want in cases:
                a = jev.parse(jev.call(jev.payload(state, q))["response"]["answers"]["decision"])
                if d["type"] == "noul":
                    hits += (a["value"] >= 0.5) == bool(want)
                elif d["type"] == "score":
                    hits += abs(a["value"] - want) < 0.5
                else:
                    hits += a["value"] == want
            results.append((hits >= 7, f"{task}: live check {hits}/8 cases match the reference answer (need 7)"))
    else:
        print("(Replay mode: structural checks only. Add TYPESAFE_API_KEY to also run your design on 8 labeled cases.)\n")
    return _report(results)


# ------------------------------------------------------------------ Exercise 2
def check_triage_exercise(attack_emails: list[dict], results=None, observation: str = "") -> bool:
    r = []
    r.append((len(attack_emails) >= 3, f"wrote at least 3 tricky emails (got {len(attack_emails)})"))
    r.append((all({"subject", "body", "expected"} <= set(e) for e in attack_emails), "each email has subject, body and your expected queue"))
    r.append((all(e.get("expected") in ("reply_now", "later", "archive") for e in attack_emails), "expected queue is reply_now, later or archive"))
    r.append((all(str(e.get("subject", "")).strip() and str(e.get("body", "")).strip() for e in attack_emails), "every email has a subject and a body"))
    ran = results is not None and len(results) > 0 and ({"jev", "gpt-4.1-mini"} & set(getattr(results, "columns", [])))
    r.append((bool(ran), "the emails were run on at least one model (needs an API key)"))
    r.append((len(str(observation).split()) >= 12, "observation compares the models in at least 12 words"))
    return _report(r)


# ------------------------------------------------------------------ Exercise 3
def check_position_bias(flip_rate: float, given_acc: float, reversed_acc: float, averaged_acc: float,
                        reference: dict) -> bool:
    r = []
    close = lambda a, b: a is not None and b is not None and math.isclose(a, b, abs_tol=1e-6)
    r.append((close(flip_rate, reference["flip_rate"]), f"flip rate {flip_rate} matches the reference {reference['flip_rate']:.4f}"))
    r.append((close(given_acc, reference["given_acc"]) and close(reversed_acc, reference["reversed_acc"]), "accuracy in the given and reversed orders matches"))
    r.append((close(averaged_acc, reference["averaged_acc"]), f"averaged accuracy {averaged_acc} matches the reference {reference['averaged_acc']:.4f}"))
    return _report(r)


# ------------------------------------------------------------------ Exercise 4
def validate_dataset(domain: dict, train: list[dict], valid: list[dict], test_texts: list[str], rows: list[dict]) -> bool:
    from collections import Counter

    from .datagen import near_duplicates
    from .prompt import LETTERS

    r = []
    labels = list(domain["choices"])
    r.append((len(labels) >= 3, f"at least 3 options (got {len(labels)})"))
    r.append((all(len(str(v).split()) >= 4 for v in domain["choices"].values()), "every option has a description of at least 4 words"))
    per = {l: len(domain["templates"].get(l, [])) for l in labels}
    r.append((all(n >= 4 for n in per.values()), f"at least 4 templates per label {per}"))
    vl = {s["expected"] for s in valid}
    r.append((set(labels) <= vl, "every label has at least one validation-only template"))
    leak = {s["template"] for s in train} & {s["template"] for s in valid}
    r.append((not leak, "no template appears in both train and validation" + (f" (shared: {sorted(leak)[:3]})" if leak else "")))
    dupes = near_duplicates([s["text"] for s in train + valid], test_texts)
    r.append((not dupes, "no training or validation text is a near-duplicate of a test case" + (f" (e.g. {sorted(dupes)[:2]})" if dupes else "")))
    c = Counter(s["expected"] for s in train)
    bal = (max(c.values()) / max(1, min(c.values()))) if c else float("inf")
    r.append((len(c) == len(labels) and bal <= 3, f"training labels are roughly balanced (max/min = {bal:.1f}, need ≤ 3)"))
    r.append((all(row["answer"] in LETTERS for row in rows), "every row ends in a single option letter"))
    return _report(r)


# ------------------------------------------------------------------ Exercise 5
def check_report(report: dict, recomputed: dict) -> bool:
    r = []
    r.append((report.get("decision") in ("deploy", "shadow", "do_not_deploy"), "decision is deploy, shadow or do_not_deploy"))
    t = report.get("threshold")
    r.append((isinstance(t, (int, float)) and 0 <= t <= 1, "a confidence threshold between 0 and 1 is chosen"))
    for k in ("accuracy", "macro_f1", "accuracy_not_flagged"):
        mine, ref = report.get(k), recomputed.get(k)
        ok = isinstance(mine, (int, float)) and ref is not None and (math.isnan(ref) or abs(mine - ref) < 0.01)
        r.append((ok, f"{k} matches the recomputed value ({ref:.3f})" if ref is not None else f"{k} reported"))
    dis = report.get("disagreements", [])
    r.append((len(dis) >= 3 and all(len(str(d.get("likely_cause", "")).split()) >= 3 for d in dis), "three disagreements, each with a likely cause"))
    r.append((len(str(report.get("missing_evidence", "")).split()) >= 8, "missing evidence is named in at least 8 words"))
    return _report(r)


# ------------------------------------------------------------------ reference implementations used by the checkers
def position_bias_reference(scorer, cases: list[dict]) -> dict:
    """Given order, reversed order and averaged accuracy, plus the flip rate, computed with dmw.scorer."""
    given, rev, avg = [], [], []
    for c in cases:
        order = list(c["choices"])
        kw = dict(question=c["question"], descriptions=c["choices"])
        given.append(scorer.score(c["context"], order, orders=[order], **kw).choice)
        rev.append(scorer.score(c["context"], order, orders=[order[::-1]], **kw).choice)
        avg.append(scorer.score(c["context"], order, orders=[order, order[::-1]], **kw).choice)
    gold = [c["expected"] for c in cases]
    acc = lambda p: sum(a == b for a, b in zip(p, gold)) / len(gold)
    return {"flip_rate": sum(a != b for a, b in zip(given, rev)) / len(cases),
            "given_acc": acc(given), "reversed_acc": acc(rev), "averaged_acc": acc(avg)}


def recompute(df, model: str) -> dict:
    """Reference metrics for Exercise 5 from a predictions DataFrame (columns: model, expected, choice, review)."""
    from . import metrics

    d = df[df.model == model]
    nf = d[~d.review]
    return {"accuracy": metrics.accuracy(d.expected.tolist(), d.choice.tolist()),
            "macro_f1": metrics.macro_f1(d.expected.tolist(), d.choice.tolist()),
            "accuracy_not_flagged": metrics.accuracy(nf.expected.tolist(), nf.choice.tolist()) if len(nf) else float("nan")}
