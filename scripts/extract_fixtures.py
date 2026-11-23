"""One-off: pull fixtures (inputs, policies, reference labels) out of the article scripts."""
import importlib.util, json, sys, types
from pathlib import Path
src = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.modules.setdefault("certifi", types.SimpleNamespace(where=lambda: ""))
def load(name):
    spec = importlib.util.spec_from_file_location(name, src / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
fx = {}
m = load("model_routing")
fx["routing"] = {"instructions": m.INSTRUCTIONS, "criteria": m.OPTIONS, "question_id": "route", "type": "choice",
                 "cases": [{"id": c, "state": {"request": t}, "expected": e} for c, t, e in m.CASES]}
m = load("inbox_triage")
cases = [{"id": c, "state": st, "expected": e} for c, st, e in m.CASES]
fx["triage"] = {"instructions": m.INSTRUCTIONS, "criteria": m.OPTIONS, "question_id": "route", "type": "choice", "now_utc": m.NOW, "cases": cases}
m = load("guardrails")
fx["guardrails"] = {"instructions": m.INSTRUCTIONS, "criteria": m.CRITERIA, "question_id": "is_injection", "type": "noul",
                    "cases": [{"id": c, "state": {"source": s, "content": t}, "expected": e} for c, s, t, e in m.CASES]}
m = load("tool_call_gating")
cases = [{"id": c, "state": st, "expected": e} for c, st, e in m.CASES]
fx["gating"] = {"instructions": m.INSTRUCTIONS, "criteria": m.OPTIONS, "question_id": "decision", "type": "choice", "cases": cases}
m = load("reranking")
fx["reranking"] = {"instructions": m.INSTRUCTIONS, "criteria": m.CRITERIA, "question_id": "relevance", "type": "score",
                   "queries": m.QUERIES}
m = load("bulk_labeling")
fx["labeling"] = {"instructions": m.INSTRUCTIONS, "criteria": m.OPTIONS, "question_id": "label", "type": "choice",
                  "cases": [{"id": c["id"], "state": {"ticket": c["ticket"]}, "expected": c["reference_label"]} for c in m.CASES]}
out.write_text(json.dumps(fx, indent=1, ensure_ascii=False))
print({k: len(v.get("cases", v.get("queries", []))) for k, v in fx.items()})
print(json.dumps(fx["triage"]["cases"][0])[:400]); print(json.dumps(fx["gating"]["cases"][0])[:600])
