"""Build data/replay/*.jsonl from the saved article runs and benchmark runs.

Keeps only what the notebooks need: inputs are in data/fixtures.json; here we keep
each provider's parsed answer, latency, estimated cost and, for Jev, the full
`answers` object exactly as the API returned it. Response IDs are dropped.
"""
import json, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

USE_CASES = {
    "routing": "results/20261006T204521.801054Z/routing_results.jsonl",
    "triage": "triage_results/20261008T190734.119991Z/triage_results.jsonl",
    "guardrails": "guardrails_results/20261006T215159.178756Z/guardrails_results.jsonl",
    "gating": "gating_results/20261006T233411.241937Z/gating_results.jsonl",
    "reranking": "reranking_results/20261008T203307.655849Z/reranking_results.jsonl",
    "evaluation": "evaluation_results/20261008T223230.791300Z/evaluation_results.jsonl",
    "labeling": "labeling_results/20261008T234632.151934Z/labeling_results.jsonl",
    "control": "control_results/20261008T235954.436393Z/control_results.jsonl",
}
DROP = {"raw_response", "usage", "returned_model", "requested_model"}

for name, rel in USE_CASES.items():
    rows = []
    for line in (src / "Jev Use Cases" / rel).read_text().splitlines():
        r = json.loads(line)
        rec = {k: v for k, v in r.items() if k not in DROP}
        rec["model"] = r.get("returned_model") or r.get("requested_model")
        raw = r.get("raw_response") or {}
        if r["provider"] == "jev" and isinstance(raw, dict):
            rec["jev_answers"] = raw.get("answers")
            rec["usage"] = raw.get("usage")
        elif r["provider"] == "openai" and isinstance(raw, dict) and raw.get("choices"):
            rec["llm_content"] = raw["choices"][0]["message"]["content"]
            rec["usage"] = {"input_tokens": raw["usage"]["prompt_tokens"], "output_tokens": raw["usage"]["completion_tokens"]}
        rows.append(rec)
    (out / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(name, len(rows))

BENCH = {
    "kev-4b": {"sst2": "kev-4b_vs_jev/sst2/20261006T094504Z-952701fe.json", "ag_news": "kev-4b_vs_jev/ag-news/20261006T095231Z-d4acae49.json",
               "boolq": "kev-4b_vs_jev/boolq/20261006T101822Z-5020abfb.json", "imdb_long": "kev-4b_vs_jev/imdb-long/20261012T090744Z-b3f34cec.json"},
    "laya": {"sst2": "laya_vs_jev/sst2/20261006T090827Z-e0cb30a5.json", "ag_news": "laya_vs_jev/ag-news/20261006T091754Z-1d3eb359.json",
             "boolq": "laya_vs_jev/boolq/20261006T092036Z-c565e297.json", "imdb_long": "laya_vs_jev/imdb-long/20261012T100252Z-f49296bc.json"},
}
bench = {}
for model, files in BENCH.items():
    for ds, rel in files.items():
        d = json.loads((src / "Jev-OSS-Comparison/results" / rel).read_text())
        ex = bench.setdefault(ds, {"dataset": d["dataset"], "examples": {}, "predictions": {}})
        for e in d["examples"]:
            ex["examples"].setdefault(e["id"], e)
        for r in d["records"]:
            p = r.get("prediction") or {}
            row = {"label": p.get("label"), "probabilities": p.get("probabilities"),
                   "latency_ms": round(1000 * r["latency_s"], 1) if r.get("latency_s") is not None else None,
                   "error": r.get("error"), "model_version": p.get("model_version")}
            ex["predictions"].setdefault(r["model"], {}).setdefault(r["example_id"], row)
        ex.setdefault("run_meta", {})[model] = {"run_id": d["run_id"], "device": d["device"], "load_s": d["local_load_s"],
                                               "summaries": d["summaries"]}
SRC = {"sst2": ("stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb", "validation"),
       "ag_news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", "test"),
       "boolq": ("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5", "validation"),
       "imdb_long": ("stanfordnlp/imdb", "e6281661ce1c48d982bc483cf8a173c1bbeb5d31", "test")}
for ds, v in bench.items():
    v["examples"] = list(v["examples"].values())
    repo, rev, split = SRC[ds]
    # Dataset texts are not redistributed; dmw.bench.attach_texts() downloads them.
    v["text_source"] = {"repo": repo, "revision": rev, "split": split, "id_is_row_index": True,
                        "note": "Example texts are not redistributed here. dmw.bench.attach_texts() downloads them from the pinned revision."}
    for e in v["examples"]:
        e["words"] = len(e.pop("text").split())
    print(ds, len(v["examples"]), {m: len(p) for m, p in v["predictions"].items()})
(out / "benchmark_local_models.json").write_text(json.dumps(bench, ensure_ascii=False))
