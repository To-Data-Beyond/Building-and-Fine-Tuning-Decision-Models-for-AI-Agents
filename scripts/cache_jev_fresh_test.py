"""Instructor prep: ask Jev once about every case in data/fresh_test.json and save the answers.

After this runs, notebook 09 shows a Jev column in replay mode too, so attendees without a key
still see base vs fine-tuned vs Jev. Needs TYPESAFE_API_KEY in .env. About 48 calls.

    python scripts/cache_jev_fresh_test.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmw import config, datagen, jev  # noqa: E402

config.load_env()
cases = datagen.load_test(config.DATA / "fresh_test.json")
rows = []
for c in cases:
    out = jev.ask(c["context"], {"d": jev.question("choice", c["question"], c["choices"])})
    a = out["answers"]["d"]
    rows.append({"id": c["id"], "group": c["group"], "expected": c["expected"], "choice": a["value"],
                 "probabilities": a["probabilities"], "confidence": a["confidence"],
                 "latency_ms": out["latency_ms"], "model": out["response"].get("model")})
    print(f"{c['id']:<4} {c['expected']:<17} → {a['value']:<17} conf={a['confidence']}")
path = config.REPLAY / "jev_fresh_test.jsonl"
path.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"\n{sum(r['choice'] == r['expected'] for r in rows)}/{len(rows)} correct · saved {path}")
