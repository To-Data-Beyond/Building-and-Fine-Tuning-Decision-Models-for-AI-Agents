"""Labeled cases and policies from the "Nine Ways to Use Jev" scripts (data/fixtures.json).

Each use case has: instructions, criteria, question_id, type, and cases with
id, state and the expected (policy) answer. Labels are never sent to a model.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _all() -> dict:
    return json.loads((config.DATA / "fixtures.json").read_text(encoding="utf-8"))


def get(use_case: str) -> dict:
    """A deep copy, so notebooks can edit criteria without touching the original."""
    return copy.deepcopy(_all()[use_case])


def questions(spec: dict) -> dict:
    return {spec["question_id"]: {"type": spec["type"], "instructions": spec["instructions"], "criteria": spec["criteria"]}}


def reranking_pairs() -> list[dict]:
    """Flatten the reranking queries into (query, passage, reference grade) rows."""
    out = []
    for q in _all()["reranking"]["queries"]:
        for i, p in enumerate(q["passages"]):
            out.append({"query": q["id"], "question": q["question"], "passage": p["id"], "text": p["text"],
                        "grade": p["grade"], "input_position": i})
    return out
