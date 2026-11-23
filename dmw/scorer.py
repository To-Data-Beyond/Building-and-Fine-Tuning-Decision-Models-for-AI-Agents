"""Letter-probability scoring for any Hugging Face causal LM.

One forward pass per option ordering: read the next-token logits for the option
letters (A, B, ...), turn them into probabilities, and average over orderings.
No text is generated, so the answer is always one of the declared options.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .prompt import LETTERS, build_messages, orderings as default_orderings

REVIEW_BELOW = 0.8  # same threshold as the Stavro server


@dataclass
class Decision:
    choice: str
    probabilities: dict[str, float]
    position_agreement: dict[str, float]
    review: bool
    flags: list[str]
    letter_mass: float          # share of the full next-token probability on the option letters
    per_order: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {"choice": self.choice, "probabilities": self.probabilities, "review": self.review,
                "flags": self.flags, "position_agreement": self.position_agreement,
                "letter_mass": round(self.letter_mass, 4), "latency_ms": round(self.latency_ms, 1)}


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render with the model's chat template, generation prompt on, thinking off.

    `enable_thinking=False` matters for hybrid Qwen3 models: the template then ends
    with an empty <think></think> block, so the next token is the answer letter.
    Templates that do not know the flag ignore it.
    """
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def letter_token_ids(tokenizer, n: int = len(LETTERS)) -> list[int]:
    ids = []
    for letter in LETTERS[:n]:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Letter {letter!r} is not a single token for this tokenizer: {toks}")
        ids.append(toks[0])
    return ids


class LetterScorer:
    def __init__(self, model, tokenizer, max_length: int = 2048):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.tok = tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.letter_ids = letter_token_ids(tokenizer)
        self.max_length = max_length

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _letter_logprobs(self, prompts: list[str], n_options: int):
        """Full-vocabulary log-probabilities of the first n letters, one row per prompt."""
        torch = self.torch
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                       truncation=False)
        if enc["input_ids"].shape[1] > self.max_length:
            raise ValueError(f"Prompt has {enc['input_ids'].shape[1]} tokens; max_length is {self.max_length}")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        return logp[:, self.letter_ids[:n_options]].cpu()

    def score(self, context: str, choices: list[str], question: str | None = None,
              descriptions: dict[str, str] | None = None, instructions: str | None = None,
              orders: list[list[str]] | None = None, n_orders: int = 2) -> Decision:
        """Score one decision. `orders` overrides the default sorted + reversed orderings."""
        orders = orders or default_orderings(choices, n_orders)
        t0 = time.perf_counter()
        prompts = [render_prompt(self.tok, build_messages(context, o, instructions, question, descriptions)) for o in orders]
        lp = self._letter_logprobs(prompts, len(choices))
        per_order, sums, wins, mass = [], {c: 0.0 for c in choices}, {c: 0 for c in choices}, 0.0
        for o, row in zip(orders, lp):
            raw = [math.exp(x) for x in row.tolist()]
            mass += sum(raw)
            probs = {c: raw[i] / sum(raw) for i, c in enumerate(o)}
            winner = max(probs, key=probs.get)
            wins[winner] += 1
            for c, p in probs.items():
                sums[c] += p
            per_order.append({"order": o, "probabilities": probs, "winner": winner})
        avg = {c: sums[c] / len(orders) for c in choices}
        choice = max(avg, key=avg.get)
        agreement = {c: wins[c] / len(orders) for c in choices}
        flags = []
        if avg[choice] < REVIEW_BELOW:
            flags.append("low_probability")
        if agreement[choice] < 1.0:
            flags.append("orderings_disagree")
        ms = (time.perf_counter() - t0) * 1000
        return Decision(choice, avg, agreement, bool(flags), flags, mass / len(orders), per_order, ms)


def score_cases(scorer: LetterScorer, cases: list[dict], **kw) -> list[dict]:
    """Score a list of {id, group, context, question, choices: {label: description}, expected}."""
    out = []
    for c in cases:
        d = scorer.score(c["context"], list(c["choices"]), question=c.get("question"),
                         descriptions=c["choices"] if any(c["choices"].values()) else None, **kw)
        out.append({"id": c["id"], "group": c.get("group"), "expected": c["expected"], **d.as_dict(),
                    "correct": d.choice == c["expected"]})
    return out


def load(model_id: str, device: str | None = None, dtype: str = "auto"):
    """Load a causal LM and tokenizer for scoring (fp16 on CUDA, fp32 elsewhere)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import config

    device = device or config.device()
    if dtype == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    return model, tok
