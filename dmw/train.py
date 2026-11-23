"""LoRA fine-tuning of a Qwen3 model into a letter-answering decision model.

The important details, each one a function so the notebook can show it:

* tokenize_rows()   prompt rendered by the SAME function the scorer uses, then the
                    answer letter and <|im_end|>. Labels are -100 on the prompt, so
                    the loss is computed on the letter only.
* check_targets()   decode the first unmasked token of some rows and assert it is a letter.
* prepare_model()   picks dtype per device, enables gradient checkpointing, adds LoRA.
* make_trainer()    a plain transformers Trainer with a padding collator.
* merge_and_save(), model_card(), push()   publish to the Hugging Face Hub.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config
from .prompt import LETTERS
from .scorer import render_prompt

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def end_token_id(tok) -> int:
    vocab = tok.get_vocab()
    return vocab["<|im_end|>"] if "<|im_end|>" in vocab else tok.eos_token_id


def tokenize_rows(rows: list[dict], tok, max_length: int = 512) -> tuple[list[dict], int]:
    """Return (examples, n_skipped). Each example has input_ids and labels."""
    end = end_token_id(tok)
    out, skipped = [], 0
    for r in rows:
        p_ids = tok(render_prompt(tok, r["messages"]), add_special_tokens=False)["input_ids"]
        a_ids = tok(r["answer"], add_special_tokens=False)["input_ids"]
        if len(a_ids) != 1:
            raise ValueError(f"Answer {r['answer']!r} is not one token: {a_ids}")
        ids = p_ids + a_ids + [end]
        if len(ids) > max_length:
            skipped += 1
            continue
        out.append({"input_ids": ids, "labels": [-100] * len(p_ids) + a_ids + [end]})
    return out, skipped


def check_targets(examples: list[dict], tok, n: int = 20) -> list[str]:
    """Decode the first trained token of n examples. Every one must be a letter A..P."""
    firsts = []
    for ex in examples[:n]:
        first = next(t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100)
        firsts.append(tok.decode([first]))
    bad = [f for f in firsts if f not in LETTERS]
    if bad:
        raise AssertionError(f"First trained token is not a letter for {len(bad)} rows, e.g. {bad[:3]!r}. "
                             "Check the chat template: the prompt must end right before the answer.")
    return firsts


class Collator:
    """Right-pad input_ids with the pad token and labels with -100."""

    def __init__(self, tok):
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, batch):
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def precision(device: str) -> dict:
    """dtype to load the base model in, and the Trainer mixed-precision flag."""
    import torch

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return {"dtype": torch.bfloat16, "bf16": True, "fp16": False}
        return {"dtype": torch.float16, "bf16": False, "fp16": True}  # e.g. Colab T4
    return {"dtype": torch.float32, "bf16": False, "fp16": False}      # mps / cpu


def prepare_model(base_model: str, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                  target_modules: list[str] | None = None, device: str | None = None):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or config.device()
    prec = precision(device)
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=prec["dtype"]).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules=target_modules or TARGET_MODULES)
    model = get_peft_model(model, lora)
    if prec["fp16"]:
        # fp16 mixed precision needs fp32 master weights for the trainable parameters.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    return model, tok, prec


def make_trainer(model, tok, train_examples, valid_examples, out_dir, prec: dict, epochs: float = 1.0,
                 lr: float = 2e-4, batch_size: int = 4, grad_accum: int = 2, eval_every: int = 25,
                 seed: int = 3033):
    import math

    from transformers import Trainer, TrainingArguments

    total_steps = math.ceil(len(train_examples) / (batch_size * grad_accum)) * epochs
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * total_steps)),
        logging_steps=5, eval_strategy="steps", eval_steps=eval_every, save_strategy="no",
        report_to=[], fp16=prec["fp16"], bf16=prec["bf16"], gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False,
        dataloader_num_workers=0, seed=seed,
    )
    return Trainer(model=model, args=args, train_dataset=train_examples, eval_dataset=valid_examples,
                   data_collator=Collator(tok))


def loss_history(trainer):
    """(train_steps, train_loss, eval_steps, eval_loss) from the Trainer log."""
    tr = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in trainer.state.log_history if "eval_loss" in h]
    return tr, ev


def merge_and_save(model, tok, out_dir) -> Path:
    """Fold the LoRA weights into the base model and save a standalone checkpoint."""
    out = Path(out_dir)
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    shutil.copy(Path(__file__).with_name("prompt.py"), out / "decision_prompt.py")
    return out


def model_card(repo_id: str, base_model: str, manifest: dict, hparams: dict, results: dict | None = None,
               hardware: str = "") -> str:
    """README.md for the Hub: what the model is, how to call it, data, evaluation and limits."""
    rows = ""
    if results:
        rows = "\n".join(f"| {g} | {v['n']} | {v['accuracy']:.1%} | {v.get('macro_f1', float('nan')):.3f} | {v.get('review_rate', float('nan')):.0%} |"
                         for g, v in results.items())
        rows = "| Group | n | Accuracy | Macro-F1 | Flagged for review |\n|---|---|---|---|---|\n" + rows
    return f"""---
base_model: {base_model}
library_name: transformers
license: apache-2.0
tags: [decision-making, classification, lora, letter-scoring]
---

# {repo_id.split('/')[-1]}

A small **decision model**: it reads a context, a question and 2–16 lettered options, and puts its
probability on the letter of the correct option. It was fine-tuned from `{base_model}` with LoRA
during the To Data & Beyond workshop *Building and Fine-Tuning Decision Models for AI Agents*.

It is not a chat model. Use it by reading the next-token probabilities of the option letters,
not by generating text.

## How to call it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from decision_prompt import build_messages   # included in this repo

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
options = ["account", "billing", "bug", "feature request"]
msgs = build_messages("Support ticket: We were charged twice in March.", options,
                      question="Which team should handle this ticket?")
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
logits = model(**tok(prompt, return_tensors="pt", add_special_tokens=False)).logits[0, -1]
letters = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(len(options))]
probs = logits[letters].softmax(-1)
print(dict(zip(options, probs.tolist())))
```

For better results, score the options in two orders (sorted and reversed), average the
probabilities, and send answers below 0.8, or where the two orders disagree, to review.

## Training

* Method: LoRA (r={hparams.get('r')}, alpha={hparams.get('alpha')}, dropout={hparams.get('dropout')}) on all attention and MLP projections, merged after training
* Loss on the answer letter only (prompt tokens masked with -100)
* {hparams.get('epochs')} epoch(s), learning rate {hparams.get('lr')}, effective batch {hparams.get('batch_size', 0) * hparams.get('grad_accum', 1)}, max length {hparams.get('max_length')}
* Data: template-generated support tickets (billing, bug, account, feature request), IT requests and hotel
  messages; labels set by construction; two option orderings per scenario; no third-party model outputs
* Rows: {manifest.get('rows')} · scenarios: {manifest.get('scenarios')} · dropped as near-duplicates of the test set: {manifest.get('dropped_near_test')}
* Hardware: {hardware or 'not recorded'}

## Evaluation

Fresh, hand-written test set (48 cases): support tickets, hard tickets (sarcasm, misleading keywords,
injected instructions), and three domains not seen in training (HR, web shop, city services).
Two option orderings, averaged.

{rows or '_Not evaluated yet._'}

## Limitations

* Small, synthetic training and test sets: expect different numbers on real traffic.
* Arithmetic hidden in the facts (totals, dates, percentages) is unreliable: compute it in code.
* Long documents and reading-comprehension questions were not part of training.
* Instructions inside the context are usually ignored, but this is not guaranteed: keep exact checks in code.
"""


def push(model_dir, repo_id: str, card: str, private: bool = True, token: str | None = None) -> str:
    """Create the repo if needed and upload the merged model folder plus README.md."""
    from huggingface_hub import HfApi

    token = token or config.key("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN (a write token) in .env or Colab secrets.")
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    Path(model_dir, "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id, commit_message="Upload decision model from the workshop")
    return f"https://huggingface.co/{repo_id}"


def save_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
