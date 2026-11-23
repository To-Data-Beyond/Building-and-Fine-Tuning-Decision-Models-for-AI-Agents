"""Helpers for the workshop "Building and Fine-Tuning Decision Models for AI Agents".

Modules
-------
config   paths, .env loading, MODE (replay or live)
jev      Jev API client with a replay store
llm      structured-output LLM baseline (OpenAI), optional
rules    the simple rule baselines used in the labs
fixtures the labeled cases from the "Nine Ways to Use Jev" article
prompt   the Stavro-compatible prompt builder (shared by training and scoring)
scorer   letter-probability scorer for a Hugging Face causal LM
datagen  template-based training data, template split, near-duplicate filter
train    LoRA fine-tuning helpers (PEFT + transformers Trainer)
metrics  accuracy, macro-F1, confusion matrices, coverage curves, latency
local    run Kev and Laya in their own virtual environments
bench    saved benchmark predictions (Kev-4B, Laya, Jev)
checks   automatic checkers for the exercises
"""

__version__ = "2026.11.22"
