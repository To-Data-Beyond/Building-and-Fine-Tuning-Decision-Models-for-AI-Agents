# Building and Fine-Tuning Decision Models for AI Agents

Workshop code · To Data & Beyond · Youssef Hosni · Sunday 22 November 2026

Notebooks for the four-hour live workshop: Jev as a decision layer, nine use cases, local decision
models (Kev-4B, Laya), fine-tuning Qwen3 into your own decision model, publishing it on Hugging Face,
and evaluating it against the base model and Jev.

## The notebooks

| # | Notebook | Type | Workshop block | Needs |
|---|---|---|---|---|
| 00 | `00_setup.ipynb` | setup | Opening | nothing |
| 01 | `01_demo_jev_anatomy.ipynb` | Demo 1 | 01 Jev and the decision layer | replay or Jev key |
| 02 | `02_demo_routing_and_triage.ipynb` | Demo 2 | 03 Labs | replay or Jev key |
| 03 | `03_exercise_design_decisions.ipynb` | Exercise 1 | 03 Labs | replay (structural checks) or Jev key |
| 04 | `04_exercise_triage_thresholds.ipynb` | Exercise 2 | 03 Labs | Jev and OpenAI keys for Part B (either one runs that model) |
| 05 | `05_demo_local_models.ipynb` | Demo 3 | 04 Local decision models | replay; optional live Kev-4B / Laya |
| 06 | `06_exercise_position_bias.ipynb` | Exercise 3 | 04 Local decision models | CPU is enough |
| 07 | `07_demo_finetune_qwen3.ipynb` | Demo 4 | 05 Fine-tuning | GPU (Colab T4), `HF_TOKEN`, `HF_USERNAME` |
| 08 | `08_exercise_build_dataset.ipynb` | Exercise 4 | 05 Fine-tuning | nothing |
| 09 | `09_demo_benchmark.ipynb` | Demo 5 | 06 Benchmarking | your model (or the prepared checkpoint) |
| 10 | `10_exercise_eval_report.ipynb` | Exercise 5 | 06 Benchmarking | output of 09, or the saved benchmark |

Solutions for every exercise are in `solutions/`.

## Run it

**Google Colab (recommended for fine-tuning).** Open `00_setup.ipynb` in Colab, set `REPO_URL` in the
first cell (or upload this folder as a zip), and run it. For notebooks 06–09 choose
*Runtime → Change runtime type → T4 GPU*. Put keys under the key icon (Secrets):
`TYPESAFE_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `HF_USERNAME`.

**Locally.** Python 3.10+:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-finetune.txt        # or requirements.txt for notebooks 01–05 and 10
cp .env.example .env                             # then fill in the keys you have
jupyter lab
```

### Replay and live

Every Jev notebook works without a key. In **replay** mode it reads the saved runs from
`data/replay/` (the recorded Jev and GPT-4.1 mini calls behind the article and the slides). With
`TYPESAFE_API_KEY` set, notebooks switch to **live** mode; live responses are cached in
`outputs/jev_cache.jsonl`, so re-running a cell does not pay twice.

## What is in the folder

```
dmw/                     helper package used by all notebooks
  prompt.py              the decision prompt (same function for training and scoring)
  scorer.py              letter-probability scorer, averaged over option orderings
  datagen.py             templates → scenarios → template split → near-duplicate filter → rows
  train.py               LoRA fine-tuning, merge, model card, push to the Hub
  jev.py, llm.py         Jev client (live + replay) and the structured-output LLM baseline
  local.py, local_worker.py   Kev and Laya in their own virtual environments
  bench.py, metrics.py, checks.py, fixtures.py, rules.py, config.py
data/
  fixtures.json          labeled cases and policies of the nine use cases
  fresh_test.json        48 hand-written test cases (tickets, hard tickets, HR, shop, city)
  airline_test.json      12 test cases for Exercise 4
  replay/                saved Jev / LLM answers and the saved local-model benchmark
scripts/                 how the data and notebooks were built, plus instructor prep
solutions/               exercise solutions
outputs/                 created when you run the notebooks (models, predictions, caches)
```

## Instructor prep before the session

1. **Prepared checkpoint.** Run `07_demo_finetune_qwen3.ipynb` once on Colab, push it publicly, and
   paste its ID into `PREPARED_CHECKPOINT` (notebooks 07 and 09). Note the training time for the slides.
2. **Jev answers for the test set.** `python scripts/cache_jev_fresh_test.py` (needs the key, ~48 calls)
   so notebook 09 shows a Jev column in replay mode.
3. **Local models.** In `05_demo_local_models.ipynb` set `RUN_LOCAL = True` once to create the
   environments and download Kev-4B and Laya before going live.
4. **Repository URL.** Set `REPO_URL` in `00_setup.ipynb` to the repository you share with attendees.

## Data and provenance

* Use-case fixtures and replays come from the scripts and saved runs of *Nine Ways to Use Jev in AI
  Applications and Agent Workflows* (runs of 6–8 October 2026, `jev-1.13.0` and `gpt-4.1-mini-2025-04-14`).
* Benchmark predictions come from the Jev vs. open models comparison (October 2026, Apple M3 Pro).
  Dataset texts are not included; `dmw.bench.attach_texts()` downloads them from the pinned revisions.
* Training templates come from the Stavro-4B build script. Labels are set by construction; no other
  model's outputs are used for training.
* The test sets in `data/` were written for this workshop. They are small and synthetic: use them to
  compare models, not to predict production accuracy.
