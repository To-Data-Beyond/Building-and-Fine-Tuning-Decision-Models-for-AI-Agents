from nbkit import SETUP, md, py, write

# =============================================================================== 05 demo local
write("05_demo_local_models.ipynb", [
    md("""
    # Demo 3 · Run Kev-4B and Laya locally

    **Part 04 · Local decision models** · about 10 minutes

    **Part 1 (always runs):** the saved head-to-head runs. Jev, Kev-4B and Laya on the same SST-2,
    AG News, BoolQ and long IMDb examples, with accuracy, macro-F1, failures and latency.

    **Part 2 (optional, live):** install Kev-4B and Laya in their own environments and run them on
    our support tickets. Kev needs `transformers>=5` and Laya needs `transformers<5`, so each runs in
    a separate worker process. Laya runs on CPU; Kev-4B needs a GPU or Apple Silicon to be practical.
    Downloads are several GB: do this before the session, not on stage.
    """),
    py(SETUP),
    py("""
    import pandas as pd
    import matplotlib.pyplot as plt
    from dmw import bench, metrics, datagen

    for ds in bench.datasets():
        meta = bench.load()[ds]
        print(f"{ds:<10} {len(meta['examples']):>3} examples · {meta['dataset']['name']}")
    """),
    md("## Part 1 · The saved head-to-head runs"),
    py("""
    tables = {ds: bench.table(ds) for ds in bench.datasets()}
    pd.concat(tables, names=["dataset"])
    """),
    py("""
    acc = pd.DataFrame({ds: t["accuracy"] for ds, t in tables.items()}).T * 100
    ax = acc.plot.bar(figsize=(8, 3.2), color=["#142B3D", "#075E91", "#1A8A9C"], rot=0)
    ax.set_ylabel("accuracy, %"); ax.set_ylim(0, 105); ax.legend(frameon=False, ncol=3)
    for c in ax.containers: ax.bar_label(c, fmt="%.0f", fontsize=8)
    plt.title("Same examples, failures count as wrong"); plt.tight_layout(); plt.show()
    """),
    md("""
    ### Where the models disagree (AG News)

    Agreement between two models says nothing about who is right: read a few rows.
    `attach_texts()` downloads the example texts from the pinned dataset revision.
    """),
    py("""
    ag = bench.frame("ag_news")
    dis = ag[(ag["kev-4b"] != ag["jev"]) | (ag["laya"] != ag["jev"])]
    print(f"{len(dis)} of {len(ag)} examples have at least one disagreement with Jev")
    try:
        dis = bench.attach_texts("ag_news", dis)
        cols = ["text", "gold", "jev", "kev-4b", "laya"]
    except Exception as e:
        print("Texts not available offline:", type(e).__name__); cols = ["id", "gold", "jev", "kev-4b", "laya"]
    dis[cols].head(8)
    """),
    md("""
    ### Long input: the encoder refuses instead of guessing

    IMDb long reviews have 1,000+ words; Laya's limit is 512 tokens. Every request is rejected,
    which the benchmark counts as a failure, not as a wrong label.
    """),
    py("""
    im = bench.frame("imdb_long")
    print("Laya failures:", im["laya"].isna().sum(), "of", len(im))
    print("Example error:", im["laya_error"].dropna().iloc[0])
    print("Words per review: median", int(im.words.median()), "| max", im.words.max())
    bench.table("imdb_long")
    """),
    md("""
    ## Part 2 · Live: run the models yourself (optional)

    Set `RUN_LOCAL = True`. Setup creates `~/.dmw_venvs/laya` and `~/.dmw_venvs/kev-4b` (or `$DMW_VENVS`)
    and installs the pinned packages. In Colab, use a GPU runtime for Kev-4B. Kev on CUDA with fp16
    (T4) is not tested upstream; if it fails, keep Laya live and use Part 1 for Kev.
    """),
    py("""
    RUN_LOCAL = False          # set True to install and run the models
    MODELS = ["laya", "kev-4b"]
    """),
    py("""
    from dmw import local
    if RUN_LOCAL:
        for m in MODELS:
            if not local.is_ready(m):
                local.setup(m)
            print(m, "ready")
    """),
    py("""
    # Shared examples: the 16 support tickets from the fresh test set (4 labels)
    tickets = [c for c in datagen.load_test("data/fresh_test.json") if c["group"] == "tickets"]
    LABELS = tickets[0]["choices"]
    INSTRUCTIONS = "Which team should handle this support ticket?"

    results = []
    if RUN_LOCAL:
        for m in MODELS:
            with local.LocalModel(m) as lm:
                print(f"{m}: loaded in {lm.load_s} s on {lm.info['device']}")
                for c in tickets:
                    out = lm.predict(c["text"], LABELS, INSTRUCTIONS)
                    results.append({"model": m, "id": c["id"], "text": c["text"][:60], "gold": c["expected"],
                                    "pred": out.get("label"), "p_top": max(out["probabilities"].values()) if out.get("probabilities") else None,
                                    "ms": out["latency_ms"], "error": out.get("error")})
        live = pd.DataFrame(results)
        display(live.pivot(index=["id", "text", "gold"], columns="model", values="pred"))
        print(live.groupby("model").apply(lambda g: pd.Series({
            "accuracy": (g.pred == g.gold).mean(), "p50 ms": g.ms.median()})))
    else:
        print("RUN_LOCAL is False: Part 1 shows the saved runs.")
    """),
    py("""
    # The long-input test: one IMDb review of 1,000+ words through Laya (needs RUN_LOCAL and internet)
    if RUN_LOCAL:
        im_t = bench.attach_texts("imdb_long", bench.frame("imdb_long").head(1))
        with local.LocalModel("laya") as lm:
            print(lm.predict(im_t.text.iloc[0], bench.labels("imdb_long"), "Classify the sentiment of this text."))
    """),
    md("""
    ## Takeaways

    * Short text: the best local models are within a few points of Jev; Laya leads AG News.
    * BoolQ and long documents separate them: encoders cannot read long input at all.
    * Load time and hardware are part of the cost of going local.
    """),
], "Demo 3")

# =============================================================================== 06 exercise position bias
EX3_INTRO = md("""
# Exercise 3 · Measure position bias in base Qwen3

**Part 04 · Local decision models** · 10 minutes

Before training anything, measure the base model with the same letter scorer we use later:

1. Score every test case with the options in the **given** order and in **reversed** order.
2. Compute how often the winner changes (the **flip rate**) and the accuracy of each order.
3. Implement **averaged** scoring over both orders.
4. Save the base model's results as the baseline for Demo 5.

Runs on CPU in a few minutes for Qwen3-0.6B; faster on a GPU.
""")
EX3_SETUP = [
    py("%pip install -q -r requirements-finetune.txt"),
    py(SETUP),
    py("""
    import json
    from dmw import scorer as S, datagen, metrics, checks
    from dmw.prompt import build_messages

    BASE_MODEL = "Qwen/Qwen3-0.6B"      # the model we fine-tune in notebook 07
    model, tok = S.load(BASE_MODEL)
    sc = S.LetterScorer(model, tok)
    cases = datagen.load_test("data/fresh_test.json")
    print(len(cases), "test cases on", config.device())
    """),
    py("""
    # What the model actually sees for one case
    c = cases[0]
    print(S.render_prompt(tok, build_messages(c["context"], list(c["choices"]), None, c["question"], c["choices"]))[-900:])
    """),
]
EX3_SAVE = [
    md("## 4 · Save the baseline for Demo 5 (sorted + reversed orderings, like Stavro)"),
    py("""
    rows = S.score_cases(sc, cases)
    summary = metrics.summarize(rows)
    json.dump({"model": BASE_MODEL, "rows": rows, "summary": summary},
              open(config.OUTPUTS / "baseline_base_model.json", "w"), indent=1)
    print("letter mass (share of probability on the option letters):",
          round(sum(r["letter_mass"] for r in rows) / len(rows), 3))
    metrics.summary_table(summary)
    """),
]
write("06_exercise_position_bias.ipynb", [
    EX3_INTRO, *EX3_SETUP,
    md("## 1 · Score one order at a time"),
    py("""
    def winner(case, order):
        \"\"\"Return the label the model picks when the options are shown in `order`.\"\"\"
        # TODO: call sc.score(...) with orders=[order]; pass question and descriptions from the case
        raise NotImplementedError

    given_pred, reversed_pred = [], []
    for c in cases:
        order = list(c["choices"])
        given_pred.append(winner(c, order))
        reversed_pred.append(winner(c, order[::-1]))
    """),
    md("## 2 · Flip rate and accuracy per order"),
    py("""
    gold = [c["expected"] for c in cases]
    flip_rate = None          # TODO: share of cases where the two orders pick different labels
    given_acc = None          # TODO
    reversed_acc = None       # TODO
    print(flip_rate, given_acc, reversed_acc)
    """),
    md("## 3 · Averaged scoring"),
    py("""
    def averaged(case):
        \"\"\"Average the option probabilities over the given and reversed orders; return the winning label.\"\"\"
        # TODO: one call to sc.score with both orders, or combine two single-order calls yourself
        raise NotImplementedError

    averaged_acc = None       # TODO
    """),
    py("""
    reference = checks.position_bias_reference(sc, cases)
    checks.check_position_bias(flip_rate, given_acc, reversed_acc, averaged_acc, reference)
    """),
    *EX3_SAVE,
    md("**Discuss:** is averaged scoring always more accurate than the better single order? What else does it give you besides accuracy?"),
], "Exercise 3")

write("solutions/06_exercise_position_bias_solution.ipynb", [
    EX3_INTRO, *EX3_SETUP,
    md("## 1 · Score one order at a time"),
    py("""
    def winner(case, order):
        return sc.score(case["context"], list(case["choices"]), question=case["question"],
                        descriptions=case["choices"], orders=[order]).choice

    given_pred, reversed_pred, p_first = [], [], []
    for c in cases:
        order = list(c["choices"])
        d = sc.score(c["context"], order, question=c["question"], descriptions=c["choices"], orders=[order])
        given_pred.append(d.choice); p_first.append(d.per_order[0]["probabilities"][order[0]])
        reversed_pred.append(winner(c, order[::-1]))
    """),
    md("## 2 · Flip rate and accuracy per order"),
    py("""
    gold = [c["expected"] for c in cases]
    flip_rate = sum(a != b for a, b in zip(given_pred, reversed_pred)) / len(cases)
    given_acc = metrics.accuracy(gold, given_pred)
    reversed_acc = metrics.accuracy(gold, reversed_pred)
    print(f"flip rate {flip_rate:.2%} | given {given_acc:.2%} | reversed {reversed_acc:.2%} | "
          f"mean P(first option) {sum(p_first)/len(p_first):.2f} (unbiased ≈ 1/n)")
    """),
    md("## 3 · Averaged scoring"),
    py("""
    def averaged(case):
        order = list(case["choices"])
        return sc.score(case["context"], order, question=case["question"], descriptions=case["choices"],
                        orders=[order, order[::-1]]).choice

    averaged_acc = metrics.accuracy(gold, [averaged(c) for c in cases])
    print(f"averaged {averaged_acc:.2%}")
    """),
    py("""
    reference = checks.position_bias_reference(sc, cases)
    checks.check_position_bias(flip_rate, given_acc, reversed_acc, averaged_acc, reference)
    """),
    *EX3_SAVE,
    md("""
    ### Discussion
    Averaging is not guaranteed to beat the better single order on every set, but it removes the
    dependence on an arbitrary choice of order, and the disagreement between orders is a free
    signal for review (`orderings_disagree`). A base model that flips often is guessing by position.
    """),
], "Exercise 3 solution")

# =============================================================================== 07 demo fine-tune
write("07_demo_finetune_qwen3.ipynb", [
    md("""
    # Demo 4 · Fine-tune Qwen3 into a decision model and publish it

    **Part 05 · Fine-tuning** · 15 minutes + training

    1. Generate the data: templates → scenarios → template split → near-duplicate filter → 2 orderings
    2. Tokenize with the loss on the answer letter only, and check the targets
    3. Train a LoRA adapter (Colab T4 or any recent GPU; Apple Silicon works, slower)
    4. Score the fresh test set with the averaged letter scorer
    5. Merge, write the model card, push to your Hugging Face account

    **Colab:** Runtime → Change runtime type → T4 GPU, before running anything.
    """),
    py("%pip install -q -r requirements-finetune.txt"),
    py(SETUP),
    md("## 0 · Settings (the only cell you need to edit)"),
    py("""
    BASE_MODEL = "Qwen/Qwen3-0.6B"              # or "Qwen/Qwen3-1.7B" with more time / GPU memory
    HF_USERNAME = config.key("HF_USERNAME")     # set in .env or Colab secrets
    REPO_NAME = "qwen3-0.6b-decision-router"
    PUSH = True                                 # needs HF_TOKEN with write access
    PRIVATE = True                              # make it public later, after reading your own model card

    HP = dict(r=16, alpha=32, dropout=0.05, lr=2e-4, epochs=1, batch_size=4, grad_accum=2, max_length=512)

    PREPARED_CHECKPOINT = ""   # backup model ID shared by the instructor if your training fails
    print("device:", config.device(), "| repo:", f"{HF_USERNAME}/{REPO_NAME}")
    """),
    md("## 1 · Data: templates → scenarios → split → filter → rows"),
    py("""
    import json, time
    import pandas as pd
    from dmw import datagen, train as T, scorer as S, metrics
    from dmw.prompt import build_messages

    test_cases = datagen.load_test("data/fresh_test.json")
    train_s, valid_s = datagen.generate()                                   # split by template
    train_s, valid_s, dropped = datagen.drop_near_test(train_s, valid_s, [c["text"] for c in test_cases])
    train_rows, valid_rows = datagen.to_rows(train_s), datagen.to_rows(valid_s, seed=7)
    manifest = datagen.manifest(train_s, valid_s, train_rows, valid_rows, dropped)
    print(json.dumps(manifest, indent=1))
    print("max similarity of any training text to a test case:",
          round(datagen.max_similarity([s["text"] for s in train_s], [c["text"] for c in test_cases]), 2))
    """),
    py("""
    pd.DataFrame(train_s)[["domain", "expected", "text"]].sample(8, random_state=1)
    """),
    md("## 2 · Tokenize: same prompt function as the scorer, loss on the letter only"),
    py("""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ex, skipped_tr = T.tokenize_rows(train_rows, tok, HP["max_length"])
    valid_ex, skipped_va = T.tokenize_rows(valid_rows, tok, HP["max_length"])
    print(f"train {len(train_ex)} (skipped {skipped_tr} over max_length) | valid {len(valid_ex)} (skipped {skipped_va})")

    ex = train_ex[0]
    print("\\n--- end of one training example (the part with labels != -100 is learned) ---")
    print(repr(tok.decode(ex["input_ids"][-40:])))
    print("first trained tokens:", T.check_targets(train_ex, tok)[:10], "← all letters, never '<think>'")
    lengths = pd.Series([len(e["input_ids"]) for e in train_ex])
    print("tokens per example: median", int(lengths.median()), "| max", lengths.max())
    """),
    md("## 3 · Model + LoRA"),
    py("""
    model, tok, prec = T.prepare_model(BASE_MODEL, r=HP["r"], alpha=HP["alpha"], dropout=HP["dropout"])
    model.print_trainable_parameters()
    print("precision:", {k: v for k, v in prec.items() if k != "dtype"}, "| dtype:", prec["dtype"])
    """),
    md("## 4 · Train  (start this early, then go back to the slides)"),
    py("""
    OUT = config.OUTPUTS / "run"
    trainer = T.make_trainer(model, tok, train_ex, valid_ex, OUT, prec, epochs=HP["epochs"], lr=HP["lr"],
                             batch_size=HP["batch_size"], grad_accum=HP["grad_accum"])
    t0 = time.time()
    trainer.train()
    TRAIN_MINUTES = round((time.time() - t0) / 60, 1)
    print("training time:", TRAIN_MINUTES, "min")
    """),
    py("""
    import matplotlib.pyplot as plt
    tr, ev = T.loss_history(trainer)
    plt.figure(figsize=(7, 3))
    plt.plot(*zip(*tr), label="train loss", color="#075E91")
    if ev: plt.plot(*zip(*ev), "o-", label="validation loss (held-out templates)", color="#C8641E")
    plt.xlabel("step"); plt.legend(frameon=False); plt.tight_layout(); plt.show()
    """),
    md("## 5 · Score the fresh test set (averaged over two orderings)"),
    py("""
    model.config.use_cache = True
    sc = S.LetterScorer(model, tok)
    ft_rows = S.score_cases(sc, test_cases)
    ft_summary = metrics.summarize(ft_rows)
    T.save_json({"model": f"{HF_USERNAME}/{REPO_NAME}", "rows": ft_rows, "summary": ft_summary},
                config.OUTPUTS / "predictions_finetuned.json")
    table = metrics.summary_table(ft_summary)
    base_file = config.OUTPUTS / "baseline_base_model.json"
    if base_file.exists():
        base = json.load(open(base_file))
        table["base accuracy"] = pd.Series({g: v["accuracy"] for g, v in base["summary"].items()}).round(3)
    table
    """),
    py("""
    wrong = [r for r in ft_rows if not r["correct"]]
    pd.DataFrame([{**{k: r[k] for k in ("id", "group", "expected", "choice", "review")},
                   "text": next(c["text"] for c in test_cases if c["id"] == r["id"])} for r in wrong])
    """),
    md("## 6 · Merge, model card, push"),
    py("""
    MODEL_DIR = config.OUTPUTS / "model"
    T.merge_and_save(model, tok, MODEL_DIR)
    hardware = config.device() if config.device() != "cuda" else __import__("torch").cuda.get_device_name(0)
    card = T.model_card(f"{HF_USERNAME}/{REPO_NAME}", BASE_MODEL, manifest, HP, ft_summary,
                        hardware=f"{hardware}, {TRAIN_MINUTES} min")
    print(card[:1200])
    """),
    py("""
    if PUSH:
        url = T.push(MODEL_DIR, f"{HF_USERNAME}/{REPO_NAME}", card, private=PRIVATE)
        print("Published:", url, "(private)" if PRIVATE else "")
    else:
        print("PUSH is False: the merged model is in", MODEL_DIR)
    """),
    md("""
    ## If training failed

    Set `PREPARED_CHECKPOINT` in the settings cell and run this, then continue with notebook 09.
    """),
    py("""
    if PREPARED_CHECKPOINT:
        m2, t2 = S.load(PREPARED_CHECKPOINT)
        rows = S.score_cases(S.LetterScorer(m2, t2), test_cases)
        T.save_json({"model": PREPARED_CHECKPOINT, "rows": rows, "summary": metrics.summarize(rows)},
                    config.OUTPUTS / "predictions_finetuned.json")
        print(metrics.summary_table(metrics.summarize(rows)))
    """),
], "Demo 4")
