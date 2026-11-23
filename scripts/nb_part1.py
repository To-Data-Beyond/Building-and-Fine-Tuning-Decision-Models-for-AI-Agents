from nbkit import SETUP, md, py, write

# =============================================================================== 00 setup
write("00_setup.ipynb", [
    md("""
    # 00 · Setup check

    **Building and Fine-Tuning Decision Models for AI Agents** · To Data & Beyond

    Run every cell. Each check prints one line: `OK`, `MISSING` or `INFO`. Post `OK` in the chat,
    or paste the first line that is not OK.

    | Mode | What runs | Keys |
    |---|---|---|
    | Replay | Saved Jev and LLM answers from the recorded runs. Every notebook works. | none |
    | Live | New calls to the Jev API (and optionally an LLM baseline) | `TYPESAFE_API_KEY` (+ `OPENAI_API_KEY`) |
    | Fine-tuning | Training Qwen3 on a GPU and publishing it | `HF_TOKEN` with **write** access, `HF_USERNAME` |

    **Where to put keys.** Locally: copy `.env.example` to `.env` and fill it in.
    In Colab: click the key icon in the left sidebar (Secrets) and add the same names.
    The notebooks never print a key.
    """),
    md("""
    ## 1 · Get the workshop files (Colab only)

    Locally, open the notebooks from the workshop folder and skip this cell.
    In Colab, set `REPO_URL` to the workshop repository, or leave it empty to upload the workshop zip.
    """),
    py("""
    import sys, os, subprocess
    from pathlib import Path

    REPO_URL = ""   # the workshop GitHub repository, shared in the chat
    TARGET = Path("/content/decision-models-workshop")

    if "google.colab" in sys.modules and not TARGET.exists():
        if REPO_URL:
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(TARGET)], check=True)
        else:
            from google.colab import files
            print("Upload the workshop zip file:")
            up = files.upload()
            zip_name = next(iter(up))
            subprocess.run(["unzip", "-q", zip_name, "-d", "/content"], check=True)
            found = next(Path("/content").glob("**/dmw/__init__.py")).parents[1]
            if found != TARGET:
                found.rename(TARGET)
        os.chdir(TARGET)
    print("Working folder:", Path.cwd())
    """),
    md("## 2 · Install packages"),
    py("""
    # Core packages for every notebook (quick). Fine-tuning packages are installed in notebook 07.
    %pip install -q -r requirements.txt
    """),
    md("## 3 · Checks"),
    py(SETUP),
    py("""
    import importlib, platform, shutil, socket

    def line(status, msg): print(f"{status:<8}{msg}")

    v = sys.version_info
    line("OK" if v >= (3, 10) else "MISSING", f"Python {platform.python_version()} (need 3.10+)")
    for pkg in ["pandas", "matplotlib", "certifi"]:
        try:
            importlib.import_module(pkg); line("OK", f"package {pkg}")
        except ImportError:
            line("MISSING", f"package {pkg}: run the install cell above")

    try:
        import torch
        dev = config.device()
        detail = torch.cuda.get_device_name(0) if dev == "cuda" else dev
        line("OK" if dev != "cpu" else "INFO", f"torch {torch.__version__}, accelerator: {detail}"
             + ("" if dev != "cpu" else "  (fine for notebooks 01-05; use a GPU runtime for 07)"))
    except ImportError:
        line("INFO", "torch not installed yet: notebook 06/07 installs it")

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    line("OK" if free_gb > 10 else "INFO", f"free disk {free_gb:.0f} GB (fine-tuning needs ~10 GB)")

    for name, why in [("TYPESAFE_API_KEY", "live Jev calls (optional)"), ("OPENAI_API_KEY", "LLM baseline (optional)"),
                      ("HF_TOKEN", "publishing your model"), ("HF_USERNAME", "your Hub repo name")]:
        line("OK" if config.key(name) else ("MISSING" if name.startswith("HF") else "INFO"),
             f"{name} {'set' if config.key(name) else 'not set'}: {why}")

    def reachable(host):
        try:
            socket.create_connection((host, 443), timeout=5).close(); return True
        except OSError:
            return False
    for host in ["api.typesafe.ai", "huggingface.co"]:
        line("OK" if reachable(host) else "MISSING", f"network: {host}")

    print("\\nRun mode for the Jev notebooks:", config.mode())
    """),
    py("""
    # Optional: confirm the Hugging Face token can write (needed in notebook 07)
    if config.key("HF_TOKEN"):
        %pip install -q huggingface_hub
        from huggingface_hub import whoami
        me = whoami(token=config.key("HF_TOKEN"))
        role = me.get("auth", {}).get("accessToken", {}).get("role")
        print("Logged in to Hugging Face as", me["name"], "| token role:", role)
        if config.key("HF_USERNAME") and config.key("HF_USERNAME") != me["name"]:
            print("Note: HF_USERNAME differs from the token owner; pushes go to HF_USERNAME, so you need write access there.")
    else:
        print("HF_TOKEN not set yet. Create one at huggingface.co → Settings → Access Tokens → Write.")
    """),
], "00 Setup check")

# =============================================================================== 01 demo
write("01_demo_jev_anatomy.ipynb", [
    md("""
    # Demo 1 · Anatomy of a decision call

    **Part 01 · Jev and the decision layer** · about 8 minutes

    We build one state, ask a **Choice**, a **Score** and a **Noul** question in a single request,
    read the raw response, and recompute confidence and score ourselves so nothing in the answer is magic.

    * **Live mode** (`TYPESAFE_API_KEY` set): new calls, cached in `outputs/jev_cache.jsonl`.
    * **Replay mode**: real saved answers from the recorded runs, one of each type.
    """),
    py(SETUP),
    py("""
    import json
    from dmw import jev

    MODE = config.mode("auto")      # "auto", "live" or "replay"
    print("MODE =", MODE)
    show = lambda x: print(json.dumps(x, indent=2, ensure_ascii=False))
    """),
    md("""
    ## 1 · State: three shapes

    The state is the evidence. A string is one piece of text, an array is a sequence of messages,
    and an object gives each fact a name. Named fields let the question refer to them directly.
    """),
    py("""
    state_string = "Checkout returns 500 errors. Customers cannot pay."
    state_array = ["Checkout returns 500 errors.", "Card payments worked yesterday."]
    state_object = {
        "message": "Checkout returns 500 errors. Customers cannot pay.",
        "service_status": "unresolved incident",
        "now_utc": "2026-10-08T09:00:00Z",
        "allowed_actions": ["read", "draft"],
    }
    show(state_object)
    """),
    md("""
    ## 2 · Three questions in one request

    The question ID (`team`, `severity`, `blocked`) is how your code reads the answer back.
    """),
    py("""
    QUESTIONS = {
        "team": jev.question("choice", "Which team handles `message`?", {
            "engineering": "Broken product behavior, failures or errors",
            "billing": "Invoice amounts, charges or refunds",
            "other": "No category fits, or the input is ambiguous",
        }),
        "severity": jev.question("score", "How severe is the issue in `message` for customers right now?", [
            "Cosmetic issue, no functional impact",
            "Degraded feature, a workaround exists",
            "Blocking issue, no workaround",
        ]),
        "blocked": jev.question("noul", "Does `message` report a currently blocked customer workflow?", {
            "true": "A customer cannot complete a task they are trying to do now.",
            "false": "A future plan, general information, or an issue that is already resolved.",
        }),
    }
    show(jev.payload(state_object, QUESTIONS))
    """),
    md("## 3 · Call Jev and read the raw response"),
    py("""
    if MODE == "live":
        out = jev.call(jev.payload(state_object, QUESTIONS))
        raw = out["response"]["answers"]
        print(f"latency {out['latency_ms']} ms  (cached: {out['cached']})")
    else:
        # Replay: one real saved answer of each type from the recorded runs
        raw = {
            "team":     jev.replay_answer("gating", "missing_permission")["answers"]["decision"],
            "severity": jev.replay_answer("reranking", None, passage="reset_d", query="reset")["answers"]["relevance"],
            "blocked":  jev.replay_answer("guardrails", "doc_override")["answers"]["is_injection"],
        }
        print("Replay: a Choice from tool gating (the case Jev found hard), a Score from reranking, a Noul from guardrails.")
    show(raw)
    """),
    md("""
    ## 4 · Recompute what the response says

    * **Choice confidence** = (p_max − 1/n) / (1 − 1/n). It is 0 for a uniform distribution and 1 for a certain one.
    * **Score** = Σ level × probability, the probability-weighted level.
    * **Noul** is P(yes) directly; P(no) = 1 − noul. There is no separate confidence field.
    """),
    py("""
    c = raw["team"]
    print("Choice:", c["choice"], "| returned confidence:", c["confidence"],
          "| recomputed:", round(jev.choice_confidence(c["probabilities"]), 3))

    s = raw["severity"]
    print("Score:", s["score"], "| recomputed expected level:", round(jev.expected_score(s["probabilities"]), 3))

    n = raw["blocked"]
    print("Noul P(yes):", n["noul"], "| P(no):", round(1 - n["noul"], 3))
    """),
    md("""
    ## 5 · A vague message gives a flat distribution

    “It does not work. Please help.” has no evidence for any team. A label is still returned,
    so the distribution and confidence are what tell code to ask a follow-up question.
    """),
    py("""
    vague = "It does not work. Please help."
    if MODE == "live":
        a = jev.ask({"message": vague}, {"team": QUESTIONS["team"]})["answers"]["team"]
        print(a)
    else:
        saved = jev.replay_answer("labeling", "t11")["answers"]["label"]   # same text in the bulk-labeling run
        print("Saved bulk-labeling answer for:", repr(vague)); show(saved)
        print("recomputed confidence:", round(jev.choice_confidence(saved["probabilities"]), 3))
    """),
    md("""
    ## 6 · Change one criterion, watch the probabilities move (live only)

    Remove the `other` option. With no fallback, the model must spread probability over two teams that
    do not fit. This is why every Choice needs a fallback option.
    """),
    py("""
    if MODE == "live":
        no_fallback = {"team": jev.question("choice", "Which team handles `message`?", {
            "engineering": "Broken product behavior, failures or errors",
            "billing": "Invoice amounts, charges or refunds",
        })}
        print("With other:   ", jev.ask({"message": vague}, {"team": QUESTIONS["team"]})["answers"]["team"]["probabilities"])
        print("Without other:", jev.ask({"message": vague}, no_fallback)["answers"]["team"]["probabilities"])
    else:
        print("Live only. In replay mode, read the saved answer above: other won because it was available.")
    """),
    md("""
    ## 7 · Confidence gating: the threshold lives in code

    The same saved answer gives a different outcome under a different threshold. No new model call.
    """),
    py("""
    t08 = jev.parse(jev.replay_answer("labeling", "t08")["answers"]["label"])
    def gate(answer, threshold):
        if answer["value"] == "other":
            return "review"                      # no handler exists for other
        return "automatic" if answer["confidence"] >= threshold else "review"
    for t in (0.80, 0.95, 1.00):
        print(f"threshold {t:.2f} → {gate(t08, t)}   (label {t08['value']}, confidence {t08['confidence']})")
    """),
    md("""
    ## Takeaways

    1. The request is **state + typed questions**; answers come back under the same IDs.
    2. Confidence and score are simple functions of the probabilities: recompute them, don't trust them blindly.
    3. A fallback option and a review path are part of the design, not an afterthought.
    """),
], "Demo 1")

# =============================================================================== 02 demo
write("02_demo_routing_and_triage.ipynb", [
    md("""
    # Demo 2 · Routing and triage: rules vs an LLM vs Jev

    **Part 03 · Labs** · about 8 minutes

    The same 12 routing requests and 12 triage emails go through three deciders:
    a simple rule, GPT-4.1 mini with a strict JSON schema, and Jev. Replay mode uses the saved
    36-call runs (12 cases × 3 repeats) from 6 and 8 October 2026.
    """),
    py(SETUP),
    py("""
    import pandas as pd
    import matplotlib.pyplot as plt
    from dmw import jev, llm, rules, fixtures

    MODE = config.mode("auto")
    LLM_LIVE = MODE == "live" and bool(config.key("OPENAI_API_KEY"))   # otherwise the LLM column is the saved run
    pd.set_option("display.max_colwidth", 70)
    print("MODE =", MODE, "| LLM baseline:", "live" if LLM_LIVE else "saved run")

    def llm_or_saved(state, spec, saved_answer):
        \"\"\"Live GPT-4.1 mini answer if possible; otherwise the saved one (and stop trying live).\"\"\"
        global LLM_LIVE
        if LLM_LIVE:
            try:
                return llm.decide(state, "route", spec["instructions"], spec["criteria"])["value"]
            except Exception as e:
                print(f"LLM baseline failed: {e}\\n→ Using the saved GPT-4.1 mini answers for the rest of this notebook.")
                LLM_LIVE = False
        return saved_answer()
    """),
    md("## 1 · The routing policy"),
    py("""
    R = fixtures.get("routing")
    print(R["instructions"], "\\n")
    for k, v in R["criteria"].items():
        print(f"{k:>7}: {v}")
    """),
    md("## 2 · Three deciders on the same 12 requests"),
    py("""
    def saved(use_case, field, repeat=1):
        df = jev.replay(use_case)
        df = df[df.repeat == repeat].set_index(["provider", "case"])
        return lambda provider, case: df.loc[(provider, case), field]

    def route_live(state):
        return jev.ask(state, fixtures.questions(R))["answers"]["route"]["value"]

    get = saved("routing", "route")
    rows = []
    for c in R["cases"]:
        req = c["state"]["request"]
        rows.append({
            "case": c["id"], "request": req[:70], "policy": c["expected"],
            "length rule": rules.length_route(req),
            "GPT-4.1 mini": llm_or_saved(c["state"], R, lambda: get("openai", c["id"])),
            "Jev": route_live(c["state"]) if MODE == "live" else get("jev", c["id"]),
        })
    routing = pd.DataFrame(rows)

    def mark(df, cols):
        return df.style.apply(lambda r: ["" if c not in cols else
                                         ("background:#e6f2eb" if r[c] == r["policy"] else "background:#f8dddd")
                                         for c in df.columns], axis=1)
    mark(routing, ["length rule", "GPT-4.1 mini", "Jev"])
    """),
    py("""
    for col in ["length rule", "GPT-4.1 mini", "Jev"]:
        print(f"{col:>13}: {(routing[col] == routing.policy).sum()}/12 match the policy")
    """),
    md("""
    **Read the red cells.** The length rule fails exactly where meaning matters: a short design request,
    a vague “Can you fix it?”, and a request that tries to choose its own tier.
    """),
    md("## 3 · Inbox triage and the urgency trap"),
    py("""
    T = fixtures.get("triage")
    get = saved("triage", "route")
    rows = []
    for c in T["cases"]:
        st = c["state"]
        rows.append({
            "case": c["id"], "subject": st["subject"], "body": st["body"][:60], "policy": c["expected"],
            "subject rule": rules.subject_triage(st["subject"]),
            "GPT-4.1 mini": llm_or_saved(st, T, lambda: get("openai", c["id"])),
            "Jev": jev.ask(st, fixtures.questions(T))["answers"]["route"]["value"] if MODE == "live" else get("jev", c["id"]),
        })
    triage = pd.DataFrame(rows)
    mark(triage, ["subject rule", "GPT-4.1 mini", "Jev"])
    """),
    py("""
    HANDLER = {"reply_now": "label_priority", "later": "queue_follow_up", "archive": "archive_candidate"}
    triage["planned action (Jev)"] = triage["Jev"].map(lambda r: HANDLER.get(r, "manual_review"))
    triage[["case", "subject", "Jev", "planned action (Jev)"]]
    """),
    md("## 4 · All 36 saved calls: agreement and latency"),
    py("""
    summary = []
    for uc in ["routing", "triage"]:
        df = jev.replay(uc)
        for prov, g in df.groupby("provider"):
            summary.append({"use case": uc, "provider": prov, "calls": len(g),
                            "policy matches": int(g.policy_match.sum()),
                            "median latency ms": round(g.latency_ms.median()),
                            "est. cost USD (36 calls)": round(g.estimated_usd.sum(), 6)})
    summary = pd.DataFrame(summary)
    summary
    """),
    py("""
    ax = summary.pivot(index="use case", columns="provider", values="median latency ms").plot.barh(
        color=["#075E91", "#8FA5B5"], figsize=(7, 2.6))
    ax.set_xlabel("median request latency, ms (saved runs)"); ax.set_ylabel("")
    for c in ax.containers: ax.bar_label(c, fmt="%d", fontsize=9)
    ax.legend(["Jev", "GPT-4.1 mini"], frameon=False); plt.tight_layout(); plt.show()
    """),
    md("""
    ## 5 · Edit a criterion and rerun only the affected cases (live)

    Try it: make `review` stricter, or remove the sentence about treating override requests as data,
    then rerun the cell and see which cases change.
    """),
    py("""
    if MODE == "live":
        R2 = fixtures.get("routing")
        R2["criteria"]["review"] = "The request has no concrete task at all."   # <- edit me
        for c in R["cases"]:
            if c["expected"] == "review" or c["id"] == "override":
                a = jev.ask(c["state"], fixtures.questions(R2))["answers"]["route"]
                print(f"{c['id']:<10} policy={c['expected']:<7} → {a['value']:<7} p={a['probabilities']}")
    else:
        print("Live only. Add TYPESAFE_API_KEY to .env to edit criteria and rerun.")
    """),
    md("""
    ## Takeaways

    * Rules break on meaning; both models follow the written policy on these 24 fixtures.
    * The trade-off between the two models here is latency, cost and the answer contract
      (typed answer + probabilities vs a JSON label), not accuracy on this small set.
    * These are synthetic fixtures, not a held-out benchmark.
    """),
], "Demo 2")

# =============================================================================== 03 exercise
# The Exercise 1 notebook is generated by scripts/nb_ex03.py

# The Exercise 1 solution notebook is generated by scripts/nb_sol03.py

# =============================================================================== 04 exercise
# The Exercise 2 notebooks are generated by scripts/nb_ex04.py
