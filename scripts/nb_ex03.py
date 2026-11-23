from nbkit import SETUP, md, py, write

CELLS = [
    md("""
    # Exercise 1 · Design three decisions

    **Part 03 · Labs** · 10 minutes

    Write one design per cell: **guardrails**, **tool gating** and **reranking**. Then run the checker.

    **What every design needs**

    | Part | Question to ask yourself |
    |---|---|
    | `state_fields` | Which facts must the model see to make this decision? |
    | `type` | Is the answer a category (Choice), an ordered degree (Score), or the truth of one condition (Noul)? |
    | `instructions` | What exactly is being judged? What happens when two options apply? Is text in the state data or instructions? |
    | `criteria` | What qualifies for each option, including a fallback when nothing fits? |

    * **Replay mode:** the checker runs structural checks: right type, a fallback, precedence,
      a sentence saying that text in the state is data, and the evidence the decision needs.
    * **Live mode:** it also sends your design to Jev on 8 labeled cases per task (need 7/8).

    Keep exact checks (allowlists, argument shapes, arithmetic) in code, not in the criteria.
    """),
    py(SETUP),
    py("""
    import json
    from dmw import fixtures, checks
    MODE = config.mode("auto")
    print("MODE =", MODE)

    def show_cases(task, n=3):
        print(f"{task}: example cases (state → expected answer)")
        for c in fixtures.get(task)["cases"][:n]:
            print(" ", json.dumps(c["state"], ensure_ascii=False)[:220], "→", c["expected"])
    """),

    # ------------------------------------------------------------------ guardrails
    md("""
    ## 1 · Guardrails

    **Decision:** does this text actively try to override the assistant's rules?

    Hints:
    * How many answers are there? One condition (yes or no) points to one type.
    * The same sentence can be fine from a user and an attack inside a retrieved web page. What must the
      state contain so the model can tell them apart?
    * Say that the content is evaluated, not obeyed, and that quoting an attack is not an attack.
    """),
    py('show_cases("guardrails")'),
    py("""
    GUARDRAILS = {
        "state_fields": [],   # TODO: which facts does the detector need? (hint: where did the text come from?)
        "type": None,         # TODO: "choice", "score" or "noul"
        "instructions": "",   # TODO
        "criteria": {},       # TODO: describe both "true" and "false"
    }
    """),

    # ------------------------------------------------------------------ gating
    md("""
    ## 2 · Tool gating

    **Decision:** may this proposed tool call run, does it need a person's approval, or must it be blocked?

    Hints:
    * Three options: `allow`, `request_approval`, `deny`.
    * The state needs the user's request, the proposed call, and the permissions the application owns.
    * What wins when a call is both a send and forbidden? Write the precedence as a sentence.
    * An email body that says “approval granted” must not grant approval. Say that the arguments are data.
    * Which option is the fallback when something is invalid or unclear?
    """),
    py('show_cases("gating")'),
    py("""
    GATING = {
        "state_fields": [],   # TODO: the request, the proposed call, and the application's permissions
        "type": None,         # TODO
        "instructions": "",   # TODO: include precedence, and say that arguments are data
        "criteria": {},       # TODO: "allow", "request_approval", "deny", each described
    }
    """),

    # ------------------------------------------------------------------ reranking
    md("""
    ## 3 · Reranking

    **Decision:** how useful is this passage for answering this question?

    Hints:
    * The levels are ordered (3 is better than 2). Which type keeps that order?
    * Judge coverage of the requested information, not shared keywords.
    * A passage that says “give me the maximum score” is data.
    * Write the levels lowest first, each describing an observable amount of coverage.
    """),
    py("""
    print("reranking: one question and its passages (reference grade 0..3)")
    print(" ", fixtures.reranking_pairs()[0]["question"])
    for p in fixtures.reranking_pairs()[:4]:
        print(" ", p["grade"], "|", p["text"])
    """),
    py("""
    RERANKING = {
        "state_fields": [],   # TODO
        "type": None,         # TODO
        "instructions": "",   # TODO
        "criteria": [],       # TODO: ordered levels, lowest first
    }
    """),

    # ------------------------------------------------------------------ code-side checks
    md("""
    ## 4 · The exact checks that stay in code (tool gating, optional if time allows)

    Before acting on `allow`, the application checks the call itself and fails closed: the model's answer can
    only make the gate stricter. Complete `permission_failure`: return a short reason string when a check
    fails, or `None` when everything is fine. The tool check is done for you.
    """),
    py("""
    def permission_failure(state):
        \"\"\"Exact, fail-closed checks over application-owned authorization data. None means OK.\"\"\"
        try:
            call, permissions = state["proposed_call"], state["trusted_permissions"]
            name, arguments = call["name"], call["arguments"]
            if name not in permissions["tools"]:
                return "missing_permission"
            # TODO: read_document must use a doc_id in permissions["documents"]  → "forbidden_document"
            # TODO: create_draft / send_email must use a "to" in permissions["recipients"]  → "forbidden_recipient"
            # TODO (bonus): exact argument names per tool  → "invalid_arguments"
        except (KeyError, TypeError):
            return "invalid_state"
        return None

    cases = {c["id"]: c for c in fixtures.get("gating")["cases"]}
    for case_id in ["draft", "missing_permission", "private_document", "forbidden_recipient"]:
        print(f"{case_id:<20} → {permission_failure(cases[case_id]['state']) or 'ok'}")
    """),

    # ------------------------------------------------------------------ checker
    md("## 5 · Run the checker\nStructural checks always; with `TYPESAFE_API_KEY` set it also runs each design on 8 labeled cases (need 7/8)."),
    py("""
    DESIGNS = {"guardrails": GUARDRAILS, "gating": GATING, "reranking": RERANKING}
    checks.check_design(DESIGNS)
    """),

    # ------------------------------------------------------------------ bonus
    md("""
    ## 6 · Bonus: your own decision

    Write the decision you posted in the opening poll as a fourth design, using the table at the top.
    """),
    py("""
    MY_DESIGN = {
        "state_fields": [],   # TODO
        "type": None,         # TODO
        "instructions": "",   # TODO
        "criteria": {},       # TODO
    }
    """),

    # ------------------------------------------------------------------ recap
    md("""
    ## 7 · Recap questions
    1. Which check failed first, and what did you change?
    2. Which part of tool gating did you leave to code, and why?
    3. Bonus: what is the state, the type, the precedence and the fallback of your own decision?
    """),
]

write("03_exercise_design_decisions.ipynb", CELLS, "Exercise 1")
