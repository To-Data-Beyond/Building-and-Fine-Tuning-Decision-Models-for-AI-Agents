"""Template-based training data for a decision model.

The ticket and domain templates come from the Stavro-4B build script
(stavro-4B/training/build_data.py). Labels are set by construction: a template
written for "bug" only ever produces bug tickets. No other model's outputs are used.

Pipeline (each step is a function you can call and inspect in the notebooks):

    generate()          templates -> scenarios, split by template (train / valid)
    drop_near_test()    remove scenarios too similar to any test case
    to_rows()           two option orderings per scenario -> chat rows with a letter target
    manifest()          counts and hashes, so a run can be reproduced
"""
from __future__ import annotations

import difflib
import hashlib
import json
import random
from collections import Counter

from .prompt import LETTERS, build_messages

rng = random.Random(30092026)  # reseeded by generate()

# ---------------------------------------------------------------------------- templates
TEAMS = {
    "billing": "payments, invoices, charges, refunds, prices, subscriptions",
    "bug": "something in the product is broken, wrong, slow or crashing",
    "account": "trouble with the customer's own login, password, two-factor code, profile or team members",
    "feature request": "asking for a new feature, setting, integration or improvement: anything the product doesn't do yet",
}
TEAMS_ALT = {  # same teams, other wording, so the model learns the idea and not one sentence
    "billing": "anything about money: what it costs, what was charged, invoices, plans and refunds",
    "bug": "the product misbehaves: errors, wrong numbers or dates, crashes, things not loading",
    "account": "who can sign in and how: passwords, 2FA, profiles, users, roles and workspaces",
    "feature request": "the customer wants the product to do something it can't do today",
}

def pick(*xs): return rng.choice(xs)
N = lambda: rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 40])
MONEY = lambda: f"{rng.choice([9, 12, 19, 24, 29, 35, 49, 59, 79, 99, 120, 240])} {pick('euro', 'EUR', 'dollars', '€', '$', 'pounds')}"
MONTH = lambda: pick("January", "March", "April", "June", "August", "October", "November")
PLAN = lambda: pick("team", "business", "starter", "pro", "enterprise", "basic")
PERSON = lambda: pick("Jonas", "Amira", "Priya", "Tom", "Ines", "Marek", "Yuki", "Lars", "Fatima", "Chen")
VIEW = lambda: pick("dashboard", "weekly report", "CSV export", "summary email", "project overview", "invoice preview", "time sheet", "analytics page")
PAGE = lambda: pick("the task list", "the calendar", "project settings", "the file browser", "the reports page", "the comments panel")

# (split, template) per team. The last template(s) of each team are validation-only.
TICKETS = {
    "billing": [
        lambda: f"How much extra would {N()} more {pick('users', 'seats', 'projects', 'guests')} cost us?",
        lambda: f"what does the {PLAN()} plan cost {pick('per month', 'per year', 'for 10 people')}?",
        lambda: f"Money is still leaving my {pick('bank account', 'account', 'card')} although I {pick('cancelled', 'closed the workspace', 'stopped using you')} in {MONTH()}.",
        lambda: f"The {pick('account', 'company account', 'main account')} was charged {MONEY()} {pick('twice', 'two times', 'again')} this month.",
        lambda: f"Can the invoice {pick('show our tax ID', 'include our cost centre', 'be addressed to our head office', 'list the seats separately')}?",
        lambda: f"Please move us to the {PLAN()} plan {pick('from next month', 'today', 'at renewal')}.",
        lambda: f"We were billed {MONEY()} but the pricing page says {MONEY()}.",
        lambda: f"Do {pick('students', 'charities', 'startups', 'teachers', 'public libraries')} get a lower price?",
        lambda: f"Our card on file {pick('expired', 'was replaced', 'was stolen')}, where do we update the payment details?",
        lambda: f"Can we pay {pick('quarterly', 'by invoice', 'in Swiss francs', 'with a purchase order', 'once a year')}?",
        lambda: f"Please refund the {MONEY()} {pick('for the months we did not use', 'from the double charge', 'for the seat we removed')}.",
        lambda: f"{pick('Oh brilliant', 'Wonderful', 'Great stuff')}, charged {pick('twice', 'three times', 'again')} for one month. {pick('Really smooth system.', 'Love it.', 'Top service.')}",
        lambda: f"is there tax on top of the {PLAN()} price",
        lambda: f"{pick('downgrade', 'cancel', 'pause')} our {pick('subscription', 'plan')} {pick('please', 'asap', 'at the end of the month')}",
        lambda: f"If we {pick('remove', 'add')} {N()} seats in the middle of the month, how is that {pick('billed', 'charged', 'refunded')}?",
        lambda: f"Where can I download the {pick('receipts', 'invoices', 'payment history')} for {pick('last year', 'Q2', 'the audit')}?",
    ],
    "bug": [
        lambda: f"The {VIEW()} shows {pick('revenue', 'hours', 'the task count', 'the budget', 'the balance')} as {pick('a negative number', 'zero', 'NaN', 'a million', 'the same value for every row')} since {pick('Monday', 'the update', 'yesterday', 'this morning')}.",
        lambda: f"{pick('Dates', 'Deadlines', 'Timestamps', 'Start times')} in the {VIEW()} are {pick('one day', 'two hours', 'a week', 'one month')} {pick('off', 'behind', 'ahead')}.",
        lambda: f"The {pick('totals', 'sums', 'subtotals', 'percentages')} in the {VIEW()} don't add up: {pick('10 + 15 shows 30', '40% + 60% shows 110%', '3 + 4 shows 34', '2.5 + 2.5 shows 4')}.",
        lambda: f"The '{pick('this month', 'last 30 days', 'this week', 'yesterday')}' filter {pick('includes', 'also shows')} {pick('last month', 'data from the week before', 'next week', 'two days')}.",
        lambda: f"{PAGE().capitalize()} {pick('crashes', 'freezes', 'shows a blank page', 'loads forever')} when I {pick('open it', 'sort by date', 'add a comment', 'switch projects')}.",
        lambda: f"The {pick('save', 'send', 'upload', 'export')} button does nothing in {pick('Firefox', 'Edge', 'Safari on iPhone', 'Chrome')}.",
        lambda: f"The progress bar says {pick('70', '85', '0', '120')}% even when every task is done.",
        lambda: f"Numbers lose their {pick('decimals', 'commas', 'minus sign', 'leading zeros')} when I save them.",
        lambda: f"{pick('Reminders', 'Scheduled posts', 'Monthly tasks')} set for {pick('the 29th', 'the 30th', 'the 31st', 'February 29')} {pick('never fire', 'disappear', 'jump to the next year')}.",
        lambda: f"{pick('search', 'sorting', 'the export', 'sync', 'login page')} broken since {pick('the update', 'yesterday', 'friday')}",
        lambda: f"The counter says {N()} {pick('new messages', 'open tasks', 'pending invites')} but the list is empty.",
        lambda: f"Duration column shows {pick('25 hours for one day', '-3 minutes', '999:99', '61 minutes in an hour')}.",
        lambda: f"{pick('Fantastic', 'Lovely', 'Perfect')}, {PAGE()} {pick('is blank', 'crashed', 'froze')} again. {pick('Third time today.', 'Every single morning.', 'Great quality.')}",
        lambda: f"Amounts in the {VIEW()} are {pick('rounded wrong', 'shown in cents', 'multiplied by 100', 'off by one')}.",
        lambda: f"The {pick('week numbers', 'age of each ticket', 'days-left count', 'time until the deadline')} {pick('is wrong by one', 'counts backwards', 'resets every Monday')}.",
        lambda: f"{pick('Charts', 'Graphs', 'Timelines')} {pick('skip', 'double count', 'hide')} {pick('weekends', 'the first day of each month', 'the last hour of the day')}.",
    ],
    "account": [
        lambda: f"How do I {pick('remove', 'add', 'invite')} {PERSON()} {pick('to our team', 'from the workspace', 'as a guest')}?",
        lambda: f"{pick('I do not get', 'Never receiving', 'Cannot get')} the {pick('2FA', 'two-factor', 'verification', 'sign-in')} code {pick('on my new phone', 'by SMS', 'in my email')}.",
        lambda: f"Please change my {pick('email address', 'name', 'username', 'profile picture')} to {pick('the new one', PERSON(), 'my work one')}.",
        lambda: f"How do I switch between {pick('my two workspaces', 'the two companies I belong to', 'my personal and work profile')}?",
        lambda: f"{pick('Our company merged with another', 'Two teams became one', 'We reorganised')}, can the two workspaces be combined?",
        lambda: f"Make {PERSON()} {pick('an admin', 'the owner', 'a viewer only', 'a project manager')}.",
        lambda: f"The password reset email {pick('never arrives', 'goes to my old address', 'link is already expired')}.",
        lambda: f"Can we have {pick('one shared login', 'a common account')} for {pick('the front desk', 'our warehouse', 'the reception tablet')}?",
        lambda: f"{pick('Delete', 'Close', 'Remove')} my {pick('profile', 'user', 'login')} {pick('permanently', 'please', 'for good')}.",
        lambda: f"{pick('locked out', 'cant log in', 'account locked', 'login not working for me')}{pick('', ' help', ' pls', '!!')}",
        lambda: f"{PERSON()} left the company, {pick('who can take over their projects', 'please disable their login', 'remove their access')}?",
        lambda: f"I {pick('lost', 'broke', 'replaced')} my phone and the authenticator app with it.",
        lambda: f"Which of our users {pick('still have admin rights', 'never accepted the invite', 'have not logged in for months')}?",
        lambda: f"My {pick('account', 'login', 'profile')} shows {pick('the name of another person','the wrong company', 'an old email')}.",
        lambda: f"{pick('Love', 'Adore', 'So enjoy')} {pick('typing my password every hour', 'the 2FA code that never comes', 'getting signed out constantly')}. {pick('Really great.', 'Wonderful.', 'Brilliant.')}",
        lambda: f"How do I give {PERSON()} access to {pick('one project only', 'the reports but nothing else', 'read-only files')}?",
    ],
    "feature request": [
        lambda: pick("kanban board", "time tracking", "custom fields", "sub-tasks", "timeline view", "emoji reactions",
                     "webhooks", "bulk edit", "print view", "PDF export", "recurring tasks", "task dependencies",
                     "multi-language UI", "tags", "a Linux app", "SSO with Okta", "passkeys", "a public API",
                     "Trello import", "Teams integration", "two-way calendar sync", "read receipts") + pick("", "?", " pls", " please", " when?", " would be nice", "!"),
        lambda: f"Any chance you'll add {pick('a Kanban view', 'time tracking', 'custom colours for labels', 'an iPad app', 'a Jira import', 'comment threads')}?",
        lambda: f"It would help us a lot if {pick('we could set custom statuses', 'reports could be scheduled', 'tasks could have checklists', 'we could lock finished projects', 'the app supported right-to-left languages')}.",
        lambda: f"Why can't I {pick('undo a delete', 'drag files between projects', 'copy a whole board', 'set a default view', 'favourite a report')} yet? {pick('Every other tool can.', 'This is basic.', 'Competitors had this for years.')}",
        lambda: f"Please let {pick('admins', 'project owners', 'guests', 'managers')} {pick('see who changed what', 'set their own dashboards', 'export everything at once', 'choose the default language')}.",
        lambda: f"Could {pick('invoices', 'reports', 'exports', 'emails')} {pick('include a chart', 'be sent in German', 'have our logo', 'group items by client')}?",
        lambda: f"Feature idea: {pick('dark theme for the mobile app', 'a focus mode', 'voice notes', 'a weekly digest', 'auto-archive old tasks')}.",
        lambda: f"{pick('Would love', 'We need', 'Please add')} {pick('an integration with HubSpot', 'a Zoom integration', 'Google Drive attachments', 'a WordPress plugin', 'GitHub sync')}.",
        lambda: f"Can you support {pick('login with a hardware key', 'SAML for our IdP', 'passwordless login', 'separate 2FA rules per team')}?",
        lambda: f"Let me {pick('mute one project', 'hide archived boards', 'rename default columns', 'pin a filter', 'choose my start page')}.",
        lambda: f"{pick('Could we choose', 'Please let us set', 'Add an option for')} {pick('which currency reports use', 'the first day of the week', 'the date format', 'a 24-hour clock')}.",
        lambda: f"{pick('mobile widgets', 'desktop notifications', 'keyboard shortcuts', 'an offline mode for the app', 'a gallery view')}{pick('', '?', ' pls')}",
        lambda: f"{pick('Is there a way to', 'Can I')} {pick('set up automatic reminders for clients', 'build my own report layout', 'bulk-assign tasks')}? If not, please add it.",
        lambda: f"Suggestion: {pick('show week numbers', 'allow comments on files', 'let tasks repeat every other week', 'add a Gantt view')}.",
    ],
}
VALID_TEMPLATES = 2  # last N templates of each team go to validation only

OTHER_TASKS = [  # other routing domains so the skill generalises beyond one inbox
    ("IT request: ", "Which IT group should take this request?", {
        "hardware": "a physical device: laptop, screen, printer, phone, keyboard or cables",
        "software": "an application, licence, update or program that misbehaves or needs installing",
        "network": "internet, Wi-Fi, VPN or connection problems",
        "access": "passwords, logins, permissions or accounts for a person"}, {
        "hardware": [lambda: f"My {pick('laptop', 'monitor', 'docking station', 'printer', 'headset')} {pick('will not turn on', 'makes a clicking noise', 'has a cracked screen', 'stopped charging')}.",
                     lambda: f"Need a {pick('second screen', 'new mouse', 'replacement charger', 'webcam')} for {PERSON()}."],
        "software": [lambda: f"{pick('Excel', 'The PDF reader', 'Outlook', 'The design tool')} {pick('crashes on start', 'asks for a licence again', 'will not update', 'freezes when saving')}.",
                     lambda: f"Please install {pick('Python', 'the new CRM client', 'a zip tool', 'Figma')} on my machine."],
        "network": [lambda: f"{pick('Wi-Fi', 'The VPN', 'Internet', 'The office network')} {pick('drops every few minutes', 'is very slow on floor 3', 'will not connect from home', 'is down in the meeting room')}.",
                    lambda: f"Can't reach {pick('the shared drive', 'any website', 'the intranet')} since I moved desks."],
        "access": [lambda: f"{PERSON()} {pick('needs access to the finance folder', 'is locked out of email', 'forgot the laptop password', 'starts Monday and needs an account')}.",
                   lambda: f"Please remove {PERSON()}'s {pick('login', 'VPN access', 'admin rights')}, they left."]}),
    ("Hotel guest message: ", "Which hotel department should handle this message?", {
        "reception": "check-in, check-out, bookings, bills, keys and general questions",
        "housekeeping": "cleaning, towels, bedding and room supplies",
        "maintenance": "something in the room is broken: heating, water, lights, doors, TV",
        "restaurant": "food, breakfast, table reservations and room service"}, {
        "reception": [lambda: f"Can we {pick('check out late', 'extend by one night', 'get a copy of the bill', 'store luggage after check-out')}?",
                      lambda: f"My key card {pick('stopped working', 'was lost', 'does not open the gym')}."],
        "housekeeping": [lambda: f"Could we get {pick('fresh towels', 'an extra pillow', 'more coffee pods', 'a cot for the baby')}?",
                         lambda: f"The room {pick('was not cleaned today', 'still has sand on the floor', 'needs new sheets')}."],
        "maintenance": [lambda: f"The {pick('shower', 'heating', 'air conditioning', 'TV', 'window lock')} in room {rng.randint(101, 480)} {pick('does not work', 'is leaking', 'makes a loud noise', 'is stuck')}.",
                        lambda: f"{pick('No hot water', 'Lights flicker', 'The door will not close')} in our room."],
        "restaurant": [lambda: f"Can we book a table for {rng.randint(2, 8)} {pick('at 8 pm', 'tomorrow evening', 'for lunch')}?",
                       lambda: f"Is breakfast {pick('gluten free friendly', 'served until 11', 'included in our rate')}?"]}),
]




# ---------------------------------------------------------------------------- pipeline
def generate(seed: int = 30092026, train_per_template: int = 10, valid_per_template: int = 4,
             other_per_template: int = 6, alt_description_rate: float = 0.3,
             valid_templates: int = VALID_TEMPLATES):
    """Render scenarios from the templates.

    The last `valid_templates` templates of each support team are validation-only,
    so validation measures new wording, not memorized templates. For the other
    domains, the last template of each label goes to validation 30% of the time.
    Returns (train, valid): lists of dicts with context, question, descriptions, expected, text, template.
    """
    rng.seed(seed)
    train, valid, seen = [], [], set()

    def add(split, text, prefix, question, descs, want, template):
        if text.lower() in seen:
            return
        seen.add(text.lower())
        (train if split == "train" else valid).append(dict(
            context=prefix + text, question=question, descriptions=dict(descs), expected=want,
            text=text, template=template, domain=prefix.strip(": ").lower()))

    for team, templates in TICKETS.items():
        for i, make in enumerate(templates):
            split = "valid" if i >= len(templates) - valid_templates else "train"
            for _ in range(train_per_template if split == "train" else valid_per_template):
                descs = TEAMS_ALT if rng.random() < alt_description_rate else TEAMS
                add(split, make(), "Support ticket: ",
                    pick("Which team should handle this ticket?", "Which team should this ticket go to?"),
                    descs, team, f"{team}#{i}")
    for prefix, question, descs, groups in OTHER_TASKS:
        for want, makes in groups.items():
            for i, make in enumerate(makes):
                for _ in range(other_per_template):
                    split = "valid" if i == len(makes) - 1 and rng.random() < 0.3 else "train"
                    add(split, make(), prefix, question, descs, want, f"{prefix.strip(': ')}/{want}#{i}")
    return train, valid


def near_duplicates(texts: list[str], test_texts: list[str], cutoff: float = 0.8) -> set[str]:
    """Texts whose closest test case has a difflib similarity >= cutoff."""
    tests = [t.lower() for t in test_texts]
    return {t for t in texts if difflib.get_close_matches(t.lower(), tests, 1, cutoff)}


def max_similarity(texts: list[str], test_texts: list[str]) -> float:
    """Highest similarity between any text and any test text (for the report)."""
    best = 0.0
    for t in texts:
        for u in test_texts:
            sm = difflib.SequenceMatcher(None, t.lower(), u.lower())
            if sm.real_quick_ratio() > best and sm.quick_ratio() > best:
                best = max(best, sm.ratio())
    return best


def drop_near_test(train: list[dict], valid: list[dict], test_texts: list[str], cutoff: float = 0.8):
    bad = near_duplicates([s["text"] for s in train + valid], test_texts, cutoff)
    keep = lambda xs: [s for s in xs if s["text"] not in bad]
    return keep(train), keep(valid), sorted(bad)


def to_rows(scenarios: list[dict], seed: int = 3033) -> list[dict]:
    """Two orderings per scenario (shuffled + reversed), each a chat row ending in the answer letter."""
    r = random.Random(seed)
    rows = []
    for s in scenarios:
        order = list(s["descriptions"])
        r.shuffle(order)
        for o in (order, order[::-1]):
            msgs = build_messages(s["context"], o, None, s["question"], {c: s["descriptions"][c] for c in o})
            rows.append({"messages": msgs, "answer": LETTERS[o.index(s["expected"])], "label": s["expected"],
                         "template": s.get("template"), "domain": s.get("domain")})
    r.shuffle(rows)
    return rows


def manifest(train_s, valid_s, train_rows, valid_rows, dropped, seed: int = 30092026) -> dict:
    h = lambda rows: hashlib.sha256("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows).encode()).hexdigest()
    return {
        "seed": seed,
        "scenarios": {"train": len(train_s), "valid": len(valid_s)},
        "rows": {"train": len(train_rows), "valid": len(valid_rows)},
        "train_by_label": dict(Counter(s["expected"] for s in train_s)),
        "valid_by_label": dict(Counter(s["expected"] for s in valid_s)),
        "dropped_near_test": len(dropped),
        "sha256": {"train": h(train_rows), "valid": h(valid_rows)},
    }


# ---------------------------------------------------------------------------- your own domain (Exercise 4)
def scenarios_from_domain(domain: dict, per_template: int = 8, seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Render scenarios for a domain you define.

    domain = {
      "prefix": "Airline customer message: ",
      "question": "Which team should handle this message?",
      "choices": {label: description, ...},
      "templates": {label: [callable or str, ...], ...},   # str may contain {slots}
      "slots": {"slot": [values, ...], ...},                 # optional
      "valid_templates_per_label": 1,
    }
    The last `valid_templates_per_label` templates of every label are validation-only.
    """
    r = random.Random(seed)
    slots = domain.get("slots", {})
    k = domain.get("valid_templates_per_label", 1)
    train, valid, seen = [], [], set()

    def render(t):
        if callable(t):
            return t()
        return t.format(**{name: r.choice(vals) for name, vals in slots.items()})

    for label, templates in domain["templates"].items():
        if label not in domain["choices"]:
            raise ValueError(f"Template label {label!r} is not one of the choices")
        for i, t in enumerate(templates):
            split = valid if i >= len(templates) - k else train
            for _ in range(per_template):
                text = render(t)
                if text.lower() in seen:
                    continue
                seen.add(text.lower())
                split.append(dict(context=domain["prefix"] + text, question=domain["question"],
                                  descriptions=dict(domain["choices"]), expected=label, text=text,
                                  template=f"{label}#{i}", domain=domain["prefix"].strip(": ").lower()))
    return train, valid


# ---------------------------------------------------------------------------- test sets
def load_test(path) -> list[dict]:
    """Load a test file with presets into flat cases: id, group, context, question, choices, expected."""
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for c in data["cases"]:
        p = data["presets"][c["preset"]] if "preset" in c else c
        out.append({"id": c["id"], "group": c.get("group", c.get("preset")), "context": p["prefix"] + c["text"],
                    "text": c["text"], "question": p["question"], "choices": dict(p["choices"]),
                    "expected": c["expected"], "note": c.get("note")})
    return out
