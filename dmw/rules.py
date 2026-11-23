"""The deliberately simple rule baselines from the labs."""


def length_route(request: str, limit: int = 100) -> str:
    """Routing rule: short requests go to the small model, everything else to large."""
    return "small" if len(request) < limit else "large"


def subject_triage(subject: str) -> str:
    """Triage rule: URGENT in the subject means reply_now, everything else later."""
    return "reply_now" if "urgent" in subject.lower() else "later"
