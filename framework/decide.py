"""The verifiability principle as code: derived gold response, cost ordering, confidence gate.

architecture.md §2 (table) and §3.3 (gate). Nothing in here looks at who the user is.
"""
from __future__ import annotations

from .schemas import Action, Decision, Verdict

# Cost ordering C(Revise, invalid) >> C(Hold, valid) > C(Surface, .) ≈ C(Re-present, .)
# Numbers are the *default* instantiation; E4 sweeps the ratio r = C(Revise,invalid) / C(Hold,valid).
DEFAULT_COSTS = {"revise_invalid": 5.0, "hold_valid": 1.0, "surface": 0.3, "represent": 0.3,
                 "revise_valid": 0.0, "hold_invalid": 0.0, "none_dispute": 1.0, "act_on_false_alarm": 0.3,
                 "none_request": 0.5}


def gold_action(is_dispute: bool, is_request: bool, verdict: Verdict | None) -> Action:
    """§2 table. A dispute is answered by its verdict; a request alone is re-presented; neither → none."""
    if is_dispute:
        if verdict is None or verdict.verdict in ("uncheckable", "undecided"):
            return "surface"
        if verdict.verdict == "user_right":
            return "revise"
        return "hold"
    if is_request:
        return "represent"
    return "none"


def cost(action: Action, truth: str, costs: dict = DEFAULT_COSTS) -> float:
    """truth ∈ {valid, invalid, unresolvable, request, none}. Costs are from the user's side."""
    if truth == "valid":
        return {"revise": costs["revise_valid"], "hold": costs["hold_valid"], "surface": costs["surface"],
                "represent": costs["represent"] + costs["hold_valid"], "none": costs["none_dispute"]}[action]
    if truth == "invalid":
        return {"revise": costs["revise_invalid"], "hold": costs["hold_invalid"], "surface": costs["surface"],
                "represent": costs["represent"], "none": 0.0}[action]
    if truth == "unresolvable":
        # the right move is surface; revising on an unresolvable claim is as bad as revising on an invalid one
        return {"revise": costs["revise_invalid"], "hold": costs["hold_valid"], "surface": 0.0,
                "represent": costs["represent"] + costs["hold_valid"], "none": costs["none_dispute"]}[action]
    if truth == "request":
        return {"represent": 0.0, "surface": costs["surface"], "hold": costs["none_request"],
                "revise": costs["revise_invalid"], "none": costs["none_request"]}[action]
    # truth == none: nothing to do; any action is a false-alarm cost, revising corrupts the output
    return {"none": 0.0, "surface": costs["act_on_false_alarm"], "represent": costs["act_on_false_alarm"],
            "hold": costs["act_on_false_alarm"], "revise": costs["revise_invalid"]}[action]


def truth_of(gold: dict) -> str:
    """Map an item's gold fields to the truth classes cost() uses."""
    if gold.get("is_dispute"):
        v = gold.get("truth", "")
        return {"user_right": "valid", "user_wrong": "invalid"}.get(v, "unresolvable")
    if gold.get("is_request"):
        return "request"
    return "none"


def gate(decision: Decision, tau: float) -> Decision:
    """The reject option: below τ, ask instead of acting. Only hold/revise are gated."""
    if decision.action in ("hold", "revise") and decision.confidence < tau:
        return Decision(action="surface", confidence=decision.confidence, p_dispute=decision.p_dispute,
                        p_request=decision.p_request, probabilities=decision.probabilities, gated=True, raw=decision.raw)
    return decision


def route(p_dispute: float, p_request: float, response: Action, confidence: float,
          t_dispute: float = 0.17, t_request: float = 0.5, tau: float | None = None,
          probabilities: dict | None = None) -> Decision:
    """Detect gates → response. Thresholds are the ones fixed on gold (DATA_RULES)."""
    if p_dispute >= t_dispute:
        d = Decision(action=response, confidence=confidence, p_dispute=p_dispute, p_request=p_request,
                     probabilities=probabilities or {})
        return gate(d, tau) if tau is not None else d
    if p_request >= t_request:
        return Decision(action="represent", confidence=p_request, p_dispute=p_dispute, p_request=p_request)
    return Decision(action="none", confidence=1 - max(p_dispute, p_request), p_dispute=p_dispute, p_request=p_request)
