"""Track 2: the decision model (Jev, TypeSafe System One) as Detect and Decide. Used as is; nothing trained.

Given an item: state = task + explanation + user turn (+ labels, + verification, + role when the
condition says so). Questions: is_dispute (noul), is_request (noul), response (choice hold/revise/surface).
Replies are templated from adapter facts (respond.py), never generated here.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..api import jev_decide
from ..decide import route
from ..schemas import Decision, GROUNDS_CHECKABLE

ROOT = Path(__file__).resolve().parents[2]
TAX = json.loads((ROOT / "data" / "taxonomy_v1.2.json").read_text())
RULES = (f"{TAX['definition']} {TAX['core_test']} Rules: " + " ".join(TAX["boundary_rules"]) +
         " Also not contestation: " + " ".join(TAX["also_no"]))
ER = TAX["explanation_request"]
REQUEST_RULES = f"{ER['definition']} Test: {ER['test']} Not a request: " + " ".join(ER["not_a_request"]) + f" {ER['relation_to_contestation']}"

RESPONSE_CRITERIA = {
    "hold": "Keep the current output and explanation; show the user the evidence that settles the point against their objection.",
    "revise": "Accept the objection: correct the output or the explanation as the user says and show what changed.",
    "surface": "Do not resolve it: show the user's view and the system's view side by side, marked unresolved, and ask the user for what would settle it.",
}
RESPONSE_INSTRUCTIONS = (
    "Choose the system's response to the user's objection. The right response depends on whether the objection can be "
    "checked against the model, its data, or the display, and on the result of that check: an objection shown to be right "
    "must be accepted (revise); one shown to be wrong must be answered with the evidence (hold); one that cannot be checked, "
    "or whose check is inconclusive, must be shown as unresolved (surface). A wrong revision corrupts the output; a wrong hold "
    "loses a correction; surfacing costs one turn. Who the user is does not change the answer."
)

QUESTIONS = {
    "is_dispute": {"type": "noul", "instructions": RULES,
                   "criteria": {"true": "The user disputes what the system asserts or does (its prediction, reasoning, evidence, data, ranking, constraint status, or a display that blocks or distorts the correct reading), calling for justification, correction, reconsideration, or acknowledgment of uncertainty.",
                                "false": "No dispute of content: incomprehension, a request for another form or level of explanation, a stated preference, personal uncertainty, reading or agreeing with the output, or small talk."}},
    "is_request": {"type": "noul", "instructions": REQUEST_RULES,
                   "criteria": {"true": "A different explanation would help the user here: another format, a plainer level, a legend, a unit, a term definition, or a datum the explanation should have shown; includes saying they cannot read or understand it.",
                                "false": "The user reads, interprets, agrees, or disputes the content on its merits, with no sign that the explanation's form or level failed them."}},
    "response": {"type": "choice", "instructions": RESPONSE_INSTRUCTIONS, "criteria": RESPONSE_CRITERIA},
}

TASKS = {
    "cfxhpo": "A conversational assistant for hyperparameter optimisation of an industrial visual anomaly detector recommended "
              "configuration options to an operator (a table of options with predicted quality and constraint status, plus a "
              "generated explanation). The operator replies. Judge the operator's TURN.",
    "fungi": "A think-aloud study participant looks at an explanation of an AI mushroom-edibility classifier (one of five formats) "
             "and comments. Judge the participant's TURN.",
}


def build_state(item: dict, episode_state: dict, labels: bool, verification: str | None, role: str | None) -> dict:
    st = {"task": TASKS[item["domain"]], "explanation": episode_state, "user_turn": item["user_text"]}
    if labels and item.get("target"):
        st["labels"] = {"target": item["target"], "interaction_act": item["interaction_act"], "grounds": item["grounds"],
                        "grounds_checkable_in_principle": GROUNDS_CHECKABLE.get(item["grounds"], "")}
    if verification is not None:
        st["verification"] = verification
    if role:
        st["user_role"] = role
    return st


def decide(item: dict, episode_state: dict, labels: bool = True, verification: str | None = None, role: str | None = None,
           repeat: int = 0, t_dispute: float = 0.17, t_request: float = 0.5, tau: float | None = None) -> Decision:
    st = build_state(item, episode_state, labels, verification, role)
    res = jev_decide(st, QUESTIONS, repeat=repeat)
    a = res["answers"]
    probs = {k: float(v) for k, v in a["response"]["probabilities"].items()}
    resp = a["response"]["choice"]
    d = route(float(a["is_dispute"]["noul"]), float(a["is_request"]["noul"]), resp, probs.get(resp, float(a["response"].get("confidence", 0))),
              t_dispute=t_dispute, t_request=t_request, tau=tau, probabilities=probs)
    d.raw = {"model": res.get("model"), "cost": res.get("usage", {}).get("cost")}
    return d
