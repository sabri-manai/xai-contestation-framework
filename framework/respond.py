"""Step 5, Respond: the text the user sees, built only from adapter facts.

Track 2 (decision model) uses the templates below verbatim. Track 1 (language model) may rephrase, but
is given only the same facts and is told to add nothing; E6 rates faithfulness against those facts.
"""
from __future__ import annotations

import json

from .api import chat
from .schemas import Decision, Verdict


def templated(action: str, user_text: str, verdict: Verdict | None, changed: dict | None, represented: str | None,
              claim_text: str = "") -> str:
    ev = "; ".join(verdict.evidence) if verdict else ""
    if action == "hold":
        return (f"I checked this against the model and its data. {ev}. "
                f"So I am keeping the current result{(' (' + claim_text + ')') if claim_text else ''}. If you have information that this check does not cover, tell me and I will show both views.")
    if action == "revise":
        ch = "; ".join(f"{k}: {v}" for k, v in (changed or {}).items())
        return f"You are right. {ev}. I have changed the result accordingly: {ch or 'the explanation now reflects this'}."
    if action == "surface":
        return (f"I cannot settle this from the model or its data. Your view: \"{user_text.strip()}\" "
                f"The system's view: {claim_text or 'the result as shown'}{(' (' + ev + ')') if ev else ''}. "
                f"I am marking this as unresolved. What would settle it for you: a retraining run, a different data source, or a specific case?")
    if action == "represent":
        return represented or "Here is the same explanation in a different form."
    return ""


LLM_REPLY_SYSTEM = (
    "You write the reply of an explainable-AI system to a user who commented on an explanation. You are given the decided "
    "ACTION and the FACTS the system established. Write 2-4 sentences in plain language addressed to the user. Use only the "
    "FACTS; add no number, claim, or promise that is not in them. hold = keep the result and give the evidence; revise = accept "
    "the user's point and say what changed; surface = present both views as unresolved and ask what would settle it; "
    "represent = re-explain in the requested form; none = a brief acknowledgement only."
)


def llm_reply(model: str, user_text: str, action: str, facts: dict, repeat: int = 0) -> str:
    msgs = [{"role": "system", "content": LLM_REPLY_SYSTEM},
            {"role": "user", "content": f"USER SAID:\n{user_text}\n\nACTION: {action}\n\nFACTS:\n{json.dumps(facts, ensure_ascii=False, indent=1)}"}]
    return chat(model, msgs, max_tokens=220, repeat=repeat, cache_kind="llm_reply")["text"]
