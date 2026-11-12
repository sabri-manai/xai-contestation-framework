"""Steps 1–5 as one interface, for the live loop (E5) and the demo.

    run_turn(adapter, user_text, track, ...) → dict(detect, labels, verdict, decision, reply)

Detect and Decide come from the chosen track; Understand maps the free-text turn onto one of the adapter's
claims (a language model does it in both tracks: Jev does not generate structure, and its structure answers
were at chance on gold); Verify is the adapter; Respond is templated (Track 2) or constrained (Track 1).
"""
from __future__ import annotations

import json
from dataclasses import asdict

from ..api import chat, parse_json
from ..decide import route
from ..respond import llm_reply, templated
from ..schemas import Decision, Grounds, Verdict
from . import decision_model as dm
from . import language_model as lm
from .understand import understand as understand_v2

UNDERSTAND_MODEL = "openai/gpt-4o-mini"

CLAIM_KINDS = {
    "cfxhpo": {
        "score_reach": "the user says an option will not reach the required minimum quality once actually trained → user_value {threshold}",
        "score_over": "the user says the table overstates an option's quality (the real value will be more than two points lower) → user_value {}",
        "rank_confirmed": "the user says a lower-ranked option will beat a higher-ranked one when trained → user_value {better, worse}",
        "satisfies": "the user says an option violates a soft constraint that the summary marks as satisfied → user_value {feature, user_says_violates: true, user_limit: {max} if the user states their own limit}",
        "rank_table": "the user misreads which option has the higher quality in the table → user_value {better, worse}",
        "feature_effect": "the user says a soft control (review budget, training fraction) makes no difference to quality → user_value {feature, new_value, user_says_no_effect: true}",
        "density": "the user says the log has no trials near an option (extrapolation) → user_value {}",
        "log_stat": "the user makes a claim about the optimisation log as a whole (share of corrupted trials etc.) → user_value {stat: corrupted_share|pretrained_off_share|n_trials, op: '>'|'<', threshold}",
        "units": "the user misreads units (fractions vs percent) → user_value {}",
        "uncheckable": "the claim is about the production line, the user's past experience, or anything the log and surrogate do not record",
        "request": "not a dispute: the user asks for a term, a different format, or plainer words → subject = feature name, 'format', or 'plain'; user_value {format: percent|plain} if applicable",
        "none": "nothing to respond to: acknowledgement, agreement, small talk",
    },
    "fungi": {
        "fact": "the user asserts something about what the slide shows (a value, a feature, a legend) → subject = the fact key from the facts list, user_value = what the user asserts",
        "display_fault": "the user points at a fault of the display (typo, duplicate, inconsistent unit) → subject = the fact key, user_value true",
        "data_stat": "the user asserts a statistic of the dataset → subject = the fact key, user_value = the asserted number or boolean",
        "model_effect": "the user claims how the model would behave if a feature were changed or removed",
        "uncheckable": "a claim from mushroom knowledge, personal experience, or about the model's competence that the slide and data cannot settle",
        "request": "not a dispute: the user asks for a legend, a term, or plainer words → subject = legend|term|plain",
        "none": "nothing to respond to",
    },
}


def understand_llm(domain: str, user_text: str, state: dict, option_names: list[str] | None = None, repeat: int = 0) -> Grounds:
    """Map the turn onto taxonomy labels and the claim it attacks. The only generative step in Track 2."""
    sys_ = ("You read one user turn about an explanation and fill a JSON record describing it for a verifier. Output exactly: "
            '{"target": <prediction|reasoning|evidence|input data|explanation representation|system competence|"">, '
            '"interaction_act": <request for justification|direct challenge|correction|alternative proposal|counterexample|rejection|request to acknowledge uncertainty|"">, '
            '"grounds": <domain knowledge|technical knowledge|prior experience|internal inconsistency|missing evidence|data-quality concern|causal implausibility|presentation ambiguity|"">, '
            '"claim_kind": <one of the kinds below>, "subject": <"Option A" etc., a feature name, a fact key, or "">, "user_value": <object or value as described>}\n'
            "Claim kinds:\n" + "\n".join(f"- {k}: {v}" for k, v in CLAIM_KINDS[domain].items()))
    user = json.dumps({"explanation": state, "user_turn": user_text, "options": option_names or []}, ensure_ascii=False)
    out = chat(UNDERSTAND_MODEL, [{"role": "system", "content": sys_}, {"role": "user", "content": user}], max_tokens=250, json_mode=True,
               repeat=repeat, cache_kind="understand")
    try:
        j = parse_json(out["text"])
        uv = j.get("user_value")
        if isinstance(uv, str):
            try:
                uv = json.loads(uv)
            except Exception:
                pass
        return Grounds(j.get("target", "") or "", j.get("interaction_act", "") or "", j.get("grounds", "") or "",
                       j.get("claim_kind", "") or "", str(j.get("subject", "") or ""), uv, extra={"raw": j})
    except Exception as e:
        return Grounds(extra={"error": str(e)})


def run_turn(adapter, domain: str, user_text: str, track: str = "decision", tau: float | None = None, role: str | None = None,
             labels: Grounds | None = None, oracle: bool = True, llm_model: str = "openai/gpt-4o-mini", repeat: int = 0,
             understand: str = "v2", clarify: bool = True) -> dict:
    state = adapter.state()
    names = [f"Option {chr(65 + i)}" for i in range(len(getattr(adapter, "options", [])))] or None
    if labels is not None:
        g = labels
    elif understand == "v2":
        g = understand_v2(adapter, domain, user_text, track=track, repeat=repeat)
    else:
        g = understand_llm(domain, user_text, state, names, repeat=repeat)
    g.extra = dict(g.extra or {}, user_text=user_text)
    if clarify and g.extra.get("clarify") and g.claim_kind not in ("none", "request"):
        # Understand is unsure which claim the user means: ask, rather than verify the wrong thing
        menu_text = next((e["text"] for e in adapter.claim_menu() if e["id"] == g.extra.get("menu_id")), "")
        d = Decision(action="surface", confidence=float(g.extra.get("choice_confidence", 0)), gated=True)
        reply = (f"I want to be sure I understand you. Do you mean that {menu_text}? "
                 f"If so I will check it against the model and its data; if not, tell me what you are objecting to.")
        return {"labels": asdict(g), "verdict": None, "decision": d.to_dict(), "reply": reply, "clarified": True}
    item = {"domain": domain, "user_text": user_text, "target": g.target, "interaction_act": g.act, "grounds": g.grounds}
    # Verify first (cheap tiers); its text is an input to Decide
    verdict = None
    if g.claim_kind not in ("", "request", "none"):
        try:
            verdict = adapter.verify(g, oracle=oracle)
        except Exception as e:
            verdict = Verdict("no", "uncheckable", [f"verifier error: {e}"], tier="none")
    vt = verdict.as_text() if verdict else None
    if track == "decision":
        d = dm.decide(item, state, labels=True, verification=vt, role=role, repeat=repeat, tau=tau)
    else:
        d = lm.decide(llm_model, item, state, labels=True, verification=vt, role=role, repeat=repeat, tau=tau)
    # Respond
    changed, represented = None, None
    if d.action == "revise" and verdict is not None and verdict.verdict == "user_right":
        changed = adapter.revise(g, verdict)["changed"]
    elif d.action == "revise":
        # the decider wants to yield but the verifier gave it nothing to yield on: say so rather than invent a change
        d = Decision(action="surface", confidence=d.confidence, p_dispute=d.p_dispute, p_request=d.p_request, probabilities=d.probabilities, gated=False)
    if d.action == "represent":
        try:
            represented = adapter.represent(g)
        except Exception as e:
            represented = f"(re-presentation failed: {e})"
    if track == "decision":
        reply = templated(d.action, user_text, verdict, changed, represented)
    else:
        reply = llm_reply(llm_model, user_text, d.action, {"verification": vt or "", "changed": changed or {}, "re-presented": represented or ""}, repeat=repeat) if d.action != "none" else ""
    return {"labels": asdict(g), "verdict": asdict(verdict) if verdict else None, "decision": d.to_dict(), "reply": reply, "clarified": False}
