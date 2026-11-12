"""E6: response quality. Builds the replies of both tracks for a stratified sample and writes blind rating sheets.

Track 2 reply = template from adapter facts (respond.templated). Track 1 reply = language model given the same
decision and facts, told to add nothing (respond.llm_reply). ~40 items per track, same items for both, order
shuffled, track hidden; the key is kept separately. Raters score appropriateness (1–5) and faithfulness
(every statement traceable to the facts: yes / partly / no).

Writes benchmark/outputs/e6_rating_sheet.csv (for raters) and e6_rating_key.csv (not for raters).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..adapters.cfxhpo import CFXHPOAdapter
from ..adapters.fungi import FungiAdapter
from ..respond import llm_reply, templated
from ..schemas import Grounds, Verdict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "outputs"
LLM = "openai/gpt-4o-mini"
SEED = 20261112
N_PER_TRACK = 40


def main() -> None:
    rng = np.random.default_rng(SEED)
    dec = pd.read_csv(OUT / "e2e3_items.csv", keep_default_na=False)
    dec = dec[(dec.decider == "jev") & (dec.condition == "oracle") & (dec.role_given.astype(str).str.lower() == "false") & (dec["repeat"].astype(int) == 0)]
    c = pd.read_csv(ROOT / "data" / "partC_cfxhpo.csv", keep_default_na=False)
    b = pd.read_csv(ROOT / "data" / "partB_pairs.csv", keep_default_na=False)
    eps = {e["episode_id"]: e for e in json.loads((ROOT / "data" / "partC_episodes.json").read_text())}
    items = pd.concat([c.assign(part="C"), b.assign(part="B")], ignore_index=True).merge(dec[["item_id", "action", "confidence"]], on="item_id")
    # stratified sample: ~8 per decided action, both parts
    sample = pd.concat([g.sample(min(8, len(g)), random_state=int(rng.integers(1e9))) for _, g in items.groupby("action")]).head(N_PER_TRACK)
    rows, key = [], []
    for _, it in sample.iterrows():
        g = Grounds(it.target, it.interaction_act, it.grounds, it.claim_kind, it.subject,
                    json.loads(it.user_value) if (it.part == "C" and it.user_value) else (it.user_value or None))
        if it.part == "C":
            ep = eps[it.episode_id]
            ad = CFXHPOAdapter(ep["dataset"], json.loads(json.dumps(ep["query"])), [dict(o) for o in ep["options"]], ep["user_request"])
            shown = ad.state()["explanation"]
        else:
            ad = FungiAdapter(it.explanation_format); shown = ad.state()["caption"]
        verdict = None; changed = None; represented = None
        if it.action in ("hold", "revise", "surface") and str(it.is_dispute).lower() == "true":
            verdict = ad.verify(g)
        if it.action == "revise" and verdict is not None:
            changed = ad.revise(g, verdict)["changed"]
        if it.action == "represent":
            represented = ad.represent(g)
        t2 = templated(it.action, it.user_text, verdict, changed, represented)
        facts = {"verification": verdict.as_text() if verdict else "", "changed": changed or {}, "re-presented": represented or ""}
        t1 = llm_reply(LLM, it.user_text, it.action, facts) if it.action != "none" else ""
        for track, reply in [("decision model (templated)", t2), ("language model (constrained)", t1)]:
            rid = f"R{len(rows) + 1:03d}"
            rows.append({"reply_id": rid, "domain": "hyperparameter optimisation" if it.part == "C" else "mushroom classifier",
                         "explanation_shown": shown, "user_said": it.user_text, "system_reply": reply,
                         "facts_available": json.dumps(facts, ensure_ascii=False),
                         "appropriateness_1_to_5": "", "faithful_yes_partly_no": "", "comment": ""})
            key.append({"reply_id": rid, "item_id": it.item_id, "track": track, "action": it.action, "gold_action": it.gold_action, "truth": it.truth})
    order = rng.permutation(len(rows))
    sheet = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    sheet.to_csv(OUT / "e6_rating_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(OUT / "e6_rating_key.csv", index=False)
    print(f"E6: {len(sheet)} replies ({len(sample)} items × 2 tracks) → e6_rating_sheet.csv; key in e6_rating_key.csv")
    print(pd.DataFrame(key).groupby(["track", "action"]).size())


if __name__ == "__main__":
    main()
