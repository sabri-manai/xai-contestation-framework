"""Part C: operator objections against CF-XHPO candidate tables, validity from the retraining bank.

Each episode = (dataset, natural-language request, parsed query, 3 options from the bank, template
explanation). Each item = one operator turn against that episode, with the taxonomy labels, the claim
it attacks, and the oracle truth. Disputes come in matched pairs (same template, one true, one false)
where the bank allows it. Written 12 Nov 2026; rule: the truth of every item is computed, never judged.

Writes data/partC_cfxhpo.csv and data/partC_episodes.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..adapters.cfxhpo import (CFXHPOAdapter, DENSITY_MIN, EFFECT_FLOOR, NOISE_FLOOR, TARGET, FEATURES, CATEGORICAL,
                               feat_label, fmt3, load_bank, option_name, _rank_records)
from ..decide import gold_action
from ..schemas import Grounds, Verdict

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261112


def make_query(opts: list[dict], rng: random.Random) -> tuple[str, dict]:
    scores = sorted(float(o[TARGET]) for o in opts)
    v = float(np.floor((scores[1] - 0.005) * 100) / 100)          # the middle option passes by a small margin
    budgets = sorted(float(o["params_soft_review_budget"]) for o in opts)
    bmax = round(float(np.ceil(budgets[1] * 200) / 200), 3)       # at most one option violates the budget
    tf = {"mode": "free"}
    corr = {"mode": "free"}
    if rng.random() < 0.5:
        corr = {"mode": "allowed", "allowed": ["none", "mild"]}
    q = {"objective": {"type": "target_min", "value": v, "eps": 0.01},
         "soft_constraints": {"params_soft_train_fraction": tf,
                              "params_soft_review_budget": {"mode": "range", "lower": 0.0, "upper": bmax},
                              "params_soft_corruption_level": corr},
         "selection": {"k": 3}}
    text = f"I want predicted F1 at least {v:.2f} and a review budget of at most {bmax:.3f}"
    text += ", corruption none or mild. Return 3 options." if corr["mode"] == "allowed" else ". Return 3 options."
    return text, q


def build(n_episodes_per_dataset: int = 14) -> tuple[pd.DataFrame, list[dict]]:
    rng = random.Random(SEED)
    bank = load_bank()
    episodes, items = [], []
    ep_no = 0

    def add(ep: dict, ad: CFXHPOAdapter, text: str, g: Grounds, is_dispute: bool, is_request: bool, kind_label: str,
            template: str, pair: str | None, truth_src: str):
        v = ad.verify(g, oracle=True) if is_dispute else None
        vb = ad.verify(g, oracle=False) if is_dispute else None
        truth = v.verdict if v else ""
        items.append({"item_id": f"C{len(items) + 1:03d}", "episode_id": ep["episode_id"], "dataset": ep["dataset"],
                      "user_text": text, "is_dispute": is_dispute, "is_request": is_request, "kind": kind_label,
                      "template": template, "pair": pair or "", "target": g.target, "interaction_act": g.act, "grounds": g.grounds,
                      "claim_kind": g.claim_kind, "subject": g.subject, "user_value": json.dumps(g.user_value) if g.user_value else "",
                      "truth": truth, "truth_source": truth_src if is_dispute else "", "checkable": v.checkable if v else "",
                      "oracle_evidence": " | ".join(v.evidence) if v else "", "built_verdict": vb.verdict if vb else "",
                      "built_evidence": " | ".join(vb.evidence) if vb else "",
                      "gold_action": gold_action(is_dispute, is_request, v)})

    for ds in ["mvtec", "proditec"]:
        pool = bank[bank.dataset == ds].reset_index(drop=True)
        # focal options: prefer the ones where the surrogate and the retraining disagree, so "user right" exists
        order = pool.reindex(pool.err_val.abs().sort_values(ascending=False).index)
        focal_ids = list(order.bank_id[: n_episodes_per_dataset])
        for fid in focal_ids:
            others = pool[pool.bank_id != fid].sample(2, random_state=rng.randint(0, 10 ** 6))
            opts_df = pd.concat([pool[pool.bank_id == fid], others]).reset_index(drop=True)
            opts = [row.to_dict() for _, row in opts_df.iterrows()]
            text, q = make_query(opts, rng)
            ad = CFXHPOAdapter(ds, q, [dict(o) for o in opts], user_request=text)
            ep_no += 1
            ep = {"episode_id": f"E{ep_no:02d}", "dataset": ds, "user_request": text, "query": q,
                  "options": [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in o.items()} for o in ad.options],
                  "state": ad.state()}
            episodes.append(ep)
            focal = ad.options[0]; fn = option_name(0)
            recs = ad.records; score_map = {r["cf_id"]: r["score"] for r in recs}
            v_req = q["objective"]["value"]
            ranked = _rank_records(recs); best = option_name(ranked[0]["cf_id"]); worst = option_name(ranked[-1]["cf_id"])

            # 1. score_reach (oracle): the system says the focal option meets the minimum
            if focal[TARGET] >= v_req:
                add(ep, ad, f"{fn} will not reach {v_req:.2f} once it is actually trained. Your table is a surrogate guess, I have seen these numbers collapse before.",
                    Grounds("prediction", "direct challenge", "prior experience", "score_reach", fn, {"threshold": v_req}),
                    True, False, "score_reach", "T1", None, "oracle")
            # 2. score_over (oracle): the table overstates by more than δ
            add(ep, ad, f"The {fmt3(focal[TARGET])} for {fn} is overstated. When you retrain it you will land more than two points lower.",
                Grounds("prediction", "direct challenge", "technical knowledge", "score_over", fn, {}),
                True, False, "score_over", "T2", None, "oracle")
            # 3. rank_confirmed (oracle): the user says a lower-ranked option will beat the best after retraining
            if best != worst:
                add(ep, ad, f"In practice {worst} will beat {best} when trained; the ordering in your table will not hold.",
                    Grounds("reasoning", "alternative proposal", "prior experience", "rank_confirmed", worst, {"better": worst, "worse": best}),
                    True, False, "rank_confirmed", "T3", None, "oracle")
            # 4. satisfies (table): user asserts a violation of the budget constraint
            f = "params_soft_review_budget"; bmax = q["soft_constraints"][f]["upper"]
            for o in ad.options[:2]:
                n = option_name(o["cf_id"]); viol = float(o[f]) > bmax
                add(ep, ad, f"{n} breaks my review-budget limit of {bmax:.3f}, but your summary says it satisfies the constraints.",
                    Grounds("explanation representation", "direct challenge", "internal inconsistency", "satisfies", n,
                            {"feature": f, "user_says_violates": True}),
                    True, False, "satisfies", "T4", f"{ep['episode_id']}-T4", "table")
            # 4b. satisfies with a stated limit that the parser snapped away (user right by construction)
            o = ad.options[1]; n = option_name(1); b = float(o[f])
            if b <= bmax:
                tight = round(b - 0.01, 3)
                add(ep, ad, f"I asked for a budget of at most {tight:.3f}, not {bmax:.3f}. {n} is over my limit even though the table marks it as fine.",
                    Grounds("explanation representation", "correction", "internal inconsistency", "satisfies", n,
                            {"feature": f, "user_says_violates": True, "user_limit": {"max": tight}}),
                    True, False, "satisfies_user_limit", "T5", None, "table")
            # 5. rank_table: the user misreads the ranking (table)
            lo, hi = min(score_map, key=score_map.get), max(score_map, key=score_map.get)
            if lo != hi:
                add(ep, ad, f"{option_name(lo)} has better quality than {option_name(hi)}, so it should be the best choice, not {option_name(hi)}.",
                    Grounds("explanation representation", "direct challenge", "presentation ambiguity", "rank_table", option_name(lo),
                            {"better": option_name(lo), "worse": option_name(hi)}),
                    True, False, "rank_table", "T6", f"{ep['episode_id']}-T6", "table")
                add(ep, ad, f"{option_name(hi)} is above {option_name(lo)} on quality, so {option_name(lo)} cannot be ranked above it.",
                    Grounds("explanation representation", "direct challenge", "presentation ambiguity", "rank_table", option_name(hi),
                            {"better": option_name(hi), "worse": option_name(lo)}),
                    True, False, "rank_table", "T6", f"{ep['episode_id']}-T6", "table")
            # 6. feature_effect (surrogate): budget / training fraction have no effect
            for feat, newv, says_no in [("params_soft_review_budget", round(max(0.05, float(focal["params_soft_review_budget"]) - 0.1), 3), True),
                                        ("params_soft_train_fraction", round(max(0.2, float(focal["params_soft_train_fraction"]) - 0.3), 2), True)]:
                add(ep, ad, f"The {feat_label(feat).lower()} makes no difference to quality here. You could set it to {newv} for {fn} and get the same F1.",
                    Grounds("reasoning", "direct challenge", "causal implausibility", "feature_effect", fn,
                            {"feature": feat, "new_value": newv, "user_says_no_effect": says_no}),
                    True, False, "feature_effect", "T7", None, "surrogate")
            # 7. density (log): extrapolation claim about the focal option and about another option
            for o in ad.options[:2]:
                n = option_name(o["cf_id"])
                add(ep, ad, f"There is nothing in the optimisation log anywhere near {n}; that number is pure extrapolation by the surrogate.",
                    Grounds("evidence", "direct challenge", "missing evidence", "density", n, {}),
                    True, False, "density", "T8", f"{ep['episode_id']}-T8", "log")
            # 8. log_stat (log): data-quality claims
            add(ep, ad, "Most of the trials in your log were run on corrupted training data, so the surrogate behind this table is not trustworthy.",
                Grounds("input data", "direct challenge", "data-quality concern", "log_stat", "log", {"stat": "corrupted_share", "op": ">", "threshold": 0.5}),
                True, False, "log_stat", "T9", f"{ep['episode_id']}-T9", "log")
            add(ep, ad, "Fewer than a quarter of the logged trials used corrupted data, so the surrogate has barely seen the corruption levels it is scoring.",
                Grounds("input data", "direct challenge", "data-quality concern", "log_stat", "log", {"stat": "corrupted_share", "op": "<", "threshold": 0.25}),
                True, False, "log_stat", "T9", f"{ep['episode_id']}-T9", "log")
            # 9. uncheckable (Surface)
            add(ep, ad, f"On our line we never see strong corruption, so {fn}'s robustness setting is irrelevant and should not count for it.",
                Grounds("reasoning", "rejection", "domain knowledge", "uncheckable", fn, {}), True, False, "uncheckable", "T10", None, "")
            add(ep, ad, f"Last year a {focal['params_backbone']} at this review budget failed badly in production. I do not trust {fn}.",
                Grounds("prediction", "rejection", "prior experience", "uncheckable", fn, {}), True, False, "uncheckable", "T11", None, "")
            # 10. requests (Re-present)
            feat = rng.choice(["params_coreset_sampling_ratio", "params_num_neighbors", "params_layers_key", "params_reduction"])
            add(ep, ad, f"What does '{feat_label(feat).lower()}' actually mean? I can't follow the table without that.",
                Grounds(claim_kind="request", subject=feat), False, True, "request_term", "T12", None, "")
            add(ep, ad, "Can you show the review budget as a percentage of images rather than a fraction? The decimals are confusing.",
                Grounds(claim_kind="request", subject="format", user_value={"format": "percent"}), False, True, "request_format", "T13", None, "")
            add(ep, ad, "I don't understand the ranking paragraph. Can you say in plain words which one I should pick and why?",
                Grounds(claim_kind="request", subject="plain", user_value={"format": "plain"}), False, True, "request_plain", "T14", None, "")
            # 11. neither (false-alarm probes)
            add(ep, ad, f"{best} looks fine, let's go with it.", Grounds(), False, False, "none", "T15", None, "")
            add(ep, ad, f"OK, so {best} has the highest quality and keeps the budget. Understood.", Grounds(), False, False, "none", "T16", None, "")
            # 12. both: a dispute that also asks for a legend (dispute wins the routing; request flagged)
            add(ep, ad, f"Your summary says {fn} satisfies all constraints, but I asked for budget under {bmax:.3f} and I cannot even see which column is the budget.",
                Grounds("explanation representation", "direct challenge", "internal inconsistency", "satisfies", fn, {"feature": f, "user_says_violates": True}),
                True, True, "satisfies_and_request", "T17", None, "table")

    df = pd.DataFrame(items)
    return df, episodes


def main() -> None:
    df, episodes = build()
    df.to_csv(ROOT / "data" / "partC_cfxhpo.csv", index=False)
    (ROOT / "data" / "partC_episodes.json").write_text(json.dumps(episodes, indent=1, ensure_ascii=False, default=float))
    print(f"Part C: {len(df)} items over {len(episodes)} episodes")
    print(pd.crosstab(df.kind, df.truth.replace("", "-")))
    print("gold actions:", df.gold_action.value_counts().to_dict())
    print("built verdict vs oracle truth (disputes with a checkable claim):")
    d = df[(df.is_dispute) & (df.truth.isin(["user_right", "user_wrong", "undecided"]))]
    print(pd.crosstab(d.truth, d.built_verdict))


if __name__ == "__main__":
    main()
