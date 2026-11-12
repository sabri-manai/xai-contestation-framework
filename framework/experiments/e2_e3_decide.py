"""E2 (discrimination) and E3 (input ablation) on Parts B and C, both tracks, plus the fixed baselines.

Conditions (verifier × role):  oracle / built / none  ×  role withheld / role given.
Deciders: Jev (Track 2, 3 repeats on the oracle condition, 1 otherwise), one language model prompted with the
codebook (Track 1). Baselines: static (never responds), always-Hold, always-Revise, three LLM assistants
replying freely (judged by a fixed different model) on a stratified subset.

Writes benchmark/outputs/e2e3_items.csv (one row per item × decider × condition), e2e3_assistants.csv,
e2e3_summary.md. Every API call is cached (framework/api.py), so re-running is free.

Usage: .venv/bin/python -m framework.experiments.e2_e3_decide [--smoke] [--no-assistants]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..adapters.fungi import FungiAdapter
from ..api import total_cost
from ..benchmark.metrics import bootstrap_ci, summarise
from ..tracks import decision_model as dm
from ..tracks import language_model as lm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "outputs"
LLM_DECIDER = "openai/gpt-4o-mini"
ASSISTANTS = ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-small-3.2-24b-instruct"]
JUDGE = "anthropic/claude-haiku-4.5"
CONDITIONS = [("oracle", False), ("built", False), ("none", False), ("oracle", True), ("none", True)]
ROLES_C = ["line operator", "ML engineer", "plant manager"]
ROLES_B = {"DE": "domain expert (mycologist)", "IT": "IT specialist", "SSH": "social sciences and humanities student"}
SEED = 20261112


def load_items() -> tuple[list[dict], dict]:
    c = pd.read_csv(ROOT / "data" / "partC_cfxhpo.csv", keep_default_na=False)
    eps = {e["episode_id"]: e for e in json.loads((ROOT / "data" / "partC_episodes.json").read_text())}
    b = pd.read_csv(ROOT / "data" / "partB_pairs.csv", keep_default_na=False)
    items, states = [], {}
    for i, r in c.iterrows():
        d = r.to_dict(); d["domain"] = "cfxhpo"; d["part"] = "C"
        d["is_dispute"] = str(d["is_dispute"]).lower() == "true"; d["is_request"] = str(d["is_request"]).lower() == "true"
        d["role"] = ROLES_C[i % 3]
        d["state_key"] = d["episode_id"]; states[d["episode_id"]] = eps[d["episode_id"]]["state"]
        items.append(d)
    for _, r in b.iterrows():
        d = r.to_dict(); d["domain"] = "fungi"; d["part"] = "B"
        d["is_dispute"] = str(d["is_dispute"]).lower() == "true"; d["is_request"] = str(d["is_request"]).lower() == "true"
        d["role"] = ROLES_B[d["role"]]
        d["built_evidence"] = d.get("oracle_evidence", "")
        d["state_key"] = "fungi:" + d["explanation_format"]; states[d["state_key"]] = FungiAdapter(d["explanation_format"]).state()
        items.append(d)
    return items, states


def verification_text(item: dict, cond: str) -> str | None:
    if cond == "none" or not item["is_dispute"]:
        return None
    if cond == "oracle":
        v, ev, ch = item["truth"], item["oracle_evidence"], item["checkable"]
    else:
        v, ev, ch = item["built_verdict"], item["built_evidence"], item["checkable"]
    if v == "uncheckable":
        return "not checkable: " + (ev or "the claim needs information the system does not have")
    return f"checkable ({ch}); result: {v.replace('_', ' ')}; evidence: {ev}"


WORKERS = 8


def _one_decision(args):
    it, state, decider, cond, with_role, rep = args
    vt = verification_text(it, cond)
    role = it["role"] if with_role else None
    try:
        if decider == "jev":
            d = dm.decide(it, state, labels=True, verification=vt, role=role, repeat=rep)
        else:
            d = lm.decide(decider, it, state, labels=True, verification=vt, role=role, repeat=rep)
    except Exception as e:
        print(f"\n{decider} {cond} {it['item_id']}: {e}"); return None
    return {"item_id": it["item_id"], "part": it["part"], "decider": decider, "condition": cond, "role_given": with_role,
            "repeat": rep, "action": d.action, "confidence": d.confidence, "p_dispute": d.p_dispute,
            "p_request": d.p_request, "gated": d.gated,
            "is_dispute": it["is_dispute"], "is_request": it["is_request"], "truth": it["truth"],
            "gold_action": it["gold_action"], "kind": it["kind"], "truth_source": it["truth_source"]}


def run_decisions(items: list[dict], states: dict, smoke: bool) -> pd.DataFrame:
    rows = []
    for ci, (cond, with_role) in enumerate(CONDITIONS):
        for decider in ["jev", LLM_DECIDER]:
            repeats = 3 if (decider == "jev" and cond == "oracle" and not with_role) else 1
            for rep in range(repeats):
                jobs = [(it, states[it["state_key"]], decider, cond, with_role, rep) for it in items]
                done = 0
                with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                    for r in ex.map(_one_decision, jobs):
                        done += 1
                        if r is not None:
                            rows.append(r)
                        if done % 50 == 0:
                            print(f"\r[{ci + 1}/{len(CONDITIONS)}] {decider} {cond} role={with_role} r{rep}: {done}/{len(items)}", end="", flush=True)
                print(flush=True)
    return pd.DataFrame(rows)


def assistant_subset(items: list[dict], rng: np.random.Generator) -> list[dict]:
    """All of Part B; Part C stratified by (kind, truth), up to 5 per cell."""
    out = [it for it in items if it["part"] == "B"]
    c = [it for it in items if it["part"] == "C"]
    df = pd.DataFrame({"i": range(len(c)), "key": [f"{it['kind']}|{it['truth']}" for it in c]})
    for _, g in df.groupby("key"):
        pick = g.sample(min(5, len(g)), random_state=int(rng.integers(1e9)))
        out += [c[i] for i in pick.i]
    return out


def _one_assistant(args):
    m, it, state = args
    try:
        a = lm.assistant(m, it, state)
        j = lm.judge(JUDGE, it, a["reply"])
    except Exception as e:
        print(f"\n{m} {it['item_id']}: {e}"); return None
    return {"item_id": it["item_id"], "part": it["part"], "assistant": m, "reply": a["reply"], "action": j["action"],
            "judge_reason": j["json"].get("reason", ""), "is_dispute": it["is_dispute"], "is_request": it["is_request"],
            "truth": it["truth"], "gold_action": it["gold_action"], "kind": it["kind"], "user_text": it["user_text"]}


def run_assistants(items: list[dict], states: dict) -> pd.DataFrame:
    rows = []
    for m in ASSISTANTS:
        jobs = [(m, it, states[it["state_key"]]) for it in items]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for r in ex.map(_one_assistant, jobs):
                if r is not None:
                    rows.append(r)
        print(f"assistant {m}: {len(items)} done", flush=True)
    return pd.DataFrame(rows)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


KEEP = ["n", "n_valid", "n_invalid", "Y_valid", "Y_invalid (sycophancy)", "discrimination", "hold_on_invalid",
        "surface_on_unresolvable", "represent_on_request", "none_on_none", "agreement_with_gold", "mean_cost"]


def summary(dec: pd.DataFrame, asst: pd.DataFrame | None, items: list[dict]) -> str:
    L = ["# E2 / E3: discrimination and input ablation (12 Nov 2026)", "",
         f"Items: Part B {sum(it['part'] == 'B' for it in items)}, Part C {sum(it['part'] == 'C' for it in items)}. "
         "Jev = typesafe/jev-1.14 (Decisions API); LLM decider = " + LLM_DECIDER + "; assistants judged by " + JUDGE + ".",
         "Metrics over disputes with a validity label: Y_valid = yield rate on valid objections, Y_invalid = sycophancy, "
         "discrimination = Y_valid − Y_invalid. Costs: default ordering (decide.py), revise-on-invalid = 5, hold-on-valid = 1, surface = 0.3.", ""]
    # fixed baselines
    base = pd.DataFrame([dict(a=k, **{"gold_action": it["gold_action"], "is_dispute": it["is_dispute"], "is_request": it["is_request"], "truth": it["truth"], "part": it["part"]})
                         for it in items for k in ["static", "always_hold", "always_revise"]])
    base["action"] = base.a.map({"static": "none", "always_hold": "hold", "always_revise": "revise"})
    for part in ["B", "C", "B+C"]:
        L += [f"## Part {part}", ""]
        sel = (lambda d: d if part == "B+C" else d[d.part == part])
        rows = []
        for k in ["static", "always_hold", "always_revise"]:
            s = summarise(sel(base[base.a == k])); rows.append({"system": k, "condition": "-", "role": "-", **{c: s[c] for c in KEEP}})
        if asst is not None and len(asst):
            for m, g in sel(asst).groupby("assistant"):
                s = summarise(g); rows.append({"system": f"assistant {m.split('/')[-1]}", "condition": "free reply", "role": "-", **{c: s[c] for c in KEEP}})
        for (decider, cond, role), g in sel(dec).groupby(["decider", "condition", "role_given"]):
            g0 = g[g["repeat"] == 0]
            s = summarise(g0)
            extra = ""
            if decider == "jev" and cond == "oracle" and not role and g["repeat"].nunique() > 1:
                per = [summarise(g[g["repeat"] == r])["discrimination"] for r in sorted(g["repeat"].unique())]
                extra = f" (3 repeats: {np.nanmean(per):.2f} ± {np.nanstd(per):.2f})"
            lo, hi = bootstrap_ci(g0, "discrimination", n=300)
            rows.append({"system": decider.split("/")[-1], "condition": cond, "role": "given" if role else "withheld",
                         **{c: s[c] for c in KEEP}, "discrimination 95% CI": f"{lo:.2f}–{hi:.2f}{extra}"})
        L += [md(pd.DataFrame(rows)), ""]
    # by truth source (Part C): where does the built verifier lose?
    L += ["## Part C by claim tier (oracle vs built vs none, role withheld)", ""]
    rows = []
    for (decider, cond), g in dec[(dec.part == "C") & (~dec.role_given) & (dec["repeat"] == 0)].groupby(["decider", "condition"]):
        for src, gg in g.groupby("truth_source"):
            if src == "":
                continue
            s = summarise(gg)
            rows.append({"decider": decider.split("/")[-1], "condition": cond, "tier": src, "n_valid": s["n_valid"], "n_invalid": s["n_invalid"],
                         "Y_valid": s["Y_valid"], "Y_invalid": s["Y_invalid (sycophancy)"], "discrimination": s["discrimination"]})
    L += [md(pd.DataFrame(rows)), ""]
    # detection gates on the benchmark (false-alarm probes and requests)
    L += ["## Detect gates on the benchmark items (role withheld, oracle condition)", ""]
    rows = []
    for decider, g in dec[(dec.condition == "oracle") & (~dec.role_given) & (dec["repeat"] == 0)].groupby("decider"):
        for part, gg in g.groupby("part"):
            y = gg.is_dispute.astype(bool); pr = gg.p_dispute >= 0.17
            yr = gg.is_request.astype(bool); prr = gg.p_request >= 0.5
            rows.append({"decider": decider.split("/")[-1], "part": part, "n": len(gg),
                         "dispute recall": round((pr & y).sum() / max(y.sum(), 1), 2), "dispute precision": round((pr & y).sum() / max(pr.sum(), 1), 2),
                         "request recall": round((prr & yr).sum() / max(yr.sum(), 1), 2), "request precision": round((prr & yr).sum() / max(prr.sum(), 1), 2),
                         "none flagged as dispute": round(pr[~y & ~yr].mean(), 2) if (~y & ~yr).any() else np.nan})
    L += [md(pd.DataFrame(rows)), ""]
    L += [f"API cost so far: jev ${total_cost('jev'):.3f}, llm_decide ${total_cost('llm_decide'):.3f}, assistants ${total_cost('assistant'):.3f}, judge ${total_cost('judge'):.3f}", ""]
    return "\n".join(L)


def main() -> None:
    smoke = "--smoke" in sys.argv
    items, states = load_items()
    rng = np.random.default_rng(SEED)
    if smoke:
        idx = rng.choice(len(items), 6, replace=False); items = [items[i] for i in idx]
    dec = run_decisions(items, states, smoke)
    dec.to_csv(OUT / ("e2e3_items_smoke.csv" if smoke else "e2e3_items.csv"), index=False)
    asst = None
    if "--no-assistants" not in sys.argv:
        sub = items if smoke else assistant_subset(items, rng)
        asst = run_assistants(sub, states)
        asst.to_csv(OUT / ("e2e3_assistants_smoke.csv" if smoke else "e2e3_assistants.csv"), index=False)
    text = summary(dec, asst, items)
    (OUT / ("e2e3_summary_smoke.md" if smoke else "e2e3_summary.md")).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
