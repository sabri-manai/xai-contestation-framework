"""E5: the mechanism inside the application, end to end, without benchmark labels.

Every Part C turn is replayed through run_turn(): Understand (language model maps the turn to a claim),
Verify (CF-XHPO adapter, three tiers; oracle = the retraining bank), Decide (track), Respond. Scored against
the derived gold action. The difference from E2/E3 is that here the labels and the claim come from the
system itself, so Understand errors propagate — this is what a live system faces.

Writes benchmark/outputs/e5_live_items.csv and e5_live.md.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..adapters.cfxhpo import CFXHPOAdapter
from ..adapters.fungi import FungiAdapter
from ..benchmark.metrics import summarise
from ..tracks.common import run_turn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "outputs"
WORKERS = 8


UNDERSTAND = "v2"   # set by main(): "v1" (free-form JSON) or "v2" (closed choice over the adapter's claim menu)
CLARIFY = True


def _one(args):
    it, ep, track, oracle = args
    if it["part"] == "C":
        ad = CFXHPOAdapter(ep["dataset"], json.loads(json.dumps(ep["query"])), [dict(o) for o in ep["options"]], ep["user_request"])
        domain = "cfxhpo"
    else:
        ad = FungiAdapter(it["explanation_format"]); domain = "fungi"
    try:
        r = run_turn(ad, domain, it["user_text"], track=track, oracle=oracle, understand=UNDERSTAND, clarify=CLARIFY)
    except Exception as e:
        print(f"\n{it['item_id']} {track}: {e}"); return None
    g = r["labels"]; v = r["verdict"] or {}
    ex = g.get("extra") or {}
    return {"item_id": it["item_id"], "part": it["part"], "track": track, "oracle": oracle, "action": r["decision"]["action"],
            "clarified": r.get("clarified", False), "menu_id": ex.get("menu_id", ""), "choice_confidence": ex.get("choice_confidence", ""),
            "confidence": r["decision"]["confidence"], "p_dispute": r["decision"]["p_dispute"], "p_request": r["decision"]["p_request"],
            "understood_kind": g["claim_kind"], "understood_subject": g["subject"], "true_kind": it["claim_kind"], "true_subject": it["subject"],
            "verifier_verdict": v.get("verdict", ""), "verifier_tier": v.get("tier", ""), "truth": it["truth"],
            "is_dispute": it["is_dispute"], "is_request": it["is_request"], "gold_action": it["gold_action"], "kind": it["kind"],
            "truth_source": it["truth_source"], "reply": r["reply"]}


def main() -> None:
    global UNDERSTAND, CLARIFY
    import sys
    UNDERSTAND = "v1" if "--v1" in sys.argv else "v2"
    CLARIFY = "--no-clarify" not in sys.argv
    tag = f"{UNDERSTAND}" + ("" if CLARIFY else "_noclarify")
    c = pd.read_csv(ROOT / "data" / "partC_cfxhpo.csv", keep_default_na=False)
    b = pd.read_csv(ROOT / "data" / "partB_pairs.csv", keep_default_na=False)
    eps = {e["episode_id"]: e for e in json.loads((ROOT / "data" / "partC_episodes.json").read_text())}
    items = []
    for _, r in c.iterrows():
        d = r.to_dict(); d["part"] = "C"; items.append(d)
    for _, r in b.iterrows():
        d = r.to_dict(); d["part"] = "B"; items.append(d)
    for d in items:
        d["is_dispute"] = str(d["is_dispute"]).lower() == "true"; d["is_request"] = str(d["is_request"]).lower() == "true"
    rows = []
    for track in ["decision", "language"]:
        for oracle in [True, False]:
            jobs = [(it, eps.get(it.get("episode_id", "")), track, oracle) for it in items]
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for r in ex.map(_one, jobs):
                    if r is not None:
                        rows.append(r)
            print(f"{track} oracle={oracle}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"e5_live_items_{tag}.csv", index=False)
    L = [f"# E5: live loop on Parts C and B, labels and claims produced by the system itself (12 Nov 2026) — Understand {tag}", "",
         ("Understand v2 = closed choice over the adapter's claim menu (Jev in the decision track, gpt-4o-mini in the language track), "
          "slots filled by gpt-4o-mini; choice probability < 0.5 → Surface-clarify (ask which claim is meant)." if UNDERSTAND == "v2" else
          "Understand v1 = gpt-4o-mini free-form JSON (kind, subject, value generated).") +
         " Verify = adapter (oracle = retraining bank available / not); Decide = track; Respond = templated (decision) or constrained LLM (language).", ""]
    rows2 = []
    for (track, oracle, part), g in df.groupby(["track", "oracle", "part"]):
        s = summarise(g)
        tk = g.true_kind.replace({"display_fault": "fact", "data_stat": "fact"})
        kind_ok = (g.understood_kind == tk).mean()
        is_opt = g.true_subject.str.startswith("Option") | (g.part == "B")
        subj_ok = ((g.understood_subject.str.strip().str.lower() == g.true_subject.str.strip().str.lower()) | ~is_opt).mean()
        rows2.append({"track": track, "oracle bank": oracle, "part": part, "n": s["n"], "claim kind understood": round(kind_ok, 2),
                      "subject understood": round(subj_ok, 2), "clarify asked": round(g.clarified.astype(str).str.lower().eq("true").mean(), 2),
                      "Y_valid": s["Y_valid"], "Y_invalid": s["Y_invalid (sycophancy)"],
                      "discrimination": s["discrimination"], "surface_on_unresolvable": s["surface_on_unresolvable"],
                      "represent_on_request": s["represent_on_request"], "none_on_none": s["none_on_none"],
                      "agreement_with_gold": s["agreement_with_gold"], "mean_cost": s["mean_cost"]})
    t = pd.DataFrame(rows2)
    L += ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    L += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values]
    L += ["", "## Where the live loop loses against E2/E3 (Part C, decision track, oracle bank): by true claim kind", ""]
    g = df[(df.track == "decision") & (df.oracle) & (df.part == "C")]
    t2 = pd.DataFrame([{"true_kind": k, "n": len(x), "kind understood": round((x.understood_kind == x.true_kind).mean(), 2),
                        "agreement with gold": round((x.action == x.gold_action).mean(), 2)} for k, x in g.groupby("true_kind")])
    L += ["| " + " | ".join(t2.columns) + " |", "|" + "---|" * len(t2.columns)]
    L += ["| " + " | ".join(str(v) for v in r) + " |" for r in t2.values]
    (OUT / f"e5_live_{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
