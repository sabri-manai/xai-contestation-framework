"""E4: does asking when unsure pay under detection noise? (C3; architecture.md §3.3, §6)

Two policies after a positive detection: ALWAYS-ACT (hold/revise as the decider says) and GATED (surface when
the decider's confidence is below τ). Expected cost per flagged turn is computed from measured pieces:

- the decider's per-item actions and confidences on the benchmark (E2/E3 output, oracle or built condition);
- the truth mix among flags: a flag is a real dispute with probability p (the detector's precision) and a
  false alarm otherwise. False alarms are drawn from the benchmark's request and "neither" items, so what
  the decider does on a false alarm is measured, not assumed;
- the cost ordering (decide.py), swept over the ratio r = C(revise, invalid) / C(hold, valid) ∈ [1, 20].

Detector precision is swept over [0.4, 0.9]; the observed operating points are marked: Jev at 0.17 on gold
(0.77), on the block majority (0.54), on the 258 test-standing labels (0.48). Also reported: the real
confidence distribution, so the gate's τ is read from precision on the oracle-verified items, not chosen by eye.

Writes benchmark/outputs/e4_cost.md and e4_cost_grid.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..decide import DEFAULT_COSTS, cost, gate, truth_of
from ..schemas import Decision

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "outputs"
OPERATING_POINTS = {"gold (run 2)": 0.77, "block majority (3 Nov)": 0.54, "258 test-standing": 0.48, "Phase 1 ensemble": 0.66}


def expected_cost(dec: pd.DataFrame, p: float, r: float, tau: float | None) -> float:
    """dec: one decider × condition; rows carry action, confidence, is_dispute, is_request, truth."""
    costs = dict(DEFAULT_COSTS); costs["revise_invalid"] = r * costs["hold_valid"]
    d = dec.copy()
    d["truth_class"] = d.apply(lambda x: truth_of({"is_dispute": x.is_dispute, "is_request": x.is_request, "truth": x.truth}), axis=1)
    if tau is not None:
        d["action"] = [gate(Decision(action=a, confidence=c), tau).action for a, c in zip(d.action, d.confidence)]
    disputes = d[d.is_dispute.astype(bool)]
    false_alarms = d[~d.is_dispute.astype(bool)]
    c_d = np.mean([cost(a, t, costs) for a, t in zip(disputes.action, disputes.truth_class)]) if len(disputes) else 0.0
    c_f = np.mean([cost(a, t, costs) for a, t in zip(false_alarms.action, false_alarms.truth_class)]) if len(false_alarms) else 0.0
    return p * c_d + (1 - p) * c_f


def tau_by_rule(dec: pd.DataFrame, floor: float = 0.75) -> float:
    """Lowest τ at which the precision of acted-on (hold/revise) decisions against the derived gold is ≥ floor."""
    d = dec[dec.action.isin(["hold", "revise"])]
    for t in np.round(np.arange(0.0, 1.001, 0.05), 2):
        s = d[d.confidence >= t]
        if len(s) and (s.action == s.gold_action).mean() >= floor:
            return float(t)
    return 1.0


def main() -> None:
    items = pd.read_csv(OUT / "e2e3_items.csv", keep_default_na=False)
    items["is_dispute"] = items.is_dispute.astype(str).str.lower() == "true"
    items["is_request"] = items.is_request.astype(str).str.lower() == "true"
    items["confidence"] = pd.to_numeric(items.confidence, errors="coerce").fillna(0)
    L = ["# E4: asking under detection noise (12 Nov 2026)", "",
         "Expected cost per flagged turn = p · cost(decider on real disputes) + (1 − p) · cost(decider on false alarms), "
         "false alarms being the benchmark's request and 'neither' items. Costs: hold-on-valid = 1, surface = 0.3, "
         "re-present = 0.3, revise-on-invalid = r. The gate turns hold/revise into surface when confidence < τ.", ""]
    grid_rows = []
    for (decider, cond), g in items[(~items.role_given.astype(str).str.lower().eq("true")) & (items["repeat"].astype(int) == 0)].groupby(["decider", "condition"]):
        if cond not in ("oracle", "built", "none"):
            continue
        tau = tau_by_rule(g)
        L += [f"## {decider.split('/')[-1]}, verifier = {cond}", "",
              f"τ by rule (lowest τ with precision of acted decisions ≥ 0.75 against derived gold): {tau:.2f}. "
              f"Confidence distribution of hold/revise decisions: " +
              ", ".join(f"p{q}={np.percentile(g[g.action.isin(['hold', 'revise'])].confidence, q):.2f}" for q in (10, 25, 50, 75, 90)), ""]
        rows = []
        for p in [0.4, 0.48, 0.54, 0.6, 0.66, 0.77, 0.9]:
            for r in [1, 2, 5, 10, 20]:
                ca = expected_cost(g, p, r, None); cg = expected_cost(g, p, r, tau)
                best_tau = min(np.round(np.arange(0.0, 1.001, 0.05), 2), key=lambda t: expected_cost(g, p, r, t))
                cb = expected_cost(g, p, r, best_tau)
                rows.append({"precision p": p, "ratio r": r, "always-act": round(ca, 3), f"gated τ={tau:.2f}": round(cg, 3),
                             "gated best τ": f"{cb:.3f} (τ={best_tau:.2f})", "gated wins": "yes" if cg < ca - 1e-9 else "no"})
                grid_rows.append({"decider": decider, "condition": cond, "p": p, "r": r, "always_act": ca, "gated_rule": cg, "tau_rule": tau,
                                  "gated_best": cb, "tau_best": best_tau})
        df = pd.DataFrame(rows)
        L += ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        L += ["| " + " | ".join(str(v) for v in rr) + " |" for rr in df.values]
        wins = df[df["gated wins"] == "yes"]
        L += ["", f"Gated policy wins at the rule's τ in {len(wins)}/{len(df)} (p, r) cells; "
              + ("smallest r at which it wins for every p ≥ 0.48: " + str(min([r for r in [1, 2, 5, 10, 20] if all(df[(df['ratio r'] == r) & (df['precision p'] >= 0.48)]['gated wins'] == 'yes')] or ['none'])) if len(wins) else "never"), ""]
    pd.DataFrame(grid_rows).to_csv(OUT / "e4_cost_grid.csv", index=False)
    L += ["Operating points observed for Jev's dispute gate at threshold 0.17: " + ", ".join(f"{k} {v}" for k, v in OPERATING_POINTS.items()) + ".", ""]
    (OUT / "e4_cost.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
