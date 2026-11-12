"""Aggregate the coders' labels on the 258 episodes into test labels, and report reliability.

Rules: data/DATA_RULES.md, entries dated 2026-11-12 (written before this script was run).

Reads  ../xai-contestation-labelling/labels/c_*.csv   (raw app files, codes only; no emails)
Writes data/coders.csv            letter, app code, coverage; no personal data
       data/human_labels.csv      every save of every coder, coder = letter (pseudonymised export)
       data/labels_258.csv        one row per episode: per-coder gates, majority, standing, coarse dims
       data/adjudication_queue_258.csv   tied episodes, with text and each coder's labels and notes
       benchmark/outputs/labels_258_summary.md   coverage, reliability (4 coders), prevalence, Jev check

Usage:  python benchmark/labels_258.py        (from the framework root; any Python with pandas+sklearn)
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT.parent / "xai-contestation-labelling"
OUT = ROOT / "benchmark" / "outputs"

# app code -> letter. A, B, C as in reliability_block.ipynb and the 3 Nov DATA_RULES entries.
CODERS = {"c_433b209090": "A", "c_bf6244ebdc": "B", "c_01534ba437": "C", "c_b59a2c33e1": "D"}
CURRENT_TEXT_FROM = "2026-10-31 10:40:00"  # UTC; cleaned sentence-unit text pushed
GATES = ["is_contestation", "explanation_request"]
DIMS = ["presence", "target", "interaction_act", "grounds", "expected_response"]
LABELS = GATES + DIMS + ["notes"]
COARSE = {  # pre-registered 3 Nov 2026 (reliability_block.ipynb §3d)
    "target": {"prediction": "conclusion", "reasoning": "account", "evidence": "account",
               "input data": "account", "explanation representation": "form",
               "system competence": "authority"},
    "grounds": {"presentation ambiguity": "checkable", "internal inconsistency": "checkable",
                "missing evidence": "checkable", "data-quality concern": "checkable",
                "causal implausibility": "checkable", "domain knowledge": "not checkable",
                "technical knowledge": "not checkable", "prior experience": "not checkable"},
    "expected_response": {"revise": "revise-type", "incorporate correction": "revise-type",
                          "justify": "hold-type", "expose conflicting evidence": "hold-type",
                          "provide additional information": "hold-type",
                          "acknowledge uncertainty": "surface-type",
                          "defer to human expertise": "surface-type"},
}
T_C, T_R = 0.17, 0.5  # fixed on gold 29 Oct 2026; not re-chosen

import json
_M = json.loads((ROOT / "data" / "taxonomy_merges_v1.2.json").read_text())
MERGED = {}  # coder-proposed OR-merges (12 Nov 2026): fine -> merged name; unmerged fine values map to themselves
for d, groups in _M["merges"].items():
    MERGED[d] = {f: name for name, g in groups.items() for f in g["fine"]}
RELATIONS = _M["relations"]
ALL_FINE = {"target": ["prediction", "reasoning", "evidence", "input data", "explanation representation", "system competence"],
            "interaction_act": ["request for justification", "direct challenge", "correction", "alternative proposal",
                                "counterexample", "rejection", "request to acknowledge uncertainty"],
            "grounds": ["domain knowledge", "technical knowledge", "prior experience", "internal inconsistency",
                        "missing evidence", "data-quality concern", "causal implausibility", "presentation ambiguity"],
            "expected_response": ["justify", "revise", "incorporate correction", "provide additional information",
                                  "acknowledge uncertainty", "expose conflicting evidence", "defer to human expertise"]}
for d, m in MERGED.items():
    for f in ALL_FINE[d]:
        m.setdefault(f, f)


# ----------------------------------------------------------------------------- load
def load_raw() -> pd.DataFrame:
    frames = []
    for code, letter in CODERS.items():
        d = pd.read_csv(LAB / "labels" / f"{code}.csv", dtype=str, keep_default_na=False)
        d["coder"] = letter
        frames.append(d.drop(columns=["annotator_code"], errors="ignore"))
    df = pd.concat(frames, ignore_index=True)
    df["is_skipped"] = df["is_skipped"].isin(["1", "1.0"]).astype(int)
    df["version"] = pd.to_numeric(df["version"]).astype(int)
    df["time_spent_s"] = pd.to_numeric(df["time_spent_s"], errors="coerce")
    return df


def export_human_labels(raw: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    """Same shape as the labelling project's export_labels.py, coder = letter."""
    df = raw.merge(order, on="episode_id", how="left").sort_values(["coder", "episode_id", "version"])
    latest = (df[df.is_skipped == 0].groupby(["coder", "episode_id"]).version.max()
              .rename("latest").reset_index())
    df = df.merge(latest, on=["coder", "episode_id"], how="left")
    df["is_latest"] = ((df.is_skipped == 0) & (df.version == df.latest)).astype(int)
    df["on_current_text"] = (df.created_at >= CURRENT_TEXT_FROM).astype(int)
    df["codebook"] = np.where(df.is_skipped == 1, "",
                              np.where(df.explanation_request == "", "v1.0", "v1.2"))
    cols = (["coder", "episode_id", "block", "queue_pos", "version", "is_latest", "is_skipped",
             "on_current_text", "created_at", "time_spent_s"] + LABELS + ["codebook"])
    return df[cols].sort_values(["queue_pos", "coder", "version"])


# ----------------------------------------------------------------------------- agreement
def fleiss_kappa(mat: np.ndarray) -> float:
    """mat: items x categories counts, equal raters per item."""
    n = mat.sum(1)[0]
    P = ((mat ** 2).sum(1) - n) / (n * (n - 1))
    pj = mat.sum(0) / mat.sum()
    pe = (pj ** 2).sum()
    return (P.mean() - pe) / (1 - pe) if pe < 1 else np.nan


def krippendorff_alpha(table: pd.DataFrame) -> float:
    """Nominal alpha; table: items x coders, NaN = missing. Items with < 2 values dropped."""
    vals = [row.dropna().tolist() for _, row in table.iterrows()]
    vals = [v for v in vals if len(v) >= 2]
    if not vals:
        return np.nan
    cats = sorted({c for v in vals for c in v})
    n_total = sum(len(v) for v in vals)
    # observed disagreement
    Do = 0.0
    for v in vals:
        m = len(v)
        cnt = Counter(v)
        pairs_dis = m * (m - 1) - sum(c * (c - 1) for c in cnt.values())
        Do += pairs_dis / (m - 1)
    Do /= n_total
    tot = Counter(c for v in vals for c in v)
    De = (n_total * (n_total - 1) - sum(c * (c - 1) for c in tot.values())) / (n_total * (n_total - 1))
    return 1 - Do / De if De > 0 else np.nan


def pairwise(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x, y in combinations(table.columns, 2):
        p = table[[x, y]].dropna()
        if len(p) < 5:
            continue
        k = cohen_kappa_score(p[x], p[y]) if p[x].nunique() + p[y].nunique() > 2 else np.nan
        rows.append({"pair": f"{x}–{y}", "n": len(p), "agreement": round((p[x] == p[y]).mean(), 2),
                     "kappa": round(k, 2)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- aggregation
def majority(values: list[str]) -> tuple[str, str, str]:
    """-> (label, votes 'k/n', status unanimous|majority|split|single|none)."""
    values = [v for v in values if v not in ("", None) and v == v]
    n = len(values)
    if n == 0:
        return "", "0/0", "none"
    cnt = Counter(values).most_common()
    top, k = cnt[0]
    if n == 1:
        return top, "1/1", "single"
    if k == n:
        return top, f"{k}/{n}", "unanimous"
    if len(cnt) > 1 and cnt[1][1] == k:
        return "", f"{k}/{n}", "split"
    if k * 2 > n:
        return top, f"{k}/{n}", "majority"
    return "", f"{k}/{n}", "split"


def aggregate(lat: pd.DataFrame, eps: pd.DataFrame, order: pd.DataFrame, splits: pd.DataFrame,
              silver: pd.DataFrame, jev_block: pd.DataFrame) -> pd.DataFrame:
    letters = sorted(CODERS.values())
    wide = {}
    for col in GATES + DIMS + ["notes", "on_current_text"]:
        w = lat.pivot(index="episode_id", columns="coder", values=col).reindex(columns=letters)
        wide[col] = w
    rows = []
    for ep in order.index:
        meta = eps.loc[ep]
        r = {"episode_id": ep, "transcript_id": meta.transcript_id,
             "participant_group": meta.participant_group, "explanation_format": meta.explanation_format,
             "sample_batch": meta.sample_batch, "block": order.loc[ep, "block"],
             "queue_pos": int(order.loc[ep, "queue_pos"]),
             "fold": splits.get(meta.transcript_id, np.nan),
             "transcript_unclear": str(meta.transcript_unclear) == "True"}
        present = [c for c in letters if wide["is_contestation"].loc[ep, c] not in ("", None)
                   and wide["is_contestation"].loc[ep, c] == wide["is_contestation"].loc[ep, c]]
        r["n_coders"] = len(present)
        r["coders"] = "".join(present)
        r["n_coders_current_text"] = int(sum(1 for c in present if wide["on_current_text"].loc[ep, c] == 1))
        for g, short in [("is_contestation", "contestation"), ("explanation_request", "request")]:
            vals = [wide[g].loc[ep, c] for c in present]
            for c in letters:
                v = wide[g].loc[ep, c]
                r[f"{g}_{c}"] = v if v == v else ""
            lab, votes, status = majority([v for v in vals if v in ("yes", "no")])
            r[short], r[f"{short}_votes"], r[f"{short}_status"] = lab, votes, status
        # any push-back = contestation or request, per coder, then majority
        pb = []
        for c in present:
            a, b = wide["is_contestation"].loc[ep, c], wide["explanation_request"].loc[ep, c]
            if b in ("yes", "no"):
                pb.append("yes" if "yes" in (a, b) else "no")
        r["pushback"], r["pushback_votes"], r["pushback_status"] = majority(pb)
        # coarse dimensions among coders who said yes
        yes_coders = [c for c in present if wide["is_contestation"].loc[ep, c] == "yes"]
        r["n_yes_coders"] = len(yes_coders)
        for d, m in COARSE.items():
            vals = [m.get(wide[d].loc[ep, c], "") for c in yes_coders]
            lab, votes, status = majority(vals)
            r[f"{d}_coarse"], r[f"{d}_coarse_status"] = lab, status
        for d, m in MERGED.items():
            vals = [m.get(wide[d].loc[ep, c], "") for c in yes_coders]
            lab, votes, status = majority(vals)
            r[f"{d}_merged"], r[f"{d}_merged_status"] = lab, status
        for d in DIMS:
            for c in letters:
                v = wide[d].loc[ep, c]
                r[f"{d}_{c}"] = v if v == v else ""
        # standing
        if r["transcript_unclear"]:
            st = "excluded_unclear"
        elif r["n_coders"] == 0:
            st = "unlabelled"
        elif r["contestation_status"] == "single" or r["request_status"] == "single":
            st = "single"
        elif r["contestation_status"] == "split" or r["request_status"] == "split":
            st = "adjudicate"
        else:
            st = "test"
        r["standing"] = st
        # Jev probabilities: block from the current-text run, rest from silver (raw text)
        if ep in jev_block.index:
            r["p_contestation_jev"], r["p_request_jev"], r["jev_source"] = (
                jev_block.loc[ep, "p_contestation"], jev_block.loc[ep, "p_request"], "block_v12")
        elif ep in silver.index:
            r["p_contestation_jev"], r["p_request_jev"], r["jev_source"] = (
                silver.loc[ep, "p_contestation"], silver.loc[ep, "p_request"], "silver_raw_text")
        else:
            r["p_contestation_jev"], r["p_request_jev"], r["jev_source"] = np.nan, np.nan, ""
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = pd.read_csv(ROOT / "data" / "episodes_to_label.csv", keep_default_na=False).set_index("episode_id")
    order = pd.read_csv(ROOT / "data" / "queue_order.csv")
    splits = pd.read_csv(ROOT / "data" / "splits.csv").set_index("transcript_id")["fold"].to_dict()
    silver = pd.read_csv(ROOT / "data" / "silver_250.csv").set_index("episode_id")
    jev_block = pd.read_csv(OUT / "jev_block_v12.csv").set_index("episode_id")

    raw = load_raw()
    human = export_human_labels(raw, order)
    human.to_csv(ROOT / "data" / "human_labels.csv", index=False)
    lat = human[human.is_latest == 1].copy()

    # coders.csv (no personal data)
    block_ids = set(order[order.queue_pos <= 100].episode_id)
    crow = []
    for code, letter in CODERS.items():
        L = lat[lat.coder == letter]
        inb = L[L.episode_id.isin(block_ids)]
        crow.append({"coder": letter, "app_code": code, "episodes_labelled": len(L),
                     "block_labelled": len(inb), "block_on_current_text": int(inb.on_current_text.sum()),
                     # 3 Nov rule as applied then: full block coded independently; A's 6 pre-cleaning
                     # block labels (31 Oct, 07:38-10:40) were reported as a subset, not a disqualifier
                     "headline_coder": bool(len(inb) == 100),
                     "first_label_utc": L.created_at.min(), "last_label_utc": L.created_at.max(),
                     "median_s_per_episode": round(L.time_spent_s.median(), 1)})
    coders = pd.DataFrame(crow)
    coders.to_csv(ROOT / "data" / "coders.csv", index=False)

    order_i = order.set_index("episode_id")
    agg = aggregate(lat, eps, order_i, splits, silver, jev_block)
    agg.to_csv(ROOT / "data" / "labels_258.csv", index=False)

    # adjudication queue
    adj = agg[agg.standing == "adjudicate"].copy()
    adj = adj.merge(eps[["text_pl", "text_en", "context_en"]], left_on="episode_id", right_index=True)
    notes = lat.pivot(index="episode_id", columns="coder", values="notes").add_prefix("notes_")
    adj = adj.merge(notes, left_on="episode_id", right_index=True, how="left")
    keep = (["episode_id", "block", "participant_group", "explanation_format", "coders"]
            + [f"is_contestation_{c}" for c in "ABCD"] + [f"explanation_request_{c}" for c in "ABCD"]
            + ["contestation_votes", "request_votes"] + [f"target_{c}" for c in "ABCD"]
            + [f"grounds_{c}" for c in "ABCD"] + [c for c in adj.columns if c.startswith("notes_")]
            + ["text_pl", "text_en", "context_en"])
    adj[keep].to_csv(ROOT / "data" / "adjudication_queue_258.csv", index=False)

    # ---------------- report
    L = ["# Labels on the 258: four coders, aggregated test labels (12 Nov 2026)", "",
         "Rules: `data/DATA_RULES.md`, 2026-11-12 entries, fixed before this was run. Output of "
         "`benchmark/labels_258.py`.", "", "## 1. Coders and coverage", "", md_table(coders), "",
         "Coder A labelled 6 block episodes between 07:38 and 10:40 UTC on 31 Oct, before the cleaned "
         "sentence-unit text was pushed; as on 3 Nov, A stays a headline coder and those 6 labels are "
         "reported, not dropped. Agreement figures below use every latest label on the block.", ""]
    cov = pd.crosstab(agg.n_coders, agg.block)
    L += ["Episodes by number of coders and block:", "", md_table(cov.reset_index()), ""]
    st = pd.crosstab(agg.standing, agg.block, margins=True)
    L += ["Standing of each episode (rule of 12 Nov):", "", md_table(st.reset_index()), ""]

    # reliability on the block, four coders, current text
    blk = agg[agg.block == "reliability_100"].set_index("episode_id")
    L += ["## 2. Reliability on the block (100 episodes, 4 coders, all on current text)", ""]
    rel = []
    for g, short in [("is_contestation", "contestation"), ("explanation_request", "request")]:
        T = blk[[f"{g}_{c}" for c in "ABCD"]].replace("", np.nan)
        T.columns = list("ABCD")
        full = T.dropna()
        mat = np.array([[(r == "yes").sum(), (r == "no").sum()] for _, r in full.iterrows()])
        for name, ids in [("block", full.index), ("flagged", full.index[blk.loc[full.index, "sample_batch"] == "flagged"]),
                          ("random_unflagged", full.index[blk.loc[full.index, "sample_batch"] == "random_unflagged"])]:
            sub = full.loc[ids]
            m = np.array([[(r == "yes").sum(), (r == "no").sum()] for _, r in sub.iterrows()])
            rel.append({"gate": short, "set": name, "n": len(sub), "Fleiss κ (4)": round(fleiss_kappa(m), 2),
                        "Krippendorff α": round(krippendorff_alpha(sub), 2),
                        "unanimous": round((sub.nunique(axis=1) == 1).mean(), 2),
                        "yes rate A/B/C/D": "/".join(f"{(sub[c] == 'yes').mean():.2f}" for c in "ABCD")})
    # any push-back
    def _pb(c):
        a, b = blk[f"is_contestation_{c}"], blk[f"explanation_request_{c}"]
        return pd.Series(np.where((a == "yes") | (b == "yes"), "yes", "no"), index=blk.index, dtype=object).where(b != "", None)
    PB = pd.DataFrame({c: _pb(c) for c in "ABCD"}, index=blk.index).dropna()
    m = np.array([[(r == "yes").sum(), (r == "no").sum()] for _, r in PB.iterrows()])
    rel.append({"gate": "any push-back", "set": "block", "n": len(PB), "Fleiss κ (4)": round(fleiss_kappa(m), 2),
                "Krippendorff α": round(krippendorff_alpha(PB), 2), "unanimous": round((PB.nunique(axis=1) == 1).mean(), 2),
                "yes rate A/B/C/D": "/".join(f"{(PB[c] == 'yes').mean():.2f}" for c in "ABCD")})
    L += [md_table(pd.DataFrame(rel)), ""]
    L += ["Pairwise Cohen κ on the block:", ""]
    pw = []
    for g, short in [("is_contestation", "contestation"), ("explanation_request", "request")]:
        T = blk[[f"{g}_{c}" for c in "ABCD"]].replace("", np.nan); T.columns = list("ABCD")
        p = pairwise(T); p.insert(0, "gate", short); pw.append(p)
    L += [md_table(pd.concat(pw)), ""]
    # three-rater reference (A, B, C) on the same block, for continuity with 3 Nov
    T = blk[[f"is_contestation_{c}" for c in "ABC"]].replace("", np.nan).dropna()
    m3 = np.array([[(r == "yes").sum(), (r == "no").sum()] for _, r in T.iterrows()])
    L += [f"Reference: three-rater (A, B, C) Fleiss κ on contestation = {fleiss_kappa(m3):.2f} (3 Nov: 0.48).", ""]

    # coarse dimensions among coders who said yes (block + all)
    L += ["## 3. Structure dimensions (coders who said contestation = yes), three levels", "",
          "fine = codebook categories; coarse = pre-registered 3 Nov collapse; merged = the coders' own "
          "OR-merges of 12 Nov (`data/taxonomy_merges_v1.2.json`). All three reported by rule.", ""]
    cd = []
    for d in ["target", "interaction_act", "grounds", "expected_response"]:
        for name, sub in [("block", blk), ("all 258", agg.set_index("episode_id"))]:
            row = {"dimension": d, "set": name}
            for level, mp in [("fine", None), ("coarse", COARSE.get(d)), ("merged", MERGED.get(d))]:
                if level != "fine" and mp is None:
                    row[f"α {level}"] = ""; continue
                T = pd.DataFrame({c: (sub[f"{d}_{c}"] if mp is None else sub[f"{d}_{c}"].map(mp)).astype(object)
                                  .where(sub[f"is_contestation_{c}"] == "yes", None) for c in "ABCD"}, index=sub.index)
                T = T.replace("", None); T = T[T.notna().sum(axis=1) >= 2]
                row["n (≥2 yes-coders)"] = len(T)
                row[f"α {level}"] = round(krippendorff_alpha(T), 2)
                row[f"unanimous {level}"] = round((T.apply(lambda r: r.dropna().nunique(), axis=1) == 1).mean(), 2)
            cd.append(row)
    L += [md_table(pd.DataFrame(cd)), ""]
    # cross-dimension relations from the coders' note: per-coder consistency, descriptive
    L += ["Cross-dimension relations from the coders' note, checked on each coder's own yes-labels "
          "(share of labels with the antecedent whose consequent holds):", ""]
    rr = []
    for rel in RELATIONS:
        (ad, av), = rel["if"].items(); (cd_, cv), = rel["then"].items()
        row = {"if": f"{ad} = {av}", "then": f"{cd_} ∈ {cv}", "endorsed": rel["endorsed"]}
        for c in "ABCD":
            Lc = lat[(lat.coder == c) & (lat.is_contestation == "yes") & (lat[ad] == av)]
            row[c] = f"{(Lc[cd_].isin(cv)).mean():.2f} (n={len(Lc)})" if len(Lc) else "—"
        rr.append(row)
    L += [md_table(pd.DataFrame(rr)), ""]

    # prevalence
    L += ["## 4. Aggregated labels (standing = test)", ""]
    tst = agg[agg.standing == "test"]
    prev = pd.DataFrame({
        "contestation yes": tst.groupby("block").apply(lambda g: (g.contestation == "yes").sum()),
        "request yes": tst.groupby("block").apply(lambda g: (g.request == "yes").sum()),
        "both": tst.groupby("block").apply(lambda g: ((g.contestation == "yes") & (g.request == "yes")).sum()),
        "neither": tst.groupby("block").apply(lambda g: ((g.contestation == "no") & (g.request == "no")).sum()),
        "n": tst.groupby("block").size()}).reset_index()
    L += [md_table(prev), ""]
    yes = tst[tst.contestation == "yes"]
    L += ["Coarse dimensions on test-standing contestations (majority among yes-coders; `split` = tie):", ""]
    for d in COARSE:
        L += [f"- {d} (coarse): " + ", ".join(f"{k} {v}" for k, v in yes[f"{d}_coarse"].replace("", "split").value_counts().items())]
    L += ["", "Merged level (coders' OR-merges), majority among yes-coders:", ""]
    for d in MERGED:
        L += [f"- {d} (merged): " + ", ".join(f"{k} {v}" for k, v in yes[f"{d}_merged"].replace("", "split").value_counts().items())]
    L += ["", "By participant group × explanation format (test-standing contestation = yes / n):", ""]
    gf = tst.groupby(["participant_group", "explanation_format"]).apply(
        lambda g: f"{(g.contestation == 'yes').sum()}/{len(g)}").unstack(fill_value="0/0").reset_index()
    L += [md_table(gf), ""]
    L += [f"Single-coder episodes (D only, reported separately): {int((agg.standing == 'single').sum())}; "
          f"contestation yes among them: {int(((agg.standing == 'single') & (agg.contestation == 'yes')).sum())}.", ""]

    # Jev vs aggregated
    L += ["## 5. Jev (Detect) against the aggregated test labels, thresholds fixed on gold (reporting only)", "",
          "Caveat: tied episodes (`adjudicate`) are absent from these rows by the aggregation rule, and they "
          "are the hardest cases, so these figures are optimistic until the adjudication is done; the "
          "block-majority figure of 3 Nov (three coders, ties broken by majority, AUC 0.88) is the "
          "conservative reference.", ""]
    jr = []
    for name, sub in [("block (current text run)", tst[tst.jev_source == "block_v12"]),
                      ("rest (silver, raw text; lower bound)", tst[tst.jev_source == "silver_raw_text"]),
                      ("all test-standing", tst[tst.jev_source != ""])]:
        for g, p, t in [("contestation", "p_contestation_jev", T_C), ("request", "p_request_jev", T_R)]:
            s = sub.dropna(subset=[p])
            y = (s[g] == "yes").astype(int); pr = (s[p] >= t).astype(int)
            tp = int((y & pr).sum()); fp = int((~y.astype(bool) & pr.astype(bool)).sum()); fn = int((y.astype(bool) & ~pr.astype(bool)).sum())
            jr.append({"set": name, "gate": g, "n": len(s), "yes": int(y.sum()),
                       "AUC": round(roc_auc_score(y, s[p]), 2) if y.nunique() > 1 else np.nan,
                       "threshold": t, "precision": round(tp / max(tp + fp, 1), 2), "recall": round(tp / max(tp + fn, 1), 2),
                       "κ": round(cohen_kappa_score(y, pr), 2)})
    L += [md_table(pd.DataFrame(jr)), ""]
    L += ["## 6. Files", "",
          "- `data/labels_258.csv`: one row per episode, per-coder gates and dimensions, majority labels, "
          "votes, status, standing, coarse and merged dimensions, fold, Jev probabilities.",
          f"- `data/adjudication_queue_258.csv`: {len(adj)} tied episodes for the adjudicator.",
          "- `data/coders.csv`, `data/human_labels.csv`: coders by letter and app code only.", ""]
    (OUT / "labels_258_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
