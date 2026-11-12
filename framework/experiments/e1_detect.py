"""E1 baselines on the 258 test-standing labels: Phase 1 detectors and an LLM prompted with the codebook.

- TF-IDF + logistic regression (Phase 1 recipe: context_pl, char+word n-grams), trained on the 76 gold
  episodes (codebook v1.0 labels, the only trained-detector training data that exists), applied to the 258.
- XLM-R (frozen, mean-pooled, left-truncated to 256 tokens) + logistic regression, same training data.
- LLM + codebook: the v1.2 codebook as prompt, P(dispute) and P(request) as numbers, one model, zero-shot.
- Jev: from data/labels_258.csv (block: current-text run; rest: silver on raw text).

Scored against the test-standing majority labels (149 episodes; ties excluded by rule → optimistic for every
system alike). Thresholds: the trained detectors use 0.5 (Phase 1 tuned per fold; no tuning on the 258 is
allowed, DATA_RULES §2) and are also reported threshold-free (AUC). Writes benchmark/outputs/e1_detect_258.md.

Run with the Phase 1 venv (torch + transformers): ../detecting-contestation-xai/.venv/bin/python -m framework.experiments.e1_detect
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

from ..api import chat, parse_json
from ..tracks.decision_model import RULES, REQUEST_RULES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "outputs"
LLM = "openai/gpt-4o-mini"


def unit_window(context: str, target: str) -> str:
    lines = [l.strip() for l in context.split("||") if l.strip()]
    t = target.strip()
    idx = next((i for i, l in enumerate(lines) if t and (t in l or l.endswith(t[-60:]))), None)
    if idx is None:
        return context
    after = []
    for l in lines[idx + 1: idx + 3]:
        if l.startswith("INT"):
            break
        after.append(l)
    return " || ".join(lines[max(0, idx - 2): idx] + [lines[idx]] + after)


def score(y: pd.Series, p: pd.Series, t: float) -> dict:
    y = y.astype(int); pr = (p >= t).astype(int)
    tp = int((y & pr).sum()); fp = int(((1 - y) & pr).sum()); fn = int((y & (1 - pr)).sum())
    return {"AUC": round(roc_auc_score(y, p), 2) if y.nunique() > 1 else np.nan, "threshold": t,
            "precision": round(tp / max(tp + fp, 1), 2), "recall": round(tp / max(tp + fn, 1), 2),
            "macro-F1": round(float(np.mean([2 * tp / max(2 * tp + fp + fn, 1), 2 * (len(y) - tp - fp - fn) / max(2 * (len(y) - tp - fp - fn) + fp + fn, 1)])), 2),
            "κ": round(cohen_kappa_score(y, pr), 2)}


def llm_codebook(ep: pd.Series, repeat: int = 0) -> dict:
    sys_ = ("You apply a coding guide to one line of a Polish think-aloud transcript about an AI mushroom classifier's explanations. "
            "Polish is authoritative, English is a translation. Lines are tagged INT: (interviewer) and P: (participant). "
            "Output one JSON object: {\"p_contestation\": <0-1>, \"p_request\": <0-1>}.\n"
            f"CONTESTATION: {RULES}\nEXPLANATION REQUEST: {REQUEST_RULES}")
    user = json.dumps({"participant_group": ep.participant_group, "explanation_format": ep.explanation_format,
                       "context_pl": unit_window(ep.context_pl, ep.text_pl), "target_pl": ep.text_pl,
                       "context_en": unit_window(ep.context_en, ep.text_en), "target_en": ep.text_en}, ensure_ascii=False)
    out = chat(LLM, [{"role": "system", "content": sys_}, {"role": "user", "content": user}], max_tokens=60, json_mode=True,
               repeat=repeat, cache_kind="llm_codebook")
    try:
        j = parse_json(out["text"])
        return {"p_c": float(j["p_contestation"]), "p_r": float(j["p_request"])}
    except Exception:
        return {"p_c": np.nan, "p_r": np.nan}


def embed(texts: list[str], model_name: str = "xlm-roberta-base", batch: int = 16, max_len: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name); tok.truncation_side = "left"
    mdl = AutoModel.from_pretrained(model_name).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i + batch]), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            vecs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
    v = np.vstack(vecs)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def main() -> None:
    gold = pd.read_csv(ROOT / "data" / "gold_standard.csv", keep_default_na=False)
    eps = pd.read_csv(ROOT / "data" / "episodes_to_label.csv", keep_default_na=False)
    lab = pd.read_csv(ROOT / "data" / "labels_258.csv", keep_default_na=False)
    test = lab[lab.standing == "test"].merge(eps[["episode_id", "text_pl", "text_en", "context_pl", "context_en"]], on="episode_id")
    yc = (test.contestation == "yes").astype(int); yr = (test.request == "yes").astype(int)
    rows = []

    # Phase 1 TF-IDF recipe, trained on gold (context_pl), applied to the 258 (cleaned context_pl)
    ytr = (gold.is_contestation == "yes").astype(int)
    tfidf = make_pipeline(make_union(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                                     TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
                          LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
    tfidf.fit(gold.context_pl.tolist(), ytr)
    p = pd.Series(tfidf.predict_proba(test.context_pl.tolist())[:, 1], index=test.index)
    rows.append({"system": "TF-IDF + LR (trained on gold 76)", "gate": "dispute", **score(yc, p, 0.5)})

    # XLM-R frozen + LR
    try:
        cache = OUT / "cache" / "xlmr_embeddings.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True); Etr, Ete = z["train"], z["test"]
            assert Etr.shape[0] == len(gold) and Ete.shape[0] == len(test)
        else:
            Etr = embed(gold.context_pl.tolist()); Ete = embed(test.context_pl.tolist())
            np.savez(cache, train=Etr, test=Ete)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(Etr, ytr)
        p = pd.Series(lr.predict_proba(Ete)[:, 1], index=test.index)
        rows.append({"system": "XLM-R frozen + LR (trained on gold 76)", "gate": "dispute", **score(yc, p, 0.5)})
        # ensemble as in Phase 1: probability average of TF-IDF and XLM-R
        pe = (pd.Series(tfidf.predict_proba(test.context_pl.tolist())[:, 1], index=test.index) + p) / 2
        rows.append({"system": "TF-IDF + XLM-R ensemble (Phase 1 best)", "gate": "dispute", **score(yc, pe, 0.5)})
    except Exception as e:
        print("XLM-R skipped:", e)

    # LLM + codebook, zero-shot
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(llm_codebook, [r for _, r in test.iterrows()]))
    pc = pd.Series([r["p_c"] for r in res], index=test.index); pr_ = pd.Series([r["p_r"] for r in res], index=test.index)
    ok = pc.notna() & pr_.notna()
    rows.append({"system": f"LLM + codebook ({LLM.split('/')[-1]}), zero-shot", "gate": "dispute", **score(yc[ok], pc[ok], 0.5)})
    rows.append({"system": f"LLM + codebook ({LLM.split('/')[-1]}), zero-shot", "gate": "request", **score(yr[ok], pr_[ok], 0.5)})
    test.loc[ok, "p_contestation_llm"] = pc[ok]; test.loc[ok, "p_request_llm"] = pr_[ok]

    # Jev (from labels_258.csv; thresholds fixed on gold)
    pj = pd.to_numeric(test.p_contestation_jev, errors="coerce"); pjr = pd.to_numeric(test.p_request_jev, errors="coerce")
    okj = pj.notna()
    rows.append({"system": "Jev 1.14 zero-shot (threshold by rule on gold)", "gate": "dispute", **score(yc[okj], pj[okj], 0.17)})
    rows.append({"system": "Jev 1.14 zero-shot (provisional 0.5)", "gate": "request", **score(yr[okj], pjr[okj], 0.5)})
    # same threshold rule applied to the LLM for a like-for-like reading (NOT tuned here: the rule was fixed on gold; we only report 0.17 too)
    rows.append({"system": f"LLM + codebook ({LLM.split('/')[-1]}), at 0.17 for comparison", "gate": "dispute", **score(yc[ok], pc[ok], 0.17)})

    df = pd.DataFrame(rows)
    test[["episode_id", "block", "contestation", "request", "p_contestation_jev", "p_request_jev", "p_contestation_llm", "p_request_llm"]].to_csv(OUT / "e1_detect_258_scores.csv", index=False)
    L = ["# E1 on the 258 test-standing labels (n = %d; 12 Nov 2026)" % len(test), "",
         "Trained detectors: Phase 1 recipes trained on the 76 gold episodes (v1.0 labels, line unit, raw text), applied to the "
         "cleaned sentence-unit text of the 258; threshold 0.5 (no tuning on the 258). LLM + codebook: v1.2 rules as prompt, "
         "zero-shot. Jev: as in labels_258.csv (block on the current text, rest on raw text). Ties excluded by the aggregation "
         "rule, so every number is optimistic alike; the comparison between systems is the point.", "",
         "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(str(v) for v in r) + " |")
    (OUT / "e1_detect_258.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
