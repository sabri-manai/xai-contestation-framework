"""CF-XHPO domain adapter: the application setting (architecture.md §4.2).

State = parsed query + candidate table + template explanation, exactly as the demo app
(Conversational-Counterfactual-Guidance-.../demo_app/app.py) shows them. The template functions are
vendored from that app (same wording) so every number in the explanation is copied from the table.

Verify runs in three tiers, cheapest first:
  1. table     — exact: constraint status, ranking, ties, best/runner-up, units  (internal inconsistency,
                 presentation ambiguity, the user's own stated limit vs the parsed query)
  2. log/surrogate — seconds: log density near an option, log statistics, surrogate partial dependence
                 on a soft control (missing evidence, data-quality concern, causal/technical claims)
  3. oracle    — retraining: the prediction itself. Here read from the frozen confirmation bank of the
                 FGCS revision (review/outputs/tables/R02_*, R03_*): 3-seed mean validation F1 per
                 configuration, noise floor δ = 0.02 (R02). Off-bank configurations would need a GPU run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..schemas import Claim, Grounds, Verdict

CFX = Path(__file__).resolve().parents[3] / "Conversational-Counterfactual-Guidance-for-Operator-Centred-Hyperparameter-Optimisation"
TABLES = CFX / "review" / "outputs" / "tables"

TARGET = "value"
FEATURES = ["params_backbone", "params_batch_size", "params_center_crop_key", "params_coreset_sampling_ratio",
            "params_image_size_key", "params_layers_key", "params_max_patches_per_image", "params_num_neighbors",
            "params_pre_trained", "params_reduction", "params_soft_corruption_level", "params_soft_review_budget",
            "params_soft_train_fraction"]
CATEGORICAL = ["params_backbone", "params_batch_size", "params_center_crop_key", "params_image_size_key",
               "params_layers_key", "params_max_patches_per_image", "params_pre_trained", "params_reduction",
               "params_soft_corruption_level"]
SOFT = ["params_soft_train_fraction", "params_soft_review_budget", "params_soft_corruption_level"]
FEATURE_LABELS = {
    "params_backbone": "Backbone", "params_batch_size": "Batch size", "params_center_crop_key": "Center crop",
    "params_coreset_sampling_ratio": "Coreset sampling ratio", "params_image_size_key": "Image size",
    "params_layers_key": "Layer set", "params_max_patches_per_image": "Max patches per image",
    "params_num_neighbors": "Nearest neighbors", "params_pre_trained": "Use pretrained weights",
    "params_reduction": "Reduction method", "params_soft_corruption_level": "Corruption level",
    "params_soft_review_budget": "Review budget", "params_soft_train_fraction": "Training fraction",
    TARGET: "Quality (F1 score)",
}
GLOSSARY = {  # Re-present material: plain-language definitions the template never gives
    "params_coreset_sampling_ratio": "the share of training patches kept in PatchCore's memory bank; smaller is faster and uses less memory, at some cost in sensitivity",
    "params_num_neighbors": "how many stored patches each test patch is compared with when scoring an anomaly",
    "params_soft_review_budget": "the fraction of images with the highest anomaly scores that an operator inspects by hand; 0.30 means the top 30%",
    "params_soft_train_fraction": "the share of the available good images used to build the model",
    "params_soft_corruption_level": "a scenario assumption about how degraded the training images are (none / mild / strong)",
    "params_layers_key": "which backbone layers' features are used (l2, l3, or both)",
    "params_reduction": "how patch features are aggregated (mean or max)",
    "params_center_crop_key": "the central crop applied before feature extraction (fraction of the image, or none)",
    "params_backbone": "the pretrained convolutional network that produces the features",
    "params_image_size_key": "the side length images are resized to, in pixels",
    "params_max_patches_per_image": "an upper bound on patches sampled per image when building the memory bank",
    "params_pre_trained": "whether the backbone starts from ImageNet weights",
    "params_batch_size": "how many images are processed together",
    TARGET: "the predicted validation F1 at the chosen review budget, i.e. how well flagged images match true defects",
}
NOISE_FLOOR = 0.02      # R02: seed sd ≈ 0.006–0.009; 0.02 is the floor the FGCS revision uses for "confirmed"
EFFECT_FLOOR = 0.01     # a soft-control effect smaller than this on the surrogate counts as "no effect"
DENSITY_RADIUS = 0.15   # normalised Gower distance within which a logged trial counts as "near"
DENSITY_MIN = 5         # fewer near trials than this = the surrogate is extrapolating (missing evidence)


# ----------------------------------------------------------------------------- vendored template
def feat_label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat.replace("params_", "").replace("_", " ").title())


def option_name(cf_id: int) -> str:
    cid = int(cf_id)
    return f"Option {chr(ord('A') + cid)}" if 0 <= cid < 26 else f"Option {cid + 1}"


def fmt3(x: Any) -> str:
    return f"{float(x):.3f}"


def human_join(items: list[str]) -> str:
    items = [s.strip() for s in items if s.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _rank_label(cf_id: int, score_map: dict[int, float]) -> str:
    score = score_map[cf_id]
    n = len(score_map)
    rank = sum(1 for s in score_map.values() if s > score + 1e-9) + 1
    tied = [option_name(i) for i, s in score_map.items() if i != cf_id and abs(s - score) <= 1e-9]
    if n == 1:
        return "It is the only option."
    if tied and rank == 1:
        return f"It is tied for the highest score with {human_join(tied)}."
    if tied:
        return f"It is tied at rank {rank} of {n}."
    if rank == 1:
        return f"It is the highest-scoring option out of {n}."
    if rank == n:
        return f"It is the lowest-scoring option out of {n}."
    return f"It ranks {rank} of {n} by score."


def _objective_text(obj: dict) -> str:
    t = obj.get("type")
    if t == "target_min":
        return f"Required quality ≥ {fmt3(obj['value'])}."
    if t == "target_band":
        return f"Required quality in [{fmt3(obj['lower'])}, {fmt3(obj['upper'])}]."
    if t == "delta_improve":
        return "Required quality: satisfy the requested improvement target."
    return "Required quality: not specified."


def _requirement_text(score: float, obj: dict, match: bool) -> str:
    t = obj.get("type")
    s = "meets" if match else "misses"
    if t == "target_min":
        return f"Quality {fmt3(score)} — {s} the minimum {fmt3(obj['value'])}."
    if t == "target_band":
        inside = "inside" if match else "outside"
        return f"Quality {fmt3(score)} — {inside} [{fmt3(obj['lower'])}, {fmt3(obj['upper'])}]."
    return f"Quality {fmt3(score)}."


def _soft_ok(row: dict, feat: str, spec: dict) -> bool:
    mode = spec.get("mode", "free")
    val = row.get(feat)
    if mode == "free":
        return True
    if mode == "fixed":
        t = spec.get("value")
        return str(val) == str(t) if feat in CATEGORICAL else abs(float(val) - float(t)) <= 1e-9
    if mode == "range":
        lo, hi = spec.get("lower"), spec.get("upper")
        return lo is not None and hi is not None and float(lo) <= float(val) <= float(hi)
    if mode == "allowed":
        return str(val) in {str(a) for a in spec.get("allowed", [])}
    return False


def _objective_ok(score: float, objective: dict) -> bool:
    t = objective.get("type")
    if t == "target_min":
        return score >= float(objective["value"])
    if t == "target_band":
        return float(objective["lower"]) <= score <= float(objective["upper"])
    return False


def _rank_records(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: (-int(r["objective_match"]), -r["soft_match_rate"], -r["score"], r["cf_id"]))


def summary_records(options: list[dict], query: dict) -> list[dict]:
    obj, sc = query["objective"], query["soft_constraints"]
    out = []
    for o in options:
        violated = [f for f in sc if not _soft_ok(o, f, sc[f])]
        out.append({"cf_id": int(o["cf_id"]), "score": float(o[TARGET]), "objective_match": _objective_ok(float(o[TARGET]), obj),
                    "soft_match_rate": 1.0 - len(violated) / max(1, len(sc)), "violated_constraints": violated})
    return out


EXPLANATION_DISCLAIMER = ("Note: this summary is generated automatically from the option table above and has not been "
                          "audited for factual accuracy. Verify the values in the table before acting on them.")


def build_explanation(records: list[dict], query: dict) -> str:
    if not records:
        return "No alternatives available to explain."
    obj = query.get("objective", {})
    score_map = {int(r["cf_id"]): float(r["score"]) for r in records}
    ranked = _rank_records(records)
    best, runner = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    all_v = [set(r["violated_constraints"]) for r in records]
    universal = set.intersection(*all_v) if all_v else set()
    lines = [_objective_text(obj)]
    lines.append("Scores: " + "; ".join(f"{option_name(r['cf_id'])} {fmt3(r['score'])}" for r in sorted(records, key=lambda r: (-r["score"], r["cf_id"]))) + ".")
    if universal:
        lines.append(f"Note: all options violate the constraint on {human_join([feat_label(v) for v in sorted(universal)])}.")
    lines.append("")
    for r in sorted(records, key=lambda x: x["cf_id"]):
        cid = int(r["cf_id"])
        per = [v for v in r["violated_constraints"] if v not in universal]
        lines.append(f"{option_name(cid)}:")
        lines.append(f"  • {_requirement_text(r['score'], obj, r['objective_match'])} {_rank_label(cid, score_map)}")
        lines.append(f"  • Violates: {human_join([feat_label(v) for v in per])}." if per else "  • Satisfies all per-option constraints.")
    lines.append("")
    bs = [v for v in best["violated_constraints"] if v not in universal]
    clause = "all per-option constraints are satisfied" if not bs else f"it violates {human_join([feat_label(v) for v in bs])}"
    lines.append(f"Best choice: {option_name(int(best['cf_id']))}. Quality {fmt3(best['score'])}, {clause}. {_rank_label(int(best['cf_id']), score_map)}")
    if runner is not None:
        rs = [v for v in runner["violated_constraints"] if v not in universal]
        rc = "all per-option constraints are satisfied" if not rs else f"it violates {human_join([feat_label(v) for v in rs])}"
        lines.append(f"Runner-up: {option_name(int(runner['cf_id']))}. Quality {fmt3(runner['score'])}, {rc}. {_rank_label(int(runner['cf_id']), score_map)}")
    return "\n".join(lines)


def table_text(options: list[dict], records: list[dict]) -> str:
    vmap = {r["cf_id"]: (human_join([feat_label(v) for v in r["violated_constraints"]]) or "none") for r in records}
    cols = [TARGET, "params_soft_review_budget", "params_soft_train_fraction", "params_soft_corruption_level",
            "params_backbone", "params_layers_key", "params_image_size_key", "params_coreset_sampling_ratio", "params_num_neighbors", "params_pre_trained"]
    head = "| Option | " + " | ".join(feat_label(c) for c in cols) + " | Violates |"
    rows = [head, "|" + "---|" * (len(cols) + 2)]
    for o in options:
        vals = []
        for c in cols:
            v = o[c]
            vals.append(fmt3(v) if c in (TARGET, "params_soft_review_budget", "params_soft_train_fraction") else str(v))
        rows.append(f"| {option_name(o['cf_id'])} | " + " | ".join(vals) + f" | {vmap[int(o['cf_id'])]} |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- oracle bank
def load_bank() -> pd.DataFrame:
    """Every configuration with a surrogate prediction and a retraining confirmation (3 seeds)."""
    rows = []
    for ds in ["mvtec", "proditec"]:
        cfs = pd.read_csv(CFX / ds / "surrogate_pkl_cfs_metadata" / f"{ds}_counterfactuals" / f"{ds}_all_methods_cfs.csv")
        agg = pd.read_csv(TABLES / f"R02_{ds}_candidates_aggregated.csv")
        m = agg.merge(cfs, left_on="id", right_on="global_cf_id")
        assert len(m) == 24 and (m.surrogate - m.surrogate_pred).abs().max() < 1e-9
        for _, r in m.iterrows():
            rows.append({"dataset": ds, "source": "candidate", "bank_id": f"{ds}_cand_{int(r.id)}", "method": r.method_x if "method_x" in r else r.method,
                         **{f: r[f] for f in FEATURES}, "surrogate": r.surrogate, "f1_val": r.f1_val, "f1_val_sd": r.f1_val_sd,
                         "f1_test": r.f1_test, "n_seeds": 3})
        sel = pd.read_csv(TABLES / f"R02_{ds}_logged_trial_selection.csv")
        rer = pd.read_csv(TABLES / f"R02_{ds}_logged_rerun.csv")
        rer["number"] = rer.id.str.replace("trial", "").astype(int)
        m = sel.merge(rer[["number", "surrogate", "f1_val", "f1_val_sd", "f1_test", "logged"]], on="number")
        assert len(m) == 48
        for _, r in m.iterrows():
            rows.append({"dataset": ds, "source": "logged", "bank_id": f"{ds}_log_{int(r.number)}", "method": "log",
                         **{f: r[f.replace("params_", "")] for f in FEATURES}, "surrogate": r.surrogate, "f1_val": r.f1_val,
                         "f1_val_sd": r.f1_val_sd, "f1_test": r.f1_test, "n_seeds": 3, "logged_value": r.logged})
    # R03 anchor sweep (MVTec only; join on anchor, method and the surrogate value)
    r3 = pd.read_csv(TABLES / "R03_multi_anchor_candidates.csv")
    mf = pd.read_csv(CFX / "mvtec" / "surrogate_pkl_cfs_metadata" / "mvtec_revision_analysis" / "mvtec_multifactual_candidates.csv")
    r3 = r3.drop(columns=["params_soft_review_budget"], errors="ignore")
    r3["k"] = r3.surrogate_pred.round(6); mf["k"] = mf.surrogate_pred.round(6)
    m = r3.merge(mf.drop(columns=["surrogate_pred"]), on=["anchor", "method", "k"], how="inner").drop_duplicates("id")
    for _, r in m.iterrows():
        rows.append({"dataset": "mvtec", "source": "anchor_sweep", "bank_id": f"mvtec_r3_{r.id}", "method": r.method,
                     **{f: r[f] for f in FEATURES}, "surrogate": r.surrogate_pred, "f1_val": r.f1_val, "f1_val_sd": np.nan,
                     "f1_test": r.f1_test, "n_seeds": 1})
    bank = pd.DataFrame(rows)
    for c in CATEGORICAL:
        bank[c] = bank[c].astype(str).str.replace(r"\.0$", "", regex=True)
    bank["err_val"] = bank.f1_val - bank.surrogate
    bank[TARGET] = bank.surrogate          # what the table shows
    return bank


# ----------------------------------------------------------------------------- the adapter
class CFXHPOAdapter:
    """One adapter per (dataset, query, options). Options are rows of the bank (so the oracle is known)."""

    def __init__(self, dataset: str, query: dict, options: list[dict], user_request: str = "",
                 bank: pd.DataFrame | None = None, surrogate=None, log: pd.DataFrame | None = None):
        self.dataset, self.query, self.user_request = dataset, query, user_request
        self.options = [dict(o, cf_id=i) for i, o in enumerate(options)]
        self.records = summary_records(self.options, query)
        self.explanation = build_explanation(self.records, query)
        self.bank = bank
        self._cb = surrogate
        self._log = log

    # ---- state and claims
    def state(self) -> dict:
        return {"dataset": self.dataset, "user_request": self.user_request, "query": self.query,
                "table": table_text(self.options, self.records), "explanation": self.explanation}

    def claims(self) -> list[Claim]:
        out = []
        obj = self.query["objective"]
        for r in self.records:
            n = option_name(r["cf_id"])
            out.append(Claim("score", n, r["score"], f"{n} has predicted quality {fmt3(r['score'])}"))
            out.append(Claim("meets_objective", n, r["objective_match"], f"{n} {'meets' if r['objective_match'] else 'misses'} the objective {_objective_text(obj)}"))
            for f in self.query["soft_constraints"]:
                ok = f not in r["violated_constraints"]
                out.append(Claim("satisfies", f"{n}:{f}", ok, f"{n} {'satisfies' if ok else 'violates'} the constraint on {feat_label(f)}"))
        ranked = _rank_records(self.records)
        out.append(Claim("best", option_name(ranked[0]["cf_id"]), True, f"Best choice: {option_name(ranked[0]['cf_id'])}"))
        return out

    def claim_menu(self) -> list[dict]:
        """Closed list of what a user turn can be about. Understand picks one entry; slots are filled after.
        Each entry: id, text (what the user would be claiming), kind, subject, user_value defaults, slots."""
        m = []
        names = [option_name(o["cf_id"]) for o in self.options]
        obj_v = float(self.query["objective"].get("value", 0))
        constrained = [f for f, spec in self.query["soft_constraints"].items() if spec.get("mode") != "free"]
        for n in names:
            m.append({"id": f"{n}:reach", "text": f"{n} will NOT reach the required minimum quality once actually trained", "kind": "score_reach",
                      "subject": n, "user_value": {"threshold": obj_v}, "slots": {"threshold": "the minimum quality the user says it will not reach (number); default the query's minimum"}})
            m.append({"id": f"{n}:over", "text": f"{n}'s quality in the table is overstated; retraining will come out clearly lower", "kind": "score_over",
                      "subject": n, "user_value": {}, "slots": {}})
            for f in constrained:
                m.append({"id": f"{n}:violates:{f}", "text": f"{n} violates the user's constraint on {feat_label(f).lower()} although the summary says it satisfies it",
                          "kind": "satisfies", "subject": n, "user_value": {"feature": f, "user_says_violates": True},
                          "slots": {"user_limit": "the limit the user states in their own words, as {'max': number} or {'min': number}; omit if none stated"}})
            for f in ["params_soft_review_budget", "params_soft_train_fraction"]:
                m.append({"id": f"{n}:noeffect:{f}", "text": f"changing the {feat_label(f).lower()} of {n} would make no difference to its quality",
                          "kind": "feature_effect", "subject": n, "user_value": {"feature": f, "user_says_no_effect": True},
                          "slots": {"new_value": f"the value of {feat_label(f).lower()} the user proposes (number)"}})
            m.append({"id": f"{n}:extrapolated", "text": f"the log has no trials near {n}; its quality is an extrapolation by the surrogate", "kind": "density",
                      "subject": n, "user_value": {}, "slots": {}})
        for a in names:
            for b in names:
                if a != b:
                    m.append({"id": f"{a}>{b}:table", "text": f"{a} has higher quality than {b} IN THE TABLE, so the ranking shown is wrong", "kind": "rank_table",
                              "subject": a, "user_value": {"better": a, "worse": b}, "slots": {}})
                    m.append({"id": f"{a}>{b}:trained", "text": f"{a} will beat {b} once both are actually trained, whatever the table says", "kind": "rank_confirmed",
                              "subject": a, "user_value": {"better": a, "worse": b}, "slots": {}})
        m.append({"id": "log:corrupted_most", "text": "most of the logged trials used corrupted training data, so the surrogate is not trustworthy", "kind": "log_stat",
                  "subject": "log", "user_value": {"stat": "corrupted_share", "op": ">", "threshold": 0.5}, "slots": {}})
        m.append({"id": "log:corrupted_few", "text": "only a small share of the logged trials used corrupted data, so the surrogate has barely seen corruption", "kind": "log_stat",
                  "subject": "log", "user_value": {"stat": "corrupted_share", "op": "<", "threshold": 0.25}, "slots": {}})
        m.append({"id": "units", "text": "the review budget or training fraction are shown in the wrong units", "kind": "units", "subject": "", "user_value": {}, "slots": {}})
        m.append({"id": "uncheckable", "text": "a claim about the production line, the user's own past experience, or anything the log and surrogate do not record", "kind": "uncheckable", "subject": "", "user_value": {}, "slots": {}})
        m.append({"id": "request:term", "text": "the user asks what a term or column means", "kind": "request", "subject": "", "user_value": {}, "slots": {"term": "the parameter or column asked about"}})
        m.append({"id": "request:percent", "text": "the user asks to see budget or fractions as percentages", "kind": "request", "subject": "format", "user_value": {"format": "percent"}, "slots": {}})
        m.append({"id": "request:plain", "text": "the user asks for the ranking or the result in plainer words", "kind": "request", "subject": "plain", "user_value": {"format": "plain"}, "slots": {}})
        m.append({"id": "none", "text": "nothing to respond to: acknowledgement, agreement, choosing an option, small talk", "kind": "none", "subject": "", "user_value": {}, "slots": {}})
        return m

    # ---- helpers
    def _opt(self, name: str) -> dict:
        idx = ord(name.strip().upper()[-1]) - ord("A") if name.strip().upper().startswith("OPTION") else int(name)
        return self.options[idx]

    def _surrogate(self):
        if self._cb is None:
            from catboost import CatBoostRegressor
            cb = CatBoostRegressor()
            cb.load_model(str(CFX / self.dataset / "surrogate_pkl_cfs_metadata" / f"{self.dataset}_surrogate_catboost.cbm"))
            self._cb = cb
        return self._cb

    def _predict(self, cfg: dict) -> float:
        cb = self._surrogate()
        X = pd.DataFrame([{f: cfg[f] for f in cb.feature_names_}])
        for c in CATEGORICAL:
            X[c] = X[c].astype(str)
        return float(cb.predict(X)[0])

    def _log_df(self) -> pd.DataFrame:
        if self._log is None:
            p = CFX / self.dataset / "surrogate_ready_dataset" / ("mvtec_4class_patchcore_surrogate_dataset.csv" if self.dataset == "mvtec" else "patchcore_surrogate_dataset_xgb.csv")
            self._log = pd.read_csv(p)
        return self._log

    def _density(self, cfg: dict) -> int:
        log = self._log_df()
        num = [f for f in FEATURES if f not in CATEGORICAL]
        d = np.zeros(len(log))
        for f in CATEGORICAL:
            d += (log[f].astype(str).str.replace(r"\.0$", "", regex=True) != str(cfg[f])).astype(float)
        for f in num:
            rng = log[f].max() - log[f].min()
            d += (log[f] - float(cfg[f])).abs() / (rng if rng > 0 else 1)
        d /= len(FEATURES)
        return int((d <= DENSITY_RADIUS).sum())

    # ---- verify
    _LABEL_TO_FEAT = {v.lower(): k for k, v in FEATURE_LABELS.items()}

    def _feat(self, name) -> str:
        """Accept a parameter name, a display label, or a loose phrase ('review budget')."""
        n = str(name or "").strip()
        if n in FEATURES:
            return n
        low = n.lower().replace("_", " ").replace("params ", "").strip()
        for k, v in FEATURE_LABELS.items():
            if low == v.lower() or low == k.replace("params_", "").replace("_", " "):
                return k
        for k in FEATURES:
            if low and low in k.replace("params_", "").replace("_", " "):
                return k
        return n

    def normalise(self, g: Grounds) -> Grounds:
        """Make an Understand output usable: option subject, parameter names, typed values."""
        uv = g.user_value if isinstance(g.user_value, dict) else ({} if g.user_value in (None, "") else {"value": g.user_value})
        text = str(g.extra.get("user_text", "")) if g.extra else ""
        opts = re.findall(r"Option ([A-Z])", text)
        subj = str(g.subject or "").strip()
        if g.claim_kind in ("score_reach", "score_over", "satisfies", "feature_effect", "density", "rank_table", "rank_confirmed"):
            if not re.fullmatch(r"Option [A-Z]", subj):
                if re.fullmatch(r"[A-Z]", subj):
                    subj = f"Option {subj}"
                elif uv.get("better") and g.claim_kind.startswith("rank"):
                    subj = str(uv["better"])
                elif opts:
                    subj = f"Option {opts[0]}"
                else:
                    subj = option_name(0)
        if "feature" in uv:
            uv["feature"] = self._feat(uv["feature"])
        elif g.claim_kind in ("satisfies", "feature_effect"):
            uv["feature"] = self._feat(g.subject) if self._feat(g.subject) in FEATURES else "params_soft_review_budget"
        if g.claim_kind == "satisfies":
            uv.setdefault("user_says_violates", True)
            lim = uv.get("user_limit")
            if isinstance(lim, (int, float)):
                uv["user_limit"] = {"max": float(lim)}
        if g.claim_kind == "feature_effect":
            uv.setdefault("user_says_no_effect", True)
            if "new_value" not in uv:
                m = re.search(r"to ([0-9]*\.?[0-9]+)", text)
                uv["new_value"] = float(m.group(1)) if m else 0.1
        if g.claim_kind == "score_reach" and "threshold" not in uv:
            m = re.search(r"reach ([0-9]*\.?[0-9]+)", text)
            uv["threshold"] = float(m.group(1)) if m else float(self.query["objective"].get("value", 0))
        if g.claim_kind == "log_stat":
            uv.setdefault("stat", "corrupted_share"); uv.setdefault("op", ">"); uv.setdefault("threshold", 0.5)
        if g.claim_kind == "request":
            if uv.get("term"):
                subj = str(uv["term"])
            if subj in ("format", "percent") or uv.get("format") == "percent":
                uv["format"] = "percent"
            elif subj == "plain" or uv.get("format") == "plain":
                uv["format"] = "plain"
            else:
                subj = self._feat(subj)
        return Grounds(g.target, g.act, g.grounds, g.claim_kind, subj, uv, g.extra)

    def verify(self, g: Grounds, oracle: bool = True) -> Verdict:
        """oracle=False withholds the retraining bank (the 'built verifier' condition of E3)."""
        g = self.normalise(g)
        kind, subj, uv = g.claim_kind, g.subject, g.user_value
        # routing is by the claim, not by the grounds: a prior-experience objection to a *prediction* is
        # checkable by retraining; a prior-experience claim about the line is not
        if kind == "uncheckable":
            return Verdict("no", "uncheckable", ["the claim concerns the production line or the operator's experience, which the log does not record"], tier="none")
        if kind == "request":
            return Verdict("no", "uncheckable", ["not a dispute"], tier="none")
        # ---- tier 1: table
        if kind == "satisfies":
            o = self._opt(subj); f = uv["feature"]; lim = uv.get("user_limit")
            val = float(o[f]) if f not in CATEGORICAL else o[f]
            spec = self.query["soft_constraints"][f]
            ok_parsed = _soft_ok(o, f, spec)
            ev = [f"{option_name(o['cf_id'])} {feat_label(f)} = {val}; parsed constraint {spec}"]
            if lim is not None and f not in CATEGORICAL:
                ok_user = float(val) <= float(lim["max"]) if "max" in lim else float(val) >= float(lim["min"])
                ev.append(f"your stated limit: {lim}")
                if ok_parsed and not ok_user:
                    return Verdict("yes", "user_right", ev + ["the parsed query does not match your request; the table's status is wrong for your limit"], tier="table")
            ev.append(("satisfies" if ok_parsed else "violates") + " the constraint as parsed")
            # the user says it violates: right iff the table shows a violation
            return Verdict("yes", "user_right" if not ok_parsed else "user_wrong", ev, tier="table")
        if kind == "rank_table":
            a, b = self._opt(uv["better"]), self._opt(uv["worse"])
            sa, sb = float(a[TARGET]), float(b[TARGET])
            ev = [f"table: {option_name(a['cf_id'])} {fmt3(sa)} vs {option_name(b['cf_id'])} {fmt3(sb)}"]
            return Verdict("yes", "user_right" if sa > sb + 1e-9 else "user_wrong", ev, tier="table")
        if kind == "units":
            return Verdict("yes", "user_wrong", ["review budget and training fraction are fractions of 1 (0.30 = 30%), as the table header states"], tier="table")
        # ---- tier 2: log and surrogate
        if kind == "density":
            o = self._opt(subj); n = self._density(o)
            ev = [f"{n} logged trials within normalised distance {DENSITY_RADIUS} of {option_name(o['cf_id'])} (threshold {DENSITY_MIN})"]
            return Verdict("yes", "user_right" if n < DENSITY_MIN else "user_wrong", ev, tier="log", detail={"n_near": n})
        if kind == "log_stat":
            log = self._log_df(); stat = uv["stat"]
            if stat == "corrupted_share":
                v = float((log.params_soft_corruption_level.astype(str) != "none").mean())
            elif stat == "pretrained_off_share":
                v = float((log.params_pre_trained.astype(str).isin(["False", "0", "0.0"])).mean())
            elif stat == "n_trials":
                v = float(len(log))
            else:
                raise ValueError(stat)
            ev = [f"log statistic {stat} = {v:.3f} over {len(log)} trials; your claim: {uv['op']} {uv['threshold']}"]
            ok = (v > uv["threshold"]) if uv["op"] == ">" else (v < uv["threshold"])
            return Verdict("yes", "user_right" if ok else "user_wrong", ev, tier="log", detail={"value": v})
        if kind == "feature_effect":
            o = self._opt(subj); f = uv["feature"]
            base = float(o[TARGET]); alt = dict(o); alt[f] = uv["new_value"]
            pred = self._predict(alt); delta = pred - base
            ev = [f"surrogate: {option_name(o['cf_id'])} with {feat_label(f)} = {uv['new_value']} → {fmt3(pred)} (now {fmt3(base)}, change {delta:+.3f}; floor {EFFECT_FLOOR})"]
            no_effect = abs(delta) <= EFFECT_FLOOR
            user_says_no_effect = uv.get("user_says_no_effect", True)
            return Verdict("yes", "user_right" if no_effect == user_says_no_effect else "user_wrong", ev, tier="surrogate", detail={"delta": delta})
        # ---- tier 3: oracle (retraining bank)
        if kind in ("score_reach", "score_over", "rank_confirmed"):
            if kind == "rank_confirmed":
                a, b = self._opt(uv["better"]), self._opt(uv["worse"])
                sa, sb = float(a[TARGET]), float(b[TARGET])
                if not oracle:
                    ev = [f"surrogate: {option_name(a['cf_id'])} {fmt3(sa)} vs {option_name(b['cf_id'])} {fmt3(sb)}; no retraining available"]
                    return Verdict("partly", "user_right" if sa > sb else "user_wrong", ev, tier="surrogate")
                ca, cb_ = float(a["f1_val"]), float(b["f1_val"])
                ev = [f"retrained (3 seeds): {option_name(a['cf_id'])} {fmt3(ca)} vs {option_name(b['cf_id'])} {fmt3(cb_)}; noise floor {NOISE_FLOOR}"]
                if ca > cb_ + NOISE_FLOOR:
                    return Verdict("yes", "user_right", ev, tier="oracle")
                if cb_ > ca + NOISE_FLOOR or abs(ca - cb_) <= NOISE_FLOOR and sa > sb:
                    return Verdict("yes", "user_wrong" if cb_ >= ca else "undecided", ev + ["within the noise floor" if abs(ca - cb_) <= NOISE_FLOOR else ""], tier="oracle")
                return Verdict("yes", "undecided", ev + ["within the noise floor"], tier="oracle")
            o = self._opt(subj); s = float(o[TARGET]); conf = float(o["f1_val"])
            if kind == "score_reach":
                v = float(uv["threshold"])
                if not oracle:
                    n = self._density(o)
                    ev = [f"surrogate predicts {fmt3(s)} ≥ {fmt3(v)}; {n} logged trials nearby; no retraining available"]
                    return Verdict("partly", "user_wrong" if s >= v else "user_right", ev, tier="surrogate")
                ev = [f"retrained (3 seeds, sd {o.get('f1_val_sd', float('nan')):.3f}): validation F1 {fmt3(conf)} vs required {fmt3(v)}; noise floor {NOISE_FLOOR}"]
                if conf < v - NOISE_FLOOR:
                    return Verdict("yes", "user_right", ev, tier="oracle")
                if conf >= v:
                    return Verdict("yes", "user_wrong", ev, tier="oracle")
                return Verdict("yes", "undecided", ev + ["within the noise floor"], tier="oracle")
            # score_over: "the table overstates it by more than δ"
            if not oracle:
                n = self._density(o)
                ev = [f"surrogate {fmt3(s)}; {n} logged trials nearby; surrogate validation RMSE ≈ 0.015; no retraining available"]
                return Verdict("partly", "user_wrong" if n >= DENSITY_MIN else "undecided", ev, tier="surrogate")
            gap = s - conf
            ev = [f"retrained (3 seeds): validation F1 {fmt3(conf)} vs table {fmt3(s)} (gap {gap:+.3f}; floor {NOISE_FLOOR})"]
            return Verdict("yes", "user_right" if gap > NOISE_FLOOR else "user_wrong", ev, tier="oracle")
        return Verdict("no", "uncheckable", [f"no check for claim kind {kind!r}"], tier="none")

    # ---- revise / hold / re-present
    def revise(self, g: Grounds, verdict: Verdict) -> dict:
        g = self.normalise(g)
        kind, subj, uv = g.claim_kind, g.subject, g.user_value
        changed = {}
        if kind in ("score_reach", "score_over", "rank_confirmed"):
            for o in self.options:
                if kind == "rank_confirmed" or option_name(o["cf_id"]) == subj:
                    o[TARGET] = float(o["f1_val"]); o["confirmed"] = True
                    changed[option_name(o["cf_id"])] = f"quality replaced by the retrained value {fmt3(o['f1_val'])} (marked confirmed)"
        elif kind == "satisfies" and uv.get("user_limit"):
            f = uv["feature"]; lim = uv["user_limit"]
            self.query["soft_constraints"][f] = {"mode": "range", "lower": lim.get("min", 0.0), "upper": lim.get("max", 1.0)}
            changed[f] = f"constraint re-parsed from your request: {self.query['soft_constraints'][f]}"
        elif kind == "feature_effect":
            changed["note"] = f"{feat_label(uv['feature'])} has no material effect on {subj} according to the surrogate; the table now says so"
        elif kind == "density":
            changed["note"] = f"{subj} is marked as extrapolated (few logged trials nearby); recommend confirming by retraining before use"
        elif kind == "log_stat":
            changed["note"] = f"the explanation now states the log statistic: {verdict.evidence[0]}"
        self.records = summary_records(self.options, self.query)
        self.explanation = build_explanation(self.records, self.query)
        return {"changed": changed, "explanation": self.explanation}

    def evidence_for_hold(self, g: Grounds, verdict: Verdict) -> list[str]:
        return list(verdict.evidence)

    def represent(self, g: Grounds) -> str:
        g = self.normalise(g)
        f = g.subject
        if f in GLOSSARY:
            return f"{feat_label(f)}: {GLOSSARY[f]}."
        if g.user_value and g.user_value.get("format") == "percent":
            rows = [f"{option_name(o['cf_id'])}: review budget {float(o['params_soft_review_budget']) * 100:.1f}% of images, training fraction {float(o['params_soft_train_fraction']) * 100:.0f}% of the good images" for o in self.options]
            return "\n".join(rows)
        if g.user_value and g.user_value.get("format") == "plain":
            ranked = _rank_records(self.records)
            b = ranked[0]
            return (f"In short: {option_name(b['cf_id'])} is the best of the {len(self.options)} options because it has the highest predicted quality "
                    f"({fmt3(b['score'])}) among those that meet your requirements. Quality is the share of flagged images that are real defects, balanced against missed defects.")
        return "Here is the same result in plainer terms: " + self.explanation.split("\n")[0]
