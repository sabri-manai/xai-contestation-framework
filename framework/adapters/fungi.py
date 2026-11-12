"""XAI-FUNGI domain adapter: the natural-episode setting (architecture.md §4.1), rendering and data tiers.

The five explanation formats the participants saw are fixed slides (detecting-contestation-xai/data/
ORIGINAL_VISUALIZATIONS_EN.pdf). The adapter holds a *fact base* read off those slides: what each slide
shows, what it says in its caption and legend, and the display faults it actually has. A user's claim
about the display or the data is checked against that base. Claims about the model's behaviour
(feature perturbation) need the XGBoost model, which the released corpus does not include, so the
model tier is **not built**: such claims come back uncheckable and the paper says so.
"""
from __future__ import annotations

from ..schemas import Claim, Grounds, Verdict

# Facts per format. Keys are propositions; values are what is true on the slide.
FACTS: dict[str, dict] = {
    "Descriptive statistics": {
        "caption": "Textual summary of the three numeric features and the missing-data count and percent per column; bar charts of value counts per categorical feature.",
        "n_rows": 61069, "n_species": 173, "partly_simulated": True,
        "cap_diameter_mean": 6.73, "cap_diameter_max": 62.34, "cap_diameter_min": 0.38, "cap_diameter_median": 5.86,
        "stem_height_mean": 6.58, "stem_height_max": 33.92, "stem_height_min": 0.0,
        "stem_width_mean": 12.15, "stem_width_max": 103.91,
        "missing_veil_type_pct": 94.8, "missing_spore_print_pct": 89.6, "missing_veil_color_pct": 87.9,
        "missing_stem_root_pct": 86.1, "missing_stem_surface_pct": 62.4, "missing_gill_spacing_pct": 41.0,
        "missing_cap_surface_pct": 30.4, "missing_gill_attachment_pct": 16.2, "missing_ring_type_pct": 4.0,
        "missing_habitat_pct": 0.0, "missing_cap_color_pct": 0.0, "missing_season_pct": 0.0,
        "more_poisonous_than_edible": True, "most_common_cap_shape": "convex", "most_common_cap_color": "brown",
        "most_common_gill_color": "white", "most_common_habitat_shown": False,
        "units_stated": True, "missing_shown_as_grey_bar": True, "percent_column_labelled": True,
    },
    "SHAP": {
        "caption": "Beeswarm: impact of individual features on the edibility prediction; points to the right (> 0) push towards poisonous, to the left towards edible. Waterfall: contribution of each feature for one mushroom that is actually poisonous.",
        "beeswarm_top3": ["stem_height_cm", "cap_diameter_cm", "stem_width_mm"],
        "beeswarm_rows_shown": 20, "beeswarm_rows_near_zero": 17, "beeswarm_legend_present": True,
        "beeswarm_right_means_poisonous": True, "beeswarm_axis_labelled": True,
        "high_stem_height_pushes_poisonous": True, "high_stem_width_pushes_poisonous": False,
        "high_cap_diameter_pushes_poisonous": True, "cap_color_among_top3": False,
        "waterfall_fx": 9.959, "waterfall_base": 9.208, "waterfall_top_feature": "stem_height_cm",
        "waterfall_stem_height_contrib": 0.78, "waterfall_stem_width_contrib": -0.22, "waterfall_cap_diameter_contrib": 0.18,
        "waterfall_other_features_collapsed": 106, "waterfall_units_of_fx_stated": False,
        "waterfall_feature_values_in_cm": False, "waterfall_instance_true_class": "poisonous",
        "waterfall_legend_present": True,
    },
    "LIME": {
        "caption": "LIME bars: feature conditions that increase (red, right) or decrease (blue, left) the probability of non-edible/poisonous for one mushroom.",
        "top_red": "stem_width_mm <= 5.20", "top_red_value": 0.07, "top_blue": "stem_width_mm > 16.56", "top_blue_value": -0.07,
        "no_bruises_pushes_poisonous": True, "bruises_pushes_poisonous": False, "white_gill_pushes_edible": True,
        "brown_stem_pushes_poisonous": True, "stem_height_over_7_74_pushes_poisonous": True,
        "n_bars": 14, "instance_shown": False, "class_probability_axis_labelled": True,
        "unit_typo_stem_height_mm": True, "edible_misspelled": True,
        "several_bins_of_same_feature_listed": True, "legend_present": False, "axis_labels_present": True,
        "bars_are_probability_changes": True,
    },
    "Counterfactual": {
        "caption": "Counterfactual analysis: what data needs to be changed for a mushroom the model (correctly) classified as non-edible/poisonous to receive the prediction edible.",
        "instance_true_class": "poisonous", "model_correct": True, "n_comparisons": 2, "comparisons_identical": True,
        "changed_feature": "stem_height_cm", "change_from": 1.43, "change_to": 23.5, "n_features_changed": 1,
        "cap_color_changed": False, "habitat_changed": False, "legend_present": True,
        "caption_says_filling_missing_data": True, "caption_matches_method": False,
        "instance_cap_diameter": 9.41, "instance_cap_color": "yellow", "instance_habitat": "forests", "instance_season": "autumn",
        "instance_has_no_data_fields": True, "n_no_data_fields": 6,
    },
    "Anchor": {
        "caption": "Anchor: a set of feature conditions whose joint presence determines the classification; below, the percentage of cases in which the model predicts the class when the highlighted conditions hold.",
        "edible_anchor_n_conditions": 9, "edible_anchor_precision": 97.2, "edible_reduced_precision": 76.2,
        "edible_reduced_highlighted": 5, "poisonous_anchor_n_conditions": 5, "poisonous_anchor_precision": 97.5,
        "poisonous_reduced_highlighted": 2, "poisonous_reduced_precision": 84.0,
        "anchor_contains_no_data_conditions": True, "edible_anchor_no_data_conditions": 2,
        "anchor_need_not_be_real_example": True, "typo_capCdiameter": True,
        "edible_example_cap_shape": "convex", "edible_example_habitat": "meadows", "edible_example_season": "spring",
        "poisonous_example_cap_color": "green", "poisonous_example_gill_attachment": "adnexed",
        "legend_present": True, "percent_explained_in_caption": True, "green_cap_in_poisonous_anchor": True,
    },
}

DISPLAY_TEXT = {k: v["caption"] for k, v in FACTS.items()}


class FungiAdapter:
    def __init__(self, fmt: str):
        assert fmt in FACTS, fmt
        self.fmt = fmt
        self.facts = dict(FACTS[fmt])

    def state(self) -> dict:
        return {"explanation_format": self.fmt, "caption": self.facts["caption"],
                "facts": {k: v for k, v in self.facts.items() if k != "caption"}}

    def claims(self) -> list[Claim]:
        return [Claim("display", k, v, f"{k} = {v}") for k, v in self.facts.items() if k != "caption"]

    def claim_menu(self) -> list[dict]:
        m = []
        for k, v in self.facts.items():
            if k == "caption":
                continue
            phrase = k.replace("_pct", " (percent)").replace("_", " ")
            if isinstance(v, bool):
                text = f"the user asserts that it is (or is not) the case that: {phrase}"
                slot = "true if the user says this IS the case, false if the user says it is NOT the case"
            elif isinstance(v, (int, float)):
                text = f"the user states a number for: {phrase}"
                slot = f"the number the user states for {phrase} (a number, not true/false)"
            else:
                text = f"the user names a value for: {phrase}"
                slot = f"the value the user names for {phrase} (a word)"
            m.append({"id": k, "text": text, "kind": "fact", "subject": k, "user_value": None, "slots": {"asserted": slot}})
        m.append({"id": "model_effect", "text": "a claim about how the model would behave if a feature were changed or removed", "kind": "model_effect", "subject": "", "user_value": None, "slots": {}})
        m.append({"id": "uncheckable", "text": "a claim from mushroom knowledge, personal experience, or about the model's competence that the slide and data cannot settle", "kind": "uncheckable", "subject": "", "user_value": None, "slots": {}})
        for k, t in [("legend", "the user ASKS what the colours, symbols, or columns mean (a question, not a claim that something is missing or wrong)"),
                     ("term", "the user ASKS what a term means"), ("plain", "the user ASKS for plainer words or a shorter version")]:
            m.append({"id": f"request:{k}", "text": t, "kind": "request", "subject": k, "user_value": None, "slots": {}})
        m.append({"id": "none", "text": "nothing to respond to: reading, agreeing, small talk", "kind": "none", "subject": "", "user_value": None, "slots": {}})
        return m

    def verify(self, g: Grounds, oracle: bool = True) -> Verdict:
        kind = g.claim_kind
        if kind in ("uncheckable", "request"):
            return Verdict("no", "uncheckable", ["the claim needs knowledge of mushrooms or of the model's internals that the slides and the released data do not contain"], tier="none")
        if kind == "model_effect":
            # would need the XGBoost model: perturb the feature and re-predict. Not released → not built.
            return Verdict("partly", "uncheckable", ["checking this needs the classifier (feature perturbation); it is not available to this verifier"], tier="none")
        if kind in ("fact", "display_fault", "data_stat"):
            key = g.subject; asserted = g.user_value
            if isinstance(asserted, dict):
                asserted = asserted.get("asserted", asserted.get("value"))
            if isinstance(asserted, str):
                low = asserted.strip().lower()
                if low in ("true", "yes"):
                    asserted = True
                elif low in ("false", "no"):
                    asserted = False
                else:
                    try:
                        asserted = float(low.replace("%", "").replace(",", ""))
                    except ValueError:
                        pass
            if key not in self.facts:
                return Verdict("no", "uncheckable", [f"no fact recorded for {key}"], tier="none")
            actual = self.facts[key]
            tier = "data" if kind == "data_stat" else "rendering"
            if isinstance(actual, bool):
                right = bool(asserted) == actual
            elif isinstance(actual, (int, float)):
                tol = 0.05 * abs(actual) + 0.05
                right = abs(float(asserted) - float(actual)) <= tol if isinstance(asserted, (int, float)) else False
            elif isinstance(actual, list):
                right = (asserted in actual) if isinstance(asserted, str) else (list(asserted) == actual)
            else:
                right = str(asserted).lower() == str(actual).lower()
            ev = [f"slide ({self.fmt}): {key} = {actual}; you said {asserted}"]
            return Verdict("yes", "user_right" if right else "user_wrong", ev, tier=tier)
        return Verdict("no", "uncheckable", [f"no check for claim kind {kind!r}"], tier="none")

    def revise(self, g: Grounds, verdict: Verdict) -> dict:
        key = g.subject
        return {"changed": {key: f"the explanation now states this explicitly: {verdict.evidence[0]}"},
                "explanation": self.facts["caption"] + f" Correction: {verdict.evidence[0]}."}

    def evidence_for_hold(self, g: Grounds, verdict: Verdict) -> list[str]:
        return list(verdict.evidence)

    def represent(self, g: Grounds) -> str:
        return {"legend": "Legend: " + self.facts["caption"],
                "plain": "In plain words: " + self.facts["caption"],
                "term": f"The term you asked about ({g.user_value}) is explained as follows: " + self.facts["caption"]}.get(g.subject, self.facts["caption"])
