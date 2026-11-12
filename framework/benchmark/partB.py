"""Part B: matched pairs of right and wrong objections against the XAI-FUNGI slides.

Each pair shares a template and a target; one member asserts something true about the slide (grounded,
user right) and the other something false (ungrounded, user wrong). Truth is read off the fact base in
adapters/fungi.py, which was transcribed from the original slides. Authored 12 Nov 2026 by the authors,
not observed; the template mix follows the grounds distribution of gold (domain knowledge, presentation
ambiguity, missing evidence, technical knowledge) and of the 258. Model-behaviour claims (feature
perturbation) are included as *uncheckable by the built verifier*; they carry no validity label.

Writes data/partB_pairs.csv.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from ..adapters.fungi import FungiAdapter
from ..decide import gold_action
from ..schemas import Grounds

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261112

# (format, template id, target, act, grounds, claim_kind, fact key, [(text, asserted_value), (text, asserted_value)])
PAIRS = [
    # ---------------- Descriptive statistics
    ("Descriptive statistics", "B01", "input data", "direct challenge", "data-quality concern", "data_stat", "missing_veil_type_pct",
     [("Veil type is missing for almost every mushroom, around 95%, so the model has basically nothing to learn from it.", 95),
      ("Veil type is missing for about half of the mushrooms, so the model is working with a half-empty column.", 50)]),
    ("Descriptive statistics", "B02", "input data", "direct challenge", "data-quality concern", "data_stat", "more_poisonous_than_edible",
     [("Your dataset has more poisonous than edible mushrooms, so the model is tilted towards calling things poisonous.", True),
      ("Your dataset has more edible than poisonous mushrooms, so the model is tilted towards calling things edible.", False)]),
    ("Descriptive statistics", "B03", "input data", "counterexample", "data-quality concern", "data_stat", "cap_diameter_max",
     [("A maximum cap diameter of 62 cm is in the data; that is not a cap mushroom, that is an error in the dataset.", 62),
      ("A maximum cap diameter of 120 cm is in the data; that is not a cap mushroom, that is an error in the dataset.", 120)]),
    ("Descriptive statistics", "B04", "evidence", "direct challenge", "missing evidence", "data_stat", "missing_habitat_pct",
     [("Habitat has no missing values at all, so the gaps you show cannot explain why habitat matters so little.", 0),
      ("Habitat is missing for most of the rows, which is why the model cannot use it.", 80)]),
    ("Descriptive statistics", "B05", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "missing_shown_as_grey_bar",
     [("The grey bars are the missing data; you never say so and it reads as just another category.", True),
      ("The missing data is not shown on the bar charts at all, only in the table.", False)]),
    ("Descriptive statistics", "B06", "input data", "direct challenge", "data-quality concern", "data_stat", "partly_simulated",
     [("Part of this dataset is simulated, so the counts do not describe real mushrooms.", True),
      ("None of this dataset is simulated; these are all field observations, so the counts are real.", False)]),
    ("Descriptive statistics", "B07", "evidence", "request for justification", "technical knowledge", "data_stat", "stem_height_min",
     [("A minimum stem height of zero is in the data; a stem of zero centimetres is a data error, not a mushroom.", 0),
      ("A minimum stem height of five centimetres is in the data, which excludes all the small species.", 5)]),
    ("Descriptive statistics", "B08", "input data", "direct challenge", "data-quality concern", "data_stat", "n_species",
     [("Only 173 species are covered, so the model knows nothing about the rest.", 173),
      ("Only 20 species are covered, so the model knows nothing about the rest.", 20)]),
    # ---------------- SHAP
    ("SHAP", "B09", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "beeswarm_legend_present",
     [("There is a colour legend, but it only says high and low, it does not say high of what for the one-hot features.", True),
      ("There is no legend at all for the colours, so I cannot tell what red means.", False)]),
    ("SHAP", "B10", "evidence", "direct challenge", "missing evidence", "fact", "beeswarm_rows_near_zero",
     [("Seventeen of the twenty rows sit on zero; the chart is mostly showing features that do nothing.", 17),
      ("Only two of the twenty rows sit on zero; almost everything shown matters.", 2)]),
    ("SHAP", "B11", "reasoning", "direct challenge", "causal implausibility", "fact", "high_stem_width_pushes_poisonous",
     [("The chart says a thick stem pushes towards edible, which is the opposite of what the stem height does.", False),
      ("The chart says a thick stem pushes towards poisonous, same as a tall stem.", True)]),
    ("SHAP", "B12", "explanation representation", "direct challenge", "internal inconsistency", "fact", "waterfall_units_of_fx_stated",
     [("The waterfall gives f(x) = 9.959 but never says what unit that is in; it is not a probability.", False),
      ("The waterfall says f(x) is a probability, but 9.959 cannot be a probability.", True)]),
    ("SHAP", "B13", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "waterfall_feature_values_in_cm",
     [("The waterfall shows stem height as 1.535, which cannot be centimetres; the values are scaled and nobody says so.", False),
      ("The waterfall shows stem height as 1.535 centimetres, which is a tiny mushroom, so the example is unrepresentative.", True)]),
    ("SHAP", "B14", "evidence", "direct challenge", "missing evidence", "fact", "waterfall_other_features_collapsed",
     [("106 features are collapsed into one line, so I cannot see whether any of them mattered.", 106),
      ("Ten features are collapsed into one line, so I cannot see whether any of them mattered.", 10)]),
    ("SHAP", "B15", "evidence", "direct challenge", "domain knowledge", "fact", "cap_color_among_top3",
     [("Cap colour, which every forager checks first, is not even in the top three of your chart.", False),
      ("Cap colour is in the top three of your chart, and that cannot be right for this dataset.", True)]),
    ("SHAP", "B16", "reasoning", "counterexample", "domain knowledge", "fact", "high_stem_height_pushes_poisonous",
     [("The chart says tall stems push towards poisonous; many edible boletes have tall stems.", True),
      ("The chart says tall stems push towards edible; many poisonous species have tall stems.", False)]),
    # ---------------- LIME
    ("LIME", "B17", "explanation representation", "direct challenge", "internal inconsistency", "display_fault", "unit_typo_stem_height_mm",
     [("One row says stem_height_mm and the others say stem_height_cm; one of them is wrong.", True),
      ("Every row uses millimetres for stem height while the data is in centimetres.", False)]),
    ("LIME", "B18", "explanation representation", "direct challenge", "presentation ambiguity", "display_fault", "several_bins_of_same_feature_listed",
     [("Three different ranges of stem width are listed at once, so I cannot tell which one this mushroom actually has.", True),
      ("Each feature appears exactly once, so there is no way to see the alternative ranges.", False)]),
    ("LIME", "B19", "evidence", "direct challenge", "missing evidence", "fact", "instance_shown",
     [("You never show the mushroom this LIME plot is about; the bars are meaningless without the instance.", False),
      ("The mushroom is shown next to the plot, but its values do not match the bars.", True)]),
    ("LIME", "B20", "reasoning", "direct challenge", "causal implausibility", "fact", "no_bruises_pushes_poisonous",
     [("The plot says no bruising pushes towards poisonous, which is backwards from how foragers use bruising.", True),
      ("The plot says bruising pushes towards poisonous, which is backwards from how foragers use bruising.", False)]),
    ("LIME", "B21", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "legend_present",
     [("There is no legend for red and blue; you only learn the meaning from the axis caption.", False),
      ("The legend says red is edible, which contradicts the axis caption.", True)]),
    ("LIME", "B22", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "top_blue_value",
     [("The strongest bar towards edible is only −0.07; these are tiny effects and the chart makes them look decisive.", -0.07),
      ("The strongest bar towards edible is −0.7; that is a huge effect for one feature.", -0.7)]),
    ("LIME", "B23", "evidence", "direct challenge", "technical knowledge", "fact", "bars_are_probability_changes",
     [("These bars are changes in class probability, not feature importances, and the title does not say that.", True),
      ("These bars are feature importances from the model, not probabilities, so the axis label is wrong.", False)]),
    ("LIME", "B24", "explanation representation", "rejection", "presentation ambiguity", "display_fault", "edible_misspelled",
     [("The axis says 'Eadible'; if the chart cannot spell its own classes, I do not trust the rest.", True),
      ("The axis labels are spelled correctly but the colours are swapped relative to the caption.", False)]),
    # ---------------- Counterfactual
    ("Counterfactual", "B25", "explanation representation", "direct challenge", "internal inconsistency", "display_fault", "comparisons_identical",
     [("Both comparisons show exactly the same change, 1.43 to 23.5; the second one adds nothing.", True),
      ("The two comparisons change different features, but the caption claims they show the same thing.", False)]),
    ("Counterfactual", "B26", "reasoning", "counterexample", "domain knowledge", "fact", "change_to",
     [("A stem of 23.5 cm on a mushroom with a 9 cm cap is absurd; no such mushroom exists.", 23.5),
      ("A stem of 50 cm on a mushroom with a 9 cm cap is absurd; no such mushroom exists.", 50)]),
    ("Counterfactual", "B27", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "caption_matches_method",
     [("The caption talks about filling in missing data, but what is shown is a change to stem height; the caption describes something else.", False),
      ("The caption correctly describes this as filling in missing values, so the stem height row is a mistake.", True)]),
    ("Counterfactual", "B28", "evidence", "direct challenge", "missing evidence", "fact", "n_features_changed",
     [("Only one feature is changed; you give no alternative with a smaller or more realistic change.", 1),
      ("Four features are changed at once, so I cannot tell which one flips the prediction.", 4)]),
    ("Counterfactual", "B29", "input data", "direct challenge", "data-quality concern", "fact", "n_no_data_fields",
     [("Six fields of this mushroom are no_data; the model is deciding on a mostly empty description.", 6),
      ("Every field of this mushroom is filled in, so missing data cannot be the issue here.", 0)]),
    ("Counterfactual", "B30", "prediction", "direct challenge", "domain knowledge", "fact", "instance_cap_color",
     [("A yellow sticky cap with orange gills in autumn forests: the model calls it poisonous and you say that is correct, but yellow caps are no sign of toxicity.", "yellow"),
      ("A red sticky cap with orange gills: the model calls it poisonous because of the red cap, and red is no sign of toxicity.", "red")]),
    ("Counterfactual", "B31", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "legend_present",
     [("The legend explains original and changed value but not why the two comparisons are shown.", True),
      ("There is no legend, so I cannot tell which column is the original value.", False)]),
    ("Counterfactual", "B32", "reasoning", "request for justification", "technical knowledge", "fact", "cap_color_changed",
     [("Why does the counterfactual never touch cap colour? Only stem height changes.", False),
      ("Why does the counterfactual change cap colour when stem height would have been enough?", True)]),
    # ---------------- Anchor
    ("Anchor", "B33", "input data", "direct challenge", "data-quality concern", "fact", "anchor_contains_no_data_conditions",
     [("Two of the anchor conditions are 'no_data'; the model is treating missing data as evidence of edibility.", True),
      ("None of the anchor conditions involves missing data, so the rule is built on real observations.", False)]),
    ("Anchor", "B34", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "edible_reduced_precision",
     [("With only the blue conditions the certainty drops to 76%, and nothing on the slide says why that is still an anchor.", 76.2),
      ("With only the blue conditions the certainty drops to 40%, which makes the reduced anchor useless.", 40)]),
    ("Anchor", "B35", "explanation representation", "direct challenge", "presentation ambiguity", "display_fault", "typo_capCdiameter",
     [("The example says 'capCdiameter'; the feature list has not even been proof-read.", True),
      ("The example lists 'cap_diameter' twice with different values.", False)]),
    ("Anchor", "B36", "reasoning", "counterexample", "domain knowledge", "fact", "green_cap_in_poisonous_anchor",
     [("A green cap is in the poisonous anchor; plenty of green-capped species are edible.", True),
      ("A brown cap is in the poisonous anchor; brown caps are the most common colour among edible species.", False)]),
    ("Anchor", "B37", "evidence", "direct challenge", "missing evidence", "fact", "edible_anchor_n_conditions",
     [("Nine conditions for one rule is not an explanation, it is a description of the whole mushroom.", 9),
      ("Three conditions for the edible rule is too few to be safe.", 3)]),
    ("Anchor", "B38", "system competence", "rejection", "technical knowledge", "fact", "anchor_need_not_be_real_example",
     [("You admit the anchor need not match any real mushroom, so the 97% is about imaginary cases.", True),
      ("You say the anchor is always a real mushroom from the data, but that cannot be true with no_data conditions.", False)]),
    ("Anchor", "B39", "explanation representation", "direct challenge", "presentation ambiguity", "fact", "percent_explained_in_caption",
     [("The caption does explain the percentage, but calls it certainty, which is not what a precision is.", True),
      ("Nothing on the slide explains what the percentage means.", False)]),
    ("Anchor", "B40", "prediction", "direct challenge", "domain knowledge", "fact", "edible_example_habitat",
     [("A white, convex, smooth-capped mushroom from a meadow in spring: the model says edible, and that matches several deadly Amanita look-alikes.", "meadows"),
      ("A white, convex, smooth-capped mushroom from a forest in spring: the model says edible, and that matches several deadly Amanita look-alikes.", "forests")]),
]

# Model-behaviour claims the built verifier cannot check (no classifier released) and domain claims; no validity label.
UNCHECKABLE = [
    ("SHAP", "U01", "reasoning", "direct challenge", "causal implausibility", "model_effect", "Stem height should not matter at all for edibility; remove it and the model will predict the same."),
    ("SHAP", "U02", "reasoning", "alternative proposal", "domain knowledge", "uncheckable", "Smell and spore print are what decide edibility in practice; a model without them cannot be trusted."),
    ("LIME", "U03", "reasoning", "direct challenge", "causal implausibility", "model_effect", "Stem colour brown pushing towards poisonous is noise; perturb it and nothing will change."),
    ("LIME", "U04", "prediction", "rejection", "prior experience", "uncheckable", "I have picked mushrooms with exactly these features for twenty years and they were fine."),
    ("Counterfactual", "U05", "reasoning", "counterexample", "domain knowledge", "uncheckable", "Young specimens of this species have short stems; the age of the mushroom explains the 1.43 cm, not toxicity."),
    ("Counterfactual", "U06", "reasoning", "direct challenge", "causal implausibility", "model_effect", "Changing habitat to meadows would have flipped the prediction just as well as the stem height."),
    ("Anchor", "U07", "prediction", "rejection", "prior experience", "uncheckable", "In my region this exact combination is always poisonous, whatever the model says."),
    ("Anchor", "U08", "reasoning", "direct challenge", "causal implausibility", "model_effect", "Drop the no_data conditions from the anchor and the 97% will not change."),
    ("Descriptive statistics", "U09", "input data", "direct challenge", "domain knowledge", "uncheckable", "These counts come from a European dataset; mushrooms here look different, so the statistics do not apply."),
    ("SHAP", "U10", "system competence", "request to acknowledge uncertainty", "technical knowledge", "uncheckable", "A model with 99.97% accuracy on partly simulated data is overfitted; it should say it is unsure."),
]

REQUESTS = [
    ("SHAP", "R01", "legend", "What do the colours mean? I do not see what high and low refer to."),
    ("SHAP", "R02", "term", "What is a SHAP effect value?"),
    ("LIME", "R03", "plain", "Can you say in plain words what this chart tells me about this mushroom?"),
    ("LIME", "R04", "term", "What does 'class probability' mean on the axis?"),
    ("Counterfactual", "R05", "plain", "I do not understand what the arrow means. Can you explain it differently?"),
    ("Anchor", "R06", "term", "What is an anchor?"),
    ("Anchor", "R07", "legend", "Which conditions are the blue ones, and why are some grey?"),
    ("Descriptive statistics", "R08", "plain", "This is too much text; can you show me just the three most important numbers?"),
    ("Descriptive statistics", "R09", "term", "What does 'std' mean in the table?"),
    ("Counterfactual", "R10", "legend", "Which column is the original value and which is the changed one?"),
]

NONE = [
    ("SHAP", "N01", "So stem height is the most important one. Makes sense."),
    ("LIME", "N02", "Right, red is poisonous and blue is edible. I see it."),
    ("Counterfactual", "N03", "OK, so a longer stem would make it edible for the model. Interesting."),
    ("Anchor", "N04", "Nine conditions and 97 percent. Fine."),
    ("Descriptive statistics", "N05", "Most of them are brown caps, as I would expect."),
    ("SHAP", "N06", "I think I understand this one now, let us move on."),
    ("Anchor", "N07", "The second anchor is shorter. OK."),
    ("Descriptive statistics", "N08", "61 thousand mushrooms. That is a lot of data."),
]


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows = []
    groups = ["DE", "IT", "SSH"]

    def add(fmt, tid, pair, text, g: Grounds, is_dispute, is_request, kind):
        ad = FungiAdapter(fmt)
        v = ad.verify(g) if is_dispute else None
        rows.append({"item_id": f"B{len(rows) + 1:03d}", "explanation_format": fmt, "template": tid, "pair": pair,
                     "user_text": text, "is_dispute": is_dispute, "is_request": is_request, "kind": kind,
                     "target": g.target, "interaction_act": g.act, "grounds": g.grounds, "claim_kind": g.claim_kind,
                     "subject": g.subject, "user_value": "" if g.user_value is None else str(g.user_value),
                     "truth": v.verdict if v else "", "truth_source": (v.tier if v else ""), "checkable": v.checkable if v else "",
                     "oracle_evidence": " | ".join(v.evidence) if v else "", "built_verdict": v.verdict if v else "",
                     "role": rng.choice(groups), "gold_action": gold_action(is_dispute, is_request, v)})

    for fmt, tid, target, act, grounds, kind, key, variants in PAIRS:
        for text, val in variants:
            add(fmt, tid, tid, text, Grounds(target, act, grounds, kind, key, val), True, False, "pair")
    for fmt, tid, target, act, grounds, kind, text in UNCHECKABLE:
        add(fmt, tid, "", text, Grounds(target, act, grounds, kind, "", None), True, False, "uncheckable")
    for fmt, tid, subj, text in REQUESTS:
        add(fmt, tid, "", text, Grounds(claim_kind="request", subject=subj, user_value=text), False, True, "request")
    for fmt, tid, text in NONE:
        add(fmt, tid, "", text, Grounds(), False, False, "none")
    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    df.to_csv(ROOT / "data" / "partB_pairs.csv", index=False)
    print(f"Part B: {len(df)} items")
    print(pd.crosstab(df.kind, df.truth.replace("", "-")))
    pr = df[df.kind == "pair"]
    bad = pr.groupby("pair").truth.apply(lambda s: sorted(s.tolist()) != ["user_right", "user_wrong"])
    print("pairs not (right, wrong):", bad[bad].index.tolist())
    print("gold actions:", df.gold_action.value_counts().to_dict())


if __name__ == "__main__":
    main()
