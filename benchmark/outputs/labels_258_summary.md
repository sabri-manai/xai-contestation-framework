# Labels on the 258: four coders, aggregated test labels (12 Nov 2026)

Rules: `data/DATA_RULES.md`, 2026-11-12 entries, fixed before this was run. Output of `benchmark/labels_258.py`.

## 1. Coders and coverage

| coder | app_code | episodes_labelled | block_labelled | block_on_current_text | headline_coder | first_label_utc | last_label_utc | median_s_per_episode |
|---|---|---|---|---|---|---|---|---|
| A | c_433b209090 | 182 | 100 | 94 | True | 2026-10-31 07:38:32 | 2026-11-02 08:55:52 | 40.20 |
| B | c_bf6244ebdc | 171 | 100 | 100 | True | 2026-10-29 18:33:27 | 2026-11-02 21:58:07 | 56.90 |
| C | c_01534ba437 | 104 | 100 | 100 | True | 2026-10-31 11:36:49 | 2026-11-03 10:00:15 | 39.90 |
| D | c_b59a2c33e1 | 258 | 100 | 100 | True | 2026-11-02 15:14:25 | 2026-11-05 16:52:33 | 47.70 |

Coder A labelled 6 block episodes between 07:38 and 10:40 UTC on 31 Oct, before the cleaned sentence-unit text was pushed; as on 3 Nov, A stays a headline coder and those 6 labels are reported, not dropped. Agreement figures below use every latest label on the block.

Episodes by number of coders and block:

| n_coders | excluded_unclear_transcript | reliability_100 | rest |
|---|---|---|---|
| 1 | 28 | 0 | 39 |
| 2 | 15 | 0 | 14 |
| 3 | 0 | 0 | 58 |
| 4 | 0 | 100 | 4 |

Standing of each episode (rule of 12 Nov):

| standing | excluded_unclear_transcript | reliability_100 | rest | All |
|---|---|---|---|---|
| adjudicate | 0 | 22 | 4 | 26 |
| excluded_unclear | 43 | 0 | 0 | 43 |
| single | 0 | 0 | 40 | 40 |
| test | 0 | 78 | 71 | 149 |
| All | 43 | 100 | 115 | 258 |

## 2. Reliability on the block (100 episodes, 4 coders, all on current text)

| gate | set | n | Fleiss κ (4) | Krippendorff α | unanimous | yes rate A/B/C/D |
|---|---|---|---|---|---|---|
| contestation | block | 100 | 0.48 | 0.48 | 0.60 | 0.29/0.41/0.28/0.22 |
| contestation | flagged | 54 | 0.35 | 0.36 | 0.43 | 0.41/0.67/0.41/0.33 |
| contestation | random_unflagged | 46 | 0.52 | 0.52 | 0.80 | 0.15/0.11/0.13/0.09 |
| request | block | 100 | 0.40 | 0.41 | 0.59 | 0.22/0.34/0.24/0.21 |
| request | flagged | 54 | 0.45 | 0.45 | 0.57 | 0.31/0.43/0.19/0.30 |
| request | random_unflagged | 46 | 0.31 | 0.31 | 0.61 | 0.11/0.24/0.30/0.11 |
| any push-back | block | 100 | 0.49 | 0.50 | 0.53 | 0.47/0.51/0.48/0.39 |

Pairwise Cohen κ on the block:

| gate | pair | n | agreement | kappa |
|---|---|---|---|---|
| contestation | A–B | 100 | 0.78 | 0.52 |
| contestation | A–C | 100 | 0.77 | 0.44 |
| contestation | A–D | 100 | 0.83 | 0.56 |
| contestation | B–C | 100 | 0.77 | 0.50 |
| contestation | B–D | 100 | 0.73 | 0.40 |
| contestation | C–D | 100 | 0.80 | 0.47 |
| request | A–B | 100 | 0.82 | 0.56 |
| request | A–C | 100 | 0.78 | 0.38 |
| request | A–D | 100 | 0.77 | 0.32 |
| request | B–C | 100 | 0.78 | 0.47 |
| request | B–D | 100 | 0.77 | 0.44 |
| request | C–D | 100 | 0.73 | 0.23 |

Reference: three-rater (A, B, C) Fleiss κ on contestation = 0.48 (3 Nov: 0.48).

## 3. Structure dimensions (coders who said contestation = yes), three levels

fine = codebook categories; coarse = pre-registered 3 Nov collapse; merged = the coders' own OR-merges of 12 Nov (`data/taxonomy_merges_v1.2.json`). All three reported by rule.

| dimension | set | n (≥2 yes-coders) | α fine | unanimous fine | α coarse | unanimous coarse | α merged | unanimous merged |
|---|---|---|---|---|---|---|---|---|
| target | block | 33 | 0.22 | 0.24 | 0.26 | 0.55 | 0.21 | 0.30 |
| target | all 258 | 59 | 0.24 | 0.25 | 0.3 | 0.54 | 0.22 | 0.31 |
| interaction_act | block | 33 | 0.04 | 0.09 |  |  | 0.14 | 0.30 |
| interaction_act | all 258 | 59 | 0.11 | 0.14 |  |  | 0.30 | 0.47 |
| grounds | block | 33 | 0.27 | 0.27 | 0.35 | 0.61 | 0.39 | 0.52 |
| grounds | all 258 | 59 | 0.26 | 0.34 | 0.41 | 0.69 | 0.34 | 0.56 |
| expected_response | block | 33 | 0.18 | 0.21 | 0.36 | 0.52 | 0.33 | 0.48 |
| expected_response | all 258 | 59 | 0.25 | 0.27 | 0.42 | 0.53 | 0.35 | 0.46 |

Cross-dimension relations from the coders' note, checked on each coder's own yes-labels (share of labels with the antecedent whose consequent holds):

| if | then | endorsed | A | B | C | D |
|---|---|---|---|---|---|---|
| interaction_act = request for justification | expected_response ∈ ['justify', 'provide additional information'] | 2 | 1.00 (n=4) | 1.00 (n=3) | 1.00 (n=8) | 0.60 (n=5) |
| interaction_act = correction | expected_response ∈ ['incorporate correction', 'revise'] | 2 | 1.00 (n=6) | 1.00 (n=1) | 1.00 (n=3) | 0.67 (n=6) |
| interaction_act = request to acknowledge uncertainty | expected_response ∈ ['acknowledge uncertainty'] | 1 | 0.00 (n=3) | 0.50 (n=2) | 1.00 (n=1) | 0.22 (n=9) |
| grounds = presentation ambiguity | target ∈ ['explanation representation'] | 2 | 0.00 (n=1) | 0.95 (n=22) | 0.00 (n=1) | 1.00 (n=1) |
| grounds = data-quality concern | target ∈ ['input data'] | 2 | 0.67 (n=3) | 1.00 (n=8) | 0.50 (n=2) | 0.70 (n=10) |

## 4. Aggregated labels (standing = test)

| block | contestation yes | request yes | both | neither | n |
|---|---|---|---|---|---|
| reliability_100 | 17 | 14 | 1 | 48 | 78 |
| rest | 25 | 18 | 4 | 32 | 71 |

Coarse dimensions on test-standing contestations (majority among yes-coders; `split` = tie):

- target (coarse): account 30, split 6, authority 5, conclusion 1
- grounds (coarse): not checkable 32, checkable 8, split 2
- expected_response (coarse): revise-type 14, split 10, hold-type 10, surface-type 8

Merged level (coders' OR-merges), majority among yes-coders:

- grounds (merged): domain knowledge or prior experience 31, split 6, data-quality concern 4, internal inconsistency 1
- interaction_act (merged): direct challenge or rejection 25, split 8, correction, counterexample or alternative proposal 5, request for justification 2, request to acknowledge uncertainty 2
- expected_response (merged): revise or incorporate correction 14, split 12, justify or provide additional information 10, defer to human expertise 6
- target (merged): split 15, evidence or input data 14, reasoning 7, system competence 5, prediction 1

By participant group × explanation format (test-standing contestation = yes / n):

| participant_group | Anchor | Counterfactual | Descriptive statistics | LIME | SHAP |
|---|---|---|---|---|---|
| DE | 3/8 | 4/7 | 8/12 | 7/9 | 6/16 |
| IT | 0/2 | 0/1 | 0/13 | 0/2 | 2/11 |
| SSH | 1/9 | 4/6 | 3/17 | 1/17 | 3/19 |

Single-coder episodes (D only, reported separately): 40; contestation yes among them: 15.

## 5. Jev (Detect) against the aggregated test labels, thresholds fixed on gold (reporting only)

Caveat: tied episodes (`adjudicate`) are absent from these rows by the aggregation rule, and they are the hardest cases, so these figures are optimistic until the adjudication is done; the block-majority figure of 3 Nov (three coders, ties broken by majority, AUC 0.88) is the conservative reference.

| set | gate | n | yes | AUC | threshold | precision | recall | κ |
|---|---|---|---|---|---|---|---|---|
| block (current text run) | contestation | 78 | 17 | 0.97 | 0.17 | 0.45 | 1.00 | 0.45 |
| block (current text run) | request | 78 | 14 | 0.97 | 0.50 | 0.69 | 0.79 | 0.67 |
| rest (silver, raw text; lower bound) | contestation | 71 | 25 | 0.91 | 0.17 | 0.51 | 0.96 | 0.38 |
| rest (silver, raw text; lower bound) | request | 71 | 18 | 0.96 | 0.50 | 0.81 | 0.72 | 0.69 |
| all test-standing | contestation | 149 | 42 | 0.94 | 0.17 | 0.48 | 0.98 | 0.43 |
| all test-standing | request | 149 | 32 | 0.96 | 0.50 | 0.75 | 0.75 | 0.68 |

## 6. Files

- `data/labels_258.csv`: one row per episode, per-coder gates and dimensions, majority labels, votes, status, standing, coarse and merged dimensions, fold, Jev probabilities.
- `data/adjudication_queue_258.csv`: 26 tied episodes for the adjudicator.
- `data/coders.csv`, `data/human_labels.csv`: coders by letter and app code only.
