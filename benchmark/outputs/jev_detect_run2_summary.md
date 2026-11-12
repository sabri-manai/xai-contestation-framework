# Jev as Detect, run 2 (codebook v1.1 question design), 29 Oct 2026

Snapshot `typesafe/jev-1.14-20261014`, unit-rule window, 3 repeats on gold, thresholds by the
DATA_RULES rule. Evaluated on gold tier A+B minus `in_guide` (n = 70; 40 yes, 30 no). Cost of
the whole run: $0.07.

## Gate 1, contestation (dispute)

| | run 2 (v1.1) | run 1 (v1.0, state carried the guide) | Phase 1 best (TF-IDF+XLM-R, CV) |
|---|---|---|---|
| AUC of P(yes) | **0.782** (per repeat 0.780 / 0.787 / 0.780) | 0.75 | — |
| threshold | 0.17 (rule: lowest with precision ≥ 0.75) | 0.5 (default) | tuned per fold |
| precision / recall (yes) | 0.77 / 0.82 | 0.85 / 0.52 | ≈0.66 / — |
| macro-F1 | **0.75** | 0.67 | 0.59 |
| κ vs gold | 0.50 on mean P; per repeat 0.455 ± 0.015 | 0.37 | — |
| run-to-run sd of P | median 0.010, max 0.053 | n/a (1 run) | — |

Tier A alone (n = 37): precision 0.70, recall 1.00, κ 0.68 — shown only beside A+B (DATA_RULES §4).

Sweep (A+B): t=0.20 → P 0.78 R 0.80 κ 0.50; t=0.30 → 0.77 / 0.68 / 0.40; t=0.50 → 0.88 / 0.55 / 0.42;
t=0.60 → 0.95 / 0.45 / 0.38. P is compressed (gold-no median 0.13, gold-yes median 0.55), so the
rule lands low; precision is flat from 0.17 to 0.30 while recall drops, which is why the rule
picks 0.17.

Window ablation (1 repeat): full ±4 context AUC 0.815 vs unit 0.782; identical at the threshold
(P 0.77, R 0.82, κ 0.50). H1's prediction that full context would lose precision on request-like
episodes did **not** hold; full context is at least as good. Decision: keep the unit window as the
default because it is what the codebook defines as the unit and what a live system sees, but
report both; revisit if the v1.1 re-adjudication changes the picture.

## Gate 2, explanation request (descriptive; no v1.1 gold yet)

On gold at 0.5: request = yes for 7 of 30 gold-no and 12 of 40 gold-yes episodes. P(request) by
gold target: explanation representation 0.55, evidence 0.41, input data 0.34, prediction 0.11,
reasoning 0.12, system competence 0.15 — the ordering the gate is meant to have.

The seven gate-1 misses at 0.17 and their P(request):

| episode | tier | P(dispute) | P(request) | run-1 review verdict |
|---|---|---|---|---|
| PK_DE_01:210 | B | 0.03 | **0.96** | both defensible (terminology) |
| MW_IT_07:68 | B | 0.05 | **0.98** | both defensible (incomprehension + info request) |
| PK_DE_09:53 | B | 0.06 | **0.97** | both defensible (uncertainty / legend) |
| PK_DE_13:147 | B | 0.05 | **0.83** | both defensible (axis labels missing) |
| DR_IT_05:271 | B | 0.09 | **0.83** | gold doubtful (garbled) |
| MW_SSH_05:79 | B | 0.08 | 0.30 | gold doubtful (affect, cut off) |
| PK_DE_01:135 | B | 0.04 | 0.04 | gold doubtful (chart reading) |

Five of seven are flagged as requests. These are the episodes the re-adjudication under v1.1 is
most likely to move from contestation to request; if it does, gate 1's recall rises without
anything being tuned. The two it does not flag are the two the run-1 review already called
doubtful gold.

## False alarms at 0.17 (10)

Six are tier A: DR_SSH_07:459 (0.19), DR_IT_05:182 (0.21), MZ_SSH_04:169 (0.32), MK_IT_06:78 (0.38),
DR_IT_05:145 (0.49), DR_SSH_06:325 (0.52). They are **not** re-tiered (tiers were fixed before the
threshold; re-tiering now would couple the gold to the model a second time). They are listed here
for the adjudicator to read with the queue; DR_SSH_06:325 ("stipe base no data (interesting)") and
MK_IT_06:78 look like plain chart reading, the others sit at the bottom of the band. Four are
tier B, including MW_SSH_05:122 (0.78), which the run-1 review and Phase 1's round-1 adjudication
both call a genuine contestation.

## Silver labels on the 250 (`data/silver_250.csv`)

At 0.17 / 0.5: contestation yes 148, no 102; request yes 59 (28 of them without contestation).
By group × format, contestation counts: DE 65, SSH 62, IT 21. Silver, not truth (DATA_RULES §1).

## What this does and does not show

- Zero-shot Jev beats the trained Phase 1 detectors on the same gold and is stable across runs;
  n = 70, so κ carries roughly ±0.2 of uncertainty and the paper says so.
- Moving the rules into the question and shrinking the state helped (0.67 → 0.75 macro-F1), but
  the threshold also moved (0.5 → 0.17 by rule), so the two changes are confounded; the sweep
  table is what separates them (run 1 at 0.2–0.3 was 0.69–0.70).
- Gate 2 cannot be scored until the coders' v1.1 labels arrive; the `export_labels.py` agreement
  check will do that automatically.
