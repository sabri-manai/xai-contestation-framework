# Gold standard changelog

`gold_standard.csv` (76 episodes, 44 contestation) is the Phase 1 reconciled gold, coded under
codebook v1.0. No label in it has been changed in Phase 2. This file records what is known about
how it was produced and what is queued for re-adjudication under codebook v1.1.

## Provenance (Phase 1, `detecting-contestation-xai/outputs/05_reliability_and_reconciliation/`)

- 36 episodes double-coded; first-pass raw agreement 0.611, Cohen's κ 0.222.
- 14 presence disagreements adjudicated in round 1: 12 → yes, 2 → no. Sources: 10 "implicit /
  hidden contestation", 3 "coder revised on review", 1 "unit rule".
- A second-round recheck (matched-context reliability recheck and representation-boundary
  recheck) reduced the positives from 46 to 44. Its per-episode decisions were not recorded.

## Labels that differ between the round-1 adjudication record and the gold file

| episode | C1 | C2 | adjudicated (round 1) | gold file | reason for change |
|---|---|---|---|---|---|
| PK_DE_04:162 | yes | no | no (unit rule: C1 coded from context) | **yes** | not recorded |
| MZ_SSH_02:273 | no | yes | yes (coder revised) | **no** | not recorded |
| MW_SSH_05:122 | no | yes | yes (coder revised) | **no** | not recorded |
| MW_IT_02:55 | yes | no | yes (hidden contestation) | **no** | not recorded |

To do (adjudicator): fill the last column from the round-2 notes, or re-adjudicate these four
under v1.1 with the reason written down.

## Tiers (`gold_tiers.csv`)

- **Tier A** (clean core): none of the flags below.
- **Tier B** (queued for re-adjudication under v1.1): presence = ambiguous, or the Phase 1 coders
  split on it, or its label changed between adjudication rounds, or the first Jev run
  (29 Oct 2026) disagreed with it, or its structure labels visibly do not fit the text.

Tier B was identified partly through Jev's disagreements, so any Jev score on tier A alone is
optimistic. Report every metric on A+B; report A separately only alongside A+B.

`in_guide` marks episodes quoted in the coding guide (v1.0 §4 worked examples and the v1.1 §1b
table). They are excluded from automatic-labeller evaluation.

## Re-adjudication queue (`readjudication_queue.csv`)

All tier-B episodes with their current gold labels and empty v1.1 columns
(`v1_1_is_contestation`, `v1_1_explanation_request`, `v1_1_notes`) for the adjudicator. The
reading in `../../xai-contestation-labelling/jev/gold_disagreement_review.md` is input, not a
label. Under v1.1 every gold episode also needs `explanation_request`; the queue covers the
contested ones first, the rest follow in the labelling app.
