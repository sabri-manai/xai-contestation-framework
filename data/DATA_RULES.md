# Data-use rules

Written once, before any Phase 2 model is scored. Changing a rule after a model has been scored
on the set the rule concerns is not allowed; add a dated note instead and report both.

## Sets

| set | file | role | may be used for |
|---|---|---|---|
| gold (76) | `gold_standard.csv` + `gold_tiers.csv` | development and reporting | choosing thresholds, prompts, question designs; reporting with CIs |
| silver (250) | `silver_250.csv` | Jev's zero-shot labels on the unlabelled sample | descriptive statistics, candidate selection for Part B, nothing that is later evaluated on these episodes |
| human (250) | `human_labels.csv` (grows as the coders work) | **test** | reporting only; no tuning; arrives via `export_labels.py` |
| Part B pairs | `partB_pairs.csv` (to build) | sycophancy benchmark | reporting; authored by us, so reported as such |
| Part C | CF-XHPO episodes (to build) | oracle validity | reporting |

## Rules

1. **Silver is never truth.** A silver label is replaced by the human label for that episode as
   soon as one exists; where two human labels disagree, the episode waits for adjudication and is
   excluded from reported numbers until then.
2. **No tuning on the 250.** Thresholds, prompts, question wording, state windows, and any other
   Detect choice are fixed on gold before the 250 are scored, and recorded in this file's
   "decisions" section with a date.
3. **Codebook v1.1 applies to everything from here on.** Phase 1 numbers stay under v1.0 and are
   quoted as such.
4. **Gold tiers.** Report every metric on tier A+B. Tier A alone may be shown next to it, never
   instead of it (tier B was identified partly through model disagreement).
5. **Episodes quoted in the coding guide** (`in_guide` in `gold_tiers.csv`) are excluded when an
   automatic labeller is scored, because the guide was part of its input.
6. **Splits.** `splits.csv` assigns every transcript (participant) to one of five folds,
   balanced within participant group, seed 20261029. Any cross-validation uses these folds; any
   held-out evaluation holds out whole transcripts. The folds are shared with the labelling
   project so human labels fall into the same folds when they arrive.
7. **Decision model only labels presence.** Jev's five structure answers (presence, target,
   act, grounds, expected response) are features, not labels: κ against gold is at chance
   (run 1, 29 Oct 2026). They stay in the `answers` column and never enter a label file.
8. **Repeats and CIs.** Any number on gold comes from ≥3 runs with mean ± sd, and n is stated.
   Jev's output varies between runs.
9. **Pinned model.** `typesafe/jev-1.14`; every output row carries the snapshot id the API
   returns (`typesafe/jev-1.14-20261014` at the time of writing). A change of snapshot is a new
   run, reported separately.
10. **Raw answers are kept.** Every Jev row stores the full answers JSON and cost.

## Detect operating point (rule, not a number)

Costs from the user's side: a missed dispute loses a correction the user tried to give; a missed
request loses an explanation the user needed; a false alarm costs one clarifying turn. So recall
is worth more than precision for both gates, as long as false alarms stay rare enough not to be
annoying. **Rule:** for each gate, the threshold is the lowest value on gold (tier A+B, excluding
`in_guide`, mean over repeats) at which precision ≥ 0.75. The notebook computes it and writes it
below; it is not chosen by eye.

## Decisions (dated)

- 2026-10-29: rules above adopted. Thresholds: to be filled by `benchmark/jev_detect.ipynb` on
  its first run under v1.1 question design.
- 2026-10-29: Detect thresholds fixed on gold (n=70, 3 repeats, window=unit, codebook v1.1): contestation 0.17 (lowest t with precision ≥ 0.75); request 0.5 (provisional: no v1.1 gold for explanation_request yet).
- 2026-10-31: Reliability block fixed before any human label on the 250 was looked at: positions 1–100 of
  `queue_order.csv` = all 46 random_unflagged + 54 flagged drawn at random with proportional
  allocation over group × format (seed 20261031, `make_queue_order.py` in the labelling project),
  shuffled together; labellers are not shown which is which. Every labeller codes these 100 first.
  Reported from this block, whatever the values: inter-coder κ on both gates and the four
  structure dimensions (independent coders, not construct developers), and the lexical filter's
  recall estimated on the 46 unflagged episodes. Codebook v1.1.
- 2026-10-31: The 250 episodes' text was cleaned before the reliability block was labelled. The
  source is uncorrected Polish automatic speech recognition; 1,856 transcript lines (every target
  and context line of the 250) were corrected for misrecognised words, casing and punctuation only
  (no content added, removed or rephrased; hedges kept; [?] = uncertain correction,
  [niezrozumiałe] = unrecoverable) and re-translated to English from the cleaned Polish, by Claude.
  Three corrections that changed meaning were reverted by hand. Originals kept in `text_pl_asr`,
  `context_pl_asr` (ASR) and `text_en_v0`, `context_en_v0` (earlier Claude translation of the raw
  ASR); per-line log in `cleaning_log.csv`. Gold, practice set and Jev runs 1–2 use the original
  text. Labels on the 250 are made on the cleaned text; Jev silver on the 250 was made on the raw text.
- 2026-10-31: Transcript-clarity exclusion, fixed before the reliability block was labelled. An
  episode is excluded from agreement and recall analyses if its judged text (target line plus the
  same speaker's continuation, unit rule) contains [?] or [niezrozumiałe] after cleaning: 39 of the
  250. In the block of 100 this removed 9 flagged and 6 unflagged episodes. Flagged ones were
  replaced at random from clear flagged episodes outside the block in the same group × format cell
  (seed 20261033); unflagged ones by a uniform redraw from the Phase 1 pool (participant lines on
  explanation slides, ≥5 words, no lexicon hit; seed 20261032; candidates taken in draw order, first
  6 clear ones used). Excluded episodes stay in the file (`transcript_unclear`) at the end of the
  queue. Consequence to report: 5 of the 6 excluded unflagged lines were IT participants (4 from
  DR_IT_05), so IT is less represented in the block (unflagged: SSH 26, IT 12, DE 8); recall and
  agreement are "on intelligible speech". 256 episodes in total (250 + 6 replacements).
- 2026-10-31 (amends the entry above, before any labelling): the 6 unflagged replacements are
  matched on participant group and explanation format instead of drawn uniformly, so the
  clarity exclusion does not change the block's composition. Per slot, 3 candidates from the same
  group × format cell of the Phase 1 unflagged pool (seed 20261034), cleaned with the same rules,
  first clear one used (3 DR_IT_05 candidates and 1 MZ_IT_06 candidate were unclear and skipped).
  The 6 uniform replacements (PK_DE_05:46, MK_SSH_02:338, PK_DE_05:178, DR_SSH_01:353, PK_DE_01:71,
  MZ_SSH_03:367) were removed unlabelled. Block composition is back to the original draw:
  SSH 48, DE 28, IT 24 (unflagged SSH 24, IT 17, DE 5). Still "on intelligible speech", and one IT
  interview (DR_IT_05) contributes fewer lines because its recording is largely unintelligible.
- 2026-10-31 (before any labelling): unit changed from transcript line to sentence for the
  episodes being labelled (codebook v1.2). The recogniser cut lines at pauses and put a full stop
  at most of them, so lines were regrouped into complete sentences by reading the Polish (Claude,
  Sonnet; same speaker only, ≤6 lines, no text changes except removing the pause full stop at a
  mid-sentence join and lowercasing the next word). Target = the sentence group containing the
  original line (`target_rows`); context = 3 groups each side. 53 targets now span more than one
  line. The clarity rule was re-applied on the sentence unit: 4 block episodes (all unflagged)
  became unclear and were replaced from the same group × format cell (seed 20261035, first clear
  of 4 candidates); two of them were same-day replacements and were removed unlabelled, the other
  two stay at the end of the queue as excluded. Totals: 258 episodes (250 + 8 replacements);
  block 100 = 54 flagged + 46 unflagged, SSH 48 / DE 28 / IT 24, 0 unclear; 115 rest; 43
  excluded as unclear. Gold, practice set, Phase 1 and Jev runs 1–2 remain line-based.
- 2026-11-03: Round-1 reliability received (two coders, block of 100 + 62 further overlapping
  episodes; snapshot `human_labels_round1.csv`, coders A/B, no emails). Registered before computing
  anything beyond the gates: the dimension disagreement is a vocabulary split (A codes "evidence"
  where B codes "reasoning"; B codes representation complaints as contestation + target
  representation where A codes request = yes). One composite was tested, stated in advance:
  form objection = request = yes OR (contestation = yes AND target = representation); prediction
  dispute = contestation = yes AND target = prediction; content dispute = any other target. Block
  results: contestation κ 0.52 (CI 0.32–0.70), request κ 0.56, form objection κ 0.52, 4-way
  form/content/prediction/none κ 0.58 (CI 0.42–0.74); fine dimensions on both-yes episodes
  κ ≤ 0.38. Beyond-block (flagged only) contestation κ 0.36; all flagged κ 0.37 (n = 116).
  Round 2, registered now: both coders re-code the dimensions on `recode_round2.csv` (either
  coder said contestation, or the gates disagree) after a calibration on the three confusions
  (`calibration_round2.md`), without discussing those episodes; round 1 and round 2 are both
  reported; the gate labels (contestation / request) are not re-coded. Headline claim restated
  at the reliable level: disputes of the prediction are rare (1/100 per coder); objections to the
  explanation's form (23–35/100) and content (26–27/100) are common.
- 2026-11-03: Round 2 re-coding is not feasible before the rebuttal. Instead, one pass of
  definition-based coarsening of the dimensions was computed on round-1 labels (both-yes
  episodes: block n = 24, all overlap n = 41), all five reported regardless of result
  (`benchmark/reliability_block.ipynb` §3d): target → conclusion / account / form / authority
  (κ 0.11 / 0.42, agreement 0.78–0.79, AC1 0.75–0.77); presence → explicit / implicit
  (κ 0.28 / 0.25); act → ask / assert wrong / propose / uncertainty (κ 0.26 / 0.38);
  grounds → checkable / not checkable (κ 0.68 / 0.69, AC1 0.79–0.86); expected response →
  hold- / revise- / surface-type (κ 0.38 / 0.44). Reported as post-hoc coarsening; the fine
  categories remain exploratory. No other merges were tried.
- 2026-11-03: Rule for additional coders, fixed before their agreement was examined: a coder is a
  headline coder only if they coded the full block of 100 independently on the current text;
  otherwise they are reported as a supplementary partial rater on the episodes they covered. Two
  label files arrived under codes c_f8596560d9 (positions 1–31) and c_01534ba437 (positions 1–3,
  35–100), apparently one person with two emails (to be confirmed); together 97/100. Reported
  either way (pairwise κ per code, three-rater Fleiss κ on jointly covered episodes,
  `benchmark/reliability_block.ipynb` §7); inclusion never depends on the value.
- 2026-11-03: Label files consolidated: one CSV per coder in the app format, codes only
  (`xai-contestation-labelling/labels/c_433b209090.csv` = A, `c_bf6244ebdc.csv` = B,
  `c_01534ba437.csv` = C, the third coder's two accounts merged, versions renumbered by time).
  Raw databases and original files kept locally in `labels_raw/` (gitignored, contain emails).
  `human_labels_round1.csv` regenerated with A/B/C. Coder C's 3 unsaved labels (positions 32–34)
  are treated as missing: labels are not reconstructed from recollection or copied from another
  coder; C may label them in the app, and only saved labels count. Three-rater Fleiss κ on the 97
  jointly covered episodes: contestation 0.48, request 0.47.
- 2026-11-03: Coder C labelled positions 32–34 in the app (10:00 UTC); block complete for all
  three coders (100/100). Headline reliability becomes the three-rater figure.
- 2026-11-03: End-to-end detection check on the block (100 episodes, cleaned sentence text):
  Jev zero-shot, v1.1 questions, threshold 0.17 as fixed on gold on 29 Oct (not re-chosen),
  `benchmark/outputs/jev_block_v12.csv`, scored against the majority of the three coders
  (`block_majority_vs_jev.csv`; 32 majority-yes, 66/100 unanimous). AUC 0.88; at 0.17 recall 0.91
  (population-weighted 0.97; 5/5 majority-contestation episodes among the unflagged), precision
  0.54 (population-weighted 0.32), κ 0.46 vs majority, 0.30–0.51 vs individual coders (human
  pairs 0.44–0.52). Lexical filter recall vs the same majority: 0.25 population-weighted. Gate
  choice between coders (dispute vs request, given both mark something): 72–74%.
- 2026-11-03: "Any push-back" (contestation = yes OR request = yes), three coders, block: Fleiss
  κ 0.60 (pairwise 0.50–0.68); on the 54 flagged episodes Fleiss κ 0.57, AC1(A–B) 0.77,
  agreement 0.72–0.87, versus Fleiss κ 0.33 for the dispute label alone; 26 of the 54 pairwise
  contestation disagreements on flagged episodes (48%) are cases where both coders saw push-back
  and differed only on dispute vs request. Unflagged 46: any push-back Fleiss κ 0.53. Added to
  `benchmark/reliability_block.ipynb` §2b.
- 2026-11-12: Fourth coder (app code `c_b59a2c33e1`, letter D) received; labelled all 258 episodes,
  the block of 100 fully on the current text (earlier saves from 27–31 Oct were superseded by
  re-labels), so D is a headline coder under the 3 Nov rule. Headline reliability becomes the
  four-rater figure; three-rater figures stay in `reliability_block.ipynb` as history. Coders are
  identified by letter (A–D) and app code in `data/coders.csv`; no email enters this project.
- 2026-11-12: **Aggregation rule for the 258 (test labels), fixed before any majority was computed**
  (`benchmark/labels_258.py` → `data/labels_258.csv`). Per episode and gate, over the latest
  non-skipped label of every coder: unanimous or majority (≥ 2 of 3, ≥ 3 of 4) → the label stands
  as a **test** label; two coders who agree → test label; a tie (1–1, 2–2) → `adjudicate` and the
  episode is excluded from reported numbers until adjudicated; one coder only → `single`, reported
  separately and never in a headline number. `transcript_unclear` episodes keep their labels but are
  excluded from every number (31 Oct rule). The structure dimensions are aggregated only at the
  coarse level pre-registered on 3 Nov (target → conclusion / account / form / authority; grounds →
  checkable / not checkable; expected response → hold- / revise- / surface-type), by majority among
  the coders who said contestation = yes; a tie is `split`. Fine categories are kept per coder and
  stay exploratory. Human majority replaces silver episode by episode (rule 1). Nothing in this
  rule depends on the values; the rule is applied once and the numbers reported whatever they are.
- 2026-11-12: Jev on the 258 is scored against the aggregated test labels for reporting only
  (block: `jev_block_v12.csv`, current text; rest: `silver_250.csv`, raw text, so the rest figure
  is a lower bound and marked as such). Thresholds stay 0.17 / 0.5 (provisional) as fixed on gold;
  the request threshold cannot be set by rule until gold carries v1.1 request labels, and the 258
  are never used to set it (rule 2).
- 2026-11-12: **Coder-proposed merges (level "merged")**, fixed before computing anything with it.
  Coders B and D wrote a note on which taxonomy categories are similar or related
  (`data/coder_notes_taxonomy_relations.md`). Their similarities are applied as OR-merges within a
  dimension, as a second coarse level beside the pre-registered 3 Nov level ("coarse"); the fine
  labels are kept untouched. Mapping in `data/taxonomy_merges_v1.2.json`:
  grounds: domain knowledge ∨ prior experience (both coders);
  interaction act: direct challenge ∨ rejection (both); correction ∨ counterexample ∨ alternative
  proposal (D only, chained through "correction");
  expected response: justify ∨ provide additional information (both); incorporate correction ∨
  revise (both);
  target: evidence ∨ input data (D only).
  Merges endorsed by one coder are applied too and marked `endorsed: 1`; reported either way. The
  cross-dimension relations in the note (act → expected response; target ↔ grounds) are not merges:
  they are checked as per-coder consistency rates and reported as such. Reliability is reported at
  all three levels (fine, coarse, merged) whatever the values; the merged level is post-hoc with
  respect to the labels but was proposed by the coders themselves, not chosen from the numbers.
