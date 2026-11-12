# Data

Every file here has one source. Nothing is edited in place: changes come through the labelling
project (`../../xai-contestation-labelling/`) and are re-exported. Rules of use are in
`DATA_RULES.md`; the gold's history is in `gold_changelog.md`.

| file | what | source | codebook |
|---|---|---|---|
| `gold_standard.csv` | 76 adjudicated episodes (44 contestation), 5 label columns + `gold_role` | Phase 1 reconciled gold, copied 29 Oct 2026 | v1.0 |
| `gold_tiers.csv` | tier A/B per gold episode, `in_guide` flag, reasons | derived (see `gold_changelog.md`) | — |
| `readjudication_queue.csv` | tier-B episodes with empty v1.1 columns for the adjudicator | derived | v1.1 target |
| `gold_disagreement_review_run1.md` | one reader's case-by-case reading of the 25 gold/Jev disagreements from run 1; input to re-adjudication, not labels | written 29 Oct 2026 | v1.0 |
| `episodes_to_label.csv` | the 258 episodes being labelled (250 + 8 replacements), sentence unit (codebook v1.2; `target_rows`), text cleaned of ASR errors 31 Oct 2026 (originals in `*_asr` / `*_v0` columns; `transcript_unclear`, `replacement_for`) | labelling project `data/` | — |
| `cleaning_log.csv` | per transcript line: original ASR, cleaned Polish, English, confidence | Claude, checked by hand | — |
| `silver_250.csv` | Jev zero-shot gate labels + probabilities on the 250 (written by `benchmark/jev_detect.ipynb`) | Jev 1.14 via OpenRouter | v1.1 |
| `human_labels.csv` | every save by every coder on the 258, one row per (coder, episode, version); `is_latest` marks the current one, `on_current_text` whether it was made on the cleaned sentence-unit text; coder = letter A–D (pseudonymised export written by `benchmark/labels_258.py` from the labelling project's `labels/*.csv`; no emails here) | `labels/*.csv` in the labelling project | `codebook` column (v1.0 rows lack `explanation_request`) |
| `coders.csv` | the four coders: letter, app code, coverage, headline status; no personal data | `benchmark/labels_258.py` | — |
| `labels_258.csv` | **the test labels**: one row per episode with every coder's gates and dimensions, the majority label per gate with votes and status (unanimous / majority / split / single), the episode's `standing` (test / adjudicate / single / excluded_unclear), coarse (3 Nov) and merged (coders' OR-merges, 12 Nov) dimensions by majority among yes-coders, fold, and Jev's probabilities (rules of 12 Nov 2026, `DATA_RULES.md`) | `benchmark/labels_258.py` | v1.2 |
| `adjudication_queue_258.csv` | the tied episodes (26 on 12 Nov 2026) with text and every coder's labels and notes, for the adjudicator | `benchmark/labels_258.py` | v1.2 |
| `splits.csv` | transcript → fold (5 folds, balanced within participant group, seed 20261029) | derived | — |
| `taxonomy_v1.1.json`, `coding_guide_v1.1.md` | codebook used by Jev runs 1–2 and the gold | labelling project | v1.1 |
| `taxonomy_v1.2.json`, `coding_guide_v1.2.md` | codebook the labellers use (sentence unit) | labelling project | v1.2 |
| `taxonomy_merges_v1.2.json` | the coders' OR-merges of similar categories and their cross-dimension relations (level "merged", with endorsement count) | coders B and D, 12 Nov 2026; rule in `DATA_RULES.md` | v1.2 |
| `coder_notes_taxonomy_relations.md` | transcription of the coders' note the merges come from; names replaced by letters | coders B and D | — |
| `phase1/` | Phase 1 reliability summary, the 14 adjudicated disagreements, detector benchmark | `detecting-contestation-xai/outputs/` | v1.0 |

To refresh after the coders push: `git -C ../../xai-contestation-labelling pull && python benchmark/labels_258.py`
(any Python with pandas and scikit-learn). It rewrites `human_labels.csv`, `coders.csv`, `labels_258.csv`,
`adjudication_queue_258.csv` and `benchmark/outputs/labels_258_summary.md`; nothing else.

Related, kept in `benchmark/outputs/`: `jev_run1_codebook_v1.0_labels.csv` (run 1, state carried the
whole guide, one repeat), `jev_detect_cache.csv` (all v1.1 calls, raw answers, gitignored),
`detect_thresholds.json` (the thresholds section 2 of the notebook fixed on gold), `jev_block_v12.csv` (Jev on the block, cleaned text, 3 Nov), `labels_258_summary.md` (coverage, four-coder reliability, prevalence, Jev vs test labels).

Episode id format: `<transcript_id>:<utterance index>`; `transcript_id` = `<coder initials>_<group>_<nn>`,
groups SSH / DE / IT. Polish text is authoritative, English is a translation (`en_source`).
