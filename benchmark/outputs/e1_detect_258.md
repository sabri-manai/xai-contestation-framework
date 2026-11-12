# E1 on the 258 test-standing labels (n = 149; 12 Nov 2026)

Trained detectors: Phase 1 recipes trained on the 76 gold episodes (v1.0 labels, line unit, raw text), applied to the cleaned sentence-unit text of the 258; threshold 0.5 (no tuning on the 258). LLM + codebook: v1.2 rules as prompt, zero-shot. Jev: as in labels_258.csv (block on the current text, rest on raw text). Ties excluded by the aggregation rule, so every number is optimistic alike; the comparison between systems is the point.

| system | gate | AUC | threshold | precision | recall | macro-F1 | κ |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR (trained on gold 76) | dispute | 0.61 | 0.5 | 0.33 | 0.9 | 0.45 | 0.12 |
| XLM-R frozen + LR (trained on gold 76) | dispute | 0.48 | 0.5 | 0.29 | 0.79 | 0.41 | 0.03 |
| TF-IDF + XLM-R ensemble (Phase 1 best) | dispute | 0.51 | 0.5 | 0.3 | 0.81 | 0.4 | 0.03 |
| LLM + codebook (gpt-4o-mini), zero-shot | dispute | 0.79 | 0.5 | 0.48 | 1.0 | 0.69 | 0.43 |
| LLM + codebook (gpt-4o-mini), zero-shot | request | 0.71 | 0.5 | 0.39 | 0.75 | 0.64 | 0.32 |
| Jev 1.14 zero-shot (threshold by rule on gold) | dispute | 0.94 | 0.17 | 0.48 | 0.98 | 0.69 | 0.43 |
| Jev 1.14 zero-shot (provisional 0.5) | request | 0.96 | 0.5 | 0.75 | 0.75 | 0.84 | 0.68 |
| LLM + codebook (gpt-4o-mini), at 0.17 for comparison | dispute | 0.79 | 0.17 | 0.48 | 1.0 | 0.69 | 0.43 |
