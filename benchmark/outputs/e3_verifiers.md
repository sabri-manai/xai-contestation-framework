# E3, verifiers scored against the benchmark truth (12 Nov 2026)

Truth = the oracle verdict (Part C: retraining bank for prediction claims, exact table / log / surrogate checks otherwise; Part B: the slide fact base). 'adapter' = Track 2's built verifier (no bank); 'LLM' = openai/gpt-4o-mini given the same state and claim, no bank, no fact base.

| part | verifier | n | exact agreement | says right when right | says wrong when wrong | says right when wrong (sycophantic verdict) | undecided/uncheckable |
|---|---|---|---|---|---|---|---|
| B | adapter (built, no bank) | 80 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| B | LLM verifier | 80 | 0.72 | 0.75 | 0.7 | 0.25 | 0.02 |
| C | adapter (built, no bank) | 391 | 0.93 | 0.91 | 0.94 | 0.0 | 0.06 |
| C | LLM verifier | 391 | 0.51 | 0.39 | 0.57 | 0.19 | 0.19 |
| C / log | adapter (built, no bank) | 112 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| C / log | LLM verifier | 112 | 0.38 | 0.0 | 0.61 | 0.0 | 0.37 |
| C / oracle | adapter (built, no bank) | 63 | 0.59 | 0.0 | 0.73 | 0.0 | 0.35 |
| C / oracle | LLM verifier | 63 | 0.3 | 0.0 | 0.37 | 0.0 | 0.56 |
| C / surrogate | adapter (built, no bank) | 56 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| C / surrogate | LLM verifier | 56 | 0.68 | 0.0 | 0.97 | 0.03 | 0.0 |
| C / table | adapter (built, no bank) | 160 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| C / table | LLM verifier | 160 | 0.61 | 0.79 | 0.48 | 0.52 | 0.0 |
| B+C uncheckable items | LLM verifier | 66 | 0.03 |  |  | 0.0 | 0.03 |

## Deciders with the LLM verifier's output as the verification input (condition built_llm)

| decider | part | n | Y_valid | Y_invalid | discrimination | agreement_with_gold | mean_cost |
|---|---|---|---|---|---|---|---|
| jev | B | 108 | 0.7 | 0.1 | 0.6 | 0.75 | 0.366 |
| jev | C | 593 | 0.227 | 0.064 | 0.163 | 0.624 | 0.381 |
| gpt-4o-mini | B | 108 | 0.725 | 0.225 | 0.5 | 0.676 | 0.619 |
| gpt-4o-mini | C | 593 | 0.383 | 0.12 | 0.263 | 0.629 | 0.503 |
