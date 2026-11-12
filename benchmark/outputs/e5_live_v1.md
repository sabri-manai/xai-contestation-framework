# E5: live loop on Parts C and B, labels and claims produced by the system itself (12 Nov 2026)

Understand = gpt-4o-mini mapping the turn to a claim; Verify = adapter (oracle = retraining bank available / not); Decide = track; Respond = templated (decision) or constrained LLM (language).

| track | oracle bank | part | n | claim kind understood | subject understood | Y_valid | Y_invalid | discrimination | surface_on_unresolvable | represent_on_request | none_on_none | agreement_with_gold | mean_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision | False | B | 108 | 0.32 | 0.56 | 0.4 | 0.125 | 0.275 | 1.0 | 1.0 | 1.0 | 0.639 | 0.358 |
| decision | False | C | 593 | 0.73 | 0.41 | 0.695 | 0.0 | 0.695 | 0.887 | 1.0 | 1.0 | 0.885 | 0.082 |
| decision | True | B | 108 | 0.32 | 0.56 | 0.4 | 0.125 | 0.275 | 1.0 | 1.0 | 1.0 | 0.639 | 0.358 |
| decision | True | C | 593 | 0.73 | 0.41 | 0.78 | 0.0 | 0.78 | 0.984 | 1.0 | 1.0 | 0.944 | 0.052 |
| language | False | B | 108 | 0.32 | 0.56 | 0.425 | 0.3 | 0.125 | 0.9 | 0.9 | 0.375 | 0.454 | 0.769 |
| language | False | C | 593 | 0.73 | 0.41 | 0.83 | 0.016 | 0.814 | 0.871 | 1.0 | 0.714 | 0.882 | 0.097 |
| language | True | B | 108 | 0.32 | 0.56 | 0.425 | 0.3 | 0.125 | 0.9 | 0.9 | 0.375 | 0.454 | 0.769 |
| language | True | C | 593 | 0.73 | 0.41 | 0.894 | 0.036 | 0.858 | 0.952 | 1.0 | 0.714 | 0.926 | 0.117 |

## Where the live loop loses against E2/E3 (Part C, decision track, oracle bank): by true claim kind

| true_kind | n | kind understood | agreement with gold |
|---|---|---|---|
|  | 56 | 0.0 | 1.0 |
| density | 56 | 0.95 | 0.66 |
| feature_effect | 56 | 1.0 | 0.95 |
| log_stat | 56 | 1.0 | 1.0 |
| rank_confirmed | 28 | 1.0 | 1.0 |
| rank_table | 56 | 0.0 | 0.98 |
| request | 84 | 0.58 | 1.0 |
| satisfies | 104 | 1.0 | 0.91 |
| score_over | 28 | 1.0 | 1.0 |
| score_reach | 13 | 1.0 | 1.0 |
| uncheckable | 56 | 0.79 | 0.98 |
