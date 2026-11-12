# E5: live loop on Parts C and B, labels and claims produced by the system itself (12 Nov 2026) — Understand v2

Understand v2 = closed choice over the adapter's claim menu (Jev in the decision track, gpt-4o-mini in the language track), slots filled by gpt-4o-mini; choice probability < 0.5 → Surface-clarify (ask which claim is meant). Verify = adapter (oracle = retraining bank available / not); Decide = track; Respond = templated (decision) or constrained LLM (language).

| track | oracle bank | part | n | claim kind understood | subject understood | clarify asked | Y_valid | Y_invalid | discrimination | surface_on_unresolvable | represent_on_request | none_on_none | agreement_with_gold | mean_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision | False | B | 108 | 0.79 | 0.76 | 0.05 | 0.725 | 0.175 | 0.55 | 1.0 | 1.0 | 0.75 | 0.722 | 0.42 |
| decision | False | C | 593 | 0.88 | 0.88 | 0.0 | 0.908 | 0.0 | 0.908 | 0.903 | 1.0 | 0.875 | 0.929 | 0.082 |
| decision | True | B | 108 | 0.79 | 0.76 | 0.05 | 0.725 | 0.175 | 0.55 | 1.0 | 1.0 | 0.75 | 0.722 | 0.42 |
| decision | True | C | 593 | 0.88 | 0.88 | 0.0 | 0.993 | 0.0 | 0.993 | 1.0 | 1.0 | 0.875 | 0.983 | 0.054 |
| language | False | B | 108 | 0.76 | 0.71 | 0.0 | 0.65 | 0.25 | 0.4 | 0.9 | 1.0 | 0.25 | 0.639 | 0.813 |
| language | False | C | 593 | 0.84 | 0.97 | 0.0 | 0.915 | 0.004 | 0.911 | 0.242 | 1.0 | 0.964 | 0.872 | 0.224 |
| language | True | B | 108 | 0.76 | 0.71 | 0.0 | 0.65 | 0.25 | 0.4 | 0.9 | 1.0 | 0.25 | 0.639 | 0.813 |
| language | True | C | 593 | 0.84 | 0.97 | 0.0 | 1.0 | 0.004 | 0.996 | 0.339 | 1.0 | 0.964 | 0.926 | 0.196 |

## Where the live loop loses against E2/E3 (Part C, decision track, oracle bank): by true claim kind

| true_kind | n | kind understood | agreement with gold |
|---|---|---|---|
|  | 56 | 0.0 | 0.88 |
| density | 56 | 1.0 | 0.98 |
| feature_effect | 56 | 1.0 | 1.0 |
| log_stat | 56 | 1.0 | 1.0 |
| rank_confirmed | 28 | 1.0 | 1.0 |
| rank_table | 56 | 1.0 | 1.0 |
| request | 84 | 1.0 | 1.0 |
| satisfies | 104 | 0.88 | 0.98 |
| score_over | 28 | 1.0 | 1.0 |
| score_reach | 13 | 1.0 | 1.0 |
| uncheckable | 56 | 1.0 | 1.0 |
