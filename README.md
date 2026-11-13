# Yield Only When You Can Check

Code, data and evidence for *Yield Only When You Can Check: Calibrated Responses to User Contestation of AI
Explanations* (Manai, Bobek, Nalepa; Jagiellonian University). When a user objects to an AI explanation, the
system should revise if it can check that the user is right, hold and show the evidence if it can check that
the user is wrong, and surface both views and ask if it cannot check at all. This repository builds that
mechanism, applies it to a conversational hyperparameter-optimisation tool (CF-XHPO), and measures it.

- Research design: [`framework/architecture.md`](framework/architecture.md)
- Data and the rules fixed before scoring: [`data/README.md`](data/README.md), [`data/DATA_RULES.md`](data/DATA_RULES.md)
- Every table and figure of the paper: [`results.ipynb`](results.ipynb)

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
tar -xzf benchmark/outputs/cache.tar.gz -C benchmark/outputs   # cached API calls: re-runs are free and identical
echo 'OPENROUTER_API_KEY=<your key>' > .env                     # only needed for calls not in the cache
```

Sibling repositories expected next to this one:

| repository | used for |
|---|---|
| `../xai-contestation-labelling` | the coders' raw label files (`benchmark/labels_258.py`) |
| `../Conversational-Counterfactual-Guidance-for-Operator-Centred-Hyperparameter-Optimisation` | CF-XHPO surrogates, optimisation logs, frozen retraining confirmations (the oracle) |
| `../detecting-contestation-xai` | Phase 1; its venv (torch, transformers, matplotlib, jupyter) runs E1's XLM-R baseline and the evidence notebook |

## Reproduce

With the cache restored, every step below runs without new API calls.

| step | command | writes |
|---|---|---|
| test labels on the 258 (four coders) and reliability | `.venv/bin/python benchmark/labels_258.py` | `data/labels_258.csv`, `benchmark/outputs/labels_258_summary.md` |
| Part C: operator objections, truth from the retraining bank | `.venv/bin/python -m framework.benchmark.partC` | `data/partC_cfxhpo.csv`, `data/partC_episodes.json` |
| Part B: matched right/wrong objections on the slides | `.venv/bin/python -m framework.benchmark.partB` | `data/partB_pairs.csv` |
| E1 detection on the 258 | `../detecting-contestation-xai/.venv/bin/python -m framework.experiments.e1_detect` | `e1_detect_258.md` |
| E2, E3 discrimination and input ablation | `.venv/bin/python -m framework.experiments.e2_e3_decide` | `e2e3_items.csv`, `e2e3_assistants.csv`, `e2e3_summary.md` |
| E3 verifiers scored against the truth | `.venv/bin/python -m framework.experiments.e3_verifiers` | `e3_verifiers.md` |
| E4 cost of asking under detection noise | `.venv/bin/python -m framework.experiments.e4_cost` | `e4_cost.md`, `e4_cost_grid.csv` |
| E5 live loop (closed-choice Understand; `--v1` for free-form) | `.venv/bin/python -m framework.experiments.e5_live` | `e5_live_v2.md`, `e5_live_items_v2.csv` |
| E6 blind rating sheets | `.venv/bin/python -m framework.experiments.e6_sheets` | `e6_rating_sheet.csv` (key regenerated locally, not tracked) |
| all tables and figures of the paper | `jupyter nbconvert --execute --inplace results.ipynb --ExecutePreprocessor.kernel_name=xai-phase1` | `paper/figures/` (local, not tracked) |

Outputs go to `benchmark/outputs/`. The notebook kernel is the Phase 1 venv, registered once with
`../detecting-contestation-xai/.venv/bin/python -m ipykernel install --user --name xai-phase1`.
`benchmark/jev_detect.ipynb` (Jev on gold, 29 Oct 2026) and `benchmark/reliability_block.ipynb` (two- and
three-coder reliability, 3 Nov 2026) are kept as the record of those earlier steps.

## Layout

```
framework/
├── architecture.md           research design (read first)
├── schemas.py                Claim, Grounds, Verdict, Decision, Episode
├── api.py                    OpenRouter: Jev Decisions API + chat completions, cached
├── decide.py                 the verifiability principle as code: derived gold, costs, confidence gate, routing
├── respond.py                templated replies and fact-constrained language-model replies
├── adapters/cfxhpo.py        the application: state, claim menu, 3-tier verify, revise, re-present
├── adapters/fungi.py         the XAI-FUNGI slides: fact base, rendering and data tiers
├── tracks/decision_model.py  Jev as Detect and Decide
├── tracks/language_model.py  language-model decider, free-replying assistants, judge
├── tracks/understand.py      Understand as a closed choice over the claim menu, with the clarify gate
├── tracks/common.py          run_turn(): the five steps as one call
├── benchmark/                partB.py, partC.py, metrics.py
└── experiments/              e1_detect, e2_e3_decide, e3_verifiers, e4_cost, e5_live, e6_sheets
benchmark/                    labels_258.py, earlier notebooks, outputs/ (results, cache.tar.gz)
data/                         episodes, labels, benchmark parts, codebooks, rules
results.ipynb                 every table and figure of the paper
archive/                      superseded versions of the design document and the paper skeleton
```
