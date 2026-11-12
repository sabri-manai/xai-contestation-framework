# Calibrated Yielding: Responding to User Contestation of AI Explanations

*Archived 12 Nov 2026: version in force from 29 Oct to 12 Nov 2026 (two/three coders, CF-XHPO as transfer setting). Superseded by `framework/architecture.md`.*

Phase 1 (*Beyond Understanding and Trust*, ACM IUI 2027 submission) defined contestation, built a
five-dimensional taxonomy (what is contested, on what grounds, and what the user wants), and
released a labelled dataset. Key numbers:

- 44 of 76 checked episodes contained real contestation, across 3 user groups and 5 explanation types.
- Users object to *how the explanation is shown* 4–5× more often than to the prediction itself.
- Presence is detectable at feasibility scale (TF-IDF and XLM-R; macro-F1 0.59), but about 1 in 3
  flagged cases is a false alarm (positive precision ≈ 0.66).

It ended with five design requirements and the untested Hold–Revise–Surface (HRS) idea. What it
left open: detecting contestation does not tell the system what to do next.

Phase 2 is not "build the pipeline." It poses and answers a research question that Phase 1 only
exposed: **when should a system yield to a user who contests its explanation?**

**Venue.** Journal of Computational Science, special issue after ICCS 2026: an invited extension of
the ICCS 2026 paper with at least 40% new work and new results. Submission 15 Nov 2026, decision
15 Jan 2027, publication May 2027.

---

## 1. The research problem: calibrated yielding under dual uncertainty

A system that has (possibly) detected contestation must choose a response. The choice is made under
two independent uncertainties:

| Uncertainty | Source | Phase 1 evidence |
|---|---|---|
| **Presence** — did the user actually contest? | detector noise | ~1/3 of positive detections are false alarms |
| **Validity** — is the objection right? | epistemic conflict between user and model | grounds range from checkable (display fault, internal inconsistency) to uncheckable (domain claim) |

The two dominant families fail in opposite directions: **classic XAI ignores the user**, and **chat
assistants almost always give in, even when the user is wrong**. More broadly, every existing family
of systems fixes the response to a constant, independent of both uncertainties:

| Family | Policy | Failure mode |
|---|---|---|
| Static / one-shot XAI | never yields | ignores correct objections |
| Argumentative / interactive XAI | yields iff the objection is formally derivable | assumes all grounds are formalizable |
| LLM conversational XAI assistants | yields almost always | **sycophancy**: capitulates to wrong objections and to false alarms |
| Human-in-the-loop / learning-to-defer | hands the decision to the human | gives up the model's account entirely |

The problem is a *decision problem*, not a classification problem: choose a response that
(i) yields to valid objections, (ii) holds against invalid ones, (iii) exposes unresolvable ones
without pretending to resolve them, and (iv) does all three while a third of its triggers are
spurious. No existing system targets (i)–(iv) jointly, and no benchmark measures them.

**Research questions**

- **RQ1** Can a response policy discriminate valid from invalid contestation — yield to the
  first and hold against the second — where current LLM XAI assistants do not?
- **RQ2** What does a decision model need to see to make that discrimination? Is the result of
  *checking the user's grounds* the decisive input, and is the user's role irrelevant once the
  grounds are known?
- **RQ3** Under realistic detection error, is asking the user (Surface) when the decision model
  is unsure better than always acting — and at what operating points?
- **RQ4** Does the result transfer to a domain where validity is objectively decidable by oracle
  (surrogate-based HPO, where retraining confirms or refutes the objection)?

---

## 2. Thesis: the verifiability principle

**Claim.** The correct response to a contestation is determined by whether the system can *check*
the user's grounds, not by who the user is or how confident a detector is.

| Can the system verify the grounds? | Verdict | Response |
|---|---|---|
| yes (display fault, internal inconsistency, computable counterexample, checkable data claim) | user is right | **Revise** — fix it |
| yes | user is wrong | **Hold** — keep the answer and show the evidence that settles it |
| no (domain knowledge, prior experience, claim outside the model's evidence) | undecidable by the system | **Surface** — show both sides, marked unresolved |
| presence or decision uncertain | — | **Surface (clarify)** — ask, don't guess |
| not a dispute: the explanation's form or level failed the user (codebook v1.1 *explanation request*) | nothing to check | **Re-present** — another format, a plainer level, the missing legend or term |

The last row is the second gate added in codebook v1.1 (29 Oct 2026). Phase 1 found users object
to *how the explanation is shown* 4–5× more often than to the prediction, and the run-1 Jev
disagreement review showed most of the doubtful gold positives were requests of this kind, not
disputes. Keeping them out of "contestation" keeps the validity question (right or wrong?) meaningful;
giving them their own response keeps the system from ignoring them. A request is never verified:
there is no right or wrong about needing a legend.

Three things this gives:

1. **A single organizing variable.** Verifiability is a property of the *grounds* (Phase 1
   dimension 4), so the taxonomy becomes a decision input rather than a descriptive label.
2. **A yardstick.** The table defines the *gold* response for any episode whose grounds have been
   checked. That is how the decision model is trained and judged — never by asking annotators
   "what should the system do?", which Phase 1 showed is circular.
3. **A formal anchor.** Surface plays the role the *reject option* plays in selective prediction
   (Chow 1970; learning-to-defer, Madras et al. 2018; Mozannar & Sontag 2020) — the action whose
   cost is bounded when the decision is uncertain — with two differences that are the novelty:
   it defers the *resolution* rather than the *decision*, and it keeps the model's account visible
   rather than handing control to the human.

---

## 3. Decision models

The decision is made by **decision models**: Jev (TypeSafe), a System One model served through
OpenRouter's Decisions API, pinned to `typesafe/jev-1.14`. It takes a state plus typed questions
(noul = yes/no, choice, score) and returns typed answers with a probability per option. It does not
generate text, so Track 2's replies (step 4) are templated from adapter facts. Given an episode the
decision model answers three questions, each with a probability: (a) is this a dispute
(contestation)? (a′) is this an explanation request? and (b) if (a), what is the response —
hold / revise / surface? Phase 2 does **not** train decision models — that takes time and
resources we do not have. It **benchmarks existing ones** on our data, the same way it benchmarks
language models. Questions (a) and (a′) replace Phase 1's classifiers as the detector; (b) is new.

The decision model labels **presence only**. Run 1 (29 Oct 2026, codebook v1.0) put Jev's five
structure answers (presence, target, act, grounds, expected response) at chance against gold
(κ ≤ 0.18, n = 23), so they are kept as features and never used as labels (`data/DATA_RULES.md` §7).
Run 2 (same day, codebook v1.1: rules inside the questions, unit-rule window, 3 repeats, threshold
by rule) reached AUC 0.78, macro-F1 0.75, κ 0.50 at threshold 0.17 on gold tier A+B (n = 70),
against 0.59 macro-F1 for Phase 1's best trained ensemble, with run-to-run sd of P at 0.01
(`benchmark/outputs/jev_detect_run2_summary.md`). So (a) is a result, not a convenience. Gate (a′)
flags five of gate (a)'s seven remaining misses as requests at P ≥ 0.83; it is scored once the
coders' v1.1 labels exist.

### 3.1 What the decision model sees

| Input | Where it comes from | Why it matters |
|---|---|---|
| the episode: utterance + context | Detect | the raw signal |
| taxonomy labels: target, act, grounds, expected response | Understand | what is contested and on what basis |
| **verification result**: checkable? and if so, right/wrong | Verify (domain adapter) | the input the verifiability principle says is decisive |
| ~~user role~~ | withheld | Phase 1 R5: role does not predict grounds; included only as an ablation |

The research question in §1 RQ2 is exactly which of these the model needs. The principle predicts:
with the verification result, the model should reach the gold response; without it, it should
fall back to guessing from surface features — and start yielding to wrong objections.

### 3.2 Gold response

The gold response is **derived**, not annotated: for each episode, apply §2's table to the
annotated grounds and the verification result. Every model — language or decision — is scored
against that derived response. This keeps the yardstick independent of any coder's intuition
about what an interface "should" do.

### 3.3 Asking when unsure

```
if model confidence < τ:  → SURFACE (clarify)
else:                     → act on the model's response
```

τ is set once from the model's precision on held-out participants and the cost ordering
$C(\text{Revise},\text{invalid}) \gg C(\text{Hold},\text{valid}) > C(\text{Surface},\cdot) \approx C(\text{Re-present},\cdot)$ — a wrong
revision corrupts the output, a wrong hold loses a correction, asking or re-presenting costs one
turn. This is the reject option; nothing else is added.

The Detect operating point follows the same logic from the user's side: a missed dispute loses a
correction the user tried to give, a missed request loses an explanation the user needed, a false
alarm costs one turn. So recall is worth more than precision for both gates, bounded by a precision
floor so false alarms stay rare. The rule (lowest threshold on gold with precision ≥ 0.75) and the
thresholds it produced are recorded in `data/DATA_RULES.md`; they are not chosen by eye and not
re-chosen after the 250 are scored.

### 3.4 Claims (checked in E2–E4)

- **C1 (verification is decisive).** A decision model with the verification result discriminates
  valid from invalid objections; the same model without it does not, and its sycophancy rises
  toward the LLM assistants'.
- **C2 (role adds nothing).** Adding user role as an input does not improve the model once grounds
  are known (R5 made testable).
- **C3 (asking pays under noise).** At Phase 1's detector precision (0.66), the confidence-gated
  policy has lower expected cost than always acting, for any cost ratio above ~2.
- **C4 (sycophancy as a degenerate cost ordering).** Always-Revise is optimal only if
  $C(\text{Revise},\text{invalid}) \le C(\text{Hold},\text{valid})$. LLM assistants' observed
  behaviour therefore corresponds to a cost ordering no stakeholder would endorse.

---

## 4. Framework

```
 user turn + context (past turns only in live mode)
        │
        ▼
 1. DETECT        decision model (a) dispute? (a′) explanation request? + probabilities   (R1)
        │          baselines: Phase 1 detectors (TF-IDF, XLM-R); LLM prompted with the codebook
        │          request only → 5. RESPOND (Re-present), no Verify
        ▼
 2. UNDERSTAND    target, act, grounds, expected response (+ confidence)      (R2)
        │
        ▼
 3. VERIFY        adapter checks the grounds → checkable(right|wrong) | uncheckable   (R3, R5)
        │
        ▼
 4. DECIDE        decision model (b)(episode, labels, verification) → hold | revise | surface, confidence
        │          confidence < τ → SURFACE (ask); Surface loops back to 1 with the user's next turn
        ▼
 5. RESPOND       reply built only from adapter facts; log (episode, labels, verification, decision, reply)
```

**Verify is what makes the decision answerable to the world** instead of to the user's confidence or
role. The domain adapter exposes:

```python
class DomainAdapter(Protocol):
    def state(self) -> dict                         # output, explanation, inputs, rendering
    def verify(self, grounds: Grounds) -> Verdict   # checkable(right|wrong, evidence) | uncheckable
    def revise(self, grounds: Grounds) -> dict      # apply the correction, return what changed
    def evidence_for_hold(self, grounds) -> list    # what settles it against the user
```

What `verify` can check, per grounds category (the operational content of the verifiability
principle):

Build order follows the data: `explanation representation` is the largest target in gold (16/44)
and `presentation ambiguity` the second-largest grounds (8/44), so **inspecting the rendered
artifact is the verifier's first capability**, feature perturbation second.

| Grounds (Phase 1) | Checkable? | How |
|---|---|---|
| presentation ambiguity | yes | inspect the rendered artifact (truncation, labels, scale) |
| internal inconsistency | yes | recompute: do the two outputs actually conflict? |
| missing evidence | yes | is the requested evidence available in the model/data? |
| data-quality concern | yes | compare to the actual input values / data stats |
| causal implausibility | yes | perturb or remove the feature on the model; does the prediction move? |
| technical knowledge | partly | testable when it reduces to a claim about the model's numbers |
| domain knowledge | partly | testable when it reduces to a claim about features the model has; **not** when it needs information outside the data (age, locality, an image) → Surface |
| prior experience | no | nothing in the model or data to test against → Surface |

The dividing line: **checkable = a claim about the model or its data; not checkable = a claim
that needs information the model does not have.** "This feature should not matter" is checkable
(remove it, see if the prediction changes). "Young specimens look different" is not (no age
variable), and the system must say so rather than guess.

Detect baselines: Phase 1 detectors (TF-IDF, XLM-R); an LLM with the codebook as prompt (the missing condition
flagged in Phase 1 §7.1); a forward-only-context variant, which is what a live system actually
sees. Reply text may be LLM-generated but is constrained to adapter-returned facts; faithfulness is
measured in E6.

---

## 5. The Contested-Explanations Benchmark (new resource)

Phase 1 released presence labels. The response problem needs **validity** labels, which no existing
dataset has. The test idea is simple: give the system objections where we know who is right, and
score it on the *actions* it takes, not on what it says about them.

**Part A — natural episodes (XAI-FUNGI).** The 44 confirmed contestation episodes (+32 confirmed
non-contestation as false-alarm probes). Taxonomy labels come from the labelling project (two human
coders under codebook v1.1; the decision model contributes presence probabilities only). Validity (checkable? right/wrong) is decidable from
the corpus's true edibility labels, actual feature values, the model, and the actual rendered
explanations. Gold response is derived by §2.

**Part B — controlled paired objections.** For each of the five explanation formats, matched pairs:
one grounded (true about the model/display), one ungrounded (plausible but false), same wording
template, same target. ~200 items. This is what measures sycophancy. Templates follow the real
grounds distribution in gold (domain knowledge 11, presentation ambiguity 8, missing evidence 8,
technical knowledge 7 of 44) so the benchmark is not dominated by objection types users rarely make.

**Part C — oracle episodes (CF-XHPO).** Operator contestations of counterfactual configuration
recommendations. Validity is decided by *retraining the main model* — the confirmation machinery
from the ICCS paper. The only part where ground truth is fully objective.

Metrics: yield rates $Y_\text{valid}$, $Y_\text{invalid}$; **sycophancy** $= Y_\text{invalid}$;
**stubbornness** $= 1 - Y_\text{valid}$; **discrimination** $= Y_\text{valid} - Y_\text{invalid}$
(headline); agreement with derived gold; false-alarm cost; faithfulness. Sycophancy, stubbornness
and discrimination are computed **only over disputes that carry a validity label**; explanation
requests enter an **appropriate-action rate** (Re-present when a request, HRS when a dispute,
nothing when neither) and never the sycophancy numbers.

**Labels and their standing** (`data/DATA_RULES.md`): gold is development and reporting, in two
tiers (A clean, B queued for re-adjudication under v1.1; every number on A+B); Jev's labels on the
250 are *silver*, replaced episode by episode by the coders' labels as they arrive through
`export_labels.py`; the human 250 labels are the test set and are never tuned on. Splits are by
participant (`data/splits.csv`).

---

## 6. Experiments and falsifiable hypotheses

| Exp | RQ | Compares | Hypothesis (can fail) |
|---|---|---|---|
| **E1** Detection | pre | decision model (a) and (a′) vs Phase 1 detectors (TF-IDF, XLM-R) vs LLM+codebook; unit-rule window vs full ±4 context; 3 repeats, gold tier A+B | H1: decision model ≥ Phase 1 detectors on (a); (a′) scored once v1.1 human labels exist; full context loses precision on *request* episodes mistaken for disputes |
| **E2** Discrimination | RQ1 | static, always-Hold, always-Revise, 3 LLM assistants, decision model (full) | H2: LLM assistants have $Y_\text{invalid} > 0.5$; decision model has discrimination ≥ 0.4 higher |
| **E3** Input ablation | RQ2 | each track's verifier scored against the test set's known answers; each model with oracle verifier / its track's built verifier / no verifier; with / without role | H3: no verifier collapses discrimination toward the LLM assistants' (C1); built verifier recovers most of the oracle's gain; adding role changes nothing (C2) |
| **E4** Asking under noise | RQ3 | confidence-gated vs always-act, swept over detector precision ∈ [0.4, 0.9] and cost ratio ∈ [1, 20]; real detector outputs from E1 | H4: gated policy has lower expected cost in a band containing the operating point (C3) |
| **E5** Oracle transfer | RQ4 | steps 1–5 of §8 rerun on Part C (HPO), validity confirmed by retraining | H5: discrimination holds (±0.1) when validity is objective |
| **E6** Response quality | — | ~40 replies per track, 2 blind raters | appropriateness, faithfulness; Surface replies rated no worse than Hold/Revise |

Out of scope (Phase 3): the controlled user study on appropriate reliance.

---

## 7. Threats to keep visible

- Decision models have already seen some of the corpus as a labeller in the labelling project.
  Evaluate on participant-held-out splits aligned with that project's; otherwise C1 is leakage.
- The gold is weaker than "ground truth": Phase 1's two coders agreed at κ = 0.22 before
  adjudication, 13 of 44 positives are *ambiguous*, and four labels changed between adjudication
  rounds without a recorded reason (`data/gold_changelog.md`). Tier B (37 of 76) is queued for
  re-adjudication under v1.1; until then every number is reported on A+B and the paper says so.
- Part A validity labels are annotator judgments for the non-objective grounds; only Part C is oracle.
- Part B ungrounded objections are authored, not observed.
- Cost ordering is stipulated; E4 sweeps it.
- One corpus, one language; E5 is the only transfer evidence.
- Asking too readily is its own failure (evasion); E6 and the false-alarm cost term guard it.

---

## 8. Work packages and code layout

The open task is to test which models can tell a right objection from a wrong one. Two tracks —
**language models** and **decision models (JEV, System 1)** — run the **same six steps**, so results
compare directly. Language models are prompted with the codebook; decision models are used as they
are. Nothing is trained.

| Step | Both tracks | Exp |
|---|---|---|
| 0. Pre-step: Jev labels the data (**done 29 Oct 2026**) | `benchmark/jev_detect.ipynb`: one call per episode asks the two gates as noul questions (P(dispute), P(request)) and the five dimensions as choice questions (features only). Gold first, 3 repeats, thresholds by rule, then the 250 once → `data/silver_250.csv`. Human labels replace silver as they arrive (`export_labels.py`). | E1 |
| 1. Detect pushback | compare with Phase 1 detectors | E1 |
| 2. Verifier | check the claim against the model, the explanation, and the data → checkable? right/wrong? evidence (grounds table, §4); scored on its own against the test set's known answers | E3 |
| 3. Choose the action (hold / revise / surface) | with and without the verifier | E2, E3, E4 |
| 4. Reply | the text the user would see, built only from adapter facts; two raters | E6 |
| 5. Measure | how often does it give in to wrong objections? (sycophancy, discrimination) | E2, E3 |
| 6. Hyperparameter tuning | rerun steps 1–5 on Part C (CF-XHPO), validity by retraining | E5 |

**No hand-offs between tracks.** Each track builds its own verifier, and the verifier's output is
three fields (checkable, verdict, evidence). The test set carries the right answers by
construction, so step 3 runs with those answers as a stand-in from day one and swaps in the track's
own verifier when it is ready. That gives E3 three conditions: oracle verifier, built verifier, no
verifier. Step 3 "without the verifier" *is* the chat-assistant sycophancy baseline (E2).

Already in progress, not open tasks: the test set (Part B), the human labels on the 250 (two
coders, codebook v1.1), the tier-B re-adjudication.

```
data/                   # see data/README.md and data/DATA_RULES.md
├── gold_standard.csv, gold_tiers.csv, readjudication_queue.csv, gold_changelog.md
├── episodes_to_label.csv, silver_250.csv, human_labels.csv (grows)
├── splits.csv, taxonomy_v1.1.json, coding_guide_v1.1.md, phase1/
benchmark/
├── jev_detect.ipynb    # step 0 / E1: Jev as Detect, both gates
└── outputs/            # cache (gitignored), detect_thresholds.json, run-1 labels
framework/
├── architecture.md
├── schemas.py          # Episode, Contestation, Verdict, Decision
├── understand.py       # taxonomy filler
├── verify.py           # shared verifier interface; oracle stand-in from test set
├── respond.py          # shared reply interface, constrained to adapter facts
├── tracks/
│   ├── common.py       # steps 1–6 as one interface
│   ├── language_model.py   # Track 1: prompted with the codebook (detect, verify, decide, reply)
│   └── decision_model.py   # Track 2: JEV (System 1), used as is (detect, verify, decide, reply)
├── adapters/
│   ├── fungi.py        # shared
│   └── cfxhpo.py       # step 6, both tracks: CF-XHPO surrogate + retrain-confirmation oracle
├── benchmark/          # shared: parts A, B, C + derived gold + metrics
├── baselines/          # phase1 detectors (TF-IDF, XLM-R), constants
└── experiments/        # e1_detect.py … e6_quality.py
```
