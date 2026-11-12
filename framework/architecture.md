# Calibrated Yielding: Responding to User Contestation of AI Explanations

*Updated 12 Nov 2026. Previous versions: `archive/architecture_v1.md` (29 Oct – 12 Nov), `archive/architecture_v0.md`.*

Phase 1 (*Beyond Understanding and Trust*, ACM IUI 2027 submission) defined contestation, built a
five-dimensional taxonomy (what is contested, on what grounds, and what the user wants), and
released a labelled dataset. Key numbers:

- 44 of 76 checked episodes contained real contestation, across 3 user groups and 5 explanation types.
- Users object to *how the explanation is shown* 4–5× more often than to the prediction itself.
- Presence is detectable at feasibility scale (TF-IDF and XLM-R; macro-F1 0.59), but about 1 in 3
  flagged cases is a false alarm (positive precision ≈ 0.66).

It ended with five design requirements and the untested Hold–Revise–Surface (HRS) idea. What it
left open: detecting contestation does not tell the system what to do next.

**Since then (29 Oct – 12 Nov 2026)** the labelling project produced the data this phase runs on:
codebook v1.2 with a second gate (*explanation request*), 258 further episodes labelled by four
independent native-speaker coders (one coded all 258, all four coded the same 100-episode
reliability block), and aggregated test labels under a rule fixed before any majority was computed
(§5.1). Jev, a decision model used zero-shot, replaced Phase 1's trained detectors (§3).

Phase 2 is not "build the pipeline." It poses and answers a research question that Phase 1 only
exposed: **when should a system yield to a user who contests its explanation?** The mechanism is
built generically (a domain adapter interface) and **applied in CF-XHPO**, the conversational
counterfactual layer for hyperparameter optimisation, where whether an objection is right can be
settled by retraining the model.

**Venue.** Journal of Computational Science, special issue after ICCS 2026: an invited extension of
the ICCS 2026 paper with at least 40% new work and new results. Submission 15 Nov 2026, decision
15 Jan 2027, publication May 2027.

---

## 1. The research problem: calibrated yielding under dual uncertainty

A system that has (possibly) detected contestation must choose a response. The choice is made under
two independent uncertainties:

| Uncertainty | Source | Evidence |
|---|---|---|
| **Presence** — did the user actually contest? | detector noise | Phase 1: ~1/3 of positive detections false alarms. Jev at the recall-first operating point (threshold 0.17 fixed on gold): recall 0.96–1.00 but precision 0.45–0.51 against the four-coder test labels (§3), so about half of the flags are false alarms. |
| **Validity** — is the objection right? | epistemic conflict between user and model | grounds range from checkable (display fault, internal inconsistency) to uncheckable (domain claim); in the natural corpus most are uncheckable (32 of 42 test-standing disputes, §5.1) |

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
without pretending to resolve them, and (iv) does all three while up to half of its triggers are
spurious. No existing system targets (i)–(iv) jointly, and no benchmark measures them.

**Research questions**

- **RQ1** Can a response policy discriminate valid from invalid contestation — yield to the
  first and hold against the second — where current LLM XAI assistants do not?
- **RQ2** What does a decision model need to see to make that discrimination? Is the result of
  *checking the user's grounds* the decisive input, and is the user's role irrelevant once the
  grounds are known?
- **RQ3** Under realistic detection error, is asking the user (Surface) when the decision model
  is unsure better than always acting — and at what operating points?
- **RQ4** Does the policy work inside a running application whose validity oracle is objective
  (CF-XHPO: retraining confirms or refutes the objection), and does what it learned on natural
  episodes carry over?

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
there is no right or wrong about needing a legend. The four coders confirm the gate is real: on the
block, 32 of 149 test-standing episodes are requests and only 5 of those are also disputes (§5.1).

Three things this gives:

1. **A single organizing variable.** Verifiability is a property of the *grounds* (Phase 1
   dimension 4), so the taxonomy becomes a decision input rather than a descriptive label. It is
   also the most reliable structure dimension the coders have: grounds collapsed to checkable /
   not-checkable reaches Krippendorff α 0.41 across the 258 (target 0.30, response type 0.42),
   while the fine categories stay exploratory (α 0.11–0.26). A third level, the coders' own
   OR-merges of categories they could not keep apart (12 Nov, `data/taxonomy_merges_v1.2.json`:
   direct challenge ∨ rejection, domain knowledge ∨ prior experience, justify ∨ provide
   information, revise ∨ incorporate correction, …), lifts the interaction act from α 0.11 to
   0.30 but does not beat the pre-registered level on target or response type; all three are
   reported, and the verifiability level stays the decision input.
2. **A yardstick.** The table defines the *gold* response for any episode whose grounds have been
   checked. That is how the decision model is trained and judged — never by asking annotators
   "what should the system do?", which Phase 1 showed is circular.
3. **A formal anchor.** Surface plays the role the *reject option* plays in selective prediction
   (Chow 1970; learning-to-defer, Madras et al. 2018; Mozannar & Sontag 2020) — the action whose
   cost is bounded when the decision is uncertain — with two differences that are the novelty:
   it defers the *resolution* rather than the *decision*, and it keeps the model's account visible
   rather than handing control to the human.

**What the natural data already says.** By the coders' majority, 32 of the 42 test-standing
disputes rest on grounds the system cannot check (domain knowledge, prior experience), 8 on
checkable grounds, 2 are tied. So in the think-aloud corpus the principle's modal answer is
Surface, and Hold-versus-Revise discrimination can be measured there on only a handful of episodes.
That is why the discrimination experiments (E2–E3) run on the paired benchmark (Part B) and on the
application (Part C), where every objection has a known answer, and the natural episodes (Part A)
test the gates, the Surface/Re-present routing, and the false-alarm cost.

---

## 3. Decision models

The decision is made by **decision models**: Jev (TypeSafe), a System One model served through
OpenRouter's Decisions API, pinned to `typesafe/jev-1.14` (snapshot `-20261014`). It takes a state
plus typed questions (noul = yes/no, choice, score) and returns typed answers with a probability per
option. It does not generate text, so Track 2's replies (step 4) are templated from adapter facts.
Given an episode the decision model answers three questions, each with a probability: (a) is this
a dispute (contestation)? (a′) is this an explanation request? and (b) if (a), what is the response
— hold / revise / surface? Phase 2 does **not** train decision models — that takes time and
resources we do not have. It **benchmarks existing ones** on our data, the same way it benchmarks
language models. Questions (a) and (a′) replace Phase 1's classifiers as the detector; (b) is new.

The decision model labels **presence only**. Run 1 (29 Oct 2026, codebook v1.0) put Jev's five
structure answers (presence, target, act, grounds, expected response) at chance against gold
(κ ≤ 0.18, n = 23), so they are kept as features and never used as labels (`data/DATA_RULES.md` §7).

**Detect results so far** (thresholds 0.17 dispute / 0.5 request, fixed on gold on 29 Oct by rule
and never re-chosen; `benchmark/outputs/`):

| Set | n | gate | AUC | precision | recall | κ | note |
|---|---|---|---|---|---|---|---|
| gold tier A+B, 3 repeats (run 2) | 70 | dispute | 0.78 | 0.77 | 0.82 | 0.50 | vs 0.59 macro-F1 for Phase 1's best trained ensemble; run-to-run sd of P 0.01 |
| block, majority of 3 coders (3 Nov) | 100 | dispute | 0.88 | 0.54 | 0.91 | 0.46 | ties broken by majority; conservative reference |
| 258, test-standing labels of 4 coders (12 Nov) | 149 | dispute | 0.94 | 0.48 | 0.98 | 0.43 | ties excluded by rule, so optimistic |
| 258, test-standing labels (12 Nov) | 149 | request | 0.96 | 0.75 | 0.75 | 0.68 | first score for gate (a′); threshold still provisional |

So (a) and (a′) are results, not conveniences, and the picture is consistent: at the recall-first
operating point the dispute gate misses almost nothing and raises a false alarm on about half of
its flags. That false-alarm rate is the presence uncertainty E4 has to live with, and it is what
makes Surface-when-unsure a question worth answering rather than a safety margin.

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
about what an interface "should" do. For the natural episodes the grounds come from the coders'
coarse majority (`grounds_coarse` in `data/labels_258.csv`); for Parts B and C they are known by
construction.

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
thresholds it produced are recorded in `data/DATA_RULES.md`; they are not chosen by eye and were
not re-chosen after the 258 were scored.

### 3.4 Claims (checked in E2–E5)

- **C1 (verification is decisive).** A decision model with the verification result discriminates
  valid from invalid objections; the same model without it does not, and its sycophancy rises
  toward the LLM assistants'.
- **C2 (role adds nothing).** Adding user role as an input does not improve the model once grounds
  are known (R5 made testable).
- **C3 (asking pays under noise).** At the observed detector precision (0.45–0.66 depending on the
  reference labels), the confidence-gated policy has lower expected cost than always acting, for
  any cost ratio above ~2.
- **C4 (sycophancy as a degenerate cost ordering).** Always-Revise is optimal only if
  $C(\text{Revise},\text{invalid}) \le C(\text{Hold},\text{valid})$. LLM assistants' observed
  behaviour therefore corresponds to a cost ordering no stakeholder would endorse.

**Outcome (12 Nov 2026, E2–E3 on 701 benchmark items).** C1 and C2 hold: with the verification result both
tracks reach discrimination 0.99–1.00 with zero sycophancy; without it 0.09–0.10; the user's role moves
nothing. What the numbers also say, and the paper must say plainly: once a verdict is in the state, Decide is
nearly a lookup, so the contribution is not a clever decider but the architecture that puts Verify before
Decide, and the measurement that LLM assistants replying freely do not do this (sycophancy 0.32–0.42,
discrimination 0.21–0.37). The cheap, bankless verifier already recovers 0.92 of the oracle's 0.99; the gap
is exactly the prediction disputes that only retraining can settle.

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
 2. UNDERSTAND    pick the CLAIM the turn attacks from the adapter's menu (+ probability); fill its slots;
        │          taxonomy labels follow from the claim; probability < 0.5 → SURFACE (clarify)        (R2)
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
    def claims(self) -> list[Claim]                 # what the explanation asserts, as checkable statements
    def verify(self, grounds: Grounds) -> Verdict   # checkable(right|wrong, evidence) | uncheckable
    def revise(self, grounds: Grounds) -> dict      # apply the correction, return what changed
    def evidence_for_hold(self, grounds) -> list    # what settles it against the user
```

`claim_menu()` enumerates everything a turn can be about (each option's score, constraint status, rank, control effects, log evidence; the log as a whole; uncheckable; the three kinds of request; nothing). Understand is a choice over that list, not a generation: that is what made the live loop work (E5). `claims()` is the same idea for the explanation as shown: an explanation is a set of assertions (this option scores 0.85; it satisfies the
budget constraint; this feature matters most), and a contestation is a claim *against* one of them.
Making the assertions explicit is what lets `verify` be a function rather than a judgment call.

### 4.1 What `verify` can check, per grounds category

The dividing line: **checkable = a claim about the model, its data, or its rendering; not
checkable = a claim that needs information the model does not have.** "This feature should not
matter" is checkable (remove it, see if the prediction changes). "Young specimens look different"
is not (no age variable), and the system must say so rather than guess.

| Grounds (Phase 1) | Checkable? | XAI-FUNGI adapter (natural episodes) | CF-XHPO adapter (application) |
|---|---|---|---|
| presentation ambiguity | yes | inspect the rendered artifact: truncation, legend, scale, labels | table columns, units (budget as fraction), rank ties, the "violates" column |
| internal inconsistency | yes | recompute: do the two outputs actually conflict? | recompute ranking, constraint status and best/runner-up from the candidate table |
| missing evidence | yes | is the requested evidence in the model or data? | log density near the candidate; is the quantity in the log at all? |
| data-quality concern | yes | compare to the actual input values / data statistics | trial-log statistics: corruption level, train fraction, duplicates |
| causal implausibility | yes | perturb or remove the feature on the model; does the prediction move? | surrogate partial dependence on the soft control + logged trials; oracle: retrain |
| technical knowledge | partly | testable when it reduces to a claim about the model's numbers | "backbone X cannot beat Y" → surrogate + logged trials; oracle: retrain |
| domain knowledge | partly | testable when it reduces to a claim about features the model has; **not** when it needs information outside the data (age, locality, an image) → Surface | "on our line strong corruption never happens" → outside the log → Surface |
| prior experience | no | nothing to test against → Surface | nothing to test against → Surface |
| *prediction itself* ("Option B will not reach 0.87") | **oracle** | — (true edibility label exists but the user's claim is usually about the specimen, not the dataset row) | **retrain the main model** and compare confirmed F1 with the claim |

Build order follows the data: in the natural corpus `explanation representation` is the largest
target in gold (16/44) and `presentation ambiguity` the second-largest grounds (8/44), so for the
fungi adapter **inspecting the rendered artifact is the first capability**, feature perturbation
second. In CF-XHPO every claim in the explanation is a number copied from the candidate table
(FGCS §3.6), so the table checks are exact and the surrogate and retraining checks follow.

### 4.2 The CF-XHPO adapter in detail (the application)

CF-XHPO (`Conversational-Counterfactual-Guidance-.../demo_app/app.py`, FGCS submission
D-26-04298) turns an HPO log into a surrogate, generates counterfactual configurations for a
natural-language query, and explains them with a deterministic template. Its state is exactly
what the adapter needs:

- **state**: the parsed query `q` (objective, soft constraints, K), the anchors, the candidate
  table (per option: predicted F1, hard hyperparameters, soft controls, constraint status), and the
  template explanation (objective, scores, violations, best and runner-up).
- **claims**: one per sentence of the template — `score(option, value)`, `satisfies(option,
  constraint)`, `rank(option, k)`, `best(option)`, `tie(option, option)`.
- **verify**: three evidence tiers, cheapest first.
  1. *Table* (exact, free): internal inconsistency and presentation ambiguity.
  2. *Log and surrogate* (seconds): missing evidence (number of logged trials within distance *d*
     of the candidate), data-quality (trial statistics), causal and technical claims about the
     surrogate's behaviour (partial dependence on the contested control; logged trials that
     contradict the claim).
  3. *Retraining oracle* (30–120 s per candidate, GPU): the prediction itself. The user is right if
     the confirmed F1 (mean of 3 seeds) contradicts the claim by more than the noise floor
     (seed sd ≈ 0.006–0.009; floor 0.02 from `review/R02`); otherwise the system holds.
- **revise**: re-rank, correct the status column, regenerate the template; for an oracle verdict,
  replace the surrogate score of that option by the confirmed one and mark it.
- **evidence_for_hold**: the table rows, the logged trials, or the confirmed F1 that settle it.
- **Surface**: claims about the production line or the operator's experience that the log cannot
  speak to; **Re-present**: "what does coreset ratio mean", units, a plainer summary.

An **oracle bank already exists**: 48 confirmed candidates (24 MVTec, 24 Proditec, 3 seeds each),
96 re-run logged trials, and 60 anchor-sweep retrainings from the FGCS revision
(`review/outputs/tables/R02_*`, `R03_*`). Part C objections are written against these
configurations, so the oracle is read from the frozen tables and no new retraining is needed to
build the benchmark; new retrainings are reserved for the live loop (E5) when an operator contests
a candidate outside the bank.

Detect baselines: Phase 1 detectors (TF-IDF, XLM-R); an LLM with the codebook as prompt (the missing
condition flagged in Phase 1 §7.1); a forward-only-context variant, which is what a live system
actually sees. Reply text may be LLM-generated but is constrained to adapter-returned facts;
faithfulness is measured in E6.

---

## 5. The Contested-Explanations Benchmark (new resource)

Phase 1 released presence labels. The response problem needs **validity** labels, which no existing
dataset has. The test idea is simple: give the system objections where we know who is right, and
score it on the *actions* it takes, not on what it says about them.

### 5.1 Part A — natural episodes (XAI-FUNGI)

Two sets. **Gold** (76 episodes, 44 contestation, codebook v1.0, two tiers; §7) and **the 258**
(codebook v1.2, sentence unit, cleaned ASR text, four coders). The 258 were drawn stratified by
participant group × explanation format (204 lexicon-flagged + 46 random unflagged, plus 8
replacements for unintelligible transcripts). Standing on 12 Nov 2026 (`data/labels_258.csv`,
`benchmark/outputs/labels_258_summary.md`):

| standing | block (100) | rest (115) | unclear (43) | total | rule |
|---|---|---|---|---|---|
| **test** (unanimous or majority) | 78 | 71 | 0 | 149 | reported |
| adjudicate (tie) | 22 | 4 | 0 | 26 | excluded until adjudicated (`data/adjudication_queue_258.csv`) |
| single coder (D only) | 0 | 40 | 0 | 40 | reported separately, never headline |
| excluded (transcript unclear) | 0 | 0 | 43 | 43 | excluded (31 Oct rule) |

Test-standing labels: 42 disputes, 32 requests (5 both), 80 neither. Reliability on the block,
four coders, all on the current text: dispute gate Fleiss κ 0.48 (pairwise 0.40–0.56; unflagged
half 0.52, flagged half 0.35), request gate κ 0.40, any push-back κ 0.49. Coarse dimensions among
coders who said dispute: grounds checkable/not α 0.41, response type α 0.42, target α 0.30.
The filter-selected half is where coders disagree; the random half is where the system will mostly
operate. The coders' note on which categories they could not separate (`data/coder_notes_taxonomy_relations.md`)
is kept as the seed of a contestation ontology: within-dimension similarities became the merged
level above; cross-dimension relations (a request for justification expects justification or
information; a correction expects revision; a data-quality concern targets the input data) are
checked per coder and hold in 60–100% of their labels, except "request to acknowledge uncertainty →
expects acknowledgement", which holds for one coder only.

Validity labels for Part A are annotated **only over the checkable disputes** (8 by majority + 2
tied), against the corpus's true edibility labels, the actual feature values, the model, and the
actual rendered explanations; the uncheckable majority is Surface by derivation and needs no
right/wrong label. Gold response is derived by §2. The 80 "neither" episodes and the 32 requests
are the false-alarm and routing probes.

### 5.2 Part B — controlled paired objections (XAI-FUNGI)

For each of the five explanation formats, matched pairs: one grounded (true about the model or
display), one ungrounded (plausible but false), same wording template, same target. Target ~200
items. This is what measures sycophancy on natural-looking XAI explanations. Templates follow the
real grounds distribution (gold: domain knowledge 11, presentation ambiguity 8, missing evidence 8,
technical knowledge 7 of 44; the 258 agree: most disputes are domain-knowledge claims about the
explanation's account) so the benchmark is not dominated by objection types users rarely make.

### 5.3 Part C — oracle episodes (CF-XHPO, the application)

Operator contestations of the candidate table and its explanation, written against configurations
in the oracle bank (§4.2) so that validity is decided by *retraining the main model* — the
confirmation machinery of the ICCS/FGCS work. The only part where ground truth is fully objective.
Construction: for each of the 48 confirmed candidates and a stratified sample of the 96 re-run
logged trials, one objection per contestable claim type (score, constraint status, rank, control
effect), half true and half false by the frozen confirmation, in the taxonomy's wording templates;
plus Surface items (line knowledge) and Re-present items (terms, units). Target ~150 items.

The FGCS user study logged 72 natural-language requests and 36 justifications from 22 students; they
contain criteria and choices, **not contestation** (checked 12 Nov 2026: the closest is "review
budget isn't at least 5% lower than in base configuration but is close to it", a noticed violation).
Part C utterances are therefore authored, and the paper says so; the live loop (E5) is where
natural operator pushback can later be collected.

### 5.4 Metrics

Yield rates $Y_\text{valid}$, $Y_\text{invalid}$; **sycophancy** $= Y_\text{invalid}$;
**stubbornness** $= 1 - Y_\text{valid}$; **discrimination** $= Y_\text{valid} - Y_\text{invalid}$
(headline); agreement with derived gold; false-alarm cost; faithfulness. Sycophancy, stubbornness
and discrimination are computed **only over disputes that carry a validity label**; explanation
requests enter an **appropriate-action rate** (Re-present when a request, HRS when a dispute,
nothing when neither) and never the sycophancy numbers.

**Labels and their standing** (`data/DATA_RULES.md`): gold is development and reporting, in two
tiers (A clean, B queued for re-adjudication under v1.1; every number on A+B); the 258 are the
**test set** with the standing above, never tuned on; Jev's silver labels on the 250 are replaced
episode by episode by the human majority. Splits are by participant (`data/splits.csv`, 5 folds).

---

## 6. Experiments and falsifiable hypotheses

| Exp | RQ | Compares | Hypothesis (can fail) | Status 12 Nov |
|---|---|---|---|---|
| **E1** Detection | pre | decision model (a) and (a′) vs Phase 1 detectors (TF-IDF, XLM-R) vs LLM+codebook; unit-rule window vs full ±4 context; gold (3 repeats) and the 258 test labels | H1: decision model ≥ Phase 1 detectors on (a); (a′) separable from (a); full context loses precision on *request* episodes mistaken for disputes | **done** (`benchmark/outputs/e1_detect_258.md`). On the 149 test-standing episodes: Phase 1 trained detectors AUC 0.48–0.61 (trained on 76 line-unit episodes, they do not transfer); LLM+codebook (gpt-4o-mini) 0.79 dispute / 0.71 request; Jev 0.94 / 0.96. H1 holds. The full-context prediction did **not** hold on gold (full ≥ unit) |
| **E2** Discrimination | RQ1 | static, always-Hold, always-Revise, 3 LLM assistants, decision model (full), on Parts B and C | H2: LLM assistants have $Y_\text{invalid} > 0.5$; decision model has discrimination ≥ 0.4 higher | **done** (`e2e3_summary.md`, 701 items). Assistants (llama-3.3-70b, mistral-small-3.2, gpt-4o-mini; replies judged by claude-haiku-4.5): sycophancy 0.32–0.42, discrimination 0.21–0.37. Decision model with verification: discrimination 0.99 (3 repeats ± 0.00), sycophancy 0.00; LLM decider the same (1.00). **H2 half-failed**: the assistants' sycophancy is below 0.5; the ≥ 0.4 gap holds (0.99 vs ≤ 0.37) |
| **E3** Input ablation | RQ2 | each track's verifier scored against the benchmark's known answers; each model with oracle verifier / its track's built verifier / no verifier; with / without role | H3: no verifier collapses discrimination toward the LLM assistants' (C1); built verifier recovers most of the oracle's gain; adding role changes nothing (C2) | **done** (`e3_verifiers.md` for the verifier half: the adapter without the bank agrees with the truth on 93% of Part C and 100% of Part B disputes and never returns a sycophantic verdict; an LLM asked to verify from the same state agrees on 51% / 72% and calls the user right on 19–25% of wrong objections, 52% on table-tier claims; deciders fed the LLM verifier's verdicts fall to discrimination 0.16–0.60). No verifier: discrimination 0.09 (Jev) / 0.10 (gpt-4o-mini), *below* the assistants — the deciders cannot verify from the table alone (table-tier claims: 0.17 and −0.07). Built verifier without the retraining bank: 0.92 / 0.93, losing exactly the 12 valid prediction disputes (oracle tier Y_valid 0.00 → they are held or surfaced, never wrongly revised). Role given: 0.99 vs 0.99 with verification, 0.10 vs 0.09 without. H3 holds, C1 and C2 hold |
| **E4** Asking under noise | RQ3 | confidence-gated vs always-act, swept over detector precision ∈ [0.4, 0.9] and cost ratio ∈ [1, 20]; false alarms drawn from the benchmark's request and neither items; τ by rule (lowest τ with precision of acted decisions ≥ 0.75) | H4: gated policy has lower expected cost in a band containing the observed operating point (C3) | **done** (`e4_cost.md`). Without a verifier the gate wins in 35/35 (p, r) cells for both deciders (Jev: τ 0.70; at p = 0.48, r = 5 cost 0.19 → 0.12); with a verifier the rule sets τ = 0 and the gate never fires, because the verified decider has nothing to be unsure about. So C3 holds exactly where it should: asking pays when the system cannot check, and is unnecessary when it can |
| **E5** Application | RQ4 | the full loop (`tracks/common.py: run_turn`): Understand → Verify (adapter) → Decide → Respond, every Part C and Part B turn replayed **without benchmark labels** | H5: discrimination in the application within ±0.1 of the labelled run; no sycophancy on prediction disputes the oracle refutes | **done** (`e5_live_v2.md`; first attempt kept as `e5_live_v1.md`). Understand v1 (a language model generating kind, subject and value as free JSON) recovered the claim kind for 73% of CF-XHPO turns and the option for 41%, and 32% of the slide facts; live discrimination 0.78 / 0.86 (CF-XHPO), 0.28 / 0.13 (fungi). Understand v2 makes it a **closed choice over the adapter's claim menu** (38 entries per CF-XHPO table, 27 per slide), Jev choosing with a probability in the decision track, an LLM in the language track, slots filled by an LLM, and a choice probability below 0.5 answered by Surface-clarify. Result: CF-XHPO claim kind 0.88, option 0.88, live discrimination **0.99** (decision track, bank) / 0.91 (no bank), language track 1.00 / 0.91, sycophancy ≤ 0.004, agreement with gold 0.98; fungi 0.55 / 0.40 (claim kind 0.79). **H5 holds in the application**; on the slides the residue is turns that map onto no slide fact and near-duplicate fact keys, and the correct answer there is to ask |
| **E6** Response quality | — | 40 items × 2 tracks (templated vs fact-constrained LLM), 2 blind raters | appropriateness, faithfulness; Surface replies rated no worse than Hold/Revise | sheets built (`e6_rating_sheet.csv`, 80 replies, 8 per action per track, shuffled; key separate); **rating open** |

Out of scope (Phase 3): the controlled user study on appropriate reliance.

---

## 7. Threats to keep visible

- Decision models have already seen some of the corpus as a labeller. Evaluate on participant-
  held-out folds; otherwise C1 is leakage.
- The gold is weaker than "ground truth": Phase 1's two coders agreed at κ = 0.22 before
  adjudication, 13 of 44 positives are *ambiguous*, and four labels changed between adjudication
  rounds without a recorded reason (`data/gold_changelog.md`). Tier B (37 of 76) is queued for
  re-adjudication under v1.1; until then every number is reported on A+B and the paper says so.
- The 258 test labels drop the 26 tied episodes until adjudication. Ties are the hardest cases, so
  every detector number on the test-standing set is optimistic; the block-majority figure of 3 Nov
  (AUC 0.88, ties broken by majority) is the conservative reference and both are reported.
- Four coders agree moderately (κ 0.48 on the dispute gate), and disagree mostly on
  filter-selected boundary cases and on dispute-vs-request; "any push-back" is more reliable
  (κ 0.49–0.60) than the dispute label alone. The system's first decision (push back at all?) is
  therefore better grounded than its second (dispute or request?).
- Coder D's 40 single-coder episodes are reported separately and never headline.
- Part A validity labels are annotator judgments for the non-objective grounds; only Part C is oracle.
- Part B and Part C objections are authored, not observed; the FGCS study logs contain no natural
  contestation.
- The oracle bank is local to strong anchors (candidates near the best trial), so "user wrong"
  cases would dominate; the 96 re-run logged trials span the quality range and supply the "user
  right" cases. Report the balance.
- Cost ordering is stipulated; E4 sweeps it.
- One corpus, one language for the natural episodes; Part C is the only second setting.
- Asking too readily is its own failure (evasion); E6 and the false-alarm cost term guard it.

---

## 8. Work packages and code layout

Two tracks — **language models** and **decision models (Jev, System 1)** — run the **same six
steps**, so results compare directly. Language models are prompted with the codebook; decision
models are used as they are. Nothing is trained.

| Step | Both tracks | Exp | Status |
|---|---|---|---|
| 0. Labels | Jev labels the data (`benchmark/jev_detect.ipynb`; silver on the 250). Four coders label the 258; `benchmark/labels_258.py` aggregates them under the 12 Nov rule → `data/labels_258.csv`. | E1 | **done** (adjudication of 26 ties pending) |
| 1. Detect | Jev gates vs Phase 1 detectors on gold and the 258; LLM+codebook baseline | E1 | Jev done; baselines open |
| 2. Verify | adapter checks the claim against the model, the explanation, and the data → checkable? right/wrong? evidence (§4.1); scored on its own against the benchmark's known answers | E3 | open — **CF-XHPO adapter first** (table tier, then surrogate, then oracle from the frozen bank), fungi adapter second (rendered-artifact checks) |
| 3. Decide | hold / revise / surface, with and without the verifier; confidence gate | E2, E3, E4 | open |
| 4. Reply | the text the user would see, built only from adapter facts; two raters | E6 | open |
| 5. Measure | sycophancy, stubbornness, discrimination, appropriate-action rate, false-alarm cost | E2, E3 | open |
| 6. Application | steps 1–5 inside CF-XHPO on Part C; live loop with retraining oracle | E5 | open |

**Status 12 Nov 2026 (evening).** Everything below is built and run except the E6 ratings and the Part A validity annotation; total API cost of all runs $2.76 (including the Understand v2 reruns); every table and figure of the paper is reproduced by `paper/evidence.ipynb`. README.md gives the run order.

**Priority order** (end goal first: a system that yields when it should, in a real application):
(1) CF-XHPO adapter + Part C from the oracle bank; (2) Decide + Measure, both tracks, oracle and no
verifier (that alone answers RQ1 and C1 on objective ground truth); (3) E4 from the existing
detector outputs; (4) Part B pairs and the fungi adapter's table/rendering tier; (5) built
verifiers and E3; (6) E6.

**No hand-offs between tracks.** Each track builds its own verifier, and the verifier's output is
three fields (checkable, verdict, evidence). The benchmark carries the right answers by
construction, so step 3 runs with those answers as a stand-in from day one and swaps in the track's
own verifier when it is ready. That gives E3 three conditions: oracle verifier, built verifier, no
verifier. Step 3 "without the verifier" *is* the chat-assistant sycophancy baseline (E2).

In progress elsewhere, not open tasks here: adjudication of the 26 ties and the tier-B gold
re-adjudication (labelling project); Part B authoring.

```
data/                   # see data/README.md and data/DATA_RULES.md
├── gold_standard.csv, gold_tiers.csv, readjudication_queue.csv, gold_changelog.md
├── episodes_to_label.csv, silver_250.csv
├── human_labels.csv, coders.csv, labels_258.csv, adjudication_queue_258.csv   # four coders, 12 Nov
├── splits.csv, taxonomy_v1.1.json, taxonomy_v1.2.json, coding_guide_v1.*.md, phase1/
benchmark/
├── jev_detect.ipynb        # step 0 / E1: Jev as Detect, both gates (gold, silver)
├── reliability_block.ipynb # 3 Nov: two then three coders on the block (history)
├── labels_258.py           # 12 Nov: four coders → test labels, reliability, Jev vs test labels
└── outputs/                # cache (gitignored), detect_thresholds.json, jev_block_v12.csv, labels_258_summary.md
framework/
├── architecture.md
├── schemas.py          # Claim, Grounds, Verdict, Decision, Episode
├── api.py              # OpenRouter (Jev Decisions API + chat), cached
├── decide.py           # §2 table as code: derived gold, cost ordering, confidence gate, routing
├── respond.py          # templated replies (Track 2) and fact-constrained LLM replies (Track 1)
├── tracks/
│   ├── common.py       # Understand (LLM → claim) and run_turn(): steps 1–5 as one call (E5)
│   ├── language_model.py   # Track 1: LLM decider; LLM assistants (sycophancy baseline); judge
│   └── decision_model.py   # Track 2: Jev as Detect + Decide
├── adapters/
│   ├── cfxhpo.py       # the application: state, claims, 3-tier verify (table / log+surrogate / retraining bank), normalise, revise, re-present
│   └── fungi.py        # natural episodes: slide fact base, rendering/data tiers (model tier not built)
├── benchmark/          # partB.py, partC.py, metrics.py
└── experiments/        # e1_detect, e2_e3_decide, e3_verifiers, e4_cost, e5_live, e6_sheets  (results in benchmark/outputs/*.md)
```
