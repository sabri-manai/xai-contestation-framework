"""Typed records shared by every step. See architecture.md §4.

An *explanation* is a set of Claims. A *contestation* is a user turn against one of them, described by
the taxonomy (target, act, grounds) and, after Verify, by a Verdict. Decide maps (episode, labels,
verdict) to a Decision. Nothing here is domain-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

Action = Literal["hold", "revise", "surface", "represent", "none"]
Checkable = Literal["yes", "no", "partly"]
VerdictValue = Literal["user_right", "user_wrong", "undecided", "uncheckable"]

# Grounds the taxonomy (v1.2) distinguishes, and whether the principle says a system can check them.
GROUNDS_CHECKABLE: dict[str, Checkable] = {
    "presentation ambiguity": "yes", "internal inconsistency": "yes", "missing evidence": "yes",
    "data-quality concern": "yes", "causal implausibility": "yes",
    "technical knowledge": "partly", "domain knowledge": "partly", "prior experience": "no",
}


@dataclass
class Claim:
    """One assertion the explanation makes, in a form a verifier can test."""
    kind: str                      # score | satisfies | rank | best | tie | feature_effect | display | data
    subject: str                   # option id, feature name, slide element
    value: Any                     # the asserted value
    text: str                      # how the explanation states it


@dataclass
class Grounds:
    """What the user contests and on what basis (Understand output or benchmark labels)."""
    target: str = ""               # taxonomy target
    act: str = ""                  # taxonomy interaction act
    grounds: str = ""              # taxonomy grounds (fine)
    claim_kind: str = ""           # which Claim.kind the objection attacks, if identifiable
    subject: str = ""              # which option / feature / element
    user_value: Any = None         # the value the user asserts instead, if any
    extra: dict = field(default_factory=dict)


@dataclass
class Verdict:
    checkable: Checkable
    verdict: VerdictValue
    evidence: list[str] = field(default_factory=list)
    tier: str = ""                 # table | log | surrogate | oracle | rendering | data | none
    detail: dict = field(default_factory=dict)

    def as_text(self) -> str:
        """Same wording as the benchmark's verification input (experiments/e2_e3_decide.verification_text)."""
        if self.verdict == "uncheckable":
            return "not checkable: " + ("; ".join(self.evidence) or "the claim needs information the system does not have")
        return f"checkable ({self.checkable}); result: {self.verdict.replace('_', ' ')}; evidence: " + " | ".join(self.evidence)


@dataclass
class Decision:
    action: Action
    confidence: float
    p_dispute: Optional[float] = None
    p_request: Optional[float] = None
    probabilities: dict = field(default_factory=dict)
    gated: bool = False            # True if the confidence gate turned the action into surface
    raw: Any = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass
class Episode:
    """A benchmark item or a live turn."""
    episode_id: str
    domain: str                    # cfxhpo | fungi
    user_text: str
    state: dict                    # adapter.state() at the time of the turn
    explanation_text: str
    labels: Grounds = field(default_factory=Grounds)
    role: str = ""                 # withheld from the model unless the ablation asks for it
    gold: dict = field(default_factory=dict)   # is_dispute, is_request, truth, gold_action, truth_source
