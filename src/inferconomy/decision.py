"""Task analysis: what a request needs, decided before anything is spent.

Every later decision in this library rests on a claim about the task in front of
it. Which strategy, how much budget, whether to escalate - all of them begin with
"what is this asking for, and does it need work". That claim has to be made
somewhere. The easy place is inside the strategy that will act on it, where it
can neither be swapped for a better one nor tested on its own, so it is made
here, behind a protocol, with the claim and the evidence for it kept together in
one :class:`Decision`.

Three properties of that object matter more than its field list.

**``unknown`` is a real answer.** There are four shapes rather than three, and
``unknown`` is one of them. With three, every request is forced into a shape the
classifier cannot support, which is the same confusion between *not supported*
and *not supported yet* that :mod:`inferconomy.capabilities` exists to prevent.
Two different kinds of not knowing are recorded and are not interchangeable:
``unknown`` with an empty :attr:`Decision.matched` means the analysis saw
nothing at all; ``unknown`` with rules in it means it saw signals that disagreed.
The first says the heuristic has no purchase on this task. The second says the
task is genuinely mixed, and a mixed task is not the same as an absent one.

**Uncertainty is reported, not resolved.** Whether ambiguity is worth paying to
remove is a question about money, not about language, so it belongs to the policy
(US-013) rather than to the thing that measures it. A decision engine that spent
more on its own uncertain inputs would be making an allocation decision nobody
asked it for and nobody could audit.

**Two kinds of not much.** ``budget_hint_tokens=0`` means the engine is confident
reasoning would be wasted. ``budget_hint_tokens=None`` means it has no opinion.
Collapsing the two turns "I do not know" into "spend nothing", which is how a
heuristic quietly becomes a policy that was never written down.

The hint follows the shape; uncertainty is what says how much of a guess the
shape is. They answer different questions on purpose.

What :class:`HeuristicDecisionEngine` is, and is not: a set of lexical rules over
the final user turn, deterministic, and free of any call. It is a floor, not a
claim about accuracy - it cannot tell "write a function that computes the
primes below n" from a creative writing prompt except by which words happen to
appear, and where its signals conflict it declines to classify rather than
guessing. That is the point of the protocol: a better engine, including one that
costs tokens to run, can be substituted without anything downstream changing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from inferconomy.contracts import Request

__all__ = [
    "CONFLICT_UNCERTAINTY",
    "DERIVED_BUDGET_HINT_TOKENS",
    "Decision",
    "DecisionEngine",
    "HeuristicDecisionEngine",
    "TaskShape",
]

MAX_CONFIDENCE = 0.8
"""What even perfect agreement is worth. A lexical heuristic is never certain."""

EVIDENCE_SATURATION = 3
"""How many agreeing signals it takes for evidence to stop adding confidence."""

CONFLICT_UNCERTAINTY = 0.9
"""Uncertainty assigned when signals fired but disagreed.

Deliberately below the 1.0 of "no signal at all". Both are ``unknown``, and
reporting them identically would throw away the only distinction the analysis
produced: a task it has no purchase on, versus a task that is genuinely mixed.
"""

DERIVED_BUDGET_HINT_TOKENS = 512
"""Starter budget suggested for a task whose answer must be derived.

A starting point, not an optimum. It is deliberately a constant rather than a
function of prompt length, because a computed number would imply a measurement
nobody has made. US-013 owns allocation; this only says where to begin.
"""


class TaskShape(str, Enum):
    """How the answer is produced relative to the prompt.

    About the shape of the work, not the subject matter. A request about poetry
    and a request about plumbing can both be ``derived``, and a request about
    physics can be ``contained``.
    """

    UNKNOWN = "unknown"
    """No shape is claimed: nothing matched, or what matched disagreed."""

    CONTAINED = "contained"
    """The answer is in the prompt. Extracting it is the whole task."""

    DERIVED = "derived"
    """The answer must be computed, concluded, or constructed from material given.

    The only shape that recommends spending on reasoning, because it is the only
    one where a derivation exists to be got right.
    """

    OPEN = "open"
    """No unique answer exists. Reasoning buys length rather than correctness."""


@dataclass(frozen=True)
class Decision:
    """What a task needs, with the evidence that says so.

    ``engine``, ``matched``, and ``reason`` exist so that a decision can be
    written into :attr:`~inferconomy.contracts.OptimizationReport.decisions`,
    read back, and argued with months later. A decision that cannot be explained
    cannot be debugged, and the decisions worth debugging are the surprising ones.
    """

    shape: TaskShape
    engine: str
    needs_reasoning: bool
    uncertainty: float
    """``[0, 1]``. 1.0 means the engine learned nothing; a tie is capped at 0.9."""
    budget_hint_tokens: int | None
    """Suggested starting budget in ``Request.max_output_tokens`` units.

    ``0`` is the confident claim that reasoning would be wasted. ``None`` is the
    refusal to have an opinion, and is what every ``unknown`` decision carries.
    """
    reason: str
    matched: tuple[str, ...] = ()
    """Rules that fired, in declaration order. Empty means nothing was seen."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", TaskShape(self.shape))
        object.__setattr__(self, "matched", tuple(str(item) for item in self.matched))

        if not isinstance(self.needs_reasoning, bool):
            raise ValueError(
                f"Decision.needs_reasoning must be a bool, got "
                f"{self.needs_reasoning!r}. It is a recommendation that costs "
                f"money, not a value to coerce."
            )
        value = float(self.uncertainty)
        if not math.isfinite(value):
            raise ValueError(
                f"Decision.uncertainty must be a finite number, got "
                f"{self.uncertainty!r}. A NaN would poison every mean it entered."
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Decision.uncertainty must lie in [0, 1], got {self.uncertainty!r}."
            )
        object.__setattr__(self, "uncertainty", value)

        hint = self.budget_hint_tokens
        if hint is not None:
            if isinstance(hint, bool) or not isinstance(hint, int):
                raise ValueError(
                    f"Decision.budget_hint_tokens must be an int or None, got {hint!r}."
                )
            if hint < 0:
                raise ValueError(
                    f"Decision.budget_hint_tokens must be non-negative, got {hint}."
                )
        if not self.engine.strip():
            raise ValueError(
                "Decision.engine must not be blank. A decision that cannot say "
                "who made it cannot be attributed to anything, which is the same "
                "as not having been made."
            )
        if not self.reason.strip():
            raise ValueError(
                "Decision.reason must not be blank. An unexplained decision is "
                "the kind that turns out to be wrong."
            )

    def audit_line(self) -> str:
        """One line for :attr:`~inferconomy.contracts.OptimizationReport.decisions`."""
        hint = (
            "none" if self.budget_hint_tokens is None else str(self.budget_hint_tokens)
        )
        matched = ", ".join(self.matched) if self.matched else "none"
        return (
            f"{self.engine}: {self.shape.value}; "
            f"reasoning={'yes' if self.needs_reasoning else 'no'}; "
            f"uncertainty={self.uncertainty:.2f}; budget_hint={hint}; "
            f"matched={matched}; {self.reason}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.value,
            "engine": self.engine,
            "needs_reasoning": self.needs_reasoning,
            "uncertainty": self.uncertainty,
            "budget_hint_tokens": self.budget_hint_tokens,
            "reason": self.reason,
            "matched": list(self.matched),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Decision:
        try:
            if not isinstance(data["needs_reasoning"], bool):
                raise ValueError(
                    "Decision 'needs_reasoning' must be a boolean, got "
                    f"{data['needs_reasoning']!r}."
                )
            raw_hint = data.get("budget_hint_tokens")
            return cls(
                shape=TaskShape(data["shape"]),
                engine=str(data["engine"]),
                needs_reasoning=data["needs_reasoning"],
                uncertainty=float(data["uncertainty"]),
                budget_hint_tokens=None if raw_hint is None else int(raw_hint),
                reason=str(data["reason"]),
                matched=tuple(data.get("matched") or ()),
            )
        except KeyError as exc:
            raise ValueError(
                f"Decision is missing required field: {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Decision has unusable values: {exc}") from exc


@runtime_checkable
class DecisionEngine(Protocol):
    """Analyses a request and says what it needs.

    Deliberately narrower than it could be. It does not choose a strategy, does
    not allocate a budget, and does not decide whether to escalate: those belong
    to :class:`~inferconomy.strategy.Strategy` and US-013's policy. An engine that
    allocated its own budget would be making the very decision the library exists
    to make auditable, and an engine that picked its own strategy could not be
    tested on its analysis alone.
    """

    @property
    def name(self) -> str:
        """Identifier recorded in decisions. Should name the engine and version,
        such as ``"heuristic-decision-2026-09"``, so a decision is traceable."""
        ...

    @property
    def deterministic(self) -> bool:
        """Whether repeated calls with identical input return identical decisions.

        Declared rather than probed, for the same reason as
        :attr:`inferconomy.judge.Judge.deterministic`: probing an engine that
        costs tokens measures the network as much as the instrument.
        """
        ...

    async def decide(self, request: Request) -> Decision: ...


@dataclass(frozen=True)
class _Rule:
    """One lexical signal, and the shape it votes for."""

    name: str
    shape: TaskShape
    test: Callable[[str], bool]


def _contains(pattern: re.Pattern[str]) -> Callable[[str], bool]:
    def test(text: str) -> bool:
        return pattern.search(text) is not None

    return test


def _any_of(*patterns: re.Pattern[str]) -> Callable[[str], bool]:
    def test(text: str) -> bool:
        return any(pattern.search(text) is not None for pattern in patterns)

    return test


def _has_multiple_parts(text: str) -> bool:
    """Two or more list items, which is two questions wearing one prompt."""
    return len(_LIST_ITEM.findall(text)) >= 2


def _answer_is_embedded(text: str) -> bool:
    """The prompt carries the material the answer is to be taken from."""
    return (
        _QUOTED_PASSAGE.search(text) is not None
        or _EMBEDDED_MARKER.search(text) is not None
    )


_DERIVATION = re.compile(
    r"\b(?:prove|proves|proven|deriv\w*|deduc\w*|calculat\w*|comput\w*|solv\w*"
    r"|simplif\w*|show that)\b",
    re.IGNORECASE,
)
_ANALYSIS = re.compile(
    r"\b(?:why|justify|analy[sz]\w*|compare|contrast|trade-?offs?|"
    r"pros and cons|which (?:one )?is better|root cause|explain how)\b",
    re.IGNORECASE,
)
_CODE = re.compile(
    r"\b(?:debug|refactor|implement|traceback|stack ?trace|unit tests?|"
    r"write a function|bug in)\b",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```")
_LIST_ITEM = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S")
_EXTRACTION = re.compile(
    r"\b(?:summar\w*|list|extract|quote|according to|fill in|outline"
    r"|which of the following|transcribe)\b",
    re.IGNORECASE,
)
_QUOTED_PASSAGE = re.compile(r"['\"\u201c][^'\"\u201c\u201d]{120,}['\"\u201d]")
_EMBEDDED_MARKER = re.compile(
    r"\b(?:the following|as (?:shown|given|provided) below)\s+"
    r"(?:text|passage|paragraph|document|code|content|message)\b",
    re.IGNORECASE,
)
_GENERATIVE = re.compile(
    r"\b(?:write|compose|draft|create|invent|imagine|brainstorm)\b[^.?!]{0,60}"
    r"\b(?:stories|story|poems?|poetry|songs?|lyrics|novels?|essays?|articles?|"
    r"blogs?|scripts?|jokes?|slogans?|taglines?|plots?|scenes?|haiku)\b",
    re.IGNORECASE,
)
_SUBJECTIVE = re.compile(
    r"\b(?:your opinion|what do you think|in your view|what would you do"
    r"|do you (?:like|prefer|believe)|be creative|open-ended|free-?writing"
    r"|whatever you (?:think|like)|from your experience)\b",
    re.IGNORECASE,
)

_RULES: tuple[_Rule, ...] = (
    _Rule("derivation-command", TaskShape.DERIVED, _contains(_DERIVATION)),
    _Rule("analytical-question", TaskShape.DERIVED, _contains(_ANALYSIS)),
    _Rule("code-task", TaskShape.DERIVED, _any_of(_FENCE, _CODE)),
    _Rule("multi-part", TaskShape.DERIVED, _has_multiple_parts),
    _Rule("extraction-request", TaskShape.CONTAINED, _contains(_EXTRACTION)),
    _Rule("answer-embedded", TaskShape.CONTAINED, _answer_is_embedded),
    _Rule("generative-request", TaskShape.OPEN, _contains(_GENERATIVE)),
    _Rule("subjective-request", TaskShape.OPEN, _contains(_SUBJECTIVE)),
)

_SHAPED_REASON = {
    TaskShape.DERIVED: "the answer has to be derived, so reasoning has work to do",
    TaskShape.CONTAINED: "the answer is in the prompt, so reasoning has nothing to add",
    TaskShape.OPEN: "there is no unique answer; reasoning buys length, not correctness",
}


def _task_text(request: Request) -> str:
    """The final user turn, falling back to the final turn if there is no user.

    Only the final user turn, matching
    :func:`inferconomy.testing.echo_response`. A system turn is standing
    configuration rather than the task, and a system prompt that says "think
    carefully" would otherwise be scored as evidence about every request the
    application ever makes.
    """
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content
    return request.messages[-1].content


class HeuristicDecisionEngine:
    """The zero-cost reference engine: lexical rules, no calls, no state.

    Deterministic by construction - no randomness, no clock, no network - so the
    same request produces the same decision on every machine, which is what makes
    a benchmark run attributable to the strategy rather than to the analysis.

    It reads only the final user turn, votes per rule, and declines to classify
    when the votes do not settle. Its uncertainty is highest exactly where it is
    least willing to spend, which is the intended default: when this library does
    not know what a task needs, it does not pay to find out.

    That default is only *safe* once US-016 sufficiency detection and US-017
    escalation exist, because those are what recover a wrong cheap answer. Until
    then a misclassified hard task is a quality loss with nothing to catch it, and
    the honest reading of ``uncertainty=1.0`` today is "the caller should decide".
    """

    name = "heuristic-decision-2026-09"
    deterministic = True

    async def decide(self, request: Request) -> Decision:
        text = _task_text(request)
        matched: list[str] = []
        counts: dict[TaskShape, int] = {}
        for rule in _RULES:
            if rule.test(text):
                matched.append(rule.name)
                counts[rule.shape] = counts.get(rule.shape, 0) + 1

        total = sum(counts.values())
        if total == 0:
            return Decision(
                shape=TaskShape.UNKNOWN,
                engine=self.name,
                needs_reasoning=False,
                uncertainty=1.0,
                budget_hint_tokens=None,
                reason=(
                    "no signal matched, so nothing is claimed and no budget is "
                    "suggested; inventing a number here would be a guess wearing "
                    "the costume of a measurement"
                ),
            )

        top = max(counts.values())
        contenders = [shape for shape, count in counts.items() if count == top]
        if len(contenders) > 1:
            return Decision(
                shape=TaskShape.UNKNOWN,
                engine=self.name,
                needs_reasoning=False,
                uncertainty=CONFLICT_UNCERTAINTY,
                budget_hint_tokens=None,
                reason=(
                    f"signals disagreed, so no shape is claimed: {', '.join(matched)}"
                ),
                matched=tuple(matched),
            )

        shape = contenders[0]
        winning = [
            rule.name for rule in _RULES if rule.name in matched and rule.shape is shape
        ]
        confidence = (
            MAX_CONFIDENCE * (top / total) * min(1.0, total / EVIDENCE_SATURATION)
        )
        return Decision(
            shape=shape,
            engine=self.name,
            needs_reasoning=shape is TaskShape.DERIVED,
            uncertainty=1.0 - confidence,
            budget_hint_tokens=self._hint(shape, request.max_output_tokens),
            reason=f"{', '.join(winning)} agree: {_SHAPED_REASON[shape]}",
            matched=tuple(matched),
        )

    @staticmethod
    def _hint(shape: TaskShape, ceiling: int | None) -> int | None:
        """A hint follows the shape. ``unknown`` gets no opinion at all."""
        if shape is not TaskShape.DERIVED:
            return 0
        if ceiling is None:
            return DERIVED_BUDGET_HINT_TOKENS
        return min(DERIVED_BUDGET_HINT_TOKENS, ceiling)
