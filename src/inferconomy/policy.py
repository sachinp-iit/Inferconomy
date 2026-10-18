"""Policy: budget allocation, escalation, and stopping.

A policy is the seam that decides *how much to spend* and *when to stop* once the
analysis is known. It does not pick the strategy, and it does not analyse the
task. Separating those three - analysis in
:mod:`inferconomy.decision`, strategy in :mod:`inferconomy.strategy`, and
allocation here - is what lets them be substituted independently.

Two constraints shape this module:

**The reference policy is deliberately conservative.** The default should
err on the side of quality. An aggressive policy that cuts budget too early can
destroy quality in a single request, and a user cannot audit what was lost after
the fact. A conservative one can be made less conservative over time by swapping
it, but the reverse is much harder to undo safely. The default here spends
little only when there is *definitive* evidence it is wasted.

**A policy makes no claims it cannot justify.** Its inputs are the decision,
the request's ceiling, the model's confirmed capabilities, and a trace of prior
calls. It does not peek at future responses or imagine judge scores. The
allocation it produces is recorded verbatim in
:attr:`~inferconomy.contracts.OptimizationReport.decisions`, so a surprising number
has its reason attached.

The policy answers three questions for a run: the initial budget, whether to
escalate given what happened, and whether to stop now. Those three are the only
levers it is allowed. More levers would make it harder to audit and impossible
to test in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from inferconomy.capabilities import CapabilityReport
from inferconomy.contracts import OptimizationReport, Request, Response, StopReason
from inferconomy.decision import Decision, TaskShape

__all__ = [
    "Allocation",
    "ConservativeReferencePolicy",
    "Policy",
]

MIN_REASONING_BUDGET = 64
"""The smallest non-zero reasoning budget that can plausibly do work.

A policy that allocates 1 token cannot meaningfully reason, so this is a floor
rather than a suggestion to spend as little as possible. The floor is explicit
so it cannot be silently set to 0."""
MAX_REASONING_BUDGET = 32_768
"""Reasonable hard cap for the reference policy.

The reference policy does not try to maximise, and it does not scale linearly
to enormous ceilings. Instead it picks a modest bound and stops escalating early
unless the runtime signals a failure."""
DEFAULT_DIRECT_BUDGET = 128
"""Starter budget for a request with no evidence that reasoning helps."""
DEFAULT_DERIVED_BUDGET = 512
"""Starter budget when reasoning is likely needed.

Deliberately conservative. A real run can escalate if insufficient."""
ESCALATION_FACTOR = 2
"""Multiplier applied to the initial budget to get one escalation's additional amount.

Deliberately a fixed multiple of the *initial* budget rather than a compounding
one: the policy is stateless across calls, so it has no running total to compound
from without reading the history, and a fixed increment is easier to audit and
test. A caller wanting compounding escalation can supply its own policy."""
MAX_ESCALATIONS = 3
"""How many times the policy is willing to escalate in total.

Three attempts is conservative enough to avoid runaway spend for the reference
policy, while still giving a difficult request two chances to recover after
the first attempt."""


@dataclass(frozen=True)
class Allocation:
    """What a policy decided for this run.

    The fields are the answers to the three questions a policy controls: initial
    budget, whether to escalate, and whether to stop. Everything is explicit so
    that :attr:`~inferconomy.contracts.OptimizationReport.decisions` can record
    why each one was taken.

    ``additional_budget`` is the amount to *add* on escalation, not the new
    total. That keeps the report's ``additional_budget`` field unambiguous: it is
    exactly what was spent on top of the initial, and ``escalated`` is true iff
    it was > 0.
    """

    initial_budget: int
    additional_budget: int
    """Amount to allocate if and when escalating. ``0`` means do not escalate."""
    should_stop: bool
    """True if this run should end now, regardless of whether it completed."""
    stop_reason: StopReason
    """Why to stop, when :attr:`should_stop` is True."""
    decision: Decision
    """The analysis this allocation was based on."""
    reason: str
    """Explanation that goes into the audit trail."""

    def __post_init__(self) -> None:
        if self.initial_budget < 0:
            raise ValueError(
                f"Allocation.initial_budget must be non-negative, got "
                f"{self.initial_budget}."
            )
        if self.additional_budget < 0:
            raise ValueError(
                f"Allocation.additional_budget must be non-negative, got "
                f"{self.additional_budget}."
            )
        if self.additional_budget == 0 and self.should_stop:
            # If there is nothing left to allocate, stopping is almost always the
            # right call; the enum just needs to say why. The policy is still
            # required to pick a reason.
            pass
        if self.stop_reason is StopReason.SUFFICIENCY and not self.should_stop:
            # Sufficiency is a stop reason, not a state to carry forward. If the
            # policy is not stopping, it should not claim sufficiency.
            raise ValueError(
                "Allocation.stop_reason cannot be SUFFICIENCY "
                "unless should_stop is True."
            )
        if not self.reason.strip():
            raise ValueError("Allocation.reason must not be blank.")
        if not isinstance(self.decision, Decision):
            raise ValueError(
                "Allocation.decision must be a Decision instance. The policy is "
                "a function of the analysis, not a replacement for it."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_budget": self.initial_budget,
            "additional_budget": self.additional_budget,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason.value,
            "decision": self.decision.to_dict(),
            "reason": self.reason,
        }


@runtime_checkable
class Policy(Protocol):
    """Allocates budget and decides when to stop.

    A policy is invoked at the beginning of a run (to get the initial allocation)
    and again after each attempt (to decide whether to escalate or stop). It must
    be deterministic with respect to its inputs - the same decision, request,
    capabilities, and history must produce the same allocation - because an audit
    trail that changes between replays is not an audit trail.

    The protocol is intentionally narrow. It does not return a strategy and does
    not mutate the request. Those are the seams for other components.
    """

    @property
    def name(self) -> str:
        """Identifier recorded in decisions and reports."""
        ...

    @property
    def deterministic(self) -> bool:
        """Whether repeated calls with identical inputs return identical results."""
        ...

    async def allocate(
        self,
        decision: Decision,
        request: Request,
        capabilities: CapabilityReport | None = None,
        *,
        history: Sequence[Response] = (),
        prior_reports: Sequence[OptimizationReport] = (),
        escalations_so_far: int = 0,
    ) -> Allocation:
        """Return the allocation for the next step of this run.

        ``history`` is the sequence of responses received so far (in order).
        ``prior_reports`` are the reports that accompanied them, in the same order.
        ``escalations_so_far`` counts how many times the policy has already
        escalated before this call. Together these let the policy decide whether
        to stop, escalate again, or end the run.
        """
        ...


def _clamp_budget(budget: int, ceiling: int | None) -> int:
    if ceiling is None:
        return min(budget, MAX_REASONING_BUDGET)
    return min(budget, ceiling)


def _decision_reason_text(decision: Decision) -> str:
    hint = (
        "none"
        if decision.budget_hint_tokens is None
        else str(decision.budget_hint_tokens)
    )
    matched = ", ".join(decision.matched) if decision.matched else "none"
    return (
        f"decision {decision.shape.value} (unc={decision.uncertainty:.2f}, "
        f"hint={hint}, matched={matched}): {decision.reason}"
    )


class ConservativeReferencePolicy:
    """A deliberately conservative allocation policy.

    The default posture is: do not spend more than necessary, but do not cut when
    the evidence is ambiguous. Where the decision engine refuses to claim a shape
    (``unknown`` with or without conflict), this policy spends a small,
    non-zero amount rather than assuming reasoning is wasted. That is the
    conservative trade-off: a cheap extra generation is usually cheaper than a
    wrong cheap one.

    Escalation only happens when there is a concrete signal the budget was
    insufficient, and only up to :attr:`MAX_ESCALATIONS`. The first escalation at
    least doubles the additional allocation, keeping the growth obvious and
    bounded.

    It never escalates for ``open`` tasks. Spending more on a task with no unique
    answer buys length, not correctness, and the runtime's sufficiency signal is
    the right place to stop.
    """

    name = "conservative-reference-2026-09"
    deterministic = True

    async def allocate(
        self,
        decision: Decision,
        request: Request,
        capabilities: CapabilityReport | None = None,
        *,
        history: Sequence[Response] = (),
        prior_reports: Sequence[OptimizationReport] = (),
        escalations_so_far: int = 0,
    ) -> Allocation:
        ceiling = request.max_output_tokens
        shape = decision.shape
        base_reason = _decision_reason_text(decision)

        if shape is TaskShape.DERIVED:
            initial = decision.budget_hint_tokens
            if initial is None or initial <= 0:
                initial = DEFAULT_DERIVED_BUDGET
            initial_budget = _clamp_budget(initial, ceiling)
            additional_budget = self._compute_additional(
                escalations_so_far, ceiling, initial_budget
            )
            return Allocation(
                initial_budget=initial_budget,
                additional_budget=additional_budget,
                should_stop=False,
                stop_reason=StopReason.STRATEGY_COMPLETE,
                decision=decision,
                reason=(
                    f"derived task; starting from decision hint or default "
                    f"({DEFAULT_DERIVED_BUDGET}); {base_reason}"
                ),
            )

        if shape is TaskShape.CONTAINED:
            return Allocation(
                initial_budget=0,
                additional_budget=0,
                should_stop=True,
                stop_reason=StopReason.STRATEGY_COMPLETE,
                decision=decision,
                reason=(
                    f"contained task: answer likely present in prompt; no reasoning "
                    f"budget warranted by the evidence; {base_reason}"
                ),
            )

        if shape is TaskShape.OPEN:
            initial_budget = _clamp_budget(DEFAULT_DIRECT_BUDGET, ceiling)
            return Allocation(
                initial_budget=initial_budget,
                additional_budget=0,
                should_stop=False,
                stop_reason=StopReason.STRATEGY_COMPLETE,
                decision=decision,
                reason=(
                    f"open task: no unique answer, so spend modestly and let "
                    f"sufficiency decide when to stop; {base_reason}"
                ),
            )

        # UNKNOWN: neither "no work" nor "much work" is claimable. Spend a small,
        # non-zero amount rather than assuming reasoning is wasted - the
        # conservative trade-off is that a cheap extra generation is usually
        # cheaper than a wrong cheap one.
        initial_budget = _clamp_budget(DEFAULT_DIRECT_BUDGET, ceiling)
        return Allocation(
            initial_budget=initial_budget,
            additional_budget=0,
            should_stop=False,
            stop_reason=StopReason.STRATEGY_COMPLETE,
            decision=decision,
            reason=(
                f"unknown task shape: conservative small allocation rather than "
                f"assuming no work; {base_reason}"
            ),
        )

    def _compute_additional(
        self, escalations_so_far: int, ceiling: int | None, last_allocation: int
    ) -> int:
        if escalations_so_far >= MAX_ESCALATIONS:
            return 0
        candidate = last_allocation * ESCALATION_FACTOR
        if candidate < MIN_REASONING_BUDGET:
            candidate = MIN_REASONING_BUDGET
        if ceiling is None:
            return min(candidate, MAX_REASONING_BUDGET)
        remaining = ceiling - last_allocation
        if remaining <= 0:
            return 0
        return min(candidate, remaining, MAX_REASONING_BUDGET)
