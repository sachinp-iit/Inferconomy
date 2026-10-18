"""Strategy: how one attempt is executed.

Analysis lives in :mod:`inferconomy.decision`, allocation in
:mod:`inferconomy.policy`. This module owns the third seam: a strategy receives
what the policy decided for one attempt and turns it into a completed provider
call, together with the audit report that explains what it did.

Two constraints shape it.

**The client is passed in, never held.** A strategy is not a connection, not a
session, and not a cache. Which client answers an attempt is the runtime's
choice - it can differ between attempts - and a strategy that held one would
make that choice for its owner. Holding nothing also keeps a strategy trivially
reusable across concurrent runs, which is what lets :func:`select_strategy` hand
out independent instances that share nothing.

**A strategy does not re-decide.** The budget and the stopping verdict arrive
from the policy, and the shape analysis came from the decision engine. A
strategy that second-guessed either would be answering a question two other
components already own, and a disagreement between them would be invisible to
the audit trail.

:class:`DirectStrategy` is deliberately the plainest thing that can be correct:
one call, no reasoning lever, the allocated budget forwarded as the output cap.
It is the comparison point every other strategy has to beat, so it must not grow
clever.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from inferconomy.capabilities import CapabilityReport
from inferconomy.client import Client, CompletionOptions
from inferconomy.contracts import (
    Capability,
    OptimizationReport,
    Request,
    Response,
    StopReason,
)
from inferconomy.policy import Allocation

__all__ = ["DirectStrategy", "Strategy", "StrategyResult", "select_strategy"]


@dataclass(frozen=True)
class StrategyResult:
    """What one attempt produced: the response, and the report explaining it.

    Always paired, because a response without its report cannot be audited and a
    report without its response cannot be checked against the text that was
    actually produced. A caller that wants only the answer reads
    :attr:`response`; a caller making a claim about cost reads both.
    """

    response: Response
    report: OptimizationReport

    def __post_init__(self) -> None:
        if not isinstance(self.response, Response):
            raise ValueError(
                f"StrategyResult.response must be a Response, got "
                f"{type(self.response).__name__}."
            )
        if not isinstance(self.report, OptimizationReport):
            raise ValueError(
                f"StrategyResult.report must be an OptimizationReport, got "
                f"{type(self.report).__name__}."
            )


@runtime_checkable
class Strategy(Protocol):
    """Executes one attempt against a single allocation.

    A strategy owns *how* the call is made - which levers to pull, how many
    calls to make - while :class:`~inferconomy.decision.DecisionEngine` owns what
    the task needs and :class:`~inferconomy.policy.Policy` owns how much to
    spend. Implementations must be deterministic with respect to their inputs,
    because a report that changes between replays cannot be compared between
    runs.

    The client arrives per call rather than at construction: a strategy holds no
    state between attempts, so two instances with the same name are
    interchangeable and safe to run concurrently.
    """

    @property
    def name(self) -> str:
        """Identifier recorded in reports and audit trails."""
        ...

    @property
    def deterministic(self) -> bool:
        """Whether repeated calls with identical inputs return identical results."""
        ...

    async def execute(
        self,
        request: Request,
        allocation: Allocation,
        client: Client,
        capabilities: CapabilityReport | None = None,
    ) -> StrategyResult:
        """Make one attempt for ``allocation`` and return its result.

        ``capabilities`` is the probed truth about the target model, or ``None``
        when nothing was probed. A strategy must not assume a lever exists
        without evidence for it; an absent report is not permission to guess.

        Exceptions from ``client`` propagate. A failed attempt is a failure, and
        the runtime is the layer that decides whether to retry, escalate, or
        report :attr:`~inferconomy.contracts.StopReason.ERROR`.
        """
        ...


class DirectStrategy:
    """One provider call, no reasoning lever, budget enforced as an output cap.

    ``direct`` is the strategy for tasks where no evidence supports paying for
    reasoning: its only job is to forward the policy's allocation faithfully.
    The allocated budget reaches the wire as
    ``CompletionOptions.max_output_tokens``, so a policy that funds little
    actually spends little rather than merely claiming to. The options it builds
    are exactly two - the output cap and the caller's stop sequences - and no
    more: which model answers is the adapter's configuration, and a reasoning
    budget or effort level would make ``direct`` not direct.

    An allocation of ``0`` funds nothing, and :class:`CompletionOptions` refuses
    a non-positive cap, so a caller's ceiling is applied instead when one was
    set; with neither, the provider's default cap applies and the audit trail
    says which happened. ``0`` means "no budget was funded", not "silence was
    guaranteed" - what the provider actually generated is recorded in ``usage``,
    and only that figure is a cost claim.

    The report always stops on :attr:`StopReason.STRATEGY_COMPLETE`, because a
    single attempt is this strategy's natural end; sufficiency and escalation
    belong to a runtime that inspects responses between attempts. For the same
    reason it does not read ``Allocation.should_stop``: a contained task still
    needs its one extraction call, and whether the run continues afterwards is
    the runtime's question to answer.
    """

    name = "direct"
    deterministic = True

    async def execute(
        self,
        request: Request,
        allocation: Allocation,
        client: Client,
        capabilities: CapabilityReport | None = None,
    ) -> StrategyResult:
        options = self._options(request, allocation)
        if capabilities is not None:
            options = capabilities.adapt(options).options
        response = await client.complete(request, options)
        return StrategyResult(
            response=response,
            report=self._report(request, allocation, response),
        )

    def _options(self, request: Request, allocation: Allocation) -> CompletionOptions:
        # A policy that funded nothing has no cap of its own to send, and
        # CompletionOptions refuses a non-positive one, so the caller's ceiling
        # is the only limit still in force. It must be, because it is a hard
        # limit the caller set rather than a budget this layer chose.
        cap = (
            allocation.initial_budget
            if allocation.initial_budget > 0
            else request.max_output_tokens
        )
        return CompletionOptions(max_output_tokens=cap, stop=request.stop)

    def _report(
        self, request: Request, allocation: Allocation, response: Response
    ) -> OptimizationReport:
        decisions = [
            allocation.decision.audit_line(),
            allocation.reason,
            f"{self.name}: one call, no reasoning; "
            f"{self._budget_text(request, allocation)}",
        ]
        if allocation.additional_budget > 0:
            decisions.append(
                "escalation was offered by the policy but not used: a single "
                "attempt does not escalate."
            )
        return OptimizationReport(
            strategy=self.name,
            capabilities_used=(
                (Capability.USAGE_REPORTING,) if response.usage.tokens_exact else ()
            ),
            usage=response.usage,
            initial_budget=allocation.initial_budget,
            additional_budget=0,
            stopped_on=StopReason.STRATEGY_COMPLETE,
            decisions=tuple(decisions),
        )

    def _budget_text(self, request: Request, allocation: Allocation) -> str:
        if allocation.initial_budget > 0:
            return f"output cap {allocation.initial_budget}"
        if request.max_output_tokens is not None:
            return (
                f"no budget funded; caller ceiling {request.max_output_tokens} "
                f"sent as the cap"
            )
        return "no budget funded; no caller ceiling; provider default cap applies"


_STRATEGIES: dict[str, Callable[[], Strategy]] = {
    "direct": DirectStrategy,
}


def select_strategy(name: str) -> Strategy:
    """Return a fresh strategy for ``name``.

    Fresh on purpose: a strategy holds no state, but a caller that pinned one
    instance across runs could still attach some. An unknown name is refused
    loudly with the valid ones listed, because a silently substituted strategy
    would attribute one strategy's results to another's.
    """
    factory = _STRATEGIES.get(name)
    if factory is None:
        valid = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"Unknown strategy {name!r}. Valid strategies: {valid}.")
    return factory()
