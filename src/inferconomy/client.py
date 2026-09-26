"""The client boundary.

This module holds the whole of Inferconomy's dependency on the outside world. A
provider adapter implements :class:`Client`; nothing else in the library knows
which provider, SDK, or HTTP client is underneath.

Two design choices are worth stating up front.

**The protocol is async.** Adaptive allocation makes a sequence of *dependent*
calls: classify, generate, judge sufficiency, escalate. Each step waits on the
last. A synchronous core would force a thread per call or a nested event loop to
bridge the async-first SDKs that every major provider ships. Latency is one of
the metrics this project reports, so paying for this debt at the start is
cheaper than paying it after the runtime exists.

**The protocol is narrow on purpose.** Three members. Anything a provider can do
that is not here is, by definition, not something the optimizer controls. The
protocol does not grow to accommodate a provider's full feature set.

Sampling parameters are deliberately absent. Inference configuration belongs to
the strategy and the policy; letting it in through the back door would put the
optimization decision back in the caller's hands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from inferconomy.contracts import Capability, Request, Response

__all__ = ["Client", "CompletionOptions"]


@dataclass(frozen=True)
class CompletionOptions:
    """How to issue one provider call.

    The task-level :class:`~inferconomy.contracts.Request` says *what* to solve;
    this says *how* to call the API. Keeping them apart matters, because the
    budget the policy allocates has to reach the wire somehow, and widening
    :class:`Request` to carry it would collapse the distinction.

    Fields a provider does not support are ignored by that provider rather than
    being an error. That is the contract that makes one adapter work across
    endpoints with different feature sets.
    """

    model: str | None = None
    """Which model to call. ``None`` means the provider's default."""

    max_output_tokens: int | None = None
    stop: tuple[str, ...] = ()
    reasoning_budget: int | None = None
    """Separate budget for reasoning tokens. See :attr:`Capability.REASONING_BUDGET`."""

    effort: str | None = None
    """Provider-native effort level. See :attr:`Capability.EFFORT_CONTROL`."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop", tuple(self.stop))
        for name in ("max_output_tokens", "reasoning_budget"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}.")
        if self.effort is not None and not self.effort.strip():
            raise ValueError("effort must not be blank when set.")


@runtime_checkable
class Client(Protocol):
    """A single-model completion endpoint.

    Implemented by provider adapters and by fakes. Note what is *not* here: no
    streaming, no tool calls, no embeddings, no batch. Those are different
    capabilities and pretending one protocol covers them would make every
    implementation dishonest about what it supports.
    """

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Levers this adapter declares, independent of the specific model.

        This is the adapter's *static* claim. Some capabilities depend on which
        model is selected, so a runtime probe still has to confirm them; this
        says what the adapter is even able to express.
        """
        ...

    async def complete(self, request: Request, options: CompletionOptions) -> Response:
        """Issue one call and return its result.

        Implementations must not raise on an unsupported option; they ignore
        what they cannot do. Genuine failures (network, authentication, refusal
        of the request itself) should raise, so the runtime can report
        :attr:`~inferconomy.contracts.StopReason.ERROR` rather than silently
        returning a truncated answer that looks successful.
        """
        ...
