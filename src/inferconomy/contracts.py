"""Core request and response contracts.

These four types are the library's public data surface: what goes in, what comes
out, what it cost, and why the optimizer decided what it did. Everything else in
Inferconomy is built on them, so they are deliberately conservative — frozen,
validated, and serializable.

Two properties are worth stating up front, because they follow from what
Inferconomy does rather than from taste:

- **Inference is not one call.** Adaptive allocation means a single user request
  may produce several LLM calls, so :class:`Usage` aggregates across calls and
  counts them. A caller that assumes one call per request will compute cost
  wrongly.
- **Not every token count is exact.** Providers do not all report usage, and
  several report it differently. :class:`Usage` therefore carries its own
  provenance in :attr:`Usage.exact` rather than leaving the reader to assume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, TypeVar

__all__ = [
    "Capability",
    "Message",
    "OptimizationReport",
    "Request",
    "Response",
    "StopReason",
    "Usage",
]


class Capability(str, Enum):
    """A lever a target model actually exposes.

    Adaptive control needs something to control. These are the levers Inferconomy
    knows how to use; the probe reports which of them are available so that the
    policy can degrade deliberately instead of silently doing nothing.
    """

    REASONING_BUDGET = "reasoning_budget"
    """Separate reasoning and output token budgets."""

    EFFORT_CONTROL = "effort_control"
    """Model-side reasoning-effort control (for example a low/medium/high knob)."""

    VISIBLE_COT = "visible_cot"
    """Chain-of-thought exposed to the caller, enabling checkpointed checks."""

    LOGPROBS = "logprobs"
    """Token logprobs or confidence, enabling probabilistic early exit."""

    USAGE_REPORTING = "usage_reporting"
    """Response reports token usage, so cost is exact rather than estimated."""


class StopReason(str, Enum):
    """Why inference stopped.

    This is Inferconomy's own decision and is distinct from a provider's
    ``finish_reason``, which is carried on :class:`Response` unchanged. The two
    are reported separately because they answer different questions: one says
    what the model did, the other says whether the optimizer was satisfied.
    """

    SUFFICIENCY = "sufficiency"
    """The output was judged sufficient and inference ended early."""

    STRATEGY_COMPLETE = "strategy_complete"
    """The selected strategy ran to its natural end."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    """Allocated budget ran out before sufficiency was reached."""

    MAX_BUDGET = "max_budget"
    """The caller's hard ceiling was reached."""

    CANCELLED = "cancelled"
    """The host application aborted the request."""

    ERROR = "error"
    """Inference stopped because of an error."""


_EnumT = TypeVar("_EnumT", bound=Enum)


def _decode_enum(enum_type: type[_EnumT], value: object, field_name: str) -> _EnumT:
    """Decode a serialized enum value, failing loudly on anything unrecognised.

    A silently ignored unknown value would let a report from a future version
    load as a plausible-looking report from this one.
    """
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (ValueError, TypeError) as exc:
        valid = ", ".join(repr(member.value) for member in enum_type)
        raise ValueError(
            f"Invalid {field_name}: {value!r}. Expected one of: {valid}."
        ) from exc


@dataclass(frozen=True)
class Message:
    """A single conversation turn."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message:
        try:
            return cls(role=str(data["role"]), content=str(data["content"]))
        except KeyError as exc:
            raise ValueError(
                f"Message is missing required field: {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True)
class Request:
    """What the host application asks for.

    A request describes the task and the caller's hard limits. It deliberately
    does not carry sampling parameters or a quality target: those belong to the
    strategy and the policy, and setting them here would put the optimization
    decision back in the caller's hands.
    """

    messages: tuple[Message, ...]
    max_output_tokens: int | None = None
    """Caller-imposed ceiling. The policy allocates within it, never above it."""

    stop: tuple[str, ...] = ()
    """Optional stop sequences, passed through to the provider."""

    def __post_init__(self) -> None:
        # Frozen dataclasses still accept mutable sequences, which would quietly
        # break the immutability guarantee callers rely on. Normalize once here.
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "stop", tuple(self.stop))

        if not self.messages:
            raise ValueError("Request requires at least one message.")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be positive, got {self.max_output_tokens}."
            )

    @classmethod
    def from_text(cls, text: str, *, system: str | None = None) -> Request:
        """Build a single-turn request, optionally preceded by a system turn."""
        if not text:
            raise ValueError("Request text must not be empty.")
        messages = []
        if system is not None:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=text))
        return cls(messages=tuple(messages))

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "max_output_tokens": self.max_output_tokens,
            "stop": list(self.stop),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Request:
        try:
            raw_messages = data["messages"]
        except KeyError as exc:
            raise ValueError("Request is missing required field: 'messages'") from exc
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, str):
            raise ValueError("Request 'messages' must be a sequence of messages.")
        return cls(
            messages=tuple(Message.from_dict(item) for item in raw_messages),
            max_output_tokens=(
                None
                if data.get("max_output_tokens") is None
                else int(data["max_output_tokens"])
            ),
            stop=tuple(data.get("stop") or ()),
        )


@dataclass(frozen=True)
class Usage:
    """Token and cost accounting, aggregated across every LLM call made.

    Adaptive inference issues more than one call per user request, so these
    figures are totals, not per-call values.

    :attr:`exact` records whether the underlying counts came from the provider
    or from an estimate. It defaults to ``False`` deliberately: a library should
    under-claim precision rather than let a caller read an estimate as a
    measurement.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    """Subset of ``output_tokens`` spent on reasoning, when separable."""

    cached_input_tokens: int = 0
    """Subset of ``input_tokens`` served from a provider cache."""

    llm_calls: int = 0
    """Number of LLM calls made. Not the number of user requests."""

    cost_usd: float | None = None
    cost_exact: bool = False
    """Whether ``cost_usd`` reflects published rates rather than an estimate."""

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "llm_calls",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError(
                f"reasoning_tokens ({self.reasoning_tokens}) cannot exceed "
                f"output_tokens ({self.output_tokens})."
            )
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(
                f"cached_input_tokens ({self.cached_input_tokens}) cannot exceed "
                f"input_tokens ({self.input_tokens})."
            )
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError(f"cost_usd must be non-negative, got {self.cost_usd}.")
        if self.cost_exact and self.cost_usd is None:
            raise ValueError("cost_exact=True requires cost_usd to be set.")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Aggregate two usage records.

        Exactness is conjunctive: a sum containing even one estimated component
        is itself an estimate, and saying otherwise would be the kind of quiet
        error this class exists to prevent.
        """
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            cost_usd=(
                None
                if self.cost_usd is None or other.cost_usd is None
                else self.cost_usd + other.cost_usd
            ),
            cost_exact=self.cost_exact and other.cost_exact,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cost_exact": self.cost_exact,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Usage:
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            reasoning_tokens=int(data.get("reasoning_tokens", 0)),
            cached_input_tokens=int(data.get("cached_input_tokens", 0)),
            llm_calls=int(data.get("llm_calls", 0)),
            cost_usd=None if data.get("cost_usd") is None else float(data["cost_usd"]),
            cost_exact=bool(data.get("cost_exact", False)),
        )


@dataclass(frozen=True)
class Response:
    """What a model produced, and what it cost.

    The provider's own ``finish_reason`` is preserved verbatim on
    :attr:`provider_finish_reason` rather than being normalized, because
    normalizing it would discard the only ground truth about what the model
    actually did.
    """

    text: str
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    provider_finish_reason: str | None = None
    reasoning_text: str | None = None
    """Chain-of-thought, when the provider exposes it separately.

    See :attr:`Capability.VISIBLE_COT`. ``None`` means the provider did not
    expose it, which is different from an empty trace.
    """

    latency_ms: float = 0.0
    """Wall-clock time for this call, excluding Inferconomy's own overhead."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "provider_finish_reason": self.provider_finish_reason,
            "reasoning_text": self.reasoning_text,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Response:
        try:
            text = str(data["text"])
        except KeyError as exc:
            raise ValueError("Response is missing required field: 'text'") from exc
        raw_usage = data.get("usage")
        return cls(
            text=text,
            usage=Usage() if raw_usage is None else Usage.from_dict(raw_usage),
            model=None if data.get("model") is None else str(data["model"]),
            provider_finish_reason=(
                None
                if data.get("provider_finish_reason") is None
                else str(data["provider_finish_reason"])
            ),
            reasoning_text=(
                None
                if data.get("reasoning_text") is None
                else str(data["reasoning_text"])
            ),
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


@dataclass(frozen=True)
class OptimizationReport:
    """Why the optimizer decided what it did.

    Separated from :class:`Response` on purpose: a response is what the model
    produced, a report is what Inferconomy chose and why. Consumers that only
    want the text should not have to traverse optimization metadata, and
    consumers that want to audit the layer should not have to parse prose.

    :attr:`total_budget` and :attr:`escalated` are derived rather than stored,
    so a report cannot claim to be internally inconsistent.
    """

    strategy: str
    capabilities_used: tuple[Capability, ...] = ()
    """Which model levers the policy actually exercised."""

    initial_budget: int = 0
    additional_budget: int = 0
    """Computation allocated on top of the initial budget after escalation."""

    stopped_on: StopReason = StopReason.STRATEGY_COMPLETE
    decisions: tuple[str, ...] = ()
    """Ordered, human-readable audit trail of decisions taken."""

    schema_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities_used", tuple(self.capabilities_used))
        object.__setattr__(self, "decisions", tuple(self.decisions))

        if self.initial_budget < 0:
            raise ValueError(
                f"initial_budget must be non-negative, got {self.initial_budget}."
            )
        if self.additional_budget < 0:
            raise ValueError(
                f"additional_budget must be non-negative, got {self.additional_budget}."
            )

    @property
    def total_budget(self) -> int:
        return self.initial_budget + self.additional_budget

    @property
    def escalated(self) -> bool:
        return self.additional_budget > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "capabilities_used": [item.value for item in self.capabilities_used],
            "initial_budget": self.initial_budget,
            "additional_budget": self.additional_budget,
            "total_budget": self.total_budget,
            "escalated": self.escalated,
            "stopped_on": self.stopped_on.value,
            "decisions": list(self.decisions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationReport:
        version = int(data.get("schema_version", cls.schema_version))
        if version > cls.schema_version:
            raise ValueError(
                f"Report schema version {version} is newer than this release "
                f"understands ({cls.schema_version}). Upgrade Inferconomy."
            )
        try:
            strategy = str(data["strategy"])
        except KeyError as exc:
            raise ValueError("Report is missing required field: 'strategy'") from exc
        raw_decisions = data.get("decisions") or ()
        if not isinstance(raw_decisions, Sequence) or isinstance(raw_decisions, str):
            raise ValueError("Report 'decisions' must be a sequence of strings.")
        return cls(
            strategy=strategy,
            capabilities_used=tuple(
                _decode_enum(Capability, item, "capability")
                for item in (data.get("capabilities_used") or ())
            ),
            initial_budget=int(data.get("initial_budget", 0)),
            additional_budget=int(data.get("additional_budget", 0)),
            stopped_on=_decode_enum(
                StopReason,
                data.get("stopped_on", StopReason.STRATEGY_COMPLETE.value),
                "stopped_on",
            ),
            decisions=tuple(str(item) for item in raw_decisions),
        )
