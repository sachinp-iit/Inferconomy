"""Runtime capability probing, and the degradation each missing capability causes.

A client advertises what it can express through :attr:`Client.capabilities`. That
is a static claim by the adapter, made before anything was called, and it is not
enough. Which levers a *particular model* honours depends on the model, and
providers ship features per deployment, per tier, and occasionally per region. So
the probe confirms at runtime rather than trusting the declaration.

The result is a three-valued answer per capability, and the third value is the
whole reason this module exists.

**Absence of evidence is not evidence of absence.** A capability can almost always
be *confirmed* by a positive observation: a response carrying usage, a trace in
``reasoning_text``, an endpoint accepting a parameter. It can rarely be *refuted*
by a negative one. A model asked a trivial arithmetic question may produce no
reasoning trace on a model that reasons perfectly well, and a small request may
never trip a logprobs threshold. Concluding "this model has no logprobs" from a
null result would be a claim the probe did not earn, and it would then select a
fallback as though it were fact.

So the outcomes are :attr:`ProbeOutcome.SUPPORTED`, :attr:`ProbeOutcome.
UNSUPPORTED`, and :attr:`ProbeOutcome.UNKNOWN`, and an unknown is never silently
downgraded to unsupported. Only a provider that got a *definitive* answer from
the endpoint may report ``False``; everything else reports ``None`` and lands in
unknown. A report full of unknowns is an honest report. A report claiming full
support because a field was left at its default is not.

Unknown is also treated as unusable when adapting a request. If a lever is
unverified, the policy does not pull it, and the degradation is recorded. Quietly
trying a lever you have not confirmed is the pretence this library exists to
avoid, and it is worse than not trying, because the caller cannot tell the
difference from a lever that worked.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from inferconomy.client import Client, CompletionOptions
from inferconomy.contracts import Capability, Request

__all__ = [
    "FALLBACKS",
    "PROBE_PROMPT",
    "Adaptation",
    "CapabilityProbeProvider",
    "CapabilityReport",
    "Degradation",
    "Disagreement",
    "ProbeEvidence",
    "ProbeOutcome",
    "probe_capabilities",
]

PROBE_PROMPT = "What is 17 * 23? Reply with the number only."
"""Deliberately a question a reasoning model will actually reason about.

A greeting would be cheaper and would tell us nothing: no reasoning model emits a
trace for "hi", so the probe could not distinguish "does not reason" from "did not
need to". Costs a handful of output tokens either way."""

RESPONSE_OBSERVABLE = (
    Capability.USAGE_REPORTING,
    Capability.VISIBLE_COT,
)
"""Capabilities a plain :class:`~inferconomy.contracts.Response` can settle."""

DELEGATED = (
    Capability.REASONING_BUDGET,
    Capability.EFFORT_CONTROL,
    Capability.LOGPROBS,
)
"""Capabilities a response cannot reveal, because a successful call looks
identical whether or not the lever did anything. These need the adapter to test
the endpoint."""


class ProbeOutcome(str, Enum):
    """What the probe learned. Three values, not two."""

    SUPPORTED = "supported"
    """Confirmed by a positive observation."""

    UNSUPPORTED = "unsupported"
    """Confirmed by a definitive answer from the endpoint, such as a rejected
    parameter. Only a provider may report this; a null observation never does."""

    UNKNOWN = "unknown"
    """Not established. The lever may work. Any policy relying on it is guessing,
    so the fallback applies and the report says so."""


FALLBACKS: Mapping[Capability, str] = MappingProxyType(
    {
        Capability.REASONING_BUDGET: (
            "The reasoning budget is dropped and the whole output allowance is used "
            "instead. Reasoning, if it happens, comes out of the ordinary budget, so "
            "the effective allowance for the answer is smaller than requested."
        ),
        Capability.EFFORT_CONTROL: (
            "The effort knob is dropped. The provider chooses its own default. A "
            "report that says an effort level was used would be false, so the "
            "degradation is carried into the run record."
        ),
        Capability.VISIBLE_COT: (
            "No chain-of-thought is available, so sufficiency checks see only the "
            "final answer. A policy that inspects a trace is running degraded and "
            "will escalate more often than one that can read its reasoning."
        ),
        Capability.LOGPROBS: (
            "No token confidences, so confidence-based early exit is unavailable. "
            "Escalation has to rest on signals in the text, which are weaker and will "
            "typically escalate later than a confidence-gated policy."
        ),
        Capability.USAGE_REPORTING: (
            "Token counts are estimated rather than measured, so cost carries an "
            "estimate and cannot be reported as exact. Every cost number from this "
            "model is a priced estimate."
        ),
    }
)
"""The documented consequence of each capability being unavailable.

Kept as data rather than prose in a docstring so a caller can surface the reason
alongside a run, and so a test can assert that every capability has one. A
capability added later without an entry here fails the test instead of shipping
with an undocumented, silent degradation. Wrapped in a proxy because a fallback
table that a caller can edit at runtime is documentation nobody can trust."""


@dataclass(frozen=True)
class ProbeEvidence:
    """What one probe attempt established."""

    capability: Capability
    outcome: ProbeOutcome
    detail: str
    calls: int = 0
    """Provider calls spent on this capability. Probing is not free, and a report
    that hid its own cost would make repeated probing look free."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbeEvidence:
        try:
            return cls(
                capability=Capability(data["capability"]),
                outcome=ProbeOutcome(data["outcome"]),
                detail=str(data.get("detail", "")),
                calls=int(data.get("calls", 0)),
            )
        except KeyError as exc:
            raise ValueError(
                f"Probe evidence is missing required field: {exc.args[0]!r}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"Probe evidence is not usable: {exc}") from exc


@runtime_checkable
class CapabilityProbeProvider(Protocol):
    """Optional adapter extension for capabilities a response cannot reveal.

    An adapter implements this to test the endpoint itself. It is deliberately
    separate from :class:`~inferconomy.client.Client`: an adapter that cannot
    answer the question must still be a usable client, and widening the main
    protocol would force every adapter to write methods whose honest answer is
    "I do not know".
    """

    async def probe_capability(
        self, capability: Capability, model: str | None
    ) -> tuple[bool | None, str]:
        """Test one capability and report ``(supported, detail)``.

        Return ``None`` for ``supported`` whenever the endpoint did not give a
        definitive answer. Returning ``False`` is a claim that it did, and a
        capability the adapter never tested is indistinguishable from one it
        failed to test, so ``None`` is the honest default and ``False`` should
        mean an explicit rejection from the provider.
        """
        ...


@dataclass(frozen=True)
class Degradation:
    """A lever the policy wanted that the target could not be confirmed to have."""

    capability: Capability
    requested: Any
    fallback: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "requested": self.requested,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class Adaptation:
    """Options that were actually sent, and what was dropped to get there."""

    options: CompletionOptions
    degradations: tuple[Degradation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "degradations", tuple(self.degradations))

    @property
    def degraded(self) -> bool:
        return bool(self.degradations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "degraded": self.degraded,
            "degradations": [d.to_dict() for d in self.degradations],
        }


@dataclass(frozen=True)
class Disagreement:
    """Where the adapter's static claim and the runtime observation differ."""

    capability: Capability
    claimed: bool
    observed: ProbeOutcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "claimed": self.claimed,
            "observed": self.observed.value,
        }


@dataclass(frozen=True)
class CapabilityReport:
    """What a target model actually does, per capability, for one adapter.

    Reusable and serializable so a probe is not repeated on every request.
    :attr:`cache_key` identifies the adapter-and-model pair the results belong
    to, because a report for one model says nothing about the next one.
    """

    client_name: str
    model: str
    evidence: tuple[ProbeEvidence, ...]
    claimed: frozenset[Capability] = frozenset()
    calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not self.model.strip():
            raise ValueError("CapabilityReport.model must not be blank.")
        if not self.client_name.strip():
            raise ValueError("CapabilityReport.client_name must not be blank.")

    def outcome(self, capability: Capability) -> ProbeOutcome:
        for item in self.evidence:
            if item.capability is capability:
                return item.outcome
        return ProbeOutcome.UNKNOWN

    def detail(self, capability: Capability) -> str:
        for item in self.evidence:
            if item.capability is capability:
                return item.detail
        return "Not probed."

    def supports(self, capability: Capability) -> bool:
        """Strictly confirmed. Unknown is not support."""
        return self.outcome(capability) is ProbeOutcome.SUPPORTED

    @property
    def supported(self) -> tuple[Capability, ...]:
        return tuple(
            item.capability
            for item in self.evidence
            if item.outcome is ProbeOutcome.SUPPORTED
        )

    @property
    def unresolved(self) -> tuple[Capability, ...]:
        """Capabilities that were not established, unknown or unsupported.

        Named rather than left implicit, because a policy about to give up a lever
        should be able to say which ones it gave up and why.
        """
        return tuple(
            item.capability
            for item in self.evidence
            if item.outcome is not ProbeOutcome.SUPPORTED
        )

    @property
    def disagreements(self) -> tuple[Disagreement, ...]:
        """Where the adapter claimed a lever the observation did not confirm.

        One direction only: overclaiming. An adapter that says it has a lever the
        target turns out not to have is a bug, and it is the dangerous direction,
        because a policy reading the declaration would pull a dead lever and the
        run record would show no reason why. Neither side is overwritten; both
        facts are kept, because the disagreement is the useful part.

        The opposite case, a lever the model has and the adapter never mentioned,
        is *not* listed here. A declaration says what the adapter is able to
        express, not everything the endpoint happens to support, so discovering
        something extra is a bonus rather than a conflict. Reporting it as one
        would fire on every confirmed capability for any minimally-declared
        adapter and bury the overclaim. Those are available on
        :attr:`discovered` instead.
        """
        return tuple(
            Disagreement(
                capability=item.capability,
                claimed=True,
                observed=item.outcome,
            )
            for item in self.evidence
            if item.capability in self.claimed
            and item.outcome is not ProbeOutcome.SUPPORTED
        )

    @property
    def discovered(self) -> tuple[Capability, ...]:
        """Confirmed capabilities the adapter never claimed.

        Worth surfacing when a new provider ships a feature Inferconomy did not
        know to ask for, since that is where new levers come from.
        """
        return tuple(
            item.capability
            for item in self.evidence
            if item.outcome is ProbeOutcome.SUPPORTED
            and item.capability not in self.claimed
        )

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {"client": self.client_name, "model": self.model}, sort_keys=True
            ).encode("utf-8")
        )
        return digest.hexdigest()[:16]

    def adapt(self, options: CompletionOptions) -> Adaptation:
        """Strip levers this target is not confirmed to have, recording each drop.

        An unknown lever is stripped too. Acting on an unverified lever produces
        a call whose effect cannot be attributed, and the caller has no way to
        tell that from a lever that worked, so the conservative reading is the
        only one that keeps the run record truthful.
        """
        drops: list[Degradation] = []
        sent = options

        if options.reasoning_budget is not None and not self.supports(
            Capability.REASONING_BUDGET
        ):
            drops.append(
                Degradation(
                    capability=Capability.REASONING_BUDGET,
                    requested=options.reasoning_budget,
                    fallback=FALLBACKS[Capability.REASONING_BUDGET],
                )
            )
            sent = CompletionOptions(
                model=sent.model,
                max_output_tokens=sent.max_output_tokens,
                stop=sent.stop,
                effort=sent.effort,
            )

        if options.effort is not None and not self.supports(Capability.EFFORT_CONTROL):
            drops.append(
                Degradation(
                    capability=Capability.EFFORT_CONTROL,
                    requested=options.effort,
                    fallback=FALLBACKS[Capability.EFFORT_CONTROL],
                )
            )
            sent = CompletionOptions(
                model=sent.model,
                max_output_tokens=sent.max_output_tokens,
                stop=sent.stop,
                reasoning_budget=sent.reasoning_budget,
            )

        return Adaptation(options=sent, degradations=tuple(drops))

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_name": self.client_name,
            "model": self.model,
            "calls": self.calls,
            "cache_key": self.cache_key,
            "claimed": sorted(c.value for c in self.claimed),
            "disagreements": [d.to_dict() for d in self.disagreements],
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityReport:
        try:
            return cls(
                client_name=str(data["client_name"]),
                model=str(data["model"]),
                evidence=tuple(
                    ProbeEvidence.from_dict(item) for item in data.get("evidence", ())
                ),
                claimed=frozenset(Capability(c) for c in data.get("claimed", ())),
                calls=int(data.get("calls", 0)),
            )
        except KeyError as exc:
            raise ValueError(
                f"Capability report is missing required field: {exc.args[0]!r}"
            ) from exc


async def probe_capabilities(
    client: Client,
    model: str | None = None,
    *,
    max_output_tokens: int = 16,
) -> CapabilityReport:
    """Find out what this target actually does, at runtime.

    Makes one minimal call to settle the two capabilities a response can answer,
    and delegates the rest to the adapter when it implements
    :class:`CapabilityProbeProvider`. Adapters that do not get ``UNKNOWN`` for
    those three, with a detail saying so, which is the honest outcome: the lever
    was not tested, not shown to be missing.

    Provider failures do not propagate. A probe exists to describe a degraded
    environment, so an exception raised while probing becomes an ``UNKNOWN`` with
    the error recorded, and the caller still gets a report. Probing during
    startup should not be the thing that stops the process.
    """
    resolved_model = model
    evidence: list[ProbeEvidence] = []
    calls = 0

    response = None
    probe_error: str | None = None
    try:
        response = await client.complete(
            Request.from_text(PROBE_PROMPT),
            CompletionOptions(model=model, max_output_tokens=max_output_tokens),
        )
        calls += 1
    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"

    if response is not None and response.model:
        resolved_model = response.model

    for capability in RESPONSE_OBSERVABLE:
        evidence.append(
            _observe_from_response(
                capability,
                response,
                probe_error,
                used_calls=calls,
            )
        )

    provider = client if isinstance(client, CapabilityProbeProvider) else None
    for capability in DELEGATED:
        evidence.append(await _delegate(capability, provider, resolved_model))

    return CapabilityReport(
        client_name=getattr(client, "name", type(client).__name__),
        model=resolved_model or model or "unknown",
        evidence=tuple(evidence),
        claimed=frozenset(getattr(client, "capabilities", frozenset())),
        calls=calls,
    )


def _observe_from_response(
    capability: Capability,
    response: Any,
    probe_error: str | None,
    *,
    used_calls: int,
) -> ProbeEvidence:
    """Settle a capability from a response, without reading absence as refusal.

    The two observable capabilities are handled differently on purpose, because
    the evidence is not the same kind of thing. A response with
    ``tokens_exact=False`` is the adapter *declaring* that its counts are
    estimates, which is a definitive answer and settles the question. A response
    with ``reasoning_text=None`` is silence, and silence does not settle whether
    the model can reason, only whether this prompt made it show its work. Treating
    the first as unknown would be excessive and would leave a known-degraded cost
    figure looking merely unverified; treating the second as unsupported would
    invent a fact.
    """
    if probe_error is not None or response is None:
        return ProbeEvidence(
            capability=capability,
            outcome=ProbeOutcome.UNKNOWN,
            detail=(
                f"Probe call failed, so nothing was observed: {probe_error}"
                if probe_error
                else "Probe call produced no response."
            ),
            calls=used_calls,
        )

    if capability is Capability.USAGE_REPORTING:
        if response.usage.tokens_exact:
            return ProbeEvidence(
                capability=capability,
                outcome=ProbeOutcome.SUPPORTED,
                detail="Response reported token usage as measured.",
                calls=used_calls,
            )
        return ProbeEvidence(
            capability=capability,
            outcome=ProbeOutcome.UNSUPPORTED,
            detail="Response carried no measured usage; the numbers are estimated.",
            calls=used_calls,
        )

    if response.reasoning_text:
        return ProbeEvidence(
            capability=capability,
            outcome=ProbeOutcome.SUPPORTED,
            detail="Response exposed a reasoning trace.",
            calls=used_calls,
        )
    return ProbeEvidence(
        capability=capability,
        outcome=ProbeOutcome.UNKNOWN,
        detail=(
            "No reasoning trace in the response. That is not evidence the model "
            "cannot reason, only that this prompt did not make it emit a trace."
        ),
        calls=used_calls,
    )


async def _delegate(
    capability: Capability, provider: Any, model: str | None
) -> ProbeEvidence:
    """Ask the adapter to test a capability its own response cannot reveal."""
    if provider is None:
        return ProbeEvidence(
            capability=capability,
            outcome=ProbeOutcome.UNKNOWN,
            detail=(
                "The adapter does not implement CapabilityProbeProvider, so this "
                "was never tested. Untested is not unsupported."
            ),
        )
    try:
        supported, detail = await provider.probe_capability(capability, model)
    except Exception as exc:
        return ProbeEvidence(
            capability=capability,
            outcome=ProbeOutcome.UNKNOWN,
            detail=f"Adapter probe raised {type(exc).__name__}: {exc}",
        )
    if supported is None:
        return ProbeEvidence(
            capability=capability,
            outcome=ProbeOutcome.UNKNOWN,
            detail=detail or "Adapter could not determine this.",
        )
    if supported:
        return ProbeEvidence(
            capability=capability,
            outcome=ProbeOutcome.SUPPORTED,
            detail=detail or "Adapter confirmed the endpoint accepted it.",
        )
    return ProbeEvidence(
        capability=capability,
        outcome=ProbeOutcome.UNSUPPORTED,
        detail=detail or "Endpoint rejected the request for this capability.",
    )
