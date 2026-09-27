"""Tests for runtime capability probing.

The load-bearing tests here are the ones about what the probe refuses to claim.
A probe that reports a full set of supports on a target that has none of them is
worse than no probe, because the policy will then pull levers that do nothing and
the run record will not say so.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from inferconomy.capabilities import (
    FALLBACKS,
    PROBE_PROMPT,
    Adaptation,
    CapabilityProbeProvider,
    CapabilityReport,
    Degradation,
    Disagreement,
    ProbeEvidence,
    ProbeOutcome,
    probe_capabilities,
)
from inferconomy.client import CompletionOptions
from inferconomy.contracts import Capability, Request, Response, Usage
from inferconomy.testing import FakeClient


def run(coro: Any) -> Any:
    """Drive one coroutine to completion.

    asyncio.run in a synchronous test keeps the async API under test without
    adding a pytest plugin and its configuration to the dev dependencies.
    """
    return asyncio.run(coro)


def probe(client: Any, model: str | None = None) -> Any:
    return probe_capabilities(client, model)


class ProbingClient:
    """A client told exactly what to report, for both halves of the probe."""

    def __init__(
        self,
        *,
        capabilities: frozenset[Capability] = frozenset(),
        usage: Usage | None = None,
        reasoning_text: str | None = None,
        answer: str = "391",
        model: str = "probe-target",
        delegate: dict[Capability, tuple[bool | None, str]] | None = None,
        complete_error: Exception | None = None,
        delegate_error: Capability | None = None,
    ) -> None:
        self._capabilities = frozenset(capabilities)
        self._usage = usage if usage is not None else Usage(tokens_exact=True)
        self._reasoning_text = reasoning_text
        self._answer = answer
        self._model = model
        self._delegate = dict(delegate or {})
        self._complete_error = complete_error
        self._delegate_error = delegate_error
        self.calls: list[CompletionOptions] = []
        self.probed: list[Capability] = []

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def complete(self, request: Request, options: CompletionOptions) -> Response:
        self.calls.append(options)
        if self._complete_error is not None:
            raise self._complete_error
        return Response(
            text=self._answer,
            usage=self._usage,
            model=self._model,
            reasoning_text=self._reasoning_text,
        )

    async def probe_capability(
        self, capability: Capability, model: str | None
    ) -> tuple[bool | None, str]:
        self.probed.append(capability)
        if self._delegate_error is capability:
            raise RuntimeError("endpoint timed out")
        return self._delegate.get(capability, (None, "adapter did not test this"))


class TestCoverage:
    """The five named capabilities are all reported, each independently."""

    def test_reports_every_named_capability(self) -> None:
        report = run(probe(ProbingClient()))
        assert {item.capability for item in report.evidence} == set(Capability)

    def test_every_capability_gets_an_outcome(self) -> None:
        report = run(probe(ProbingClient()))
        for capability in Capability:
            assert isinstance(report.outcome(capability), ProbeOutcome)

    def test_an_unprobed_capability_is_unknown_rather_than_missing(self) -> None:
        report = run(probe(ProbingClient(), model=None))
        assert report.outcome(Capability.LOGPROBS) is ProbeOutcome.UNKNOWN


class TestAbsenceIsNotEvidence:
    """The reason the outcome is three-valued rather than two."""

    def test_no_reasoning_trace_is_unknown_not_unsupported(self) -> None:
        report = run(probe(ProbingClient(reasoning_text=None)))
        assert report.outcome(Capability.VISIBLE_COT) is ProbeOutcome.UNKNOWN
        assert not report.supports(Capability.VISIBLE_COT)

    def test_the_detail_says_why_absence_proves_nothing(self) -> None:
        detail = run(probe(ProbingClient(reasoning_text=None))).detail(
            Capability.VISIBLE_COT
        )
        assert "not evidence" in detail

    def test_a_reasoning_trace_confirms_visible_cot(self) -> None:
        report = run(probe(ProbingClient(reasoning_text="17*23=391")))
        assert report.supports(Capability.VISIBLE_COT)

    def test_estimated_usage_is_unsupported_not_unknown(self) -> None:
        """A declaration is evidence; silence is not.

        tokens_exact=False is the adapter saying its counts are estimates, which
        settles the question. Reporting UNKNOWN here would leave a known-degraded
        cost figure looking merely unverified.
        """
        report = run(probe(ProbingClient(usage=Usage(tokens_exact=False))))
        assert report.outcome(Capability.USAGE_REPORTING) is ProbeOutcome.UNSUPPORTED

    def test_measured_usage_confirms_usage_reporting(self) -> None:
        report = run(probe(ProbingClient(usage=Usage(tokens_exact=True))))
        assert report.supports(Capability.USAGE_REPORTING)

    def test_an_adapter_that_cannot_test_reports_unknown_not_unsupported(self) -> None:
        """The FakeClient implements no probe method, so nothing was tested.

        Reporting "unsupported" here would attribute a fact about the model to a
        limitation of the adapter, which is a different claim entirely.
        """
        report = run(probe(FakeClient()))
        for capability in (
            Capability.REASONING_BUDGET,
            Capability.EFFORT_CONTROL,
            Capability.LOGPROBS,
        ):
            assert report.outcome(capability) is ProbeOutcome.UNKNOWN
            assert not report.supports(capability)

    def test_untested_detail_says_untested_is_not_unsupported(self) -> None:
        report = run(probe(FakeClient()))
        for capability in (Capability.LOGPROBS, Capability.EFFORT_CONTROL):
            assert "not unsupported" in report.detail(capability)

    def test_a_client_without_a_probe_method_is_never_asked_to_probe(self) -> None:
        client = FakeClient()
        report = run(probe(client))
        assert client.calls
        assert not hasattr(client, "probe_capability")
        assert report.outcome(Capability.LOGPROBS) is ProbeOutcome.UNKNOWN


class TestAdapterAnswers:
    """A null answer from the adapter is respected; only the endpoint may refuse."""

    def test_adapter_false_may_claim_unsupported(self) -> None:
        client = ProbingClient(
            delegate={Capability.LOGPROBS: (False, "endpoint said no")}
        )
        report = run(probe(client))
        assert report.outcome(Capability.LOGPROBS) is ProbeOutcome.UNSUPPORTED
        assert "endpoint said no" in report.detail(Capability.LOGPROBS)

    def test_adapter_true_confirms_the_capability(self) -> None:
        client = ProbingClient(
            delegate={Capability.EFFORT_CONTROL: (True, "endpoint accepted effort")}
        )
        assert run(probe(client)).supports(Capability.EFFORT_CONTROL)

    def test_adapter_none_stays_unknown(self) -> None:
        client = ProbingClient(delegate={Capability.LOGPROBS: (None, "no signal")})
        report = run(probe(client))
        assert report.outcome(Capability.LOGPROBS) is ProbeOutcome.UNKNOWN

    def test_adapter_true_gets_a_default_detail(self) -> None:
        client = ProbingClient(delegate={Capability.LOGPROBS: (True, "")})
        assert run(probe(client)).detail(Capability.LOGPROBS).strip()

    def test_adapter_false_gets_a_default_detail(self) -> None:
        client = ProbingClient(delegate={Capability.LOGPROBS: (False, "")})
        assert run(probe(client)).detail(Capability.LOGPROBS).strip()

    def test_a_raising_probe_does_not_take_the_report_down_with_it(self) -> None:
        client = ProbingClient(delegate_error=Capability.REASONING_BUDGET)
        report = run(probe(client))
        assert report.outcome(Capability.REASONING_BUDGET) is ProbeOutcome.UNKNOWN
        assert "RuntimeError" in report.detail(Capability.REASONING_BUDGET)
        assert report.outcome(Capability.LOGPROBS) is ProbeOutcome.UNKNOWN

    def test_only_capabilities_a_response_cannot_settle_are_delegated(self) -> None:
        client = ProbingClient()
        run(probe(client))
        assert set(client.probed) == {
            Capability.REASONING_BUDGET,
            Capability.EFFORT_CONTROL,
            Capability.LOGPROBS,
        }

    def test_the_adapter_is_handed_the_resolved_model(self) -> None:
        seen: list[str | None] = []

        class Recording(ProbingClient):
            async def probe_capability(
                self, capability: Capability, model: str | None
            ) -> tuple[bool | None, str]:
                seen.append(model)
                return None, "no"

        run(probe(Recording(model="served-model")))
        assert seen == ["served-model"] * 3


class TestFailedProbeCall:
    """Probing describes a degraded environment, so it must not raise."""

    def test_yields_unknown_everywhere_rather_than_an_exception(self) -> None:
        report = run(probe(ProbingClient(complete_error=RuntimeError("401"))))
        for capability in Capability:
            assert report.outcome(capability) is ProbeOutcome.UNKNOWN

    def test_records_the_error_text(self) -> None:
        report = run(probe(ProbingClient(complete_error=RuntimeError("401"))))
        assert "401" in report.detail(Capability.USAGE_REPORTING)

    def test_still_probes_the_delegated_capabilities(self) -> None:
        client = ProbingClient(complete_error=RuntimeError("401"))
        run(probe(client))
        assert len(client.probed) == 3


class TestDocumentedFallbacks:
    """Every capability needs a consequence someone can read."""

    def test_every_capability_has_a_documented_fallback(self) -> None:
        assert set(FALLBACKS) == set(Capability)

    @pytest.mark.parametrize("capability", list(Capability))
    def test_no_fallback_is_empty(self, capability: Capability) -> None:
        assert FALLBACKS[capability].strip()

    @pytest.mark.parametrize("capability", list(Capability))
    def test_every_fallback_says_what_actually_happens(
        self, capability: Capability
    ) -> None:
        """A fallback that does not describe a consequence is a shrug."""
        assert len(FALLBACKS[capability].split()) > 8

    def test_the_logprobs_fallback_names_the_lost_signal(self) -> None:
        assert "exit" in FALLBACKS[Capability.LOGPROBS]

    def test_the_usage_fallback_says_cost_becomes_an_estimate(self) -> None:
        assert "estimate" in FALLBACKS[Capability.USAGE_REPORTING]

    def test_the_cot_fallback_says_the_trace_is_gone(self) -> None:
        assert "final" in FALLBACKS[Capability.VISIBLE_COT]

    def test_fallbacks_cannot_be_reassigned(self) -> None:
        with pytest.raises(TypeError):
            FALLBACKS[Capability.LOGPROBS] = "changed"  # type: ignore[index]


class TestAdaptation:
    """Unconfirmed levers are dropped, and the drop is recorded."""

    def test_a_confirmed_lever_is_sent_untouched(self) -> None:
        client = ProbingClient(
            delegate={
                Capability.REASONING_BUDGET: (True, "ok"),
                Capability.EFFORT_CONTROL: (True, "ok"),
            }
        )
        adaptation = run(probe(client)).adapt(
            CompletionOptions(reasoning_budget=512, effort="high")
        )
        assert adaptation.options.reasoning_budget == 512
        assert adaptation.options.effort == "high"
        assert not adaptation.degraded

    def test_an_unconfirmed_lever_is_dropped_and_recorded(self) -> None:
        adaptation = run(probe(ProbingClient())).adapt(
            CompletionOptions(effort="high", reasoning_budget=512)
        )
        assert adaptation.options.effort is None
        assert adaptation.options.reasoning_budget is None
        assert adaptation.degraded

    def test_an_unknown_lever_is_dropped_too(self) -> None:
        """Unverified is not usable.

        Pulling an unconfirmed lever produces a call whose effect cannot be
        attributed, and the caller cannot tell that from a lever that worked.
        """
        report = run(probe(ProbingClient()))
        assert report.outcome(Capability.EFFORT_CONTROL) is ProbeOutcome.UNKNOWN
        assert report.adapt(CompletionOptions(effort="low")).options.effort is None

    def test_a_definitively_unsupported_lever_is_dropped(self) -> None:
        client = ProbingClient(delegate={Capability.EFFORT_CONTROL: (False, "no")})
        report = run(probe(client))
        assert report.outcome(Capability.EFFORT_CONTROL) is ProbeOutcome.UNSUPPORTED
        assert report.adapt(CompletionOptions(effort="low")).options.effort is None

    def test_each_drop_carries_its_documented_fallback(self) -> None:
        adaptation = run(probe(ProbingClient())).adapt(
            CompletionOptions(effort="high", reasoning_budget=256)
        )
        for degradation in adaptation.degradations:
            assert degradation.fallback == FALLBACKS[degradation.capability]
            assert degradation.requested is not None

    def test_a_dropped_lever_records_what_was_asked_for(self) -> None:
        adaptation = run(probe(ProbingClient())).adapt(CompletionOptions(effort="high"))
        assert adaptation.degradations[0].capability is Capability.EFFORT_CONTROL
        assert adaptation.degradations[0].requested == "high"

    def test_both_levers_can_be_dropped_at_once(self) -> None:
        adaptation = run(probe(ProbingClient())).adapt(
            CompletionOptions(effort="high", reasoning_budget=256)
        )
        assert {d.capability for d in adaptation.degradations} == {
            Capability.EFFORT_CONTROL,
            Capability.REASONING_BUDGET,
        }

    def test_a_dropped_lever_does_not_take_its_neighbour_with_it(self) -> None:
        client = ProbingClient(delegate={Capability.EFFORT_CONTROL: (True, "ok")})
        adaptation = run(probe(client)).adapt(
            CompletionOptions(effort="high", reasoning_budget=256)
        )
        assert adaptation.options.effort == "high"
        assert adaptation.options.reasoning_budget is None
        assert len(adaptation.degradations) == 1

    def test_options_that_are_not_levers_survive(self) -> None:
        adaptation = run(probe(ProbingClient())).adapt(
            CompletionOptions(
                model="m", max_output_tokens=64, stop=("X",), effort="high"
            )
        )
        assert adaptation.options.model == "m"
        assert adaptation.options.max_output_tokens == 64
        assert adaptation.options.stop == ("X",)

    def test_no_levers_requested_means_no_degradation(self) -> None:
        report = run(probe(ProbingClient()))
        assert not report.adapt(CompletionOptions(max_output_tokens=8)).degraded

    def test_adaptation_does_not_mutate_the_original_options(self) -> None:
        options = CompletionOptions(effort="high")
        run(probe(ProbingClient())).adapt(options)
        assert options.effort == "high"


class TestDisagreements:
    """A static claim and a runtime observation that differ are both worth keeping."""

    def test_an_overclaiming_adapter_is_surfaced(self) -> None:
        client = ProbingClient(
            capabilities=frozenset({Capability.LOGPROBS}),
            delegate={Capability.LOGPROBS: (None, "no")},
        )
        report = run(probe(client))
        claim = next(
            d for d in report.disagreements if d.capability is Capability.LOGPROBS
        )
        assert claim.claimed is True
        assert claim.observed is ProbeOutcome.UNKNOWN

    def test_an_unclaimed_capability_the_model_has_is_discovered_not_disputed(
        self,
    ) -> None:
        """A discovery is a bonus, not a conflict.

        Flagging it as a disagreement would fire on every confirmed capability
        for any adapter with a short declaration, and bury the overclaim that
        actually matters.
        """
        client = ProbingClient(delegate={Capability.EFFORT_CONTROL: (True, "ok")})
        report = run(probe(client))
        assert Capability.EFFORT_CONTROL in report.discovered
        assert not any(
            d.capability is Capability.EFFORT_CONTROL for d in report.disagreements
        )

    def test_an_agreeing_adapter_raises_no_disagreement(self) -> None:
        client = ProbingClient(
            capabilities=frozenset({Capability.LOGPROBS}),
            delegate={Capability.LOGPROBS: (True, "ok")},
        )
        assert not run(probe(client)).disagreements

    def test_a_declared_capability_the_model_lacks_is_surfaced(self) -> None:
        client = ProbingClient(
            capabilities=frozenset({Capability.EFFORT_CONTROL}),
            delegate={Capability.EFFORT_CONTROL: (False, "rejected")},
        )
        report = run(probe(client))
        assert any(
            d.capability is Capability.EFFORT_CONTROL for d in report.disagreements
        )

    def test_a_client_declaring_nothing_and_being_asked_nothing_agrees(self) -> None:
        client = ProbingClient(capabilities=frozenset({Capability.USAGE_REPORTING}))
        report = run(probe(client))
        assert report.supports(Capability.USAGE_REPORTING)
        assert not report.disagreements


class TestCost:
    """Probing is not free, and a report that hid its cost would look free."""

    def test_the_probe_makes_exactly_one_call(self) -> None:
        client = ProbingClient()
        run(probe(client))
        assert len(client.calls) == 1

    def test_the_probe_asks_for_a_small_allowance(self) -> None:
        client = ProbingClient()
        run(probe(client))
        assert client.calls[0].max_output_tokens == 16

    def test_the_allowance_is_configurable(self) -> None:
        client = ProbingClient()
        run(probe_capabilities(client, max_output_tokens=8))
        assert client.calls[0].max_output_tokens == 8

    def test_the_prompt_is_one_a_reasoning_model_would_reason_about(self) -> None:
        """A greeting cannot distinguish 'does not reason' from 'did not need to'."""
        assert PROBE_PROMPT.strip().endswith("number only.")
        assert any(char.isdigit() for char in PROBE_PROMPT)

    def test_the_prompt_is_not_empty(self) -> None:
        assert PROBE_PROMPT.strip()

    def test_the_report_counts_the_calls_it_spent(self) -> None:
        assert run(probe(ProbingClient())).calls == 1

    def test_a_failed_call_is_not_counted_as_spent(self) -> None:
        report = run(probe(ProbingClient(complete_error=RuntimeError("x"))))
        assert report.calls == 0

    def test_the_probe_sends_no_levers_of_its_own(self) -> None:
        """A probe asking for effort would taint the test of whether effort works."""
        client = ProbingClient()
        run(probe(client))
        assert client.calls[0].effort is None
        assert client.calls[0].reasoning_budget is None


class TestCaching:
    """Probing on every request would cost more than it saves."""

    def test_the_cache_key_is_stable_for_the_same_pair(self) -> None:
        one = run(probe(ProbingClient(model="m")))
        two = run(probe(ProbingClient(model="m")))
        assert one.cache_key == two.cache_key

    def test_a_failed_probe_does_not_reuse_the_requested_model(self) -> None:
        """A call that never reached the endpoint cannot say which model served it.

        Keying the report on the model that was asked for would let a failed
        probe seed the cache with a model it never actually reached.
        """
        failed = run(probe(ProbingClient(model="m", complete_error=RuntimeError("x"))))
        assert failed.model == "unknown"
        assert failed.cache_key != run(probe(ProbingClient(model="m"))).cache_key

    def test_a_report_for_one_model_does_not_speak_for_another(self) -> None:
        evidence = (ProbeEvidence(Capability.LOGPROBS, ProbeOutcome.UNKNOWN, "no"),)
        one = CapabilityReport("c", "model-a", evidence)
        two = CapabilityReport("c", "model-b", evidence)
        assert one.cache_key != two.cache_key

    def test_a_report_for_one_adapter_does_not_speak_for_another(self) -> None:
        evidence = (ProbeEvidence(Capability.LOGPROBS, ProbeOutcome.UNKNOWN, "no"),)
        one = CapabilityReport("adapter-a", "m", evidence)
        two = CapabilityReport("adapter-b", "m", evidence)
        assert one.cache_key != two.cache_key

    def test_the_cache_key_is_short_enough_to_log(self) -> None:
        assert len(CapabilityReport("c", "m", ()).cache_key) == 16


class TestUnresolved:
    """A policy about to give up a lever should be able to name what it gave up."""

    def test_unresolved_lists_what_the_policy_gave_up(self) -> None:
        report = run(probe(ProbingClient()))
        assert set(report.unresolved) == {
            c for c in Capability if report.outcome(c) is not ProbeOutcome.SUPPORTED
        }

    def test_supported_lists_only_confirmed_capabilities(self) -> None:
        report = run(probe(ProbingClient()))
        assert set(report.supported) == {c for c in Capability if report.supports(c)}

    def test_a_fully_capable_client_resolves_nothing_as_unresolved(self) -> None:
        client = ProbingClient(
            capabilities=frozenset(Capability),
            usage=Usage(tokens_exact=True),
            reasoning_text="thinking",
            delegate={c: (True, "ok") for c in Capability},
        )
        report = run(probe(client))
        assert set(report.supported) == set(Capability)
        assert not report.unresolved

    def test_supports_is_false_for_an_unprobed_capability(self) -> None:
        report = CapabilityReport("c", "m", ())
        assert not report.supports(Capability.LOGPROBS)
        assert report.detail(Capability.LOGPROBS) == "Not probed."


class TestSerialization:
    def test_a_report_round_trips(self) -> None:
        original = run(probe(ProbingClient(capabilities=frozenset(Capability))))
        restored = CapabilityReport.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )
        assert restored.to_dict() == original.to_dict()

    def test_a_restored_report_answers_the_same_questions(self) -> None:
        original = run(probe(ProbingClient()))
        restored = CapabilityReport.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )
        assert restored.supports(Capability.USAGE_REPORTING)
        assert restored.cache_key == original.cache_key

    def test_the_payload_carries_the_disagreements_with_the_answers(self) -> None:
        client = ProbingClient(
            capabilities=frozenset({Capability.LOGPROBS}),
            delegate={Capability.LOGPROBS: (False, "no")},
        )
        payload = run(probe(client)).to_dict()
        assert payload["disagreements"][0]["capability"] == "logprobs"

    def test_a_report_from_dict_requires_a_name_and_model(self) -> None:
        with pytest.raises(ValueError, match="client_name"):
            CapabilityReport.from_dict({"model": "m"})
        with pytest.raises(ValueError, match="model"):
            CapabilityReport.from_dict({"client_name": "c"})

    def test_a_blank_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="model"):
            CapabilityReport("c", "  ", ())

    def test_a_blank_client_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="client_name"):
            CapabilityReport("  ", "m", ())

    def test_evidence_from_dict_rejects_a_missing_field(self) -> None:
        with pytest.raises(ValueError, match="outcome"):
            ProbeEvidence.from_dict({"capability": "logprobs"})

    def test_evidence_from_dict_rejects_a_missing_capability(self) -> None:
        with pytest.raises(ValueError, match="capability"):
            ProbeEvidence.from_dict({"outcome": "supported"})

    def test_evidence_from_dict_rejects_an_unknown_capability(self) -> None:
        with pytest.raises(ValueError, match="not usable"):
            ProbeEvidence.from_dict(
                {"capability": "telepathy", "outcome": "supported", "detail": "x"}
            )


class TestStructure:
    def test_a_fully_loaded_client_is_recognised_as_a_probe_provider(self) -> None:
        assert isinstance(ProbingClient(), CapabilityProbeProvider)

    def test_a_plain_client_is_not_mistaken_for_a_probe_provider(self) -> None:
        assert not isinstance(FakeClient(), CapabilityProbeProvider)

    def test_evidence_is_immutable(self) -> None:
        item = ProbeEvidence(Capability.LOGPROBS, ProbeOutcome.UNKNOWN, "no")
        with pytest.raises(AttributeError):
            item.outcome = ProbeOutcome.SUPPORTED  # type: ignore[misc]

    def test_report_evidence_is_coerced_to_a_tuple(self) -> None:
        evidence = [ProbeEvidence(Capability.LOGPROBS, ProbeOutcome.UNKNOWN, "no")]
        report = CapabilityReport("c", "m", evidence)  # type: ignore[arg-type]
        assert isinstance(report.evidence, tuple)

    def test_degradation_serializes_the_capability_and_the_reason(self) -> None:
        payload = Degradation(
            Capability.LOGPROBS, True, FALLBACKS[Capability.LOGPROBS]
        ).to_dict()
        assert payload["capability"] == "logprobs"
        assert payload["fallback"] == FALLBACKS[Capability.LOGPROBS]

    def test_adaptation_serialization_reports_degraded_state(self) -> None:
        adaptation = Adaptation(
            options=CompletionOptions(),
            degradations=(
                Degradation(
                    Capability.EFFORT_CONTROL,
                    "high",
                    FALLBACKS[Capability.EFFORT_CONTROL],
                ),
            ),
        )
        payload = adaptation.to_dict()
        assert payload["degraded"] is True
        assert payload["degradations"][0]["requested"] == "high"

    def test_an_undegraded_adaptation_says_so(self) -> None:
        assert Adaptation(options=CompletionOptions()).to_dict()["degraded"] is False

    def test_adaptation_coerces_its_degradations_to_a_tuple(self) -> None:
        adaptation = Adaptation(
            options=CompletionOptions(),
            degradations=[Degradation(Capability.LOGPROBS, True, "x")],  # type: ignore[arg-type]
        )
        assert isinstance(adaptation.degradations, tuple)

    def test_disagreement_serializes_both_sides(self) -> None:
        payload = Disagreement(
            Capability.LOGPROBS, True, ProbeOutcome.UNKNOWN
        ).to_dict()
        assert payload == {
            "capability": "logprobs",
            "claimed": True,
            "observed": "unknown",
        }
