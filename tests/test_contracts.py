"""Contract tests: round-trip serialization, validation, and domain invariants.

Round-tripping is the load-bearing property here. These objects are going into
telemetry, reproducibility artifacts, and user logs, so a value that survives
``to_dict`` / ``from_dict`` is what a stored report will still mean six months
and several releases from now.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from inferconomy import (
    Capability,
    Message,
    OptimizationReport,
    Request,
    Response,
    StopReason,
    Usage,
)

# Representative instances, each with a unique id. Two share a type so that
# parametrized ids stay readable.
SAMPLES: list[tuple[str, Any, str]] = [
    ("message", Message(role="user", content="hello"), "Message"),
    (
        "request-conversation",
        Request(
            messages=(
                Message(role="system", content="be brief"),
                Message("user", "hi"),
            ),
            max_output_tokens=512,
            stop=("</done>",),
        ),
        "Request",
    ),
    (
        "request-minimal",
        Request(messages=(Message(role="user", content="hi"),)),
        "Request",
    ),
    (
        "usage-populated",
        Usage(
            input_tokens=100,
            output_tokens=40,
            reasoning_tokens=25,
            cached_input_tokens=10,
            llm_calls=2,
            cost_usd=0.0032,
            cost_exact=True,
        ),
        "Usage",
    ),
    ("usage-empty", Usage(), "Usage"),
    (
        "response-populated",
        Response(
            text="eventual consistency means...",
            usage=Usage(input_tokens=12, output_tokens=300, llm_calls=1),
            model="some-model",
            provider_finish_reason="stop",
            reasoning_text="thinking...",
            latency_ms=840.5,
        ),
        "Response",
    ),
    ("response-empty", Response(text=""), "Response"),
    (
        "report-escalated",
        OptimizationReport(
            strategy="reason_verify",
            capabilities_used=(Capability.REASONING_BUDGET, Capability.USAGE_REPORTING),
            initial_budget=400,
            additional_budget=260,
            stopped_on=StopReason.BUDGET_EXHAUSTED,
            decisions=("classified as reasoning task", "budget 400", "escalated"),
        ),
        "OptimizationReport",
    ),
    ("report-minimal", OptimizationReport(strategy="direct"), "OptimizationReport"),
]

SAMPLE_IDS = [sample[0] for sample in SAMPLES]


def _load(type_name: str) -> Any:
    return globals()[type_name]


class TestRoundTrip:
    @pytest.mark.parametrize(
        ("sample_id", "instance", "type_name"), SAMPLES, ids=SAMPLE_IDS
    )
    def test_dict_round_trip_preserves_value(
        self, sample_id: str, instance: Any, type_name: str
    ) -> None:
        assert _load(type_name).from_dict(instance.to_dict()) == instance

    @pytest.mark.parametrize(
        ("sample_id", "instance", "type_name"), SAMPLES, ids=SAMPLE_IDS
    )
    def test_dict_round_trip_survives_json(
        self, sample_id: str, instance: Any, type_name: str
    ) -> None:
        """Serialization must survive a real JSON round trip, not just a dict copy."""
        encoded = json.dumps(instance.to_dict())
        assert _load(type_name).from_dict(json.loads(encoded)) == instance

    @pytest.mark.parametrize(
        ("sample_id", "instance", "type_name"), SAMPLES, ids=SAMPLE_IDS
    )
    def test_to_dict_is_json_native(
        self, sample_id: str, instance: Any, type_name: str
    ) -> None:
        """No enum, tuple, or dataclass may leak into serialized output."""
        json.dumps(instance.to_dict(), allow_nan=False)


class TestImmutability:
    @pytest.mark.parametrize(
        ("instance", "field", "value"),
        [
            (Message(role="user", content="hi"), "content", "mutated"),
            (Request(messages=(Message("user", "hi"),)), "messages", ()),
            (Usage(input_tokens=5), "input_tokens", 99),
            (Response(text="hi"), "text", "mutated"),
            (OptimizationReport(strategy="direct"), "strategy", "mutated"),
        ],
        ids=["Message", "Request", "Usage", "Response", "OptimizationReport"],
    )
    def test_fields_cannot_be_reassigned(
        self, instance: Any, field: str, value: Any
    ) -> None:
        with pytest.raises(FrozenInstanceError):
            setattr(instance, field, value)

    def test_request_normalizes_mutable_sequences(self) -> None:
        """A caller passing lists must not be able to mutate a frozen Request."""
        messages = [Message(role="user", content="hi")]
        request = Request(messages=messages, stop=["a"])  # type: ignore[arg-type]
        messages.append(Message(role="user", content="injected"))
        assert len(request.messages) == 1
        assert isinstance(request.messages, tuple)
        assert isinstance(request.stop, tuple)


class TestRequest:
    def test_from_text_builds_single_turn(self) -> None:
        request = Request.from_text("hello")
        assert request.messages == (Message(role="user", content="hello"),)

    def test_from_text_prepends_system_turn(self) -> None:
        request = Request.from_text("hello", system="be terse")
        assert [m.role for m in request.messages] == ["system", "user"]

    def test_rejects_empty_messages(self) -> None:
        with pytest.raises(ValueError, match="at least one message"):
            Request(messages=())

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Request.from_text("")

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_non_positive_output_limit(self, limit: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            Request(messages=(Message("user", "hi"),), max_output_tokens=limit)

    def test_from_dict_rejects_bare_string_as_messages(self) -> None:
        with pytest.raises(ValueError, match="must be a sequence"):
            Request.from_dict({"messages": "not a list"})

    def test_from_dict_reports_missing_field(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'messages'"):
            Request.from_dict({})


class TestUsage:
    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            Usage(input_tokens=-1)

    def test_reasoning_tokens_are_a_subset_of_output(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed output_tokens"):
            Usage(output_tokens=10, reasoning_tokens=11)

    def test_cached_tokens_are_a_subset_of_input(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed input_tokens"):
            Usage(input_tokens=10, cached_input_tokens=11)

    def test_exact_cost_requires_a_cost(self) -> None:
        with pytest.raises(ValueError, match="requires cost_usd"):
            Usage(cost_exact=True)

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValueError, match="cost_usd must be non-negative"):
            Usage(cost_usd=-1.0)

    def test_exactness_defaults_to_false(self) -> None:
        """A library should under-claim precision, never assume a measurement."""
        assert Usage().cost_exact is False

    def test_total_tokens_excludes_reasoning_double_counting(self) -> None:
        usage = Usage(input_tokens=10, output_tokens=20, reasoning_tokens=15)
        assert usage.total_tokens == 30

    def test_addition_aggregates_across_calls(self) -> None:
        first = Usage(
            input_tokens=10, output_tokens=20, reasoning_tokens=5, llm_calls=1
        )
        second = Usage(input_tokens=7, output_tokens=3, llm_calls=1)
        total = first + second
        assert total.input_tokens == 17
        assert total.output_tokens == 23
        assert total.reasoning_tokens == 5
        assert total.llm_calls == 2

    def test_addition_conjuncts_exactness(self) -> None:
        exact = Usage(cost_usd=0.01, cost_exact=True)
        estimate = Usage(cost_usd=0.02, cost_exact=False)
        assert (exact + exact).cost_exact is True
        assert (exact + estimate).cost_exact is False

    def test_addition_propagates_unknown_cost(self) -> None:
        known = Usage(cost_usd=0.01, cost_exact=True)
        unknown = Usage()
        assert (known + unknown).cost_usd is None
        assert (known + unknown).cost_exact is False


class TestOptimizationReport:
    def test_total_budget_is_derived_not_stored(self) -> None:
        report = OptimizationReport(
            strategy="direct", initial_budget=180, additional_budget=20
        )
        assert report.total_budget == 200

    def test_escalation_is_derived_from_additional_budget(self) -> None:
        assert OptimizationReport(strategy="direct").escalated is False
        assert OptimizationReport(strategy="direct", additional_budget=1).escalated

    def test_rejects_negative_budgets(self) -> None:
        with pytest.raises(ValueError, match="initial_budget must be non-negative"):
            OptimizationReport(strategy="direct", initial_budget=-1)
        with pytest.raises(ValueError, match="additional_budget must be non-negative"):
            OptimizationReport(strategy="direct", additional_budget=-1)

    def test_reports_a_schema_version(self) -> None:
        assert OptimizationReport(strategy="direct").to_dict()["schema_version"] == 1

    def test_rejects_newer_schema_version(self) -> None:
        """Silently loading a future report would misrepresent it as this version."""
        with pytest.raises(ValueError, match="newer than this release understands"):
            OptimizationReport.from_dict(
                {"schema_version": 99, "strategy": "direct", "initial_budget": 10}
            )

    def test_rejects_unknown_capability(self) -> None:
        with pytest.raises(ValueError, match="Invalid capability"):
            OptimizationReport.from_dict(
                {
                    "strategy": "direct",
                    "capabilities_used": ["telepathy"],
                    "additional_budget": 5,
                }
            )

    @pytest.mark.parametrize(
        "match", ["Invalid stopped_on", "sufficiency"], ids=["message", "valid-values"]
    )
    def test_rejects_unknown_stop_reason(self, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            OptimizationReport.from_dict({"strategy": "direct", "stopped_on": "vibes"})

    def test_rejects_missing_strategy(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'strategy'"):
            OptimizationReport.from_dict({})

    def test_rejects_non_sequence_decisions(self) -> None:
        with pytest.raises(ValueError, match="'decisions' must be a sequence"):
            OptimizationReport.from_dict({"strategy": "direct", "decisions": "one"})


class TestEnumDecoding:
    def test_accepts_an_enum_member_instead_of_its_value(self) -> None:
        """A caller round-tripping through Python objects, not JSON, should work."""
        report = OptimizationReport.from_dict(
            {
                "strategy": "direct",
                "stopped_on": StopReason.SUFFICIENCY,
                "capabilities_used": [Capability.LOGPROBS],
            }
        )
        assert report.stopped_on is StopReason.SUFFICIENCY
        assert report.capabilities_used == (Capability.LOGPROBS,)

    @pytest.mark.parametrize(
        "bad", [None, 42, ["sufficiency"]], ids=["none", "int", "list"]
    )
    def test_rejects_non_scalar_enum_values(self, bad: object) -> None:
        with pytest.raises(ValueError, match="Invalid stopped_on"):
            OptimizationReport.from_dict({"strategy": "direct", "stopped_on": bad})


class TestFromDictRequiredFields:
    def test_message_reports_missing_field(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'content'"):
            Message.from_dict({"role": "user"})

    def test_response_reports_missing_field(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'text'"):
            Response.from_dict({"model": "some-model"})


class TestStopReason:
    def test_stop_reason_is_distinct_from_provider_finish_reason(self) -> None:
        """Our decision and the model's behaviour are separate facts."""
        response = Response(text="x", provider_finish_reason="length")
        report = OptimizationReport(
            strategy="direct", stopped_on=StopReason.BUDGET_EXHAUSTED
        )
        assert response.provider_finish_reason == "length"
        assert report.stopped_on.value == "budget_exhausted"
