"""Token accounting and provenance tests.

The property under test is not arithmetic. It is that a caller can never be led
to believe an estimate is a measurement, by any path in this module.
"""

from __future__ import annotations

import pytest

from inferconomy import (
    Message,
    OptimizationReport,
    Request,
    Usage,
    UsageBasis,
)
from inferconomy.testing import estimate_tokens, user_request
from inferconomy.tokens import (
    HeuristicTokenEstimator,
    TokenEstimator,
    estimate_input_tokens,
    estimate_usage,
)


class WordCountEstimator:
    """A stand-in for a real tokenizer, to prove the protocol is pluggable."""

    def count(self, text: str) -> int:
        return len(text.split())


class TestEstimatorProtocol:
    def test_heuristic_satisfies_the_protocol(self) -> None:
        assert isinstance(HeuristicTokenEstimator(), TokenEstimator)

    def test_third_party_estimator_satisfies_the_protocol(self) -> None:
        assert isinstance(WordCountEstimator(), TokenEstimator)

    def test_estimate_usage_accepts_a_pluggable_estimator(self) -> None:
        usage = estimate_usage(
            user_request("one two three four"),
            estimator=WordCountEstimator(),
        )
        assert usage.input_tokens == 4


class TestHeuristicEstimator:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("", 0), ("abcd", 1), ("abcde", 2), ("a" * 8, 2)],
        ids=["empty", "exact", "rounds-up", "double"],
    )
    def test_counts_by_character_length(self, text: str, expected: int) -> None:
        assert HeuristicTokenEstimator().count(text) == expected

    def test_honours_a_custom_ratio(self) -> None:
        assert HeuristicTokenEstimator(chars_per_token=2).count("abcd") == 2

    @pytest.mark.parametrize("ratio", [0, -1], ids=["zero", "negative"])
    def test_rejects_non_positive_ratio(self, ratio: int) -> None:
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            HeuristicTokenEstimator(chars_per_token=ratio)

    def test_is_deterministic(self) -> None:
        estimator = HeuristicTokenEstimator()
        assert estimator.count("stable") == estimator.count("stable")

    def test_never_claims_exactness(self) -> None:
        """A tokenizer cannot upgrade an estimate into a measurement."""
        usage = estimate_usage(user_request("a" * 40), "b" * 40)
        assert usage.tokens_exact is False
        assert usage.cost_exact is False
        assert usage.basis is UsageBasis.ESTIMATED


class TestInputTokenEstimation:
    def test_counts_every_turn_not_just_the_last(self) -> None:
        """Ignoring prior turns is a classic source of flattering under-reporting."""
        request = Request(
            messages=(
                Message(role="system", content="ab"),
                Message(role="user", content="cd"),
                Message(role="assistant", content="ef"),
            )
        )
        assert estimate_input_tokens(request) == 3

    def test_empty_content_costs_nothing(self) -> None:
        request = Request(messages=(Message(role="user", content=""),))
        assert estimate_input_tokens(request) == 0


class TestEstimateUsage:
    def test_never_marks_tokens_exact(self) -> None:
        assert estimate_usage(user_request("hi"), "there").tokens_exact is False

    def test_never_invents_a_cost(self) -> None:
        usage = estimate_usage(user_request("hi"), "there")
        assert usage.cost_usd is None
        assert usage.cost_exact is False

    def test_counts_one_call(self) -> None:
        assert estimate_usage(user_request("hi"), "there").llm_calls == 1

    def test_estimates_reasoning_separately_when_present(self) -> None:
        usage = estimate_usage(
            user_request("hi"), "answer", reasoning_text="thinking hard"
        )
        assert usage.reasoning_tokens > 0
        assert usage.reasoning_tokens <= usage.output_tokens

    def test_clamps_reasoning_that_exceeds_output(self) -> None:
        """A provider that bills reasoning inside output must not break this."""
        usage = estimate_usage(
            user_request("hi"),
            output_text="ab",
            reasoning_text="a very long chain of thought indeed",
        )
        assert usage.reasoning_tokens == usage.output_tokens

    def test_optional_reasoning_defaults_to_zero(self) -> None:
        assert estimate_usage(user_request("hi"), "answer").reasoning_tokens == 0

    def test_handles_an_empty_generation(self) -> None:
        usage = estimate_usage(user_request("hi"), "")
        assert usage.output_tokens == 0
        assert usage.input_tokens > 0


class TestProvenanceFlags:
    def test_both_flags_default_false(self) -> None:
        usage = Usage()
        assert usage.tokens_exact is False
        assert usage.cost_exact is False
        assert usage.exact is False

    def test_exact_requires_both(self) -> None:
        assert Usage(input_tokens=1, tokens_exact=True).exact is False
        assert Usage(cost_usd=1.0, cost_exact=True).exact is False
        assert Usage(
            input_tokens=1, cost_usd=1.0, tokens_exact=True, cost_exact=True
        ).exact

    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            (
                Usage(input_tokens=1, tokens_exact=True, cost_usd=1.0, cost_exact=True),
                UsageBasis.REPORTED,
            ),
            (Usage(input_tokens=1, tokens_exact=True), UsageBasis.REPORTED_UNPRICED),
            (
                Usage(input_tokens=1, cost_usd=1.0, cost_exact=True),
                UsageBasis.PRICED_FROM_ESTIMATE,
            ),
            (Usage(input_tokens=1), UsageBasis.ESTIMATED),
            (Usage(), UsageBasis.ABSENT),
        ],
        ids=["reported", "unpriced", "priced-estimate", "estimated", "absent"],
    )
    def test_basis_summarizes_provenance(
        self, usage: Usage, expected: UsageBasis
    ) -> None:
        assert usage.basis is expected

    def test_exact_tokens_without_a_price_still_keeps_the_measurement(self) -> None:
        """A real token count must not be discarded because a price is missing."""
        usage = Usage(input_tokens=100, tokens_exact=True)
        assert usage.basis is UsageBasis.REPORTED_UNPRICED
        assert usage.input_tokens == 100

    def test_zero_tokens_reported_exactly_is_not_absent(self) -> None:
        assert Usage(tokens_exact=True).basis is UsageBasis.REPORTED_UNPRICED

    def test_summing_degrades_exactness(self) -> None:
        """Escalation means several calls; one estimate contaminates the total."""
        exact = Usage(input_tokens=1, tokens_exact=True, cost_usd=1.0, cost_exact=True)
        estimated = Usage(input_tokens=1)
        assert (exact + exact).basis is UsageBasis.REPORTED
        assert (exact + estimated).basis is UsageBasis.ESTIMATED

    def test_summing_loses_cost_when_either_side_lacks_it(self) -> None:
        total = Usage(cost_usd=1.0, cost_exact=True) + Usage(input_tokens=5)
        assert total.cost_usd is None
        assert total.cost_exact is False


class TestReportProvenance:
    def test_absent_usage_is_flagged(self) -> None:
        report = OptimizationReport(strategy="direct")
        assert report.usage_basis is UsageBasis.ABSENT
        assert report.citable is False

    def test_estimated_report_is_not_citable(self) -> None:
        report = OptimizationReport(strategy="direct", usage=Usage(input_tokens=10))
        assert report.usage_basis is UsageBasis.ESTIMATED
        assert report.citable is False

    def test_only_a_fully_reported_record_is_citable(self) -> None:
        report = OptimizationReport(
            strategy="direct",
            usage=Usage(
                input_tokens=10, tokens_exact=True, cost_usd=0.01, cost_exact=True
            ),
        )
        assert report.usage_basis is UsageBasis.REPORTED
        assert report.citable is True

    def test_unpriced_report_is_not_citable(self) -> None:
        report = OptimizationReport(
            strategy="direct", usage=Usage(input_tokens=10, tokens_exact=True)
        )
        assert report.citable is False

    def test_provenance_survives_serialization(self) -> None:
        report = OptimizationReport(strategy="direct", usage=Usage(input_tokens=10))
        payload = report.to_dict()
        assert payload["usage_basis"] == "estimated"
        assert payload["citable"] is False
        restored = OptimizationReport.from_dict(payload)
        assert restored.usage_basis is report.usage_basis
        assert restored.citable is report.citable

    def test_provenance_is_derived_not_stored(self) -> None:
        """A caller cannot hand a report a provenance it does not deserve."""
        payload = OptimizationReport(strategy="direct").to_dict()
        payload["usage_basis"] = "reported"
        payload["citable"] = True
        assert OptimizationReport.from_dict(payload).citable is False


class TestTestingHelperParity:
    def test_estimate_tokens_matches_the_default_estimator(self) -> None:
        assert estimate_tokens("abcdefgh") == HeuristicTokenEstimator().count(
            "abcdefgh"
        )
