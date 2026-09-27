"""Oracle budgeter tests.

The oracle exists to tell a good result from a mediocre one, which only works if
it cannot be mistaken for an achievement. So the property under test is that the
bound is arithmetically correct and that every reason it is unattainable stays
attached to it, survives serialization, and blocks citability.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from inferconomy import UsageBasis
from inferconomy.benchmark import (
    ORACLE_CAVEATS,
    MetricRow,
    OracleResult,
    OracleRow,
    RunResult,
    compute_oracle,
)

MODEL = "bench-model"


def a_row(
    task_id: str = "t",
    *,
    category: str = "summarization",
    repetition: int = 0,
    budget: int = 64,
    cost_usd: float | None = 0.01,
    citable: bool = True,
) -> MetricRow:
    return MetricRow(
        task_id=task_id,
        category=category,
        repetition=repetition,
        strategy="direct",
        budget=budget,
        input_tokens=10,
        output_tokens=20,
        reasoning_tokens=0,
        total_tokens=30,
        llm_calls=1,
        cost_usd=cost_usd,
        usage_basis=UsageBasis.REPORTED,
        citable=citable,
        latency_ms=100.0,
    )


def a_run(
    *rows: MetricRow,
    budget: int = 64,
    strategy: str = "direct",
    version: str = "test-1",
    fingerprint: str = "fp",
) -> RunResult:
    return RunResult(
        config_fingerprint=fingerprint,
        strategy=strategy,
        budget=budget,
        price_table_version=version,
        price_table_verified=True,
        rows=tuple(rows),
    )


def two_runs() -> tuple[RunResult, RunResult]:
    """Baseline at 64 costs 0.01; a cheaper run at 16 costs 0.004."""
    baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
    cheaper = a_run(a_row("t1", budget=16, cost_usd=0.004), budget=16)
    return baseline, cheaper


class TestBoundArithmetic:
    def test_reports_the_cheapest_cost_per_task(self) -> None:
        baseline, cheaper = two_runs()

        result = compute_oracle([baseline, cheaper])

        assert result.rows[0].oracle_cost_usd == pytest.approx(0.004)
        assert result.rows[0].chosen_budget == 16
        assert result.max_savings_usd == pytest.approx(0.006)
        assert result.max_savings_fraction == pytest.approx(0.6)

    def test_savings_are_never_negative(self) -> None:
        """The baseline is its own candidate, so it can never lose to itself."""
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
        pricier = a_run(a_row("t1", budget=128, cost_usd=0.05), budget=128)

        result = compute_oracle([baseline, pricier])

        assert result.rows[0].savings_usd == 0.0
        assert result.rows[0].chosen_budget == 64

    def test_fraction_never_exceeds_one(self) -> None:
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
        free = a_run(a_row("t1", budget=1, cost_usd=0.0), budget=1)

        assert compute_oracle([baseline, free]).max_savings_fraction == pytest.approx(
            1.0
        )

    def test_zero_cost_baseline_divides_by_nothing(self) -> None:
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.0), budget=64)
        other = a_run(a_row("t1", budget=1, cost_usd=0.0), budget=1)

        result = compute_oracle([baseline, other])

        assert result.rows[0].savings_fraction == 0.0
        assert result.max_savings_fraction == 0.0

    def test_ties_resolve_to_the_smaller_budget(self) -> None:
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
        tied_small = a_run(a_row("t1", budget=8, cost_usd=0.004), budget=8)
        tied_large = a_run(a_row("t1", budget=32, cost_usd=0.004), budget=32)

        result = compute_oracle([baseline, tied_large, tied_small])

        assert result.rows[0].chosen_budget == 8
        assert result.rows[0].runs_considered == 3

    def test_identical_runs_report_zero_saving(self) -> None:
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
        clone = replace(baseline, budget=64)

        assert compute_oracle([baseline, clone]).max_savings_usd == pytest.approx(0.0)

    def test_allocates_per_task_rather_than_a_single_budget(self) -> None:
        """The whole point: different tasks get different budgets."""
        baseline = a_run(
            a_row("hard", category="reasoning", budget=64, cost_usd=0.02),
            a_row("easy", category="summarization", budget=64, cost_usd=0.02),
            budget=64,
        )
        split = a_run(
            a_row("hard", category="reasoning", budget=64, cost_usd=0.02),
            a_row("easy", category="summarization", budget=4, cost_usd=0.001),
            budget=4,
        )

        result = compute_oracle([baseline, split])
        by_task = {row.task_id: row for row in result.rows}

        assert by_task["hard"].savings_usd == pytest.approx(0.0)
        assert by_task["easy"].savings_usd == pytest.approx(0.019)
        assert result.max_savings_fraction == pytest.approx(0.475)

    def test_sums_across_repetitions(self) -> None:
        baseline = a_run(
            a_row("t1", repetition=0, budget=64, cost_usd=0.01),
            a_row("t1", repetition=1, budget=64, cost_usd=0.01),
            budget=64,
        )
        cheaper = a_run(
            a_row("t1", repetition=0, budget=8, cost_usd=0.005),
            a_row("t1", repetition=1, budget=8, cost_usd=0.005),
            budget=8,
        )

        result = compute_oracle([baseline, cheaper])

        assert len(result.rows) == 2
        assert result.max_savings_usd == pytest.approx(0.01)

    def test_explicit_baseline_is_used_over_run_order(self) -> None:
        baseline, cheaper = two_runs()

        result = compute_oracle([cheaper, baseline], baseline=cheaper)

        assert result.baseline_budget == 16
        assert result.max_savings_usd == pytest.approx(0.0)


class TestUnattainabilityLabelling:
    def test_never_claims_attainability(self) -> None:
        result = compute_oracle(list(two_runs()))

        assert result.attainable is False
        assert result.to_dict()["attainable"] is False

    def test_attainability_survives_deserialization(self) -> None:
        """A result read from a file must not arrive claiming it was achieved."""
        restored = OracleResult.from_dict(
            json.loads(json.dumps(compute_oracle(list(two_runs())).to_dict()))
        )

        assert restored.attainable is False

    def test_caveats_travel_with_the_data(self) -> None:
        result = compute_oracle(list(two_runs()))

        assert result.caveats == ORACLE_CAVEATS
        assert len(result.caveats) >= 3
        assert any("quality" in caveat.lower() for caveat in result.caveats)
        assert any("foresight" in caveat.lower() for caveat in result.caveats)

    def test_caveats_are_in_the_serialized_payload(self) -> None:
        """A caveat left in a docstring is a caveat nobody reads downstream."""
        payload = compute_oracle(list(two_runs())).to_dict()

        assert payload["caveats"] == list(ORACLE_CAVEATS)
        assert json.dumps(payload).count("foresight") == 1

    def test_caveats_survive_deserialization(self) -> None:
        restored = OracleResult.from_dict(
            json.loads(json.dumps(compute_oracle(list(two_runs())).to_dict()))
        )

        assert restored.caveats == ORACLE_CAVEATS

    def test_a_hand_written_attainability_claim_is_overwritten(self) -> None:
        data = compute_oracle(list(two_runs())).to_dict()
        data["attainable"] = True

        restored = OracleResult.from_dict(data)

        assert restored.attainable is False
        assert restored.to_dict()["attainable"] is False


class TestCitability:
    def test_estimated_costs_do_not_yield_a_citable_saving(self) -> None:
        """A saving derived from estimates is an estimate of an estimate."""
        baseline = a_run(
            a_row("t1", budget=64, cost_usd=0.01, citable=False), budget=64
        )
        cheaper = a_run(a_row("t1", budget=8, cost_usd=0.004, citable=False), budget=8)

        result = compute_oracle([baseline, cheaper])

        assert result.rows[0].citable is False
        assert result.citable is False
        assert result.max_savings_usd == pytest.approx(0.006)

    def test_fully_measured_runs_are_citable(self) -> None:
        assert compute_oracle(list(two_runs())).citable is True

    def test_a_single_unmeasured_task_blocks_citability(self) -> None:
        baseline = a_run(
            a_row("t1", budget=64, cost_usd=0.01),
            a_row("t2", budget=64, cost_usd=0.01),
            budget=64,
        )
        cheaper = a_run(
            a_row("t1", budget=8, cost_usd=0.004),
            a_row("t2", budget=8, cost_usd=0.004, citable=False),
            budget=8,
        )

        assert compute_oracle([baseline, cheaper]).citable is False

    def test_an_empty_result_is_not_citable(self) -> None:
        result = OracleResult(
            baseline_fingerprint="fp", baseline_budget=64, price_table_version="v"
        )

        assert result.citable is False


class TestIncomparableTasks:
    def test_unpriced_rows_are_excluded_and_counted(self) -> None:
        baseline = a_run(
            a_row("t1", budget=64, cost_usd=0.01),
            a_row("t2", budget=64, cost_usd=None),
            budget=64,
        )
        cheaper = a_run(
            a_row("t1", budget=8, cost_usd=0.004),
            a_row("t2", budget=8, cost_usd=None),
            budget=8,
        )

        result = compute_oracle([baseline, cheaper])
        dropped = next(row for row in result.rows if row.task_id == "t2")

        assert dropped.comparable is False
        assert result.incomparable_rows == 1
        assert result.citable is False

    def test_dropped_tasks_do_not_silently_understate_the_bound(self) -> None:
        baseline = a_run(
            a_row("t1", budget=64, cost_usd=0.01),
            a_row("t2", budget=64, cost_usd=0.01),
            budget=64,
        )
        cheaper = a_run(
            a_row("t1", budget=8, cost_usd=0.004),
            a_row("t2", budget=8, cost_usd=None),
            budget=8,
        )

        result = compute_oracle([baseline, cheaper])

        # t1 alone, with t2 reported as dropped rather than priced at zero.
        assert result.baseline_cost_usd == pytest.approx(0.01)
        assert result.incomparable_rows == 1
        assert len(result.rows) == 2

    def test_comparable_rows_excludes_dropped_tasks(self) -> None:
        baseline = a_run(
            a_row("t1", budget=64, cost_usd=0.01),
            a_row("t2", budget=64, cost_usd=None),
            budget=64,
        )
        cheaper = a_run(
            a_row("t1", budget=8, cost_usd=0.004),
            a_row("t2", budget=8, cost_usd=None),
            budget=8,
        )

        result = compute_oracle([baseline, cheaper])

        assert [row.task_id for row in result.comparable_rows] == ["t1"]


class TestCategoryReporting:
    def test_reports_per_category(self) -> None:
        baseline = a_run(
            a_row("a", category="code", budget=64, cost_usd=0.01),
            a_row("b", category="summary", budget=64, cost_usd=0.02),
            budget=64,
        )
        cheaper = a_run(
            a_row("a", category="code", budget=8, cost_usd=0.005),
            a_row("b", category="summary", budget=8, cost_usd=0.002),
            budget=8,
        )

        summaries = compute_oracle([baseline, cheaper]).by_category()

        assert [s.category for s in summaries] == ["code", "summary"]
        assert summaries[0].mean_savings_usd == pytest.approx(0.005)
        assert summaries[1].mean_savings_usd == pytest.approx(0.018)

    def test_categories_are_sorted(self) -> None:
        baseline = a_run(
            a_row("a", category="zeta", budget=64, cost_usd=0.01),
            a_row("b", category="alpha", budget=64, cost_usd=0.01),
            budget=64,
        )
        cheaper = a_run(
            a_row("a", category="zeta", budget=8, cost_usd=0.005),
            a_row("b", category="alpha", budget=8, cost_usd=0.005),
            budget=8,
        )

        summaries = compute_oracle([baseline, cheaper]).by_category()

        assert [s.category for s in summaries] == ["alpha", "zeta"]

    def test_counts_only_citable_tasks_per_category(self) -> None:
        baseline = a_run(
            a_row("a", category="code", budget=64, cost_usd=0.01),
            a_row("b", category="code", budget=64, cost_usd=0.01),
            budget=64,
        )
        cheaper = a_run(
            a_row("a", category="code", budget=8, cost_usd=0.005),
            a_row("b", category="code", budget=8, cost_usd=0.005, citable=False),
            budget=8,
        )

        summary = compute_oracle([baseline, cheaper]).by_category()[0]

        assert summary.tasks == 2
        assert summary.citable_tasks == 1

    def test_dropped_categories_are_absent_from_summaries(self) -> None:
        baseline = a_run(
            a_row("a", category="code", budget=64, cost_usd=0.01), budget=64
        )
        cheaper = a_run(a_row("a", category="code", budget=8, cost_usd=None), budget=8)

        assert compute_oracle([baseline, cheaper]).by_category() == ()

    def test_no_grand_total_is_offered(self) -> None:
        """The bound is per category or it is a measurement of the task mix."""
        names = {name for name in dir(OracleResult) if not name.startswith("_")}

        assert "by_category" in names
        assert "mean_savings_usd" not in names
        assert "overall_mean" not in names

    def test_category_summary_serializes(self) -> None:
        baseline = a_run(
            a_row("a", category="code", budget=64, cost_usd=0.01), budget=64
        )
        cheaper = a_run(a_row("a", category="code", budget=8, cost_usd=0.005), budget=8)

        payload = compute_oracle([baseline, cheaper]).by_category()[0].to_dict()

        assert payload["category"] == "code"
        assert payload["mean_savings_usd"] == pytest.approx(0.005)


class TestIncomparableRuns:
    def test_one_run_is_not_an_oracle(self) -> None:
        with pytest.raises(ValueError, match="at least one other budget"):
            compute_oracle([a_run(a_row("t1"), budget=64)])

    def test_no_runs_is_not_an_oracle(self) -> None:
        with pytest.raises(ValueError, match="at least one other budget"):
            compute_oracle([])

    def test_different_task_sets_are_refused(self) -> None:
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
        other = a_run(a_row("different", budget=8, cost_usd=0.004), budget=8)

        with pytest.raises(ValueError, match="different set of tasks"):
            compute_oracle([baseline, other])

    def test_a_repetition_count_mismatch_is_refused(self) -> None:
        baseline = a_run(
            a_row("t1", repetition=0, budget=64, cost_usd=0.01),
            a_row("t1", repetition=1, budget=64, cost_usd=0.01),
            budget=64,
        )
        other = a_run(a_row("t1", repetition=0, budget=8, cost_usd=0.004), budget=8)

        with pytest.raises(ValueError, match="different set of tasks"):
            compute_oracle([baseline, other])

    def test_mixed_strategies_are_refused(self) -> None:
        """A strategy gap is not a budget gap."""
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
        other = a_run(
            a_row("t1", budget=8, cost_usd=0.004), budget=8, strategy="reason"
        )

        with pytest.raises(ValueError, match="not a budget difference"):
            compute_oracle([baseline, other])

    def test_two_price_tables_are_refused(self) -> None:
        """Two price tables would make the difference measure the tables."""
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
        other = a_run(a_row("t1", budget=8, cost_usd=0.004), budget=8, version="other")

        with pytest.raises(ValueError, match="measures the tables"):
            compute_oracle([baseline, other])

    def test_the_offending_run_is_named(self) -> None:
        baseline = a_run(a_row("t1", budget=64, cost_usd=0.01), budget=64)
        other = a_run(a_row("t1", budget=8, cost_usd=0.004), budget=8, version="other")

        with pytest.raises(ValueError, match="budget 8"):
            compute_oracle([baseline, other])


class TestSerialization:
    def test_round_trips_through_json(self) -> None:
        original = compute_oracle(list(two_runs()))

        restored = OracleResult.from_dict(json.loads(json.dumps(original.to_dict())))

        assert restored.rows == original.rows
        assert restored.max_savings_usd == pytest.approx(original.max_savings_usd)
        assert restored.baseline_fingerprint == original.baseline_fingerprint
        assert restored.incomparable_rows == original.incomparable_rows

    def test_row_round_trips(self) -> None:
        row = OracleRow(
            task_id="t",
            category="c",
            repetition=1,
            baseline_budget=64,
            baseline_cost_usd=0.01,
            oracle_cost_usd=0.004,
            chosen_budget=8,
            runs_considered=2,
            savings_usd=0.006,
            savings_fraction=0.6,
            comparable=False,
            citable=False,
        )

        assert OracleRow.from_dict(row.to_dict()) == row

    def test_a_row_missing_a_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing required field"):
            OracleRow.from_dict({"task_id": "t", "category": "c"})

    def test_a_result_missing_a_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing required field"):
            OracleResult.from_dict({"baseline_budget": 64})

    def test_payload_records_the_price_table(self) -> None:
        baseline = a_run(a_row("t1"), budget=64, version="2026.09.26-1")
        other = a_run(a_row("t1", budget=8), budget=8, version="2026.09.26-1")

        assert compute_oracle([baseline, other]).price_table_version == "2026.09.26-1"

    def test_payload_records_the_baseline_fingerprint(self) -> None:
        baseline = a_run(a_row("t1"), budget=64, fingerprint="abc123")
        other = a_run(a_row("t1", budget=8), budget=8, fingerprint="zzz999")

        result = compute_oracle([baseline, other])

        assert result.baseline_fingerprint == "abc123"
        assert result.to_dict()["baseline_fingerprint"] == "abc123"
