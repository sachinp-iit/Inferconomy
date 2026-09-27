"""Fixed-budget baseline runner tests.

The property under test is that the runner cannot quietly become adaptive, and
that two runs configured identically are recognisably so. A baseline that
improves its own output, or a run that cannot be reconstructed from its config,
would leave every later comparison built on sand.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from inferconomy import CostTableError, Response, UnknownModel, Usage, UsageBasis
from inferconomy.benchmark import (
    CONFIG_SCHEMA_VERSION,
    BaselineConfig,
    CategorySummary,
    MetricRow,
    RunResult,
    Task,
    run_baseline,
)
from inferconomy.costs import ModelPrice, PriceTable
from inferconomy.testing import FakeClient, echo_response

MODEL = "bench-model"
VERIFIED = ModelPrice(input_per_million=3.0, output_per_million=15.0, verified=True)
UNVERIFIED = ModelPrice(input_per_million=3.0, output_per_million=15.0)


def verified_table() -> PriceTable:
    return PriceTable(version="test-1", models={MODEL: VERIFIED})


def unverified_table() -> PriceTable:
    return PriceTable(version="test-unverified", models={MODEL: UNVERIFIED})


class ReportsItsModel(FakeClient):
    """A provider that names the model it used, as an adapter should."""

    async def complete(self, request: Any, options: Any) -> Any:
        return replace(echo_response(request, options), model=MODEL)


class ReportsNoModel(FakeClient):
    """A provider that names nothing, which makes the run unpriceable."""

    async def complete(self, request: Any, options: Any) -> Any:
        return replace(echo_response(request, options), model=None)


class Broken(FakeClient):
    async def complete(self, request: Any, options: Any) -> Any:
        raise RuntimeError("provider down")


class ReportsUsage(FakeClient):
    """A provider that reports usage, as a real adapter does.

    The shipped echo fake does not, and must not: an estimate dressed as a
    measurement is the failure this library exists to prevent.
    """

    def __init__(self) -> None:
        super().__init__(
            [
                Response(
                    text="answer",
                    usage=Usage(
                        input_tokens=10, output_tokens=5, llm_calls=1, tokens_exact=True
                    ),
                )
            ],
            repeat_last=True,
        )


def a_task(task_id: str = "t1", category: str = "summarization") -> Task:
    return Task(task_id=task_id, prompt="Summarise the text.", category=category)


def a_config(**kwargs: Any) -> BaselineConfig:
    defaults: dict[str, Any] = {
        "tasks": (a_task(),),
        "budget": 64,
        "model": MODEL,
    }
    return BaselineConfig(**{**defaults, **kwargs})


def a_row(
    task_id: str = "t",
    category: str = "c",
    output_tokens: int = 10,
    cost_usd: float | None = 0.0,
    citable: bool = True,
) -> MetricRow:
    return MetricRow(
        task_id=task_id,
        category=category,
        repetition=0,
        strategy="direct",
        budget=10,
        input_tokens=1,
        output_tokens=output_tokens,
        reasoning_tokens=0,
        total_tokens=output_tokens + 1,
        llm_calls=1,
        cost_usd=cost_usd,
        usage_basis=UsageBasis.REPORTED if citable else UsageBasis.PRICED_UNVERIFIED,
        citable=citable,
        latency_ms=1.0,
    )


def a_result(*rows: MetricRow) -> RunResult:
    return RunResult("fp", "direct", 10, "test-1", True, rows)


def run(coro: Any) -> Any:
    """Drive one coroutine to completion.

    asyncio.run in a synchronous test keeps the async API under test without
    adding a pytest plugin and its configuration to the dev dependencies.
    """
    return asyncio.run(coro)


class TestTask:
    def test_builds_a_request_with_its_system_prompt(self) -> None:
        request = Task("t", "Explain.", "analysis", system="Be terse.").to_request()
        assert [message.role for message in request.messages] == ["system", "user"]
        assert request.messages[1].content == "Explain."

    def test_omits_an_absent_system_prompt(self) -> None:
        assert len(a_task().to_request().messages) == 1

    def test_rejects_a_blank_task_id(self) -> None:
        with pytest.raises(ValueError, match="task_id must not be blank"):
            Task("  ", "p", "c")

    def test_rejects_a_blank_category(self) -> None:
        with pytest.raises(ValueError, match="category must not be blank"):
            Task("t", "p", " ")

    def test_rejects_a_blank_prompt(self) -> None:
        with pytest.raises(ValueError, match="blank prompt"):
            Task("t", "", "c")

    def test_requires_a_task_id_on_load(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'task_id'"):
            Task.from_dict({"prompt": "p", "category": "c"})

    def test_round_trips(self) -> None:
        task = Task("t", "p", "c", system="s")
        assert Task.from_dict(task.to_dict()) == task


class TestConfig:
    def test_fingerprint_is_stable_across_calls(self) -> None:
        assert a_config().fingerprint == a_config().fingerprint

    def test_fingerprint_survives_a_json_round_trip(self) -> None:
        original = a_config()
        assert (
            BaselineConfig.from_json(original.to_json()).fingerprint
            == original.fingerprint
        )

    def test_fingerprint_ignores_key_order(self) -> None:
        config = a_config()
        shuffled = dict(reversed(list(json.loads(config.to_json()).items())))
        assert BaselineConfig.from_dict(shuffled).fingerprint == config.fingerprint

    @pytest.mark.parametrize(
        "changed",
        [
            {"budget": 65},
            {"model": "other-model"},
            {"repetitions": 2},
            {"strategy": "reason"},
            {"price_file": "prices.json"},
            {
                "tasks": [
                    {"task_id": "t1", "prompt": "Other.", "category": "summarization"}
                ]
            },
        ],
        ids=["budget", "model", "repetitions", "strategy", "price-file", "prompt"],
    )
    def test_fingerprint_changes_when_anything_that_matters_changes(
        self, changed: dict[str, Any]
    ) -> None:
        payload = {**a_config().to_dict(), **changed}
        assert BaselineConfig.from_dict(payload).fingerprint != a_config().fingerprint

    def test_rejects_duplicate_task_ids(self) -> None:
        with pytest.raises(ValueError, match=r"Duplicate task_id.*dup"):
            a_config(tasks=(a_task("dup"), a_task("dup", "analysis")))

    def test_rejects_a_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="budget must be positive"):
            a_config(budget=0)

    def test_rejects_non_positive_repetitions(self) -> None:
        with pytest.raises(ValueError, match="repetitions must be positive"):
            a_config(repetitions=0)

    def test_rejects_a_blank_strategy(self) -> None:
        with pytest.raises(ValueError, match="strategy must not be blank"):
            a_config(strategy=" ")

    def test_rejects_an_empty_task_list(self) -> None:
        with pytest.raises(ValueError, match="at least one task"):
            a_config(tasks=())

    def test_rejects_a_non_task(self) -> None:
        with pytest.raises(ValueError, match="is not a Task"):
            a_config(tasks=({"task_id": "t"},))

    def test_rejects_a_newer_schema(self) -> None:
        payload = {**a_config().to_dict(), "schema_version": CONFIG_SCHEMA_VERSION + 1}
        with pytest.raises(ValueError, match="newer than this release"):
            BaselineConfig.from_dict(payload)

    def test_rejects_an_unknown_schema(self) -> None:
        payload = {**a_config().to_dict(), "schema_version": 0}
        with pytest.raises(ValueError, match="Unsupported config schema version"):
            BaselineConfig.from_dict(payload)

    def test_requires_tasks_on_load(self) -> None:
        payload = a_config().to_dict()
        del payload["tasks"]
        with pytest.raises(ValueError, match="missing a 'tasks' array"):
            BaselineConfig.from_dict(payload)

    def test_requires_a_budget_on_load(self) -> None:
        payload = a_config().to_dict()
        del payload["budget"]
        with pytest.raises(ValueError, match="missing required field: 'budget'"):
            BaselineConfig.from_dict(payload)

    def test_rejects_a_non_array_tasks_field(self) -> None:
        with pytest.raises(ValueError, match="missing a 'tasks' array"):
            BaselineConfig.from_dict({"schema_version": 1, "budget": 1, "tasks": "t1"})

    def test_reports_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            BaselineConfig.from_json("{oops")

    def test_loads_tasks_as_a_tuple(self) -> None:
        assert isinstance(BaselineConfig.from_json(a_config().to_json()).tasks, tuple)


class TestRun:
    def test_emits_one_row_per_task(self) -> None:
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        assert len(result.rows) == 1

    def test_makes_one_call_per_task(self) -> None:
        client = FakeClient()
        config = a_config(tasks=(a_task("a"), a_task("b")))
        run(run_baseline(config, client, table=verified_table()))
        assert client.call_count == 2

    def test_the_budget_reaches_the_provider(self) -> None:
        client = FakeClient()
        run(run_baseline(a_config(budget=64), client, table=verified_table()))
        assert client.last_options.max_output_tokens == 64

    def test_the_model_reaches_the_provider(self) -> None:
        client = FakeClient()
        run(run_baseline(a_config(), client, table=verified_table()))
        assert client.last_options.model == MODEL

    def test_records_the_config_fingerprint(self) -> None:
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        assert result.config_fingerprint == a_config().fingerprint

    def test_rows_are_sorted_by_category_then_task(self) -> None:
        config = a_config(
            tasks=(
                a_task("z", "summarization"),
                a_task("a", "analysis"),
                a_task("m", "summarization"),
            )
        )
        result = run(run_baseline(config, FakeClient(), table=verified_table()))
        assert [(row.category, row.task_id) for row in result.rows] == [
            ("analysis", "a"),
            ("summarization", "m"),
            ("summarization", "z"),
        ]

    def test_repetitions_produce_indexed_rows(self) -> None:
        config = a_config(repetitions=3)
        result = run(run_baseline(config, FakeClient(), table=verified_table()))
        assert [row.repetition for row in result.rows] == [0, 1, 2]

    def test_never_escalates(self) -> None:
        """One call per task is the definition; a second would be adaptation."""
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        assert all(row.escalated is False for row in result.rows)

    def test_reports_zero_decision_overhead(self) -> None:
        """A constant, so an adaptive run has a zero point to beat."""
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        assert all(row.decision_overhead_ms == 0.0 for row in result.rows)

    def test_carries_no_quality_score(self) -> None:
        """Absent, not zero: no judge exists, and zero would read as a failure."""
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        assert all(row.quality is None for row in result.rows)

    def test_rows_carry_provenance_from_the_table(self) -> None:
        result = run(run_baseline(a_config(), ReportsUsage(), table=verified_table()))
        row = result.rows[0]
        assert row.citable is True
        assert row.usage_basis is UsageBasis.REPORTED

    def test_a_run_of_reported_usage_and_a_verified_table_is_citable(self) -> None:
        result = run(run_baseline(a_config(), ReportsUsage(), table=verified_table()))
        assert result.citable is True
        assert result.price_table_verified is True

    def test_estimated_usage_is_not_laundered_by_the_harness(self) -> None:
        """The shipped fake does not report usage, and its rows must say so."""
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        assert result.rows[0].usage_basis is UsageBasis.PRICED_FROM_ESTIMATE
        assert result.citable is False

    def test_estimated_usage_at_an_unverified_rate_is_worse_still(self) -> None:
        result = run(run_baseline(a_config(), FakeClient(), table=unverified_table()))
        assert result.rows[0].usage_basis is UsageBasis.ESTIMATED
        assert result.citable is False

    def test_an_unverified_table_makes_the_run_uncitable(self) -> None:
        result = run(run_baseline(a_config(), ReportsUsage(), table=unverified_table()))
        assert result.citable is False
        assert result.price_table_verified is False
        assert result.rows[0].usage_basis is UsageBasis.PRICED_UNVERIFIED

    def test_records_the_price_table_version(self) -> None:
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        assert result.price_table_version == "test-1"

    def test_prices_the_model_the_provider_reported(self) -> None:
        result = run(
            run_baseline(
                a_config(model=None), ReportsItsModel(), table=verified_table()
            )
        )
        assert result.rows[0].cost_usd is not None

    def test_refuses_to_price_an_unnamed_model(self) -> None:
        with pytest.raises(CostTableError, match="names no model"):
            run(
                run_baseline(
                    a_config(model=None), ReportsNoModel(), table=verified_table()
                )
            )

    def test_propagates_an_unknown_model(self) -> None:
        with pytest.raises(UnknownModel):
            run(
                run_baseline(
                    a_config(model="ghost"), FakeClient(), table=verified_table()
                )
            )

    def test_a_provider_error_aborts_rather_than_producing_partial_rows(
        self,
    ) -> None:
        with pytest.raises(RuntimeError, match="provider down"):
            run(run_baseline(a_config(), Broken(), table=verified_table()))

    def test_prices_with_the_bundled_table_by_default(self) -> None:
        config = a_config(model="gpt-4o")
        result = run(run_baseline(config, FakeClient()))
        assert result.rows[0].cost_usd is not None
        assert result.price_table_verified is False

    def test_loads_a_price_file_named_by_the_config(self, tmp_path: Path) -> None:
        path = tmp_path / "prices.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "version": "file-1",
                    "models": {
                        MODEL: {
                            "input_per_million": 1.0,
                            "output_per_million": 2.0,
                            "verified": True,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        result = run(run_baseline(a_config(price_file=str(path)), FakeClient()))
        assert result.price_table_version == "file-1"
        assert result.citable is False


class TestReproducibility:
    def test_two_identical_runs_agree(self) -> None:
        config = a_config(tasks=(a_task("a"), a_task("b", "analysis")), repetitions=2)
        first = run(run_baseline(config, FakeClient(), table=verified_table()))
        second = run(run_baseline(config, FakeClient(), table=verified_table()))
        assert first.to_dict() == second.to_dict()

    def test_a_result_survives_a_json_round_trip(self) -> None:
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        assert RunResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result

    def test_a_config_loaded_from_disk_reproduces_the_run(self, tmp_path: Path) -> None:
        config = a_config()
        path = tmp_path / "baseline.json"
        path.write_text(config.to_json(), encoding="utf-8")
        reloaded = BaselineConfig.from_json(path.read_text(encoding="utf-8"))
        first = run(run_baseline(config, FakeClient(), table=verified_table()))
        second = run(run_baseline(reloaded, FakeClient(), table=verified_table()))
        assert first.to_dict() == second.to_dict()

    def test_a_result_carries_no_wall_clock(self) -> None:
        """A timestamp changing every run would make the comparison vacuous."""
        result = run(run_baseline(a_config(), FakeClient(), table=verified_table()))
        serialised = json.dumps(result.to_dict())
        assert "timestamp" not in serialised
        assert "started_at" not in serialised


class TestMetricRow:
    def test_round_trips(self) -> None:
        row = a_row(cost_usd=0.5)
        assert MetricRow.from_dict(row.to_dict()) == row

    def test_keeps_an_absent_cost_absent(self) -> None:
        assert MetricRow.from_dict(a_row(cost_usd=None).to_dict()).cost_usd is None

    def test_keeps_an_absent_quality_absent(self) -> None:
        assert MetricRow.from_dict(a_row().to_dict()).quality is None

    def test_rejects_an_unrecognised_basis(self) -> None:
        payload = a_row().to_dict()
        payload["usage_basis"] = "roughly_right"
        with pytest.raises(ValueError, match="Invalid usage_basis"):
            MetricRow.from_dict(payload)

    def test_requires_its_identity_fields_on_load(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'task_id'"):
            MetricRow.from_dict({"category": "c", "strategy": "s", "budget": 1})


class TestRunResult:
    def test_is_not_citable_with_no_rows(self) -> None:
        assert a_result().citable is False

    def test_is_uncitable_when_any_row_is(self) -> None:
        assert a_result(a_row(), a_row(task_id="b", citable=False)).citable is False

    def test_groups_by_category_in_order(self) -> None:
        rows = (
            a_row(category="summarization"),
            a_row(category="analysis", task_id="b"),
        )
        assert [s.category for s in a_result(*rows).by_category()] == [
            "analysis",
            "summarization",
        ]

    def test_averages_tokens_within_a_category(self) -> None:
        rows = (a_row(output_tokens=10), a_row(task_id="b", output_tokens=20))
        summary = a_result(*rows).by_category()[0]
        assert summary.mean_output_tokens == 15.0
        assert summary.mean_total_tokens == 16.0
        assert summary.tasks == 2

    def test_averages_cost_over_priced_rows_only(self) -> None:
        """Averaging over unpriced rows would report an unknown cost as free."""
        rows = (
            a_row(cost_usd=1.0),
            a_row(task_id="b", cost_usd=3.0),
            a_row(task_id="d", cost_usd=None),
        )
        summary = a_result(*rows).by_category()[0]
        assert summary.mean_cost_usd == 2.0
        assert summary.priced_tasks == 2
        assert summary.tasks == 3

    def test_mean_cost_is_absent_when_nothing_is_priced(self) -> None:
        assert a_result(a_row(cost_usd=None)).by_category()[0].mean_cost_usd is None

    def test_counts_citable_rows(self) -> None:
        rows = (a_row(), a_row(task_id="b", citable=False))
        assert a_result(*rows).by_category()[0].citable_tasks == 1

    def test_offers_no_cross_category_aggregate(self) -> None:
        """One number over heterogeneous categories is not a measurement."""
        serialised = a_result(
            a_row(category="a"), a_row(category="b", task_id="b")
        ).to_dict()
        aggregates = {k for k in serialised if k not in {"rows", "by_category"}}
        assert not any("overall" in key or "grand" in key for key in aggregates)

    def test_round_trips(self) -> None:
        result = a_result(a_row())
        assert RunResult.from_dict(result.to_dict()) == result

    def test_rederives_summaries_on_load(self) -> None:
        payload = a_result(a_row()).to_dict()
        payload["by_category"] = []
        assert (
            RunResult.from_dict(payload).by_category()
            == a_result(a_row()).by_category()
        )

    def test_reports_a_missing_field(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'budget'"):
            RunResult.from_dict({"config_fingerprint": "f", "strategy": "s"})

    def test_summary_serialises_its_priced_count(self) -> None:
        assert (
            CategorySummary("c", 1, 1.0, 2.0, 0.5, 3.0, 1, 1).to_dict()["priced_tasks"]
            == 1
        )
