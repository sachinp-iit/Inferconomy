"""Frontier publication-gate tests.

A frontier is the number this project will be judged on, and it is also the
easiest to flatter. So the property under test is refusal: a frontier missing any
of the four controls must not be publishable, and the reason must name the control
that failed. A gate that fails silently is worse than no gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from inferconomy.benchmark import (
    MetricRow,
    OracleResult,
    OracleRow,
    RunResult,
    compute_oracle,
)
from inferconomy.contracts import UsageBasis
from inferconomy.frontier import (
    CONTROLS,
    FRONTIER_SCHEMA_VERSION,
    TOKEN_SHORTFALL_TOLERANCE,
    Frontier,
    FrontierCategory,
    FrontierConfig,
    UnpublishableFrontier,
    build_frontier,
)
from inferconomy.judge import (
    JudgeBasis,
    JudgeRequest,
    NoiseReport,
    NullResult,
    QualityDelta,
    judge_noise,
    null_condition,
)
from inferconomy.testing import FunctionJudge

BUDGETS = (16, 64, 256)


def a_row(
    task_id: str,
    category: str,
    budget: int,
    *,
    cost: float,
    tokens: int = 100,
    citable: bool = True,
) -> MetricRow:
    return MetricRow(
        task_id=task_id,
        category=category,
        repetition=0,
        strategy="direct",
        budget=budget,
        input_tokens=tokens // 2,
        output_tokens=tokens // 2,
        reasoning_tokens=0,
        total_tokens=tokens,
        llm_calls=1,
        cost_usd=cost,
        usage_basis=UsageBasis.REPORTED,
        citable=citable,
        latency_ms=10.0,
    )


def a_run(
    budget: int,
    *,
    categories: tuple[str, ...] = ("code", "summary"),
    cost: float = 0.01,
    tokens: int = 100,
    citable: bool = True,
) -> RunResult:
    return RunResult(
        config_fingerprint=f"fp-{budget}",
        strategy="direct",
        budget=budget,
        price_table_version="test-1",
        price_table_verified=True,
        rows=tuple(
            a_row(
                f"{category}-1",
                category,
                budget,
                cost=cost,
                tokens=tokens,
                citable=citable,
            )
            for category in categories
        ),
    )


def a_config(**kwargs: Any) -> FrontierConfig:
    defaults: dict[str, Any] = {
        "budgets": BUDGETS,
        "model": "test-model",
        "judge_name": "test-judge",
    }
    return FrontierConfig(**{**defaults, **kwargs})


def runs_with_rising_quality() -> dict[int, RunResult]:
    """Cost and tokens rise with budget, quality rises with it too."""
    return {
        16: a_run(16, cost=0.01, tokens=100),
        64: a_run(64, cost=0.04, tokens=400),
        256: a_run(256, cost=0.16, tokens=1600),
    }


def quality_with_rising_quality() -> dict[int, dict[str, float]]:
    return {
        16: {"code": 0.70, "summary": 0.72},
        64: {"code": 0.85, "summary": 0.80},
        256: {"code": 0.90, "summary": 0.88},
    }


def an_oracle(categories: tuple[str, ...] = ("code", "summary")) -> OracleResult:
    """An oracle whose reference arm matches the config's baseline budget."""
    return OracleResult(
        baseline_fingerprint="fp-256",
        baseline_budget=256,
        price_table_version="test-1",
        rows=tuple(
            OracleRow(
                task_id=f"{category}-1",
                category=category,
                repetition=0,
                baseline_budget=256,
                baseline_cost_usd=0.16,
                oracle_cost_usd=0.08,
                chosen_budget=64,
                runs_considered=2,
                savings_usd=0.08,
                savings_fraction=0.5,
            )
            for category in categories
        ),
    )


def a_passing_null() -> NullResult:
    return NullResult(
        self_delta=0.0,
        self_passed=True,
        order_delta=0.0,
        order_assertable=True,
        order_passed=True,
        by_category=(
            QualityDelta(
                category="code",
                items=1,
                mean_earlier=0.7,
                mean_later=0.7,
                delta=0.0,
                citable_items=1,
            ),
            QualityDelta(
                category="summary",
                items=1,
                mean_earlier=0.72,
                mean_later=0.72,
                delta=0.0,
                citable_items=1,
            ),
        ),
        repeats=2,
        judge_name="test-judge",
    )


def a_noise_report(stdev: float = 0.005) -> NoiseReport:
    from inferconomy.judge import BenchmarkNoise

    return NoiseReport(
        per_category=tuple(
            BenchmarkNoise(
                category=category,
                items=1,
                repeats=3,
                mean_score=0.8,
                mean_stdev=stdev,
                max_range=stdev * 2,
                unanimous_fraction=1.0,
                position_bias=0.0,
                citable_items=1,
            )
            for category in ("code", "summary")
        ),
        repeats=3,
        judge_name="test-judge",
        basis_counts={JudgeBasis.MODEL: 6},
        min_detectable_delta=stdev,
    )


def a_complete_frontier(**kwargs: Any) -> Frontier:
    defaults: dict[str, Any] = {
        "config": a_config(),
        "runs": runs_with_rising_quality(),
        "quality": quality_with_rising_quality(),
        "oracle": an_oracle(),
        "null": a_passing_null(),
        "noise": a_noise_report(),
    }
    return build_frontier(**{**defaults, **kwargs})


class TestPublishGate:
    def test_a_complete_frontier_publishes(self) -> None:
        frontier = a_complete_frontier()

        assert frontier.publishable is True
        assert frontier.publish()["publishable"] is True

    def test_publishing_marks_the_frontier(self) -> None:
        frontier = a_complete_frontier()
        assert frontier.published is False

        frontier.publish()

        assert frontier.published is True

    def test_an_unpublished_frontier_says_so(self) -> None:
        """An assembled frontier must not be mistaken for a vetted one."""
        assert a_complete_frontier().published is False
        assert a_complete_frontier().to_dict()["published"] is False

    def test_all_four_controls_are_required(self) -> None:
        frontier = a_complete_frontier()

        assert tuple(c.name for c in frontier.controls) == CONTROLS
        assert len(CONTROLS) == 4

    def test_publishing_without_an_oracle_is_refused(self) -> None:
        frontier = a_complete_frontier(oracle=None)

        with pytest.raises(UnpublishableFrontier, match="oracle_bound"):
            frontier.publish()

    def test_publishing_without_a_null_is_refused(self) -> None:
        frontier = a_complete_frontier(null=None)

        with pytest.raises(UnpublishableFrontier, match="null_and_noise"):
            frontier.publish()

    def test_publishing_without_a_noise_report_is_refused(self) -> None:
        frontier = a_complete_frontier(noise=None)

        with pytest.raises(UnpublishableFrontier, match="null_and_noise"):
            frontier.publish()

    def test_publishing_with_nothing_at_all_is_refused(self) -> None:
        frontier = build_frontier(
            config=a_config(),
            runs=runs_with_rising_quality(),
            quality=quality_with_rising_quality(),
        )

        with pytest.raises(UnpublishableFrontier):
            frontier.publish()

    def test_the_error_names_every_failing_control(self) -> None:
        frontier = build_frontier(
            config=a_config(),
            runs=runs_with_rising_quality(),
            quality=quality_with_rising_quality(),
        )

        with pytest.raises(UnpublishableFrontier) as excinfo:
            frontier.publish()

        assert "oracle_bound" in str(excinfo.value)
        assert "null_and_noise" in str(excinfo.value)

    def test_a_category_with_no_baseline_point_is_flagged(self) -> None:
        """Defensive: build_frontier always supplies a baseline point, so this
        guards a hand-assembled category rather than a public-API input."""
        from inferconomy.frontier import FrontierPoint, _check_controls

        orphan = FrontierCategory(
            category="code",
            points=(
                FrontierPoint(
                    category="code",
                    budget=16,
                    cost_usd=0.01,
                    quality=0.8,
                    tasks=1,
                    is_baseline=False,
                    total_tokens=100,
                    noise_floor=0.0,
                    delta_vs_baseline=0.1,
                    resolvable=True,
                    cost_citable=True,
                    quality_citable=True,
                    price_table_version="test-1",
                ),
            ),
            baseline_budget=256,
            baseline_cost_usd=0.16,
            oracle_cost_usd=0.08,
            max_savings_usd=0.08,
        )

        controls = _check_controls(
            a_config(), [orphan], an_oracle(), a_passing_null(), a_noise_report(), True
        )

        control = next(c for c in controls if c.name == "cost_matched_baseline")
        assert control.passed is False
        assert "no baseline point" in control.detail

    def test_a_treatment_beating_the_baseline_on_tokens_blocks_publication(
        self,
    ) -> None:
        """A treatment that out-tokened the reference arm beat a hobbled baseline."""
        runs = runs_with_rising_quality()
        overran = a_run(64, cost=0.04, tokens=9999)
        frontier = build_frontier(
            config=a_config(),
            runs={**runs, 64: overran},
            quality=quality_with_rising_quality(),
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(),
        )

        with pytest.raises(UnpublishableFrontier, match="cost_matched_baseline"):
            frontier.publish()

    def test_a_treatment_beating_the_baseline_on_tokens_is_named(self) -> None:
        runs = runs_with_rising_quality()
        overran = a_run(64, cost=0.04, tokens=9999)
        frontier = build_frontier(
            config=a_config(),
            runs={**runs, 64: overran},
            quality=quality_with_rising_quality(),
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(),
        )

        control = next(
            c for c in frontier.controls if c.name == "cost_matched_baseline"
        )
        assert control.passed is False
        assert "9999" in control.detail

    def test_a_small_token_shortfall_is_tolerated(self) -> None:
        """Two runs of the same budget differ slightly; that is not a strawman."""
        runs = runs_with_rising_quality()
        slightly_less = a_run(256, cost=0.16, tokens=1580)
        frontier = build_frontier(
            config=a_config(),
            runs={**runs, 256: slightly_less},
            quality=quality_with_rising_quality(),
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(),
        )

        control = next(
            c for c in frontier.controls if c.name == "cost_matched_baseline"
        )
        assert control.passed is True
        assert 1580 * (1 + TOKEN_SHORTFALL_TOLERANCE) >= 400

    def test_a_failed_null_blocks_publication(self) -> None:
        broken = NullResult(
            self_delta=0.4,
            self_passed=False,
            order_delta=0.0,
            order_assertable=True,
            order_passed=True,
            by_category=(),
            repeats=2,
            judge_name="broken",
        )
        frontier = a_complete_frontier(null=broken)

        with pytest.raises(UnpublishableFrontier, match="null_and_noise"):
            frontier.publish()

    def test_a_proxy_judge_blocks_publication(self) -> None:
        proxy = a_noise_report()
        object.__setattr__(proxy, "basis_counts", {JudgeBasis.PROXY: 6})
        frontier = a_complete_frontier(noise=proxy)

        with pytest.raises(UnpublishableFrontier, match="null_and_noise"):
            frontier.publish()

    def test_an_oracle_missing_a_category_blocks_publication(self) -> None:
        frontier = a_complete_frontier(oracle=an_oracle(categories=("code",)))

        with pytest.raises(UnpublishableFrontier, match="oracle_bound"):
            frontier.publish()

    def test_unpublishable_cannot_claim_citable(self) -> None:
        frontier = a_complete_frontier(oracle=None)

        assert frontier.publishable is False
        assert frontier.citable is False

    def test_the_failure_reason_is_carried_in_the_payload(self) -> None:
        """A reader of a results file must meet the objections, not just the curve."""
        payload = a_complete_frontier(oracle=None).to_dict()

        oracle_control = next(
            c for c in payload["controls"] if c["name"] == "oracle_bound"
        )
        assert oracle_control["passed"] is False
        assert "upper bound" in oracle_control["detail"]


class TestCurve:
    def test_one_point_per_budget_per_category(self) -> None:
        frontier = a_complete_frontier()

        code = frontier.category("code")
        assert [p.budget for p in code.points] == [16, 64, 256]

    def test_points_are_ordered_by_cost(self) -> None:
        frontier = a_complete_frontier()

        costs = [p.cost_usd for p in frontier.category("code").points]
        assert costs == sorted(costs)

    def test_the_baseline_is_the_most_generous_budget(self) -> None:
        """Savings are measured against the arm that was given the most compute."""
        frontier = a_complete_frontier()

        baseline = next(p for p in frontier.category("code").points if p.is_baseline)
        assert baseline.budget == 256
        assert baseline.total_tokens == max(
            p.total_tokens for p in frontier.category("code").points
        )

    def test_a_larger_budget_shows_up_cheaper_when_it_does(self) -> None:
        """The curve is sorted by cost, not by budget, so a regression is visible."""
        runs = runs_with_rising_quality()
        runs[256] = a_run(256, cost=0.005, tokens=1600)
        frontier = build_frontier(
            config=a_config(),
            runs=runs,
            quality=quality_with_rising_quality(),
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(),
        )

        points = frontier.category("code").points
        assert points[0].budget == 256

    def test_quality_deltas_are_measured_against_the_baseline(self) -> None:
        frontier = a_complete_frontier()

        cheapest = frontier.category("code").point(16)
        assert cheapest.delta_vs_baseline == pytest.approx(-0.20)

    def test_the_baseline_point_has_no_delta(self) -> None:
        frontier = a_complete_frontier()

        assert frontier.category("code").point(256).delta_vs_baseline == pytest.approx(
            0.0
        )

    def test_a_delta_inside_the_noise_floor_is_unresolvable(self) -> None:
        """A quality gain the instrument cannot see is not a finding."""
        quality = {
            16: {"code": 0.70, "summary": 0.72},
            64: {"code": 0.8995, "summary": 0.8795},
            256: {"code": 0.90, "summary": 0.88},
        }
        frontier = build_frontier(
            config=a_config(),
            runs=runs_with_rising_quality(),
            quality=quality,
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(stdev=0.01),
        )

        assert frontier.category("code").point(64).resolvable is False
        assert frontier.category("code").point(16).resolvable is True

    def test_the_baseline_point_is_always_resolvable(self) -> None:
        """It defines the reference, so there is no delta to resolve."""
        frontier = build_frontier(
            config=a_config(),
            runs=runs_with_rising_quality(),
            quality=quality_with_rising_quality(),
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(stdev=0.5),
        )

        assert frontier.category("code").point(256).resolvable is True

    def test_unresolvable_points_are_listed_not_hidden(self) -> None:
        """The point stays on the curve; hiding it would misreport the shape."""
        quality = {
            16: {"code": 0.70, "summary": 0.72},
            64: {"code": 0.8995, "summary": 0.8795},
            256: {"code": 0.90, "summary": 0.88},
        }
        frontier = build_frontier(
            config=a_config(),
            runs=runs_with_rising_quality(),
            quality=quality,
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(stdev=0.01),
        )

        unresolvable = frontier.unresolvable()
        assert {(p.category, p.budget) for p in unresolvable} == {
            ("code", 64),
            ("summary", 64),
        }

    def test_estimated_costs_block_citability(self) -> None:
        runs = runs_with_rising_quality()
        runs[64] = a_run(64, cost=0.04, tokens=400, citable=False)
        frontier = build_frontier(
            config=a_config(),
            runs=runs,
            quality=quality_with_rising_quality(),
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(),
        )

        assert frontier.publishable is True
        assert frontier.citable is False
        assert frontier.category("code").point(64).citable is False

    def test_captured_savings_is_bounded_by_the_oracle(self) -> None:
        frontier = a_complete_frontier()

        captured = frontier.category("code").captured_savings_fraction
        assert captured is not None
        assert captured <= 1.0

    def test_captured_savings_is_none_without_an_oracle(self) -> None:
        frontier = a_complete_frontier(oracle=None)

        assert frontier.category("code").captured_savings_fraction is None
        assert frontier.category("code").max_savings_usd is None

    def test_the_oracle_cost_sits_below_the_baseline(self) -> None:
        frontier = a_complete_frontier()

        category = frontier.category("code")
        assert category.oracle_cost_usd is not None
        assert category.oracle_cost_usd < category.baseline_cost_usd

    def test_captured_savings_is_none_when_the_oracle_found_nothing(self) -> None:
        zero = OracleResult(
            baseline_fingerprint="fp",
            baseline_budget=16,
            price_table_version="test-1",
            rows=(
                OracleRow(
                    task_id="code-1",
                    category="code",
                    repetition=0,
                    baseline_budget=16,
                    baseline_cost_usd=0.01,
                    oracle_cost_usd=0.01,
                    chosen_budget=16,
                    runs_considered=2,
                    savings_usd=0.0,
                    savings_fraction=0.0,
                ),
                OracleRow(
                    task_id="summary-1",
                    category="summary",
                    repetition=0,
                    baseline_budget=16,
                    baseline_cost_usd=0.01,
                    oracle_cost_usd=0.01,
                    chosen_budget=16,
                    runs_considered=2,
                    savings_usd=0.0,
                    savings_fraction=0.0,
                ),
            ),
        )
        frontier = a_complete_frontier(oracle=zero)

        assert frontier.category("code").captured_savings_fraction is None

    def test_no_aggregate_curve_is_exposed(self) -> None:
        """One number for the whole curve measures the category mix."""
        names = {name for name in dir(Frontier) if not name.startswith("_")}

        assert not any("overall" in name or "mean" in name for name in names)
        assert "category" in names

    def test_categories_are_reported_separately(self) -> None:
        frontier = a_complete_frontier()

        assert [c.category for c in frontier.categories] == ["code", "summary"]

    def test_an_unknown_category_lookup_fails_loudly(self) -> None:
        with pytest.raises(KeyError, match="No frontier for category"):
            a_complete_frontier().category("translation")

    def test_an_unknown_budget_lookup_fails_loudly(self) -> None:
        with pytest.raises(KeyError, match="No point at budget"):
            a_complete_frontier().category("code").point(999)


class TestConfigFingerprint:
    def test_identical_configs_agree(self) -> None:
        assert a_config().fingerprint == a_config().fingerprint

    def test_a_different_budget_changes_the_fingerprint(self) -> None:
        assert a_config().fingerprint != a_config(budgets=(16, 64)).fingerprint

    def test_a_different_model_changes_the_fingerprint(self) -> None:
        assert a_config().fingerprint != a_config(model="other").fingerprint

    def test_a_different_judge_changes_the_fingerprint(self) -> None:
        assert a_config().fingerprint != a_config(judge_name="other").fingerprint

    def test_the_fingerprint_is_recorded_on_the_frontier(self) -> None:
        config = a_config()
        frontier = build_frontier(
            config=config,
            runs=runs_with_rising_quality(),
            quality=quality_with_rising_quality(),
            oracle=an_oracle(),
            null=a_passing_null(),
            noise=a_noise_report(),
        )

        assert frontier.config_fingerprint == config.fingerprint

    def test_repeats_change_the_fingerprint(self) -> None:
        assert a_config().fingerprint != a_config(repetitions=3).fingerprint


class TestConfigValidation:
    def test_no_budgets_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one budget"):
            FrontierConfig(budgets=(), model="m", judge_name="j")

    def test_duplicate_budgets_are_refused(self) -> None:
        with pytest.raises(ValueError, match="Duplicate budgets"):
            FrontierConfig(budgets=(16, 16), model="m", judge_name="j")

    def test_a_non_positive_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            FrontierConfig(budgets=(0,), model="m", judge_name="j")

    def test_a_blank_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="model must not be blank"):
            FrontierConfig(budgets=(16,), model="  ", judge_name="j")

    def test_a_blank_judge_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="judge_name must not be blank"):
            FrontierConfig(budgets=(16,), model="m", judge_name=" ")

    def test_an_unnamed_judge_is_refused_because_it_is_not_a_measurement(self) -> None:
        with pytest.raises(ValueError, match="not a measurement"):
            FrontierConfig(budgets=(16,), model="m", judge_name="")

    def test_zero_repetitions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            FrontierConfig(budgets=(16,), model="m", judge_name="j", repetitions=0)

    def test_a_baseline_outside_the_budgets_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not among budgets"):
            FrontierConfig(
                budgets=(16, 64), model="m", judge_name="j", baseline_budget=999
            )

    def test_the_baseline_defaults_to_the_most_generous_budget(self) -> None:
        """The arm a saving is measured against must not be the starved one."""
        assert a_config(budgets=(256, 16, 64)).baseline_budget == 256

    def test_duplicate_categories_are_refused(self) -> None:
        with pytest.raises(ValueError, match="Duplicate categories"):
            a_config(categories=("code", "code"))


class TestConfigSerialization:
    def test_round_trips_through_json(self) -> None:
        config = a_config(price_file="prices.json", categories=("code", "summary"))

        restored = FrontierConfig.from_dict(json.loads(config.to_json()))

        assert restored == config
        assert restored.fingerprint == config.fingerprint

    def test_written_and_re_read_from_disk(self, tmp_path: Path) -> None:
        config = a_config(price_file="prices.json")
        path = config.to_file(str(tmp_path / "frontier.json"))

        assert FrontierConfig.from_file(path) == config

    def test_the_written_file_ends_with_a_newline(self, tmp_path: Path) -> None:
        path = a_config().to_file(str(tmp_path / "frontier.json"))

        assert Path(path).read_text(encoding="utf-8").endswith("}\n")

    def test_a_missing_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing required field"):
            FrontierConfig.from_dict({"budgets": [16]})

    def test_the_schema_version_is_recorded(self) -> None:
        assert a_config().to_dict()["schema_version"] == FRONTIER_SCHEMA_VERSION

    def test_the_shipped_example_config_is_valid(self) -> None:
        """The committed config must load, or the reproducibility claim is empty."""
        root = Path(__file__).resolve().parent.parent
        config = FrontierConfig.from_file(str(root / "benchmarks" / "frontier.json"))

        assert config.budgets
        assert config.judge_name.strip()
        assert config.fingerprint


class TestBuildValidation:
    def test_a_missing_run_is_refused(self) -> None:
        runs = runs_with_rising_quality()
        del runs[256]

        with pytest.raises(ValueError, match="needs a run"):
            build_frontier(
                config=a_config(), runs=runs, quality=quality_with_rising_quality()
            )

    def test_quality_for_an_unknown_budget_is_refused(self) -> None:
        quality = quality_with_rising_quality()
        quality[999] = {"code": 0.5, "summary": 0.5}

        with pytest.raises(ValueError, match="Quality covers budgets"):
            build_frontier(
                config=a_config(),
                runs=runs_with_rising_quality(),
                quality=quality,
            )

    def test_a_category_missing_at_one_budget_is_refused(self) -> None:
        quality = quality_with_rising_quality()
        del quality[64]["code"]

        with pytest.raises(ValueError, match="is missing at budget"):
            build_frontier(
                config=a_config(), runs=runs_with_rising_quality(), quality=quality
            )

    def test_a_category_missing_from_a_run_is_refused(self) -> None:
        runs = runs_with_rising_quality()
        runs[64] = a_run(64, categories=("code",), cost=0.04, tokens=400)

        with pytest.raises(ValueError, match="is missing at budget"):
            build_frontier(
                config=a_config(), runs=runs, quality=quality_with_rising_quality()
            )

    def test_config_categories_must_match_the_data(self) -> None:
        with pytest.raises(ValueError, match="but the data covers"):
            a_complete_frontier(config=a_config(categories=("code",)))

    def test_an_impossible_quality_score_is_refused(self) -> None:
        quality = quality_with_rising_quality()
        quality[64]["code"] = 1.5

        with pytest.raises(ValueError, match="not a usable score"):
            build_frontier(
                config=a_config(), runs=runs_with_rising_quality(), quality=quality
            )

    def test_a_nan_quality_score_is_refused(self) -> None:
        quality = quality_with_rising_quality()
        quality[64]["code"] = math.nan

        with pytest.raises(ValueError, match="not a usable score"):
            build_frontier(
                config=a_config(), runs=runs_with_rising_quality(), quality=quality
            )


class TestEndToEnd:
    def test_a_frontier_from_real_harness_pieces_publishes(self) -> None:
        """The same objects a real run produces, wired together."""
        items = [
            JudgeRequest(task_id="code-1", category="code", prompt="p", response="a"),
            JudgeRequest(
                task_id="summary-1", category="summary", prompt="p", response="b"
            ),
        ]
        judge = FunctionJudge(lambda r: 0.8, name="test-judge", basis=JudgeBasis.MODEL)
        null = null_condition_sync(judge, items)
        noise = judge_noise_sync(judge, items, repeats=3)

        frontier = build_frontier(
            config=a_config(),
            runs=runs_with_rising_quality(),
            quality={
                16: {"code": 0.8, "summary": 0.8},
                64: {"code": 0.8, "summary": 0.8},
                256: {"code": 0.8, "summary": 0.8},
            },
            oracle=an_oracle(),
            null=null,
            noise=noise,
        )

        assert frontier.publishable is True
        assert frontier.publish()["per_category"]

    def test_a_failed_null_from_a_liar_blocks_publication(self) -> None:
        items = [
            JudgeRequest(task_id="code-1", category="code", prompt="p", response="a"),
            JudgeRequest(
                task_id="summary-1", category="summary", prompt="p", response="b"
            ),
        ]
        seen: dict[str, int] = {}

        def drifting(request: Any) -> float:
            n = seen.get(request.task_id, 0)
            seen[request.task_id] = n + 1
            return 0.4 + 0.1 * n

        liar = FunctionJudge(drifting, name="liar", deterministic=True)
        frontier = build_frontier(
            config=a_config(),
            runs=runs_with_rising_quality(),
            quality={
                16: {"code": 0.8, "summary": 0.8},
                64: {"code": 0.8, "summary": 0.8},
                256: {"code": 0.8, "summary": 0.8},
            },
            oracle=an_oracle(),
            null=null_condition_sync(liar, items),
            noise=judge_noise_sync(liar, items, repeats=3),
        )

        with pytest.raises(UnpublishableFrontier, match="null_and_noise"):
            frontier.publish()

    def test_the_oracle_from_real_runs_bounds_the_frontier(self) -> None:
        items = [
            JudgeRequest(task_id="code-1", category="code", prompt="p", response="a"),
            JudgeRequest(
                task_id="summary-1", category="summary", prompt="p", response="b"
            ),
        ]
        judge = FunctionJudge(lambda r: 0.8, name="test-judge")

        def run_at(budget: int, cost: float) -> RunResult:
            return RunResult(
                config_fingerprint=f"fp-{budget}",
                strategy="direct",
                budget=budget,
                price_table_version="test-1",
                price_table_verified=True,
                rows=(
                    a_row("code-1", "code", budget, cost=cost, tokens=budget),
                    a_row("summary-1", "summary", budget, cost=cost, tokens=budget),
                ),
            )

        # The oracle sees a cheaper budget than any the frontier swept, so perfect
        # allocation could have beaten even the curve's best point.
        oracle = compute_oracle([run_at(256, 0.16), run_at(8, 0.01)])
        frontier = build_frontier(
            config=a_config(),
            runs={16: run_at(16, 0.10), 64: run_at(64, 0.13), 256: run_at(256, 0.16)},
            quality={b: {"code": 0.8, "summary": 0.8} for b in BUDGETS},
            oracle=oracle,
            null=null_condition_sync(judge, items),
            noise=judge_noise_sync(judge, items, repeats=3),
        )

        category = frontier.category("code")
        captured = category.captured_savings_fraction
        assert category.max_savings_usd == pytest.approx(0.15)
        assert captured == pytest.approx(0.06 / 0.15)
        assert captured is not None and captured < 1.0

    def test_a_misaligned_oracle_blocks_publication(self) -> None:
        """A bound from a different baseline is a ratio of unrelated numbers."""
        misaligned = OracleResult(
            baseline_fingerprint="fp-8",
            baseline_budget=8,
            price_table_version="test-1",
            rows=an_oracle().rows,
        )
        frontier = a_complete_frontier(oracle=misaligned)

        with pytest.raises(UnpublishableFrontier, match="oracle_bound"):
            frontier.publish()

    def test_a_misaligned_oracle_says_which_budgets(self) -> None:
        misaligned = OracleResult(
            baseline_fingerprint="fp-8",
            baseline_budget=8,
            price_table_version="test-1",
            rows=an_oracle().rows,
        )
        frontier = a_complete_frontier(oracle=misaligned)

        control = next(c for c in frontier.controls if c.name == "oracle_bound")
        assert "budget 8" in control.detail
        assert "budget 256" in control.detail

    def test_a_misaligned_oracle_yields_no_captured_fraction(self) -> None:
        misaligned = OracleResult(
            baseline_fingerprint="fp-8",
            baseline_budget=8,
            price_table_version="test-1",
            rows=an_oracle().rows,
        )
        frontier = a_complete_frontier(oracle=misaligned)

        assert frontier.category("code").captured_savings_fraction is None
        assert frontier.category("code").max_savings_usd is None


def null_condition_sync(judge: Any, items: Any) -> NullResult:
    import asyncio

    return asyncio.run(null_condition(judge, items))


def judge_noise_sync(judge: Any, items: Any, repeats: int = 3) -> NoiseReport:
    import asyncio

    return asyncio.run(judge_noise(judge, items, repeats=repeats))
