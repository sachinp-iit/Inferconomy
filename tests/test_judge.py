"""Null condition and judge noise tests.

The property under test is that a question containing no difference must produce
a difference of exactly zero, and that a judge with no error bars is exposed
before its numbers are published. Both are things that go wrong silently: a
nonzero null gets attributed to the model, and an unmeasured judge gets quoted as
though it were exact.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterator
from typing import Any

import pytest

from inferconomy.judge import (
    MIN_REPEATS,
    BenchmarkNoise,
    Judge,
    JudgeBasis,
    JudgeRequest,
    JudgeScore,
    NoiseReport,
    NullResult,
    QualityDelta,
    compare_scores,
    judge_noise,
    null_condition,
)
from inferconomy.testing import FunctionJudge


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def an_item(
    task_id: str = "t1",
    category: str = "summarization",
    response: str = "a good answer",
) -> JudgeRequest:
    return JudgeRequest(
        task_id=task_id,
        category=category,
        prompt="Summarise the text.",
        response=response,
    )


def items(*specs: tuple[str, str]) -> tuple[JudgeRequest, ...]:
    return tuple(an_item(task_id, category) for task_id, category in specs)


def constant_judge(value: float = 0.8, **kwargs: Any) -> FunctionJudge:
    return FunctionJudge(lambda _: value, **kwargs)


class CountingJudge:
    """A judge that hands out a scripted score per task, and records call order.

    Stands in for a model judge: the same input can yield a different score on a
    later call, and the answer can depend on where the item sat in the batch.
    """

    def __init__(
        self,
        values: dict[str, list[float]],
        *,
        basis: JudgeBasis = JudgeBasis.MODEL,
        deterministic: bool = False,
        position_bias: float = 0.0,
    ) -> None:
        self._values = values
        self._basis = basis
        self._deterministic = deterministic
        self._position_bias = position_bias
        self._calls: dict[str, int] = {}
        self.orders: list[str] = []

    @property
    def name(self) -> str:
        return "counting-judge"

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @property
    def call_count(self) -> int:
        return sum(self._calls.values())

    async def score(self, request: JudgeRequest) -> JudgeScore:
        seen = self._calls.get(request.task_id, 0)
        self._calls[request.task_id] = seen + 1
        self.orders.append(request.task_id)
        series = self._values[request.task_id]
        value = series[min(seen, len(series) - 1)]
        if self._position_bias:
            position = len(self.orders) - 1
            value = min(1.0, max(0.0, value + self._position_bias * position))
        return JudgeScore(score=value, basis=self._basis)


class TestScoreValidation:
    def test_rejects_a_score_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            JudgeScore(score=1.5, basis=JudgeBasis.MODEL)

    def test_rejects_a_negative_score(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            JudgeScore(score=-0.1, basis=JudgeBasis.MODEL)

    def test_rejects_nan(self) -> None:
        """A NaN would silently poison every mean it entered."""
        with pytest.raises(ValueError, match="finite"):
            JudgeScore(score=math.nan, basis=JudgeBasis.MODEL)

    def test_rejects_infinity(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            JudgeScore(score=math.inf, basis=JudgeBasis.MODEL)

    def test_accepts_the_boundaries(self) -> None:
        assert JudgeScore(score=0.0, basis=JudgeBasis.HEURISTIC).score == 0.0
        assert JudgeScore(score=1.0, basis=JudgeBasis.HEURISTIC).score == 1.0

    def test_coerces_an_integer_score(self) -> None:
        assert JudgeScore(score=1, basis=JudgeBasis.HEURISTIC).score == 1.0

    def test_rejects_a_blank_task_id(self) -> None:
        with pytest.raises(ValueError, match="task_id"):
            JudgeRequest(task_id="  ", category="c", prompt="p", response="r")

    def test_rejects_a_blank_response(self) -> None:
        with pytest.raises(ValueError, match="response must not be blank"):
            JudgeRequest(task_id="t", category="c", prompt="p", response="   ")

    def test_score_round_trips_through_json(self) -> None:
        score = JudgeScore(score=0.5, basis=JudgeBasis.MODEL, rationale="looks right")

        assert JudgeScore.from_dict(json.loads(json.dumps(score.to_dict()))) == score

    def test_a_missing_score_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing required field"):
            JudgeScore.from_dict({"basis": "model"})

    def test_an_unknown_basis_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unusable basis"):
            JudgeScore.from_dict({"score": 0.5, "basis": "vibes"})


class TestNullCondition:
    def test_a_question_with_no_difference_answers_zero(self) -> None:
        """The provable null: two arms that are the same object."""
        result = run(null_condition(constant_judge(0.8), items(("t1", "code"))))

        assert result.self_delta == 0.0
        assert result.self_passed is True
        assert result.passed is True

    def test_the_null_holds_across_every_category(self) -> None:
        subject = items(("t1", "code"), ("t2", "summary"), ("t3", "code"))

        result = run(null_condition(constant_judge(0.6), subject))

        assert all(d.delta == 0.0 for d in result.by_category)
        assert [d.category for d in result.by_category] == ["code", "summary"]

    def test_a_deterministic_judge_has_no_order_delta(self) -> None:
        subject = items(("t1", "code"), ("t2", "code"), ("t3", "code"))

        result = run(null_condition(constant_judge(0.9), subject))

        assert result.order_delta == 0.0
        assert result.order_assertable is True
        assert result.order_passed is True
        assert result.passed is True

    def test_a_stochastic_judge_is_reported_rather_than_asserted(self) -> None:
        """Order stability is not assertable against a judge that promised nothing."""
        judge = CountingJudge(
            {"t1": [0.5, 0.9], "t2": [0.4, 0.8], "t3": [0.6, 0.7]},
        )
        subject = items(("t1", "code"), ("t2", "code"), ("t3", "code"))

        result = run(null_condition(judge, subject))

        assert result.self_passed is True
        assert result.order_assertable is False
        assert result.passed is True

    def test_position_bias_is_measured_even_when_stochastic(self) -> None:
        judge = CountingJudge(
            {"t1": [0.5, 0.5], "t2": [0.5, 0.5], "t3": [0.5, 0.5]},
            position_bias=0.1,
        )
        subject = items(("t1", "code"), ("t2", "code"), ("t3", "code"))

        result = run(null_condition(judge, subject))

        assert result.order_delta > 0.0
        assert result.order_assertable is False

    def test_a_deterministic_judge_with_position_bias_fails(self) -> None:
        """If a judge promises determinism and then depends on batch position, that
        is a broken promise rather than a passing test."""
        judge = CountingJudge(
            {"t1": [0.5], "t2": [0.5], "t3": [0.5]},
            deterministic=True,
            position_bias=0.2,
        )
        subject = items(("t1", "code"), ("t2", "code"), ("t3", "code"))

        result = run(null_condition(judge, subject))

        assert result.order_assertable is True
        assert result.order_passed is False
        assert result.passed is False

    def test_the_items_are_actually_judged_twice(self) -> None:
        judge = constant_judge(0.5)

        run(null_condition(judge, items(("t1", "code"), ("t2", "code"))))

        assert judge.call_count == 4

    def test_a_broken_self_comparison_fails_even_without_position_bias(self) -> None:
        """A harness that cannot even subtract a set from itself is broken, and no
        order check should rescue it."""
        result = NullResult(
            self_delta=0.5,
            self_passed=False,
            order_delta=0.0,
            order_assertable=True,
            order_passed=True,
            by_category=(
                QualityDelta(
                    category="code",
                    items=1,
                    mean_earlier=0.5,
                    mean_later=0.5,
                    delta=0.5,
                    citable_items=1,
                ),
            ),
            repeats=2,
            judge_name="broken",
        )

        assert result.passed is False
        assert result.citable is False

    def test_an_unknown_category_lookup_fails_loudly(self) -> None:
        result = run(null_condition(constant_judge(), items(("t1", "code"))))

        with pytest.raises(KeyError, match="No null delta"):
            result.category("summary")

    def test_a_duplicate_task_id_is_refused(self) -> None:
        """Two items sharing an id would overwrite each other silently."""
        with pytest.raises(ValueError, match="Duplicate task_id"):
            run(null_condition(constant_judge(), (an_item("t1"), an_item("t1"))))

    def test_an_empty_item_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Nothing to judge"):
            run(null_condition(constant_judge(), ()))

    def test_one_repeat_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            run(null_condition(constant_judge(), items(("t1", "code")), repeats=1))

    def test_the_result_is_citable_only_when_the_null_passes(self) -> None:
        subject = items(("t1", "code"), ("t2", "code"))

        passing = run(null_condition(constant_judge(), subject))
        failing = run(null_condition(PositionDependent(), subject))

        assert passing.citable is True
        assert failing.citable is False

    def test_null_result_serializes_its_own_verdict(self) -> None:
        payload = run(null_condition(constant_judge(), items(("t1", "code")))).to_dict()

        assert payload["self_passed"] is True
        assert payload["self_delta"] == 0.0
        assert payload["order_assertable"] is True
        assert len(payload["by_category"]) == 1

    def test_a_failing_null_serializes_as_failed(self) -> None:
        payload = run(
            null_condition(PositionDependent(), items(("t1", "code")))
        ).to_dict()

        assert payload["passed"] is False
        assert payload["order_passed"] is False
        assert payload["citable"] is False


class PositionDependent(FunctionJudge):
    """Scores by how many times it has seen an item, so order must matter.

    Declares itself deterministic, which is the point: a judge promising
    determinism and then depending on call order is making a promise it breaks.
    """

    def __init__(self) -> None:
        super().__init__(lambda _: 0.5, name="position-dependent")
        self._seen: dict[str, int] = {}

    async def score(self, request: JudgeRequest) -> JudgeScore:
        count = self._seen.get(request.task_id, 0)
        self._seen[request.task_id] = count + 1
        return JudgeScore(score=0.2 + 0.3 * count, basis=JudgeBasis.MODEL)


class TestNoiseMeasurement:
    def test_a_perfectly_stable_judge_reports_no_noise(self) -> None:
        subject = items(("t1", "code"), ("t2", "code"))

        report = run(judge_noise(constant_judge(0.7), subject, repeats=3))

        entry = report.category("code")
        assert entry.mean_stdev == 0.0
        assert entry.max_range == 0.0
        assert entry.unanimous_fraction == 1.0
        assert report.min_detectable_delta == 0.0

    def test_a_noisy_judge_reports_its_spread(self) -> None:
        judge = CountingJudge({"t1": [0.2, 0.8, 0.5], "t2": [0.4, 0.4, 0.4]})

        report = run(
            judge_noise(judge, items(("t1", "code"), ("t2", "code")), repeats=3)
        )

        entry = report.category("code")
        # Averaged over both items: t1 wobbles 0.3, t2 does not wobble at all.
        assert entry.mean_stdev == pytest.approx(0.15)
        assert entry.max_range == pytest.approx(0.6)
        assert entry.unanimous_fraction == pytest.approx(0.5)

    def test_noise_is_reported_per_benchmark(self) -> None:
        subject = items(("t1", "code"), ("t2", "summary"))

        report = run(judge_noise(constant_judge(0.5), subject, repeats=2))

        assert [e.category for e in report.per_category] == ["code", "summary"]

    def test_a_quiet_category_cannot_hide_a_noisy_one(self) -> None:
        """The floor is a maximum, not a mean, so the worst category sets it."""
        judge = CountingJudge({"t1": [0.5, 0.5], "t2": [0.1, 0.9]})
        subject = items(("t1", "code"), ("t2", "summary"))

        report = run(judge_noise(judge, subject, repeats=2))

        assert report.category("code").mean_stdev == pytest.approx(0.0)
        # Sample stdev at n=2 divides by n-1, so a 0.8 spread reads as 0.8/sqrt(2).
        assert report.category("summary").mean_stdev == pytest.approx(
            0.8 / math.sqrt(2)
        )
        assert report.min_detectable_delta == pytest.approx(0.8 / math.sqrt(2))

    def test_the_floor_is_the_size_of_a_delta_that_means_nothing(self) -> None:
        judge = CountingJudge({"t1": [0.30, 0.34, 0.32]})

        report = run(judge_noise(judge, items(("t1", "code")), repeats=3))

        assert report.min_detectable_delta > 0.01
        assert report.min_detectable_delta == pytest.approx(
            report.category("code").mean_stdev
        )

    def test_item_counts_are_kept_separate_per_category(self) -> None:
        subject = items(("t1", "code"), ("t2", "code"), ("t3", "summary"))

        report = run(judge_noise(constant_judge(), subject, repeats=2))

        assert report.category("code").items == 2
        assert report.category("summary").items == 1

    def test_every_item_is_judged_once_per_repeat(self) -> None:
        judge = constant_judge()

        run(judge_noise(judge, items(("t1", "code"), ("t2", "code")), repeats=4))

        assert judge.call_count == 8

    def test_judging_is_sequential_so_order_is_observable(self) -> None:
        """Concurrent judging would confound batch order with scheduling."""
        judge = constant_judge()
        subject = items(("t1", "code"), ("t2", "code"), ("t3", "code"))

        run(judge_noise(judge, subject, repeats=2))

        assert judge.order_seen == ["t1", "t2", "t3", "t1", "t2", "t3"]

    def test_a_proxy_basis_makes_the_report_uncitable(self) -> None:
        """A proxy is correlated with quality, not a measurement of it."""
        judge = FunctionJudge(lambda _: 0.5, basis=JudgeBasis.PROXY)

        assert (
            run(judge_noise(judge, items(("t1", "code")), repeats=2)).citable is False
        )

    def test_a_model_judge_is_citable(self) -> None:
        judge = FunctionJudge(lambda _: 0.5, basis=JudgeBasis.MODEL)

        assert run(judge_noise(judge, items(("t1", "code")), repeats=2)).citable is True

    def test_basis_counts_are_recorded(self) -> None:
        judge = FunctionJudge(lambda _: 0.5, basis=JudgeBasis.MODEL)

        report = run(judge_noise(judge, items(("t1", "code")), repeats=2))

        assert report.basis_counts[JudgeBasis.MODEL] == 2

    def test_position_bias_is_reported_per_benchmark(self) -> None:
        judge = CountingJudge(
            {"t1": [0.5, 0.5], "t2": [0.5, 0.5]}, position_bias=0.25, deterministic=True
        )

        report = run(
            judge_noise(judge, items(("t1", "code"), ("t2", "code")), repeats=2)
        )

        assert report.category("code").position_bias > 0.0

    def test_position_bias_is_zero_for_a_stable_judge(self) -> None:
        report = run(judge_noise(constant_judge(0.5), items(("t1", "code")), repeats=3))

        assert report.category("code").position_bias == pytest.approx(0.0)

    def test_a_stable_judge_is_unanimous(self) -> None:
        report = run(judge_noise(constant_judge(0.5), items(("t1", "code")), repeats=3))

        assert report.category("code").unanimous_fraction == 1.0

    def test_the_judge_name_is_recorded(self) -> None:
        judge = FunctionJudge(lambda _: 0.5, name="gpt-judge-2026-09")

        report = run(judge_noise(judge, items(("t1", "code")), repeats=2))

        assert report.judge_name == "gpt-judge-2026-09"
        assert report.to_dict()["judge_name"] == "gpt-judge-2026-09"

    def test_the_repeat_count_is_recorded(self) -> None:
        report = run(judge_noise(constant_judge(), items(("t1", "code")), repeats=5))

        assert report.repeats == 5
        assert report.to_dict()["repeats"] == 5

    def test_below_two_repeats_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            run(judge_noise(constant_judge(), items(("t1", "code")), repeats=1))

    def test_the_minimum_repeat_count_is_usable(self) -> None:
        report = run(
            judge_noise(constant_judge(), items(("t1", "code")), repeats=MIN_REPEATS)
        )

        assert report.repeats == MIN_REPEATS

    def test_an_unknown_category_lookup_fails_loudly(self) -> None:
        report = run(judge_noise(constant_judge(), items(("t1", "code")), repeats=2))

        with pytest.raises(KeyError, match="No noise reported"):
            report.category("summary")

    def test_the_report_serializes(self) -> None:
        report = run(judge_noise(constant_judge(0.5), items(("t1", "code")), repeats=2))

        payload = report.to_dict()

        assert payload["per_category"][0]["category"] == "code"
        assert payload["min_detectable_delta"] == 0.0
        assert payload["citable"] is True
        assert json.loads(json.dumps(payload)) == payload

    def test_no_grand_mean_is_offered(self) -> None:
        """One noise number across benchmarks measures the category mix.

        ``per_category`` is a dataclass field, so it is absent from ``dir()`` on
        the class; the aggregate a reader could reach by mistake is a property,
        and those are what this checks for.
        """
        properties = {
            name
            for name in dir(NoiseReport)
            if not name.startswith("_")
            and isinstance(getattr(NoiseReport, name, None), property)
        }

        assert properties == {"citable"}

    def test_the_report_holds_one_entry_per_benchmark(self) -> None:
        subject = items(("t1", "code"), ("t2", "summary"))

        report = run(judge_noise(constant_judge(), subject, repeats=2))

        assert len(report.per_category) == 2

    def test_noise_entry_serializes(self) -> None:
        entry = BenchmarkNoise(
            category="code",
            items=2,
            repeats=3,
            mean_score=0.5,
            mean_stdev=0.1,
            max_range=0.2,
            unanimous_fraction=0.5,
            position_bias=0.0,
            citable_items=2,
        )

        assert entry.to_dict()["max_range"] == 0.2


class TestCompareScores:
    def _scores(
        self, subject: tuple[JudgeRequest, ...], value: float
    ) -> dict[str, JudgeScore]:
        return {
            i.task_id: JudgeScore(score=value, basis=JudgeBasis.MODEL) for i in subject
        }

    def test_reports_later_minus_earlier(self) -> None:
        subject = items(("t1", "code"))

        deltas = compare_scores(
            subject, self._scores(subject, 0.4), subject, self._scores(subject, 0.7)
        )

        assert deltas[0].delta == pytest.approx(0.3)
        assert deltas[0].mean_earlier == pytest.approx(0.4)
        assert deltas[0].mean_later == pytest.approx(0.7)

    def test_a_worse_later_arm_reports_a_negative_delta(self) -> None:
        subject = items(("t1", "code"))

        deltas = compare_scores(
            subject, self._scores(subject, 0.7), subject, self._scores(subject, 0.4)
        )

        assert deltas[0].delta == pytest.approx(-0.3)

    def test_deltas_are_per_category(self) -> None:
        subject = items(("t1", "code"), ("t2", "summary"))

        deltas = compare_scores(
            subject, self._scores(subject, 0.4), subject, self._scores(subject, 0.6)
        )

        assert [d.category for d in deltas] == ["code", "summary"]
        assert all(d.delta == pytest.approx(0.2) for d in deltas)

    def test_different_item_sets_are_refused(self) -> None:
        left = items(("t1", "code"))
        right = items(("t2", "code"))

        with pytest.raises(ValueError, match="different item sets"):
            compare_scores(
                left, self._scores(left, 0.4), right, self._scores(right, 0.6)
            )

    def test_a_missing_score_is_refused(self) -> None:
        subject = items(("t1", "code"))

        with pytest.raises(ValueError, match="never judged"):
            compare_scores(subject, {}, subject, self._scores(subject, 0.6))

    def test_mismatched_categories_between_arms_are_refused(self) -> None:
        """Same task ids, but the arms disagree about which benchmark they are in.

        Only reachable with hand-built score maps, since the two arms arrive as
        their own item lists. Worth pinning because a category silently changing
        hands would attribute a delta to the wrong benchmark.
        """
        earlier = items(("t1", "code"))
        later = items(("t1", "summary"))

        with pytest.raises(ValueError, match="Categories differ between arms"):
            compare_scores(
                earlier, self._scores(earlier, 0.4), later, self._scores(later, 0.9)
            )

    def test_input_order_does_not_change_the_answer(self) -> None:
        """Summation order changes the last bits of a float mean, and a harness
        whose answer moves with input order would report its own noise."""
        subject = items(("t1", "code"), ("t2", "code"), ("t3", "code"))
        scores = {
            "t1": JudgeScore(score=0.1, basis=JudgeBasis.MODEL),
            "t2": JudgeScore(score=0.2, basis=JudgeBasis.MODEL),
            "t3": JudgeScore(score=0.3, basis=JudgeBasis.MODEL),
        }

        forward = compare_scores(subject, scores, subject, scores)
        backward = compare_scores(tuple(reversed(subject)), scores, subject, scores)

        assert forward[0].mean_earlier == backward[0].mean_earlier
        assert forward[0].delta == backward[0].delta

    def test_delta_serializes(self) -> None:
        delta = QualityDelta(
            category="code",
            items=2,
            mean_earlier=0.4,
            mean_later=0.6,
            delta=0.2,
            citable_items=2,
        )

        assert delta.to_dict()["delta"] == 0.2


class TestFunctionJudge:
    def test_wraps_a_plain_float_function(self) -> None:
        judge = FunctionJudge(lambda request: 0.25 if request.task_id == "t1" else 0.75)

        assert run(judge.score(an_item("t1"))).score == 0.25
        assert run(judge.score(an_item("t2"))).score == 0.75

    def test_wraps_a_full_score(self) -> None:
        judge = FunctionJudge(
            lambda _: JudgeScore(score=0.9, basis=JudgeBasis.MODEL, rationale="r")
        )

        score = run(judge.score(an_item()))

        assert score.basis is JudgeBasis.MODEL
        assert score.rationale == "r"

    def test_defaults_to_deterministic(self) -> None:
        assert FunctionJudge(lambda _: 0.5).deterministic is True

    def test_determinism_can_be_declared_false(self) -> None:
        """A user wrapping a model judge must be able to say so."""
        assert FunctionJudge(lambda _: 0.5, deterministic=False).deterministic is False

    def test_counts_its_calls(self) -> None:
        judge = FunctionJudge(lambda _: 0.5)

        run(judge.score(an_item()))
        run(judge.score(an_item()))

        assert judge.call_count == 2

    def test_satisfies_the_judge_protocol(self) -> None:
        assert isinstance(FunctionJudge(lambda _: 0.5), Judge)

    def test_rejects_an_impossible_score_from_the_function(self) -> None:
        """A broken user function is caught at the boundary, not in the mean."""
        judge = FunctionJudge(lambda _: 2.0)

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            run(judge.score(an_item()))

    def test_repr_names_the_instrument(self) -> None:
        text = repr(FunctionJudge(lambda _: 0.5, name="j1", basis=JudgeBasis.MODEL))

        assert "j1" in text
        assert "model" in text

    def test_a_known_category_lookup_returns_the_delta(self) -> None:
        result = run(
            null_condition(constant_judge(), items(("t1", "code"), ("t2", "sum")))
        )

        assert result.category("sum").category == "sum"
        assert result.category("sum").delta == 0.0


class TestIteratorSafety:
    def test_a_generator_of_items_is_accepted(self) -> None:
        """Callers should not have to materialise a list to measure noise."""

        def stream() -> Iterator[JudgeRequest]:
            yield an_item("t1", "code")
            yield an_item("t2", "code")

        report = run(judge_noise(constant_judge(), tuple(stream()), repeats=2))

        assert report.category("code").items == 2
