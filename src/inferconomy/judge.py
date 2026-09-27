"""Judging, the null condition, and the noise floor.

Every quality claim this project will ever make passes through a judge, and a
judge is an instrument with error bars. This module exists to measure those error
bars before anyone is allowed to publish a number that depends on them.

The distinction it draws is between two things that are routinely confused.

**A null condition proves the harness is sound.** It asks a question with no
difference in it and requires the answer to be exactly zero. That catches the
bugs that would otherwise be blamed on the model: mis-zipped joins, a category
assigned to the wrong row, a mean taken over the wrong denominator, units mixed
between arms. A null is only *provable* when the two arms are the same object, in
which case the difference is zero by arithmetic identity no matter how noisy the
judge is. Anything else is only empirically zero, and the size of its failure is
the noise, not a defect.

**The noise floor characterises the judge.** Judging the same response repeatedly
and watching the spread answers a different question: how large a quality delta
this instrument can resolve at all. That number, :attr:`NoiseReport.
min_detectable_delta`, is a ceiling on what may be claimed. A frontier that
reports a 0.4% quality gain against a 1.2% noise floor has reported nothing.

Three things are deliberately refused here. A score outside ``[0, 1]`` or
``NaN`` is rejected rather than averaged, because a judge returning ``1.5`` is
broken and averaging hides it. Aggregation is per category with no grand mean,
for the same reason as everywhere else in the library: one number across
heterogeneous benchmarks measures the category mix. And judging runs
*sequentially*, because the order-invariance check measures a judge's positional
bias, and concurrent judging would confound the order with scheduling.

A judge must declare its own determinism. The null condition can only be
*asserted* against a judge that promises to be deterministic; against a
stochastic one the same check still runs and is reported, as a measurement of
position bias, but its failure is not held against the harness.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BenchmarkNoise",
    "Judge",
    "JudgeBasis",
    "JudgeRequest",
    "JudgeScore",
    "NoiseReport",
    "NullResult",
    "QualityDelta",
    "compare_scores",
    "judge_noise",
    "null_condition",
]

MIN_REPEATS = 2
"""Below this a standard deviation is not defined, so noise cannot be reported."""


class JudgeBasis(str, Enum):
    """What kind of instrument produced a score.

    Recorded per score rather than per run, because a pipeline that mixes a model
    judge with a string-length proxy has not measured quality and should not be
    able to present itself as though it had.
    """

    HUMAN = "human"
    MODEL = "model"
    """A language model scoring the response. Noisy, and honest about being so."""

    HEURISTIC = "heuristic"
    """A deterministic rule, such as an exact-match checker. Reliable and narrow."""

    PROXY = "proxy"
    """Something correlated with quality rather than measuring it, such as output
    length. Not a quality claim."""


@dataclass(frozen=True)
class JudgeRequest:
    """One response to be scored."""

    task_id: str
    category: str
    prompt: str
    response: str
    reference: str | None = None
    """A known-good answer, where the benchmark has one. Judges that can use it
    should; judges that cannot should say so in their basis rather than
    pretending the comparison is symmetric."""

    def __post_init__(self) -> None:
        for name in ("task_id", "category"):
            value = getattr(self, name)
            if not value.strip():
                raise ValueError(f"JudgeRequest.{name} must not be blank.")
        if not self.response.strip():
            raise ValueError(
                "JudgeRequest.response must not be blank. An empty response cannot "
                "be scored, and scoring it as zero would report a judging failure "
                "as a model failure."
            )


@dataclass(frozen=True)
class JudgeScore:
    """One score, with the provenance of the instrument that produced it."""

    score: float
    basis: JudgeBasis
    rationale: str | None = None

    def __post_init__(self) -> None:
        value = float(self.score)
        if not math.isfinite(value):
            raise ValueError(
                f"JudgeScore.score must be a finite number, got {self.score!r}. A "
                f"NaN would poison every mean it entered."
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"JudgeScore.score must lie in [0, 1], got {self.score!r}. Scores "
                f"outside that range cannot be averaged with scores inside it."
            )
        object.__setattr__(self, "score", value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "basis": self.basis.value,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JudgeScore:
        try:
            return cls(
                score=float(data["score"]),
                basis=JudgeBasis(data["basis"]),
                rationale=data.get("rationale"),
            )
        except KeyError as exc:
            raise ValueError(
                f"Judge score is missing required field: {exc.args[0]!r}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"Judge score has an unusable basis: {exc}") from exc


@runtime_checkable
class Judge(Protocol):
    """Scores one response.

    Deliberately narrower than a typical evaluation harness. A judge returns a
    score and says what kind of instrument it is; it does not aggregate, average,
    or decide whether a delta is significant. Those are this module's job,
    because a judge that summarises its own results is a judge nobody can audit.
    """

    @property
    def name(self) -> str:
        """Identifier recorded in results. Should name the instrument and version,
        such as ``"gpt-judge-2026-09"``, so a score is traceable to its source."""
        ...

    @property
    def deterministic(self) -> bool:
        """Whether repeated calls with identical input return identical scores.

        Declared, not probed. Probing costs a judge call per item and, against a
        model judge, measures the network as much as the instrument. A judge that
        lies about this gets its null condition reported rather than asserted,
        which is a worse but not dishonest outcome.
        """
        ...

    async def score(self, request: JudgeRequest) -> JudgeScore: ...


@dataclass(frozen=True)
class QualityDelta:
    """Quality difference for one category, later arm minus earlier arm."""

    category: str
    items: int
    mean_earlier: float
    mean_later: float
    delta: float
    """``mean_later - mean_earlier``, so a positive delta means the later
    configuration scored higher. The sign is stated rather than implied because
    for cost the good direction is negative and for quality it is positive, and a
    delta whose good direction depends on the metric is a delta that will be
    quoted backwards."""

    citable_items: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "items": self.items,
            "mean_earlier": self.mean_earlier,
            "mean_later": self.mean_later,
            "delta": self.delta,
            "citable_items": self.citable_items,
        }


@dataclass(frozen=True)
class BenchmarkNoise:
    """How much one judge moves on one benchmark."""

    category: str
    items: int
    repeats: int
    mean_score: float
    mean_stdev: float
    """Average, over items, of the per-item sample standard deviation. The typical
    disagreement between two runs of the judge."""

    max_range: float
    """Largest single-item spread seen. The tail, which ``mean_stdev`` hides."""

    unanimous_fraction: float
    """Share of items where every repeat returned a bit-identical score. A
    stochastic judge drives this toward zero, and that is information rather than
    failure."""

    position_bias: float
    """Mean absolute shift in per-item score when the same items are judged in
    reverse order. Nonzero means the judge's verdict depends on where the item
    sat in the batch, which is a real defect in a model judge and worth measuring
    even when it cannot be asserted."""

    citable_items: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "items": self.items,
            "repeats": self.repeats,
            "mean_score": self.mean_score,
            "mean_stdev": self.mean_stdev,
            "max_range": self.max_range,
            "unanimous_fraction": self.unanimous_fraction,
            "position_bias": self.position_bias,
            "citable_items": self.citable_items,
        }


@dataclass(frozen=True)
class NoiseReport:
    """Judge noise per benchmark, and the smallest delta it can resolve."""

    per_category: tuple[BenchmarkNoise, ...]
    repeats: int
    judge_name: str
    basis_counts: Mapping[JudgeBasis, int] = field(default_factory=dict)
    min_detectable_delta: float = 0.0
    """Largest per-category ``mean_stdev``, as a floor on claimable deltas.

    A maximum across categories rather than a mean, because a mean would let a
    quiet category hide a noisy one. A delta smaller than this cannot be
    distinguished from the instrument's own wobble, and publishing it would be
    reporting the judge rather than the system.
    """

    @property
    def citable(self) -> bool:
        """False if any score came from a proxy.

        A proxy is correlated with quality rather than measuring it, so a report
        containing one is not a quality measurement even though the arithmetic is
        fine.
        """
        return not self.basis_counts.get(JudgeBasis.PROXY, 0)

    def category(self, name: str) -> BenchmarkNoise:
        for entry in self.per_category:
            if entry.category == name:
                return entry
        raise KeyError(
            f"No noise reported for category {name!r}. Known: "
            f"{[e.category for e in self.per_category]}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "repeats": self.repeats,
            "citable": self.citable,
            "min_detectable_delta": self.min_detectable_delta,
            "basis_counts": {
                basis.value: n
                for basis, n in sorted(
                    self.basis_counts.items(), key=lambda kv: kv[0].value
                )
            },
            "per_category": [entry.to_dict() for entry in self.per_category],
        }


@dataclass(frozen=True)
class NullResult:
    """Outcome of asking a question with no difference in it."""

    self_delta: float
    """Difference between a score set and itself. Zero by arithmetic identity, so
    a nonzero value is a defect in this module and not a fact about the judge."""

    self_passed: bool
    order_delta: float
    """Difference between judging the items in order and in reverse. Not trivially
    zero, so it measures the judge's positional stability."""

    order_assertable: bool
    """Whether the judge declared itself deterministic, which is the only case in
    which ``order_passed`` is held against the run."""

    order_passed: bool
    by_category: tuple[QualityDelta, ...]
    repeats: int
    judge_name: str

    @property
    def passed(self) -> bool:
        """The provable part must hold; the order part counts only when the judge
        promised determinism."""
        if not self.self_passed:
            return False
        return self.order_passed or not self.order_assertable

    @property
    def citable(self) -> bool:
        return self.passed and bool(self.by_category)

    def category(self, name: str) -> QualityDelta:
        for entry in self.by_category:
            if entry.category == name:
                return entry
        raise KeyError(
            f"No null delta for category {name!r}. Known: "
            f"{[e.category for e in self.by_category]}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "repeats": self.repeats,
            "passed": self.passed,
            "citable": self.citable,
            "self_delta": self.self_delta,
            "self_passed": self.self_passed,
            "order_delta": self.order_delta,
            "order_passed": self.order_passed,
            "order_assertable": self.order_assertable,
            "by_category": [entry.to_dict() for entry in self.by_category],
        }


async def _score_in_order(
    judge: Judge, items: Sequence[JudgeRequest]
) -> dict[str, JudgeScore]:
    """Judge ``items`` one at a time, keyed by task id.

    Sequential on purpose. The order-invariance check asks whether a judge's
    verdict depends on batch position, and issuing the calls concurrently would
    make the interleaving a second, invisible source of variation.
    """
    scored: dict[str, JudgeScore] = {}
    for item in items:
        if item.task_id in scored:
            raise ValueError(
                f"Duplicate task_id {item.task_id!r} in judge items. Two items "
                f"sharing an id would silently overwrite each other."
            )
        scored[item.task_id] = await judge.score(item)
    return scored


def _means_by_category(
    scored: Mapping[str, JudgeScore], items: Sequence[JudgeRequest]
) -> dict[str, tuple[float, int, int]]:
    """Mean score per category, with item and citable counts.

    Iterates in a canonical ``(category, task_id)`` order so the result cannot
    depend on the order items arrived in. Summation order changes the last bits
    of a float mean, and a harness whose answer moves with input order would
    report noise that is its own.
    """
    by_category: dict[str, list[JudgeScore]] = {}
    for item in sorted(items, key=lambda i: (i.category, i.task_id)):
        score = scored.get(item.task_id)
        if score is None:
            raise ValueError(f"Item {item.task_id!r} was never judged.")
        by_category.setdefault(item.category, []).append(score)
    return {
        category: (
            statistics.fmean(s.score for s in scores),
            len(scores),
            sum(1 for s in scores if s.basis is not JudgeBasis.PROXY),
        )
        for category, scores in by_category.items()
    }


def _stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def compare_scores(
    earlier: Sequence[JudgeRequest],
    earlier_scores: Mapping[str, JudgeScore],
    later: Sequence[JudgeRequest],
    later_scores: Mapping[str, JudgeScore],
) -> tuple[QualityDelta, ...]:
    """Per-category quality difference, later arm minus earlier arm.

    Both arms must cover the same task ids. Comparing different item sets would
    report the difference between the benchmarks rather than between the
    configurations.
    """
    if {item.task_id for item in earlier} != {item.task_id for item in later}:
        raise ValueError(
            "The two arms cover different task ids. A quality delta between "
            "different item sets measures the item sets."
        )
    means_a = _means_by_category(earlier_scores, earlier)
    means_b = _means_by_category(later_scores, later)
    if set(means_a) != set(means_b):
        raise ValueError(
            f"Categories differ between arms: {sorted(means_a)} vs {sorted(means_b)}."
        )
    return tuple(
        QualityDelta(
            category=category,
            items=means_a[category][1],
            mean_earlier=means_a[category][0],
            mean_later=means_b[category][0],
            delta=means_b[category][0] - means_a[category][0],
            citable_items=min(means_a[category][2], means_b[category][2]),
        )
        for category in sorted(means_a)
    )


def _validate_items(items: Sequence[JudgeRequest], repeats: int) -> None:
    if not items:
        raise ValueError("Nothing to judge. An empty item set measures no noise.")
    if repeats < MIN_REPEATS:
        raise ValueError(
            f"repeats must be at least {MIN_REPEATS} to measure noise, got {repeats}. "
            f"A single judging pass has no spread to report."
        )


async def judge_noise(
    judge: Judge, items: Sequence[JudgeRequest], repeats: int = 3
) -> NoiseReport:
    """Judge every item ``repeats`` times and quantify the spread, per category.

    Three separate things get measured, because they fail differently. ``mean_stdev``
    is the typical wobble between two runs. ``max_range`` is the worst single item,
    which a mean hides. ``unanimous_fraction`` says how often the judge agreed with
    itself at all, which is the difference between an instrument and a coin.

    The last repeat is the one reported as the score, so a caller that only wants a
    score can use this without judging twice.
    """
    _validate_items(items, repeats)

    passes: list[dict[str, JudgeScore]] = []
    for _ in range(repeats):
        passes.append(await _score_in_order(judge, items))
    final = passes[-1]

    per_category: list[BenchmarkNoise] = []
    for category in sorted({item.category for item in items}):
        members = sorted(
            (i for i in items if i.category == category), key=lambda i: i.task_id
        )
        series = [[pass_[i.task_id].score for pass_ in passes] for i in members]
        stdevs = [_stdev(values) for values in series]
        ranges = [max(values) - min(values) for values in series]
        final_scores = [final[i.task_id].score for i in members]
        shifts = [
            abs(
                statistics.fmean(p[i.task_id].score for p in (passes[0], passes[-1]))
                - final_scores[position]
            )
            for position, i in enumerate(members)
        ]
        per_category.append(
            BenchmarkNoise(
                category=category,
                items=len(members),
                repeats=repeats,
                mean_score=statistics.fmean(final_scores),
                mean_stdev=statistics.fmean(stdevs),
                max_range=max(ranges),
                unanimous_fraction=sum(1 for r in ranges if r == 0.0) / len(members),
                position_bias=statistics.fmean(shifts),
                citable_items=sum(
                    1 for i in members if final[i.task_id].basis is not JudgeBasis.PROXY
                ),
            )
        )

    basis_counts: dict[JudgeBasis, int] = {}
    for scores in passes:
        for item in items:
            basis = scores[item.task_id].basis
            basis_counts[basis] = basis_counts.get(basis, 0) + 1

    return NoiseReport(
        per_category=tuple(per_category),
        repeats=repeats,
        judge_name=judge.name,
        basis_counts=basis_counts,
        min_detectable_delta=max((e.mean_stdev for e in per_category), default=0.0),
    )


async def null_condition(
    judge: Judge, items: Sequence[JudgeRequest], repeats: int = 2
) -> NullResult:
    """Ask a question with no difference in it, and require the answer to be zero.

    Two checks, deliberately different in kind.

    The first compares a score set against itself. Zero is guaranteed by
    arithmetic, so a nonzero result can only mean this module is miscounting. It
    is a hard assertion and is always enforced.

    The second judges the same items in reverse order. Zero is *not* guaranteed
    here — it depends on the judge — so it measures positional stability. It is
    asserted only when the judge declared itself deterministic, and reported
    either way, because a model judge that shifts with batch position is a real
    finding rather than a failed test.
    """
    _validate_items(items, repeats)

    forward = await _score_in_order(judge, items)
    reversed_scores = await _score_in_order(judge, list(reversed(items)))

    by_category = compare_scores(items, forward, items, forward)
    order_by_category = compare_scores(items, forward, items, reversed_scores)

    order_delta = max((abs(d.delta) for d in order_by_category), default=0.0)
    assertable = bool(getattr(judge, "deterministic", False))
    return NullResult(
        self_delta=max((abs(d.delta) for d in by_category), default=0.0),
        self_passed=all(d.delta == 0.0 for d in by_category),
        order_delta=order_delta,
        order_assertable=assertable,
        order_passed=order_delta == 0.0,
        by_category=by_category,
        repeats=repeats,
        judge_name=judge.name,
    )
