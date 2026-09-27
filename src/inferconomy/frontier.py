"""Cost-quality frontiers, and the gate that decides whether one may be published.

A frontier is a curve: spend more, get more. Stated that way it is the easiest
number in inference to manufacture, because you can pick where to start the
x-axis, which points to draw, and which metric counts as quality. The curve will
look excellent in all three cases.

So the shape of this module is defensive. A :class:`Frontier` cannot be
published unless it carries evidence for all four of the project's controls, and
:meth:`Frontier.publish` raises rather than returning a payload that is missing
one. The controls travel inside the serialized output, so a reader of a results
file meets them at the same time as the curve rather than trusting that they were
checked.

The four, and what each one actually rules out:

1. **Oracle bound.** How much could perfect allocation have saved? Without it a
   30% saving is uninterpretable, because the maximum might have been 90%.
2. **Cost-matched baseline.** The baseline is allowed the same tokens. This is
   checked numerically rather than asserted: a baseline that used fewer tokens
   than the treatment is a strawman, and a strawman flatters every treatment.
3. **Null condition and calibrated judging.** A null configuration must measure
   as a difference of zero, and the judge must be a measuring instrument rather
   than a proxy. A delta smaller than the measured noise is not a finding.
4. **Per-domain reporting.** Per category, never aggregated. Enforced by the
   shape of the API: there is no method that returns one number for the curve.

One consequence is worth stating plainly. A point whose quality delta falls below
the judge's measured noise is *not resolvable*, and :attr:`FrontierPoint.citable`
is False for it. The point still appears on the curve, because hiding it would
misrepresent the shape, but a resolver below the floor is not a result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from inferconomy.benchmark import OracleResult, RunResult
from inferconomy.judge import NoiseReport, NullResult

__all__ = [
    "CONTROLS",
    "FRONTIER_SCHEMA_VERSION",
    "Control",
    "Frontier",
    "FrontierCategory",
    "FrontierConfig",
    "FrontierPoint",
    "UnpublishableFrontier",
    "build_frontier",
]

FRONTIER_SCHEMA_VERSION = 1

CONTROLS = (
    "oracle_bound",
    "cost_matched_baseline",
    "null_and_noise",
    "per_domain_reporting",
)

TOKEN_SHORTFALL_TOLERANCE = 0.05
"""How far a treatment may exceed the baseline's tokens before the comparison is
a strawman. 5% absorbs the difference between two measured runs of the same
budget without allowing a treatment to quietly spend more and still be called the
cheaper one."""


class UnpublishableFrontier(ValueError):
    """Raised when a frontier is missing evidence for a control.

    A subclass of :class:`ValueError` because a frontier that cannot be published
    is a malformed result, not a runtime failure.
    """


@dataclass(frozen=True)
class FrontierConfig:
    """The committed description of a frontier run.

    Fingerprinted for the same reason :class:`~inferconomy.benchmark.BaselineConfig`
    is: a claim of reproducibility should be a comparison of two hashes.
    """

    budgets: tuple[int, ...]
    model: str
    judge_name: str
    price_file: str | None = None
    repetitions: int = 1
    categories: tuple[str, ...] = ()
    baseline_budget: int | None = None
    """Which budget is the reference arm. Defaults to the *largest*.

    This is the arm a savings claim is measured against, so it must be the one
    that was given the most compute. Defaulting to the smallest would make the
    project's own cost-matched-baseline control unsatisfiable: on a budget sweep
    the cheapest arm is by definition the one holding the fewest tokens, so every
    other point would look like it had beaten a starved baseline. Measuring
    savings against the generous arm is also the question that matters, which is
    what better allocation would save against always spending enough.
    """

    schema_version: int = FRONTIER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "budgets", tuple(self.budgets))
        object.__setattr__(self, "categories", tuple(self.categories))
        if not self.budgets:
            raise ValueError("A frontier needs at least one budget.")
        if len(set(self.budgets)) != len(self.budgets):
            raise ValueError(f"Duplicate budgets in {self.budgets}.")
        if any(b <= 0 for b in self.budgets):
            raise ValueError(f"Budgets must be positive, got {self.budgets}.")
        if not self.model.strip():
            raise ValueError("FrontierConfig.model must not be blank.")
        if not self.judge_name.strip():
            raise ValueError(
                "FrontierConfig.judge_name must not be blank. A quality number with "
                "no named instrument attached is not a measurement."
            )
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be at least 1, got {self.repetitions}.")
        resolved = self.baseline_budget
        if resolved is None:
            resolved = max(self.budgets)
        if resolved not in self.budgets:
            raise ValueError(
                f"baseline_budget {resolved} is not among budgets {self.budgets}."
            )
        object.__setattr__(self, "baseline_budget", resolved)
        if self.categories and len(set(self.categories)) != len(self.categories):
            raise ValueError(f"Duplicate categories in {self.categories}.")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "budgets": list(self.budgets),
            "model": self.model,
            "judge_name": self.judge_name,
            "price_file": self.price_file,
            "repetitions": self.repetitions,
            "categories": list(self.categories),
            "baseline_budget": self.baseline_budget,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrontierConfig:
        try:
            return cls(
                budgets=tuple(int(b) for b in data["budgets"]),
                model=str(data["model"]),
                judge_name=str(data["judge_name"]),
                price_file=data.get("price_file"),
                repetitions=int(data.get("repetitions", 1)),
                categories=tuple(str(c) for c in data.get("categories", ())),
                baseline_budget=(
                    int(data["baseline_budget"])
                    if data.get("baseline_budget")
                    else None
                ),
                schema_version=int(data.get("schema_version", FRONTIER_SCHEMA_VERSION)),
            )
        except KeyError as exc:
            raise ValueError(
                f"Frontier config is missing required field: {exc.args[0]!r}"
            ) from exc

    @classmethod
    def from_file(cls, path: str) -> FrontierConfig:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_file(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path


@dataclass(frozen=True)
class Control:
    """One of the four checks, with the reason it passed or failed."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class FrontierPoint:
    """One measured point on the curve."""

    category: str
    budget: int
    cost_usd: float
    quality: float
    tasks: int
    is_baseline: bool
    total_tokens: int
    noise_floor: float
    """The judge's per-category ``mean_stdev`` for this benchmark. A quality delta
    smaller than this cannot be told apart from the instrument's own wobble."""

    delta_vs_baseline: float
    resolvable: bool
    """Whether ``delta_vs_baseline`` exceeds the noise floor in magnitude. False
    means the point is on the curve but its quality difference is not a finding.
    The baseline point itself is always resolvable: it defines the reference, so
    there is no delta to resolve and calling it unresolvable would bury the
    signal in one meaningless entry per category."""

    cost_citable: bool
    quality_citable: bool
    price_table_version: str

    @property
    def citable(self) -> bool:
        """Both halves of the point must be trustworthy for either number to be."""
        return self.cost_citable and self.quality_citable

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "budget": self.budget,
            "cost_usd": self.cost_usd,
            "quality": self.quality,
            "tasks": self.tasks,
            "is_baseline": self.is_baseline,
            "total_tokens": self.total_tokens,
            "noise_floor": self.noise_floor,
            "delta_vs_baseline": self.delta_vs_baseline,
            "resolvable": self.resolvable,
            "cost_citable": self.cost_citable,
            "quality_citable": self.quality_citable,
            "citable": self.citable,
            "price_table_version": self.price_table_version,
        }


@dataclass(frozen=True)
class FrontierCategory:
    """A whole curve for one category, with its own savings accounting."""

    category: str
    points: tuple[FrontierPoint, ...]
    baseline_budget: int
    baseline_cost_usd: float
    oracle_cost_usd: float | None
    max_savings_usd: float | None
    """How much perfect allocation could have saved at the baseline, from the
    oracle. ``None`` when no oracle was supplied, which fails the first control
    rather than defaulting to an unbounded claim."""

    @property
    def captured_savings_fraction(self) -> float | None:
        """Share of the oracle's bound that the curve's best point captures.

        Bounded above by 1.0 by construction, and ``None`` without an oracle
        rather than reported against an assumed ceiling of 100%.
        """
        if self.max_savings_usd is None or self.max_savings_usd <= 0:
            return None
        best = min(point.cost_usd for point in self.points)
        saved = self.baseline_cost_usd - best
        return min(1.0, saved / self.max_savings_usd)

    def point(self, budget: int) -> FrontierPoint:
        for entry in self.points:
            if entry.budget == budget:
                return entry
        raise KeyError(
            f"No point at budget {budget} for category {self.category!r}. "
            f"Known: {[p.budget for p in self.points]}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "baseline_budget": self.baseline_budget,
            "baseline_cost_usd": self.baseline_cost_usd,
            "oracle_cost_usd": self.oracle_cost_usd,
            "max_savings_usd": self.max_savings_usd,
            "captured_savings_fraction": self.captured_savings_fraction,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass(frozen=True)
class Frontier:
    """A cost-quality curve, with the evidence required to publish it."""

    config_fingerprint: str
    categories: tuple[FrontierCategory, ...]
    controls: tuple[Control, ...]
    judge_name: str
    min_detectable_delta: float
    published: bool = False
    """Whether :meth:`publish` has been called successfully. False by default, so
    an assembled frontier cannot be mistaken for a vetted one by inspecting the
    object."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", tuple(self.categories))
        object.__setattr__(self, "controls", tuple(self.controls))

    @property
    def failed_controls(self) -> tuple[Control, ...]:
        return tuple(control for control in self.controls if not control.passed)

    @property
    def publishable(self) -> bool:
        """All four controls passing. Not sufficient on its own to have published."""
        return len(self.controls) == len(CONTROLS) and not self.failed_controls

    @property
    def citable(self) -> bool:
        """Whether every point on the curve has trustworthy cost and quality."""
        return self.publishable and all(
            point.citable for category in self.categories for point in category.points
        )

    def category(self, name: str) -> FrontierCategory:
        for entry in self.categories:
            if entry.category == name:
                return entry
        raise KeyError(
            f"No frontier for category {name!r}. Known: "
            f"{[c.category for c in self.categories]}."
        )

    def unresolvable(self) -> tuple[FrontierPoint, ...]:
        """Points whose quality difference is inside the judge's noise."""
        return tuple(
            point
            for category in self.categories
            for point in category.points
            if not point.resolvable
        )

    def publish(self) -> dict[str, Any]:
        """Return the publishable payload, or refuse.

        Raises :class:`UnpublishableFrontier` naming every failing control. A
        frontier missing its oracle, its null condition, or a cost-matched
        baseline is not a weak result, it is an unusable one, and the error says
        which check to go and satisfy.
        """
        if not self.publishable:
            failed = ", ".join(c.name for c in self.failed_controls)
            raise UnpublishableFrontier(
                f"Frontier cannot be published: failing controls {failed}. "
                f"{len(self.failed_controls)} of {len(CONTROLS)} required."
            )
        object.__setattr__(self, "published", True)
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_fingerprint": self.config_fingerprint,
            "judge_name": self.judge_name,
            "min_detectable_delta": self.min_detectable_delta,
            "published": self.published,
            "publishable": self.publishable,
            "citable": self.citable,
            "controls": [control.to_dict() for control in self.controls],
            "per_category": [category.to_dict() for category in self.categories],
        }


def _category_means(
    run: RunResult, quality: Mapping[str, float]
) -> dict[str, tuple[float, int, int, bool]]:
    """Per category: mean cost, item count, token sum, and whether costs are citable."""
    grouped: dict[str, list[Any]] = {}
    for row in run.rows:
        grouped.setdefault(row.category, []).append(row)
    out: dict[str, tuple[float, int, int, bool]] = {}
    for category, rows in grouped.items():
        priced = [r for r in rows if r.cost_usd is not None]
        costs = [r.cost_usd or 0.0 for r in priced]
        tokens = sum(r.total_tokens for r in priced)
        out[category] = (
            sum(costs) / len(costs) if costs else 0.0,
            len(priced),
            tokens,
            bool(priced) and all(r.citable for r in priced),
        )
    return out


def build_frontier(
    config: FrontierConfig,
    runs: Mapping[int, RunResult],
    quality: Mapping[int, Mapping[str, float]],
    *,
    oracle: OracleResult | None = None,
    null: NullResult | None = None,
    noise: NoiseReport | None = None,
) -> Frontier:
    """Assemble a frontier from measured runs and their judged quality.

    ``runs`` maps budget to the run at that budget; ``quality`` maps the same
    budget to category to mean judged score. The two must agree on budgets, since
    a point with a cost and no quality is not a point on a cost-quality curve.

    ``oracle``, ``null``, and ``noise`` are optional in the signature and
    required in practice. Passing none of them produces a frontier that cannot be
    published, which is the intended outcome rather than an argument to relax the
    gate.
    """
    if set(runs) != set(config.budgets):
        raise ValueError(
            f"Runs cover budgets {sorted(runs)} but config declares "
            f"{sorted(config.budgets)}. Every declared budget needs a run."
        )
    if set(quality) != set(config.budgets):
        raise ValueError(
            f"Quality covers budgets {sorted(quality)} but config declares "
            f"{sorted(config.budgets)}."
        )
    baseline_budget = config.baseline_budget
    assert baseline_budget is not None  # narrowed in __post_init__

    baseline_means = _category_means(runs[baseline_budget], quality[baseline_budget])
    all_categories = sorted(
        {c for budget in config.budgets for c in quality[budget]} | set(baseline_means)
    )
    if config.categories and set(config.categories) != set(all_categories):
        raise ValueError(
            f"Config declares categories {sorted(config.categories)} but the data "
            f"covers {sorted(all_categories)}."
        )

    noise_floor = {
        entry.category: entry.mean_stdev
        for entry in (noise.per_category if noise else ())
    }
    oracle_savings: dict[str, float] = {}
    oracle_aligned = oracle is not None and oracle.baseline_budget == baseline_budget
    if oracle is not None and oracle_aligned:
        for row in oracle.comparable_rows:
            oracle_savings[row.category] = row.savings_usd

    categories: list[FrontierCategory] = []
    for category in all_categories:
        points: list[FrontierPoint] = []
        for budget in sorted(config.budgets):
            means = _category_means(runs[budget], quality[budget])
            if category not in means or category not in quality[budget]:
                raise ValueError(
                    f"Category {category!r} is missing at budget {budget}. A curve "
                    f"with a hole in it is not a curve."
                )
            mean_cost, count, tokens, cost_citable = means[category]
            score = float(quality[budget][category])
            if not 0.0 <= score <= 1.0 or not math.isfinite(score):
                raise ValueError(
                    f"Quality for {category!r} at budget {budget} is {score!r}, "
                    f"which is not a usable score."
                )
            baseline_score = float(quality[baseline_budget][category])
            floor = noise_floor.get(category, 0.0)
            delta = score - baseline_score
            points.append(
                FrontierPoint(
                    category=category,
                    budget=budget,
                    cost_usd=mean_cost,
                    quality=score,
                    tasks=count,
                    is_baseline=budget == baseline_budget,
                    total_tokens=tokens,
                    noise_floor=floor,
                    delta_vs_baseline=delta,
                    resolvable=budget == baseline_budget or abs(delta) >= floor,
                    cost_citable=cost_citable,
                    quality_citable=noise is None or noise.citable,
                    price_table_version=runs[budget].price_table_version,
                )
            )
        points.sort(key=lambda p: p.cost_usd)
        categories.append(
            FrontierCategory(
                category=category,
                points=tuple(points),
                baseline_budget=baseline_budget,
                baseline_cost_usd=baseline_means[category][0]
                if category in baseline_means
                else 0.0,
                oracle_cost_usd=(
                    baseline_means[category][0] - oracle_savings[category]
                    if category in baseline_means and category in oracle_savings
                    else None
                ),
                max_savings_usd=oracle_savings.get(category),
            )
        )

    return Frontier(
        config_fingerprint=config.fingerprint,
        categories=tuple(categories),
        controls=_check_controls(
            config, categories, oracle, null, noise, oracle_aligned
        ),
        judge_name=config.judge_name,
        min_detectable_delta=noise.min_detectable_delta if noise else 0.0,
    )


def _check_controls(
    config: FrontierConfig,
    categories: Sequence[FrontierCategory],
    oracle: OracleResult | None,
    null: NullResult | None,
    noise: NoiseReport | None,
    oracle_aligned: bool,
) -> tuple[Control, ...]:
    """Evaluate the four controls. Each returns a reason, not just a boolean."""
    controls: list[Control] = []

    if oracle is None:
        controls.append(
            Control(
                "oracle_bound",
                False,
                "No oracle supplied, so there is no upper bound on what better "
                "allocation could have saved. Every saving below is "
                "uninterpretable without one.",
            )
        )
    elif not oracle_aligned:
        assert oracle is not None
        controls.append(
            Control(
                "oracle_bound",
                False,
                f"The oracle's bound was measured against budget "
                f"{oracle.baseline_budget} but this frontier's reference arm is "
                f"budget {config.baseline_budget}. A saving divided by a bound "
                f"computed from a different baseline is a ratio of two "
                f"unrelated numbers.",
            )
        )
    else:
        covered = {c.category for c in categories if c.max_savings_usd is not None}
        controls.append(
            Control(
                "oracle_bound",
                covered == {c.category for c in categories},
                f"Oracle bound supplied for {len(covered)} of "
                f"{len(categories)} categories.",
            )
        )

    starved: list[str] = []
    for category in categories:
        baseline = next((p for p in category.points if p.is_baseline), None)
        if baseline is None:
            starved.append(f"{category.category} (no baseline point)")
            continue
        ceiling = baseline.total_tokens * (1.0 + TOKEN_SHORTFALL_TOLERANCE)
        for point in category.points:
            if point.total_tokens > ceiling:
                starved.append(
                    f"{category.category}@{point.budget} used {point.total_tokens} "
                    f"tokens against a baseline of {baseline.total_tokens}"
                )
    controls.append(
        Control(
            "cost_matched_baseline",
            not starved,
            "Baseline allowed at least as many tokens as every treatment."
            if not starved
            else f"Baseline was starved: {'; '.join(starved)}.",
        )
    )

    reasons: list[str] = []
    if null is None:
        reasons.append("no null condition was run")
    elif not null.passed:
        reasons.append(f"null condition failed (self_delta={null.self_delta})")
    if noise is None:
        reasons.append("judge noise was not measured")
    elif not noise.citable:
        reasons.append("judge was a proxy rather than a measuring instrument")
    controls.append(
        Control(
            "null_and_noise",
            not reasons,
            "Null condition passed and judge noise measured."
            if not reasons
            else f"Judging is not established: {'; '.join(reasons)}.",
        )
    )

    controls.append(
        Control(
            "per_domain_reporting",
            len(categories) > 0,
            f"Reported for {len(categories)} categories, with no aggregate curve "
            f"exposed by this module.",
        )
    )
    assert tuple(c.name for c in controls) == CONTROLS
    return tuple(controls)
