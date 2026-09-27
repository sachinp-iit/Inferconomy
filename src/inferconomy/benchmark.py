"""Fixed-budget baseline runs.

A baseline is the reference point every adaptive result is measured against, so
its most important property is not that it is simple but that it is *fixed*. The
same task, the same budget, the same prompt, every time, with no inspection of
the output and no second attempt. If the baseline can improve its own result,
there is nothing left to measure against.

Two things follow from that, and both are enforced here rather than documented
and hoped for.

**A run is reproducible from its config alone.** :class:`BaselineConfig`
serializes to JSON, and :attr:`BaselineConfig.fingerprint` is a hash of that
canonical form. The fingerprint is recorded in the result, so a claim of
reproducibility is a comparison of two hashes rather than a promise. A wall-clock
timestamp is deliberately absent from the result: a field that changes on every
run would make the comparison meaningless.

**There is no single headline number.** :meth:`RunResult.by_category` returns
per-category summaries and nothing else. Gains on code reasoning do not transfer
to summarization, and a mean across twelve categories is a number that looks
like evidence while being a weighted average of whatever the mix happened to be.

Quality is not part of this story. Every row carries ``quality=None`` rather
than a placeholder score, so a row cannot be read as "passed" before a judge
exists to say so.

:func:`compute_oracle` builds the upper bound on what any adaptive policy could
have saved, using the same runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from inferconomy.client import Client, CompletionOptions
from inferconomy.contracts import Request, UsageBasis, decode_enum
from inferconomy.costs import CostTableError, PriceTable, load_price_table

__all__ = [
    "ORACLE_CAVEATS",
    "BaselineConfig",
    "CategorySummary",
    "MetricRow",
    "OracleCategorySummary",
    "OracleResult",
    "OracleRow",
    "RunResult",
    "Task",
    "compute_oracle",
    "run_baseline",
]

CONFIG_SCHEMA_VERSION = 1
"""Shape of the config file. Bumped only when the format changes."""


@dataclass(frozen=True)
class Task:
    """One benchmark item.

    :attr:`category` is not decoration. Per-category reporting is the only
    reporting this module offers, so every task must declare which domain it
    belongs to or it cannot be summarised honestly.
    """

    task_id: str
    prompt: str
    category: str
    system: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "category"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank.")
        if not self.prompt.strip():
            raise ValueError(f"Task {self.task_id!r} has a blank prompt.")

    def to_request(self) -> Request:
        return Request.from_text(self.prompt, system=self.system)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "category": self.category,
            "system": self.system,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Task:
        try:
            return cls(
                task_id=str(data["task_id"]),
                prompt=str(data["prompt"]),
                category=str(data["category"]),
                system=None if data.get("system") is None else str(data["system"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"Task is missing required field: {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True)
class BaselineConfig:
    """Everything a baseline run depends on, and nothing it does not.

    A config that omitted any of these would not determine a run, and a run that
    could not be reconstructed from its config could not be checked. The
    fingerprint exists to make that check mechanical.
    """

    tasks: tuple[Task, ...]
    budget: int
    """Output tokens allowed per call. The thing being held fixed."""

    strategy: str = "direct"
    """Label for the condition being measured, not a request to be adaptive."""

    model: str | None = None
    price_file: str | None = None
    """Path to a user price table. ``None`` means the bundled snapshot."""

    repetitions: int = 1
    """Independent samples per task. Real providers are not deterministic; a
    single sample is not evidence."""

    schema_version: ClassVar[int] = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        for index, task in enumerate(self.tasks):
            if not isinstance(task, Task):
                raise ValueError(f"Task {index} is not a Task: {task!r}.")
        if not self.tasks:
            raise ValueError(
                "A baseline needs at least one task; an empty run proves nothing."
            )
        duplicates = _duplicates(task.task_id for task in self.tasks)
        if duplicates:
            raise ValueError(
                "Duplicate task_id(s), which would make rows ambiguous: "
                + ", ".join(sorted(duplicates))
            )
        if self.budget <= 0:
            raise ValueError(f"budget must be positive, got {self.budget}.")
        if self.repetitions <= 0:
            raise ValueError(f"repetitions must be positive, got {self.repetitions}.")
        if not self.strategy.strip():
            raise ValueError("strategy must not be blank.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "model": self.model,
            "budget": self.budget,
            "price_file": self.price_file,
            "repetitions": self.repetitions,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_json(self) -> str:
        """Canonical JSON, so the same config always fingerprints identically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BaselineConfig:
        version = int(data.get("schema_version", 0))
        if version > CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Config schema version {version} is newer than this release "
                f"understands ({CONFIG_SCHEMA_VERSION}). Upgrade Inferconomy."
            )
        if version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported config schema version: {version}.")
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, str):
            raise ValueError("Config is missing a 'tasks' array.")
        for field_name in ("budget",):
            if field_name not in data:
                raise ValueError(f"Config is missing required field: {field_name!r}.")
        return cls(
            tasks=tuple(Task.from_dict(item) for item in raw_tasks),
            budget=int(data["budget"]),
            strategy=str(data.get("strategy", "direct")),
            model=None if data.get("model") is None else str(data["model"]),
            price_file=None
            if data.get("price_file") is None
            else str(data["price_file"]),
            repetitions=int(data.get("repetitions", 1)),
        )

    @classmethod
    def from_json(cls, text: str) -> BaselineConfig:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Config is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @property
    def fingerprint(self) -> str:
        """Hash of the canonical config.

        Two runs with the same fingerprint were configured identically. This is
        not a claim that the results will match — only that the inputs did, which
        is the part a reader has to be able to check.
        """
        digest = hashlib.sha256(self.to_json().encode("utf-8"))
        return digest.hexdigest()[:16]

    def price_table(self) -> PriceTable:
        return (
            load_price_table()
            if self.price_file is None
            else load_price_table(self.price_file)
        )


def _duplicates(values: Any) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


@dataclass(frozen=True)
class MetricRow:
    """The full metric row for one task on one repetition.

    Every field a later comparison needs is present, including the ones this
    baseline fills with a constant. :attr:`decision_overhead_ms` is 0.0 and
    :attr:`escalated` is False because a fixed-budget run does neither, and
    recording them as constants is what gives an adaptive run something to be
    measured against.
    """

    task_id: str
    category: str
    repetition: int
    strategy: str
    budget: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    llm_calls: int
    cost_usd: float | None
    usage_basis: UsageBasis
    citable: bool
    latency_ms: float
    decision_overhead_ms: float = 0.0
    escalated: bool = False
    quality: float | None = None
    """Absent, not zero. A judge does not exist yet, and a zero here would read
    as a task that scored nothing rather than one that was never scored."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "repetition": self.repetition,
            "strategy": self.strategy,
            "budget": self.budget,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "cost_usd": self.cost_usd,
            "usage_basis": self.usage_basis.value,
            "citable": self.citable,
            "latency_ms": self.latency_ms,
            "decision_overhead_ms": self.decision_overhead_ms,
            "escalated": self.escalated,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricRow:
        try:
            task_id = str(data["task_id"])
            strategy = str(data["strategy"])
            budget = int(data["budget"])
        except KeyError as exc:
            raise ValueError(
                f"Metric row is missing required field: {exc.args[0]!r}"
            ) from exc
        return cls(
            task_id=task_id,
            category=str(data["category"]),
            repetition=int(data.get("repetition", 0)),
            strategy=strategy,
            budget=budget,
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            reasoning_tokens=int(data.get("reasoning_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            llm_calls=int(data.get("llm_calls", 0)),
            cost_usd=None if data.get("cost_usd") is None else float(data["cost_usd"]),
            usage_basis=decode_enum(
                UsageBasis, data.get("usage_basis", UsageBasis.ABSENT), "usage_basis"
            ),
            citable=bool(data.get("citable", False)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            decision_overhead_ms=float(data.get("decision_overhead_ms", 0.0)),
            escalated=bool(data.get("escalated", False)),
            quality=None if data.get("quality") is None else float(data["quality"]),
        )


@dataclass(frozen=True)
class CategorySummary:
    """Means for one category. Never merged with another category."""

    category: str
    tasks: int
    mean_output_tokens: float
    mean_total_tokens: float
    mean_cost_usd: float | None
    mean_latency_ms: float
    priced_tasks: int
    """How many rows in this category carry a cost. The mean is taken over these
    rows only, because averaging over the rest would report an unpriced task as a
    free one."""
    citable_tasks: int
    """How many rows in this category are citable. A category whose rows are
    mostly estimated is not evidence, and this is how a reader finds out."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "tasks": self.tasks,
            "mean_output_tokens": self.mean_output_tokens,
            "mean_total_tokens": self.mean_total_tokens,
            "mean_cost_usd": self.mean_cost_usd,
            "mean_latency_ms": self.mean_latency_ms,
            "priced_tasks": self.priced_tasks,
            "citable_tasks": self.citable_tasks,
        }


@dataclass(frozen=True)
class RunResult:
    """Everything one baseline run produced, and the config that produced it."""

    config_fingerprint: str
    strategy: str
    budget: int
    price_table_version: str
    price_table_verified: bool
    """Whether the rate behind every cost in this run was checked. A run priced
    from the unverified bundled snapshot is a hypothesis about money."""

    rows: tuple[MetricRow, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))

    @property
    def citable(self) -> bool:
        """True only if every row in the run is citable."""
        return all(row.citable for row in self.rows) and bool(self.rows)

    def by_category(self) -> tuple[CategorySummary, ...]:
        """Per-category means, sorted by category.

        Deliberately the only aggregation offered. There is no overall mean
        because a single number across heterogeneous categories is not a
        measurement of anything.
        """
        grouped: dict[str, list[MetricRow]] = {}
        for row in self.rows:
            grouped.setdefault(row.category, []).append(row)

        summaries = []
        for category in sorted(grouped):
            rows = grouped[category]
            count = len(rows)
            priced = [row for row in rows if row.cost_usd is not None]
            summaries.append(
                CategorySummary(
                    category=category,
                    tasks=count,
                    mean_output_tokens=sum(r.output_tokens for r in rows) / count,
                    mean_total_tokens=sum(r.total_tokens for r in rows) / count,
                    mean_cost_usd=(
                        None
                        if not priced
                        else sum(r.cost_usd or 0.0 for r in priced) / len(priced)
                    ),
                    mean_latency_ms=sum(r.latency_ms for r in rows) / count,
                    priced_tasks=len(priced),
                    citable_tasks=sum(1 for row in rows if row.citable),
                )
            )
        return tuple(summaries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_fingerprint": self.config_fingerprint,
            "strategy": self.strategy,
            "budget": self.budget,
            "price_table_version": self.price_table_version,
            "price_table_verified": self.price_table_verified,
            "citable": self.citable,
            "rows": [row.to_dict() for row in self.rows],
            "by_category": [summary.to_dict() for summary in self.by_category()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunResult:
        try:
            rows = tuple(MetricRow.from_dict(row) for row in data.get("rows", ()))
            return cls(
                config_fingerprint=str(data["config_fingerprint"]),
                strategy=str(data["strategy"]),
                budget=int(data["budget"]),
                price_table_version=str(data.get("price_table_version", "")),
                price_table_verified=bool(data.get("price_table_verified", False)),
                rows=rows,
            )
        except KeyError as exc:
            raise ValueError(
                f"Run result is missing required field: {exc.args[0]!r}"
            ) from exc


async def run_baseline(
    config: BaselineConfig, client: Client, *, table: PriceTable | None = None
) -> RunResult:
    """Run ``config`` against ``client`` and return the metric rows.

    One call per task per repetition, at the fixed budget, with no inspection of
    the output and no retry. That restraint is the baseline: the moment this
    function is allowed to react to a result it stops being the reference.

    A provider error propagates rather than becoming a failed row. A partial run
    that looks complete is worse than a crash, because a crash cannot be mistaken
    for a result.
    """
    prices = table if table is not None else config.price_table()
    options = CompletionOptions(model=config.model, max_output_tokens=config.budget)
    rows: list[MetricRow] = []

    for task in config.tasks:
        request = task.to_request()
        for repetition in range(config.repetitions):
            response = await client.complete(request, options)
            model = config.model or response.model
            if not model:
                raise CostTableError(
                    "Cannot price this run: the config names no model and the "
                    "provider reported none. Set the model in the config, or "
                    "include it in your price table under the name the provider "
                    "returns."
                )
            priced = prices.price_usage(response.usage, model)
            rows.append(
                MetricRow(
                    task_id=task.task_id,
                    category=task.category,
                    repetition=repetition,
                    strategy=config.strategy,
                    budget=config.budget,
                    input_tokens=priced.input_tokens,
                    output_tokens=priced.output_tokens,
                    reasoning_tokens=priced.reasoning_tokens,
                    total_tokens=priced.total_tokens,
                    llm_calls=priced.llm_calls,
                    cost_usd=priced.cost_usd,
                    usage_basis=priced.basis,
                    citable=priced.exact,
                    latency_ms=response.latency_ms,
                )
            )

    rows.sort(key=lambda row: (row.category, row.task_id, row.repetition))
    return RunResult(
        config_fingerprint=config.fingerprint,
        strategy=config.strategy,
        budget=config.budget,
        price_table_version=prices.version,
        price_table_verified=all(price.verified for price in prices.models.values()),
        rows=tuple(rows),
    )


ORACLE_CAVEATS = (
    "Assumes task quality is preserved at every budget. No judge exists yet, so "
    "this assumption is unverified, and a real judge may find that the cheapest "
    "run for a task is not the best one.",
    "Assumes a per-task budget chosen with perfect foresight of every run's "
    "outcome. No deployed policy can see all runs before choosing a budget for "
    "the next request.",
    "A bound on savings is only as exact as the costs inside it. If the underlying "
    "usage figures are estimated, or the price table is unverified, this is an "
    "estimate of an estimate.",
    "The bound is computed against one baseline budget. Comparing it to a saving "
    "measured against a different baseline divides two unrelated numbers.",
)
"""The reasons this number cannot be achieved, carried inside the data.

A caveat that lives only in a docstring is a caveat nobody reads once the result
has been serialized into a results file and passed around. These travel with the
result so that anyone reading the JSON later meets the objections at the same
time as the number.
"""


@dataclass(frozen=True)
class OracleRow:
    """What perfect allocation would have saved on one task."""

    task_id: str
    category: str
    repetition: int
    baseline_budget: int
    baseline_cost_usd: float
    oracle_cost_usd: float
    """Cheapest observed cost for this task, across every run including the
    baseline. Including the baseline guarantees this is never above it, so the
    resulting saving is a bound rather than a wild extrapolation."""

    chosen_budget: int
    """Budget the oracle would have allocated. Ties resolve to the smaller
    budget, so the choice is deterministic."""

    runs_considered: int
    savings_usd: float
    savings_fraction: float
    comparable: bool = True
    """False when this task had no cost in some run, so no honest comparison was
    possible. Excluded from the totals rather than silently scored as zero."""

    citable: bool = True
    """False when the costs underneath were estimated or unpriced. A saving
    computed from numbers nobody can vouch for is not a citable saving."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "repetition": self.repetition,
            "baseline_budget": self.baseline_budget,
            "baseline_cost_usd": self.baseline_cost_usd,
            "oracle_cost_usd": self.oracle_cost_usd,
            "chosen_budget": self.chosen_budget,
            "runs_considered": self.runs_considered,
            "savings_usd": self.savings_usd,
            "savings_fraction": self.savings_fraction,
            "comparable": self.comparable,
            "citable": self.citable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OracleRow:
        try:
            return cls(
                task_id=str(data["task_id"]),
                category=str(data["category"]),
                repetition=int(data.get("repetition", 0)),
                baseline_budget=int(data["baseline_budget"]),
                baseline_cost_usd=float(data["baseline_cost_usd"]),
                oracle_cost_usd=float(data["oracle_cost_usd"]),
                chosen_budget=int(data["chosen_budget"]),
                runs_considered=int(data.get("runs_considered", 0)),
                savings_usd=float(data.get("savings_usd", 0.0)),
                savings_fraction=float(data.get("savings_fraction", 0.0)),
                comparable=bool(data.get("comparable", True)),
                citable=bool(data.get("citable", True)),
            )
        except KeyError as exc:
            raise ValueError(
                f"Oracle row is missing required field: {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True)
class OracleCategorySummary:
    """Bounds for one category. Never merged with another category."""

    category: str
    tasks: int
    mean_savings_usd: float
    mean_savings_fraction: float
    citable_tasks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "tasks": self.tasks,
            "mean_savings_usd": self.mean_savings_usd,
            "mean_savings_fraction": self.mean_savings_fraction,
            "citable_tasks": self.citable_tasks,
        }


@dataclass(frozen=True)
class OracleResult:
    """The upper bound on achievable savings, and its own refutations.

    Read :attr:`caveats` before :attr:`max_savings_fraction`. That number is the
    point of the comparison and the reason it must not be believed.
    """

    baseline_fingerprint: str
    baseline_budget: int
    price_table_version: str
    rows: tuple[OracleRow, ...] = field(default_factory=tuple)
    caveats: tuple[str, ...] = ORACLE_CAVEATS
    incomparable_rows: int = 0
    """Tasks that could not be compared and are therefore missing from the totals.
    A bound computed while quietly dropping tasks is not a bound."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "caveats", tuple(self.caveats))

    @property
    def comparable_rows(self) -> tuple[OracleRow, ...]:
        return tuple(row for row in self.rows if row.comparable)

    @property
    def attainable(self) -> bool:
        """Always False.

        A property rather than a stored field, so a result deserialized from a
        file cannot arrive carrying a claim of attainability.
        """
        return False

    @property
    def baseline_cost_usd(self) -> float:
        return sum(row.baseline_cost_usd for row in self.comparable_rows)

    @property
    def oracle_cost_usd(self) -> float:
        return sum(row.oracle_cost_usd for row in self.comparable_rows)

    @property
    def max_savings_usd(self) -> float:
        return self.baseline_cost_usd - self.oracle_cost_usd

    @property
    def max_savings_fraction(self) -> float:
        """Bound as a fraction of baseline cost, never above 1.0."""
        baseline = self.baseline_cost_usd
        if baseline <= 0:
            return 0.0
        return min(1.0, self.max_savings_usd / baseline)

    @property
    def citable(self) -> bool:
        """True only if every row is both comparable and citable.

        Completeness and trustworthiness are separate requirements: a complete
        comparison built on estimated costs is still not a citable claim.
        """
        return (
            bool(self.rows)
            and self.incomparable_rows == 0
            and all(row.comparable and row.citable for row in self.rows)
        )

    def by_category(self) -> tuple[OracleCategorySummary, ...]:
        """Per-category bounds, on the same terms as :meth:`RunResult.by_category`."""
        grouped: dict[str, list[OracleRow]] = {}
        for row in self.comparable_rows:
            grouped.setdefault(row.category, []).append(row)
        summaries = []
        for category in sorted(grouped):
            rows = grouped[category]
            summaries.append(
                OracleCategorySummary(
                    category=category,
                    tasks=len(rows),
                    mean_savings_usd=sum(row.savings_usd for row in rows) / len(rows),
                    mean_savings_fraction=(
                        sum(row.savings_fraction for row in rows) / len(rows)
                    ),
                    citable_tasks=sum(1 for row in rows if row.citable),
                )
            )
        return tuple(summaries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_fingerprint": self.baseline_fingerprint,
            "baseline_budget": self.baseline_budget,
            "price_table_version": self.price_table_version,
            "attainable": self.attainable,
            "citable": self.citable,
            "baseline_cost_usd": self.baseline_cost_usd,
            "oracle_cost_usd": self.oracle_cost_usd,
            "max_savings_usd": self.max_savings_usd,
            "max_savings_fraction": self.max_savings_fraction,
            "incomparable_rows": self.incomparable_rows,
            "caveats": list(self.caveats),
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OracleResult:
        try:
            return cls(
                baseline_fingerprint=str(data["baseline_fingerprint"]),
                baseline_budget=int(data["baseline_budget"]),
                price_table_version=str(data.get("price_table_version", "")),
                rows=tuple(OracleRow.from_dict(row) for row in data.get("rows", ())),
                incomparable_rows=int(data.get("incomparable_rows", 0)),
            )
        except KeyError as exc:
            raise ValueError(
                f"Oracle result is missing required field: {exc.args[0]!r}"
            ) from exc


def _row_key(row: MetricRow) -> tuple[str, int]:
    return (row.task_id, row.repetition)


def compute_oracle(
    runs: Sequence[RunResult], baseline: RunResult | None = None
) -> OracleResult:
    """Upper bound on savings from perfect per-task budget allocation.

    Supply the baseline run plus every run at another budget. For each task the
    oracle takes the cheapest observed cost and calls the difference a saving.

    The result is a bound, not a target, and the reasons why travel with it in
    :data:`ORACLE_CAVEATS`. The dominant one is that quality is assumed rather
    than measured, because no judge exists yet, which means this function can say
    how much money better allocation might have saved and cannot say whether the
    answer would still have been right.

    The baseline is included among the candidates, so a task whose baseline was
    already its cheapest reports zero saving rather than a negative one.

    Raises :class:`ValueError` when the runs are not comparable: different tasks,
    different strategies, or different price tables. Comparing costs computed
    with two price tables measures the tables rather than the allocator, and
    comparing two strategies against each other reports the strategy gap as though
    it were a budget gap.
    """
    if len(runs) < 2:
        raise ValueError(
            "An oracle needs a baseline and at least one other budget. A single run "
            "has nothing to be better than."
        )
    base = baseline if baseline is not None else runs[0]
    others = [run for run in runs if run is not base]
    _require_comparable(base, others)
    candidates = [base, *others]

    rows: list[OracleRow] = []
    for task_id, repetition in sorted(_row_key(row) for row in base.rows):
        observed = [
            row
            for run in candidates
            for row in run.rows
            if _row_key(row) == (task_id, repetition)
        ]
        reference = next(
            row for row in base.rows if _row_key(row) == (task_id, repetition)
        )
        priced = [row for row in observed if row.cost_usd is not None]
        cheapest = min(
            priced, key=lambda row: (row.cost_usd or 0.0, row.budget), default=None
        )

        baseline_cost = reference.cost_usd or 0.0
        oracle_cost = baseline_cost
        if cheapest is not None:
            oracle_cost = min(baseline_cost, cheapest.cost_usd or 0.0)
        savings = baseline_cost - oracle_cost
        rows.append(
            OracleRow(
                task_id=task_id,
                category=reference.category,
                repetition=repetition,
                baseline_budget=base.budget,
                baseline_cost_usd=baseline_cost,
                oracle_cost_usd=oracle_cost,
                chosen_budget=cheapest.budget if cheapest is not None else base.budget,
                runs_considered=len(observed),
                savings_usd=savings,
                savings_fraction=0.0 if baseline_cost <= 0 else savings / baseline_cost,
                comparable=len(priced) == len(observed),
                citable=all(row.citable for row in observed),
            )
        )

    return OracleResult(
        baseline_fingerprint=base.config_fingerprint,
        baseline_budget=base.budget,
        price_table_version=base.price_table_version,
        rows=tuple(rows),
        incomparable_rows=sum(1 for row in rows if not row.comparable),
    )


def _require_comparable(base: RunResult, others: Sequence[RunResult]) -> None:
    keys = {_row_key(row) for row in base.rows}
    for run in others:
        if {_row_key(row) for row in run.rows} != keys:
            raise ValueError(
                f"Run at budget {run.budget} evaluated a different set of tasks than "
                f"the baseline at budget {base.budget}. An oracle can only compare "
                f"runs that answered the same questions."
            )
        if run.strategy != base.strategy:
            raise ValueError(
                f"Run at budget {run.budget} used strategy {run.strategy!r} but the "
                f"baseline used {base.strategy!r}. That difference is not a budget "
                f"difference; vary one thing at a time."
            )
        if run.price_table_version != base.price_table_version:
            raise ValueError(
                f"Run at budget {run.budget} was priced with table "
                f"{run.price_table_version!r} but the baseline used "
                f"{base.price_table_version!r}. Comparing costs from two price "
                f"tables measures the tables, not the allocator."
            )
