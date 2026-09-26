"""Provider pricing, loaded from a data file rather than from code.

Price data changes on a provider's schedule, not on ours. Embedding it in Python
would mean a release every time a vendor changes a number, which is both
municipal work and an excuse to ship a new version for a one-character change.
So it lives in JSON, carries its own version independent of the package version,
and can be replaced by a file the operator supplies.

**A price is a claim, not a measurement.** The library cannot check a vendor's
invoice, so every price in the bundled file ships with ``verified: false`` and
:meth:`PriceTable.price_usage` will not call a cost exact on the strength of an
unverified number. An operator who supplies a table they have checked against
an invoice gets exact costs; everyone else gets a defensible estimate and a
``usage_basis`` that says so.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

from inferconomy.contracts import Usage

__all__ = [
    "BUNDLED_PRICE_RESOURCE",
    "PRICE_TABLE_SCHEMA_VERSION",
    "CostTableError",
    "InvalidPriceTable",
    "ModelPrice",
    "PriceTable",
    "UnknownModel",
    "bundled_price_table",
    "load_price_table",
    "price_usage",
]

PRICE_TABLE_SCHEMA_VERSION = 1
"""Shape of the file, bumped only when the format itself changes.

Independent of the table's ``version``, which tracks price data, and of the
package version, which tracks code. A data correction must never require a
release.
"""

BUNDLED_PRICE_RESOURCE = "prices.json"
_TOKENS_PER_UNIT = 1_000_000.0
_MONEY_PRECISION = 12
"""Decimal places kept when converting tokens to currency.

Not cosmetic. Binary floats cannot represent most decimal fractions, so
unrounded sums of per-million divisions accumulate error in the low digits of
every cost, and two runs that should agree stop agreeing. Twelve places is far
below any real invoice and far above the noise.
"""


class CostTableError(Exception):
    """Base class for pricing failures."""


class InvalidPriceTable(CostTableError, ValueError):
    """The file is unreadable, malformed, or internally inconsistent."""


class UnknownModel(CostTableError, LookupError):
    """The table has no price for this model.

    Raised instead of returning zero. A silent zero would turn "we do not know
    what this model costs" into "this model is free", which is the most expensive
    kind of wrong.
    """


@dataclass(frozen=True)
class ModelPrice:
    """One model's rates, in a single currency per million tokens."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    """Cache-read rate. ``None`` means the vendor publishes no discount, so the
    full input rate applies. Not the same as zero, which would be free."""

    reasoning_per_million: float | None = None
    """Reasoning-token rate. ``None`` means reasoning is billed as output."""

    source: str = ""
    """Where this number came from. Required to be non-empty in practice: a
    price with no source cannot be audited or refreshed."""

    as_of: str = ""
    """ISO date the price was recorded, so staleness is visible in the data."""

    verified: bool = False
    """Whether an operator confirmed this price against an actual invoice.

    Only ``True`` allows a derived cost to be reported as exact.
    """

    def __post_init__(self) -> None:
        for name in (
            "input_per_million",
            "output_per_million",
            "cached_input_per_million",
            "reasoning_per_million",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

    @property
    def cached_input_rate(self) -> float:
        return (
            self.input_per_million
            if self.cached_input_per_million is None
            else self.cached_input_per_million
        )

    @property
    def reasoning_rate(self) -> float:
        return (
            self.output_per_million
            if self.reasoning_per_million is None
            else self.reasoning_per_million
        )

    def cost_of(self, usage: Usage) -> float:
        """Cost of ``usage`` at these rates.

        Reasoning and cache reads are billed as subsets of their parents, so they
        are re-priced and subtracted rather than added. A usage record that
        already claims a cost is priced from its token counts alone, which keeps
        re-pricing from compounding a previous estimate.
        """
        billable_input = usage.input_tokens - usage.cached_input_tokens
        billable_output = usage.output_tokens - usage.reasoning_tokens
        total = (
            billable_input * self.input_per_million
            + usage.cached_input_tokens * self.cached_input_rate
            + billable_output * self.output_per_million
            + usage.reasoning_tokens * self.reasoning_rate
        )
        return round(total / _TOKENS_PER_UNIT, _MONEY_PRECISION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_per_million": self.input_per_million,
            "output_per_million": self.output_per_million,
            "cached_input_per_million": self.cached_input_per_million,
            "reasoning_per_million": self.reasoning_per_million,
            "source": self.source,
            "as_of": self.as_of,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelPrice:
        if not isinstance(data, Mapping):
            raise InvalidPriceTable(f"Price entry must be an object, got {data!r}.")
        for field_name in ("input_per_million", "output_per_million"):
            if field_name not in data:
                raise InvalidPriceTable(
                    f"Price entry is missing required field: {field_name!r}."
                )

        def optional_float(name: str) -> float | None:
            raw = data.get(name)
            return None if raw is None else float(raw)

        try:
            return cls(
                input_per_million=float(data["input_per_million"]),
                output_per_million=float(data["output_per_million"]),
                cached_input_per_million=optional_float("cached_input_per_million"),
                reasoning_per_million=optional_float("reasoning_per_million"),
                source=str(data.get("source", "")),
                as_of=str(data.get("as_of", "")),
                verified=bool(data.get("verified", False)),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidPriceTable(f"Price entry is malformed: {exc}") from exc


@dataclass(frozen=True)
class PriceTable:
    """A versioned set of model prices.

    Immutable: a table shared across concurrent requests cannot be edited out from
    under one of them, which is the sort of race that turns into a plausible
    number in a report.
    """

    version: str
    models: Mapping[str, ModelPrice]
    currency: str = "USD"
    schema_version: int = PRICE_TABLE_SCHEMA_VERSION
    origin: str = "bundled"
    """Where this table came from. A path for a user file, for audit trails."""

    def __post_init__(self) -> None:
        for model, price in self.models.items():
            if not isinstance(model, str) or not model:
                raise InvalidPriceTable(f"Model name must be non-empty, got {model!r}.")
            if not isinstance(price, ModelPrice):
                raise InvalidPriceTable(
                    f"Price for {model!r} is not a ModelPrice: {price!r}."
                )
        object.__setattr__(self, "models", dict(self.models))

    def __contains__(self, model: object) -> bool:
        return model in self.models

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.models))

    def __len__(self) -> int:
        return len(self.models)

    def __getitem__(self, model: str) -> ModelPrice:
        return self.get(model)

    def get(self, model: str, default: ModelPrice | None = None) -> ModelPrice:
        """Price for ``model``, or ``default`` if absent and one was given.

        The exception is raised even when a default exists, because a caller
        passing a default is asking for a fallback while the caller who did not
        is asking a question they need answered.
        """
        try:
            return self.models[model]
        except KeyError:
            if default is not None:
                return default
        known = ", ".join(sorted(self.models)) or "none"
        raise UnknownModel(
            f"No price for {model!r} in {self.origin} (version {self.version}). "
            f"Known models: {known}."
        )

    def price_usage(self, usage: Usage, model: str) -> Usage:
        """Attach a cost to ``usage`` for ``model``.

        Token counts and their exactness are preserved untouched. ``cost_exact``
        reports only whether the *rate* was verified, never whether the resulting
        number is citable: a verified rate applied to estimated tokens still
        yields a cost that inherits the token error, and claiming otherwise would
        be the mistake this library exists to prevent. :attr:`Usage.exact` and
        :attr:`Usage.basis` are the checks that require both.
        """
        price = self.get(model)
        return replace(
            usage,
            cost_usd=price.cost_of(usage),
            cost_exact=price.verified,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "currency": self.currency,
            "models": {
                model: self.models[model].to_dict() for model in sorted(self.models)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriceTable:
        if not isinstance(data, Mapping):
            raise InvalidPriceTable(f"Price table must be a JSON object, got {data!r}.")
        schema_version = int(data.get("schema_version", 0))
        if schema_version > PRICE_TABLE_SCHEMA_VERSION:
            raise InvalidPriceTable(
                f"Price table schema version {schema_version} is newer than this "
                f"release understands ({PRICE_TABLE_SCHEMA_VERSION}). "
                f"Upgrade Inferconomy."
            )
        if schema_version != PRICE_TABLE_SCHEMA_VERSION:
            raise InvalidPriceTable(
                f"Unsupported price table schema version: {schema_version}."
            )
        raw_models = data.get("models")
        if not isinstance(raw_models, Mapping):
            raise InvalidPriceTable("Price table is missing a 'models' object.")
        try:
            version = str(data["version"])
        except KeyError as exc:
            raise InvalidPriceTable(
                "Price table is missing required field: 'version'"
            ) from exc
        return cls(
            version=version,
            models={
                str(model): ModelPrice.from_dict(entry)
                for model, entry in raw_models.items()
            },
            currency=str(data.get("currency", "USD")),
            schema_version=schema_version,
        )


def _parse(text: str, origin: str) -> PriceTable:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidPriceTable(f"{origin} is not valid JSON: {exc}") from exc
    return replace(PriceTable.from_dict(data), origin=origin)


def load_price_table(path: str | os.PathLike[str] | None = None) -> PriceTable:
    """Load a price table.

    With no argument, the bundled snapshot. With a path, the operator's file
    entirely: a user table is never merged with the bundled one, because a merge
    would silently resurrect bundled prices for models the operator meant to
    exclude.
    """
    if path is not None:
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InvalidPriceTable(
                f"Cannot read price table at {file_path}: {exc}"
            ) from exc
        return _parse(text, str(file_path))

    resource = (
        resources.files("inferconomy").joinpath("data").joinpath(BUNDLED_PRICE_RESOURCE)
    )
    try:
        with resources.as_file(resource) as bundled:
            text = bundled.read_text(encoding="utf-8")
    except (OSError, ModuleNotFoundError) as exc:
        raise InvalidPriceTable(
            f"Bundled price table {BUNDLED_PRICE_RESOURCE!r} is missing from this "
            f"installation: {exc}. Reinstall the package or supply a price file."
        ) from exc
    return _parse(text, f"bundled:{BUNDLED_PRICE_RESOURCE}")


_BUNDLED: PriceTable | None = None


def bundled_price_table() -> PriceTable:
    """The bundled snapshot, parsed once per process.

    Cached because parsing is pure and the result is immutable. Re-reading a
    file on every request would be pure overhead for a table that cannot change
    while the process runs.
    """
    global _BUNDLED
    if _BUNDLED is None:
        _BUNDLED = load_price_table()
    return _BUNDLED


def price_usage(usage: Usage, model: str, table: PriceTable | None = None) -> Usage:
    """Convenience wrapper: price ``usage`` using the bundled table by default."""
    return (table or bundled_price_table()).price_usage(usage, model)
