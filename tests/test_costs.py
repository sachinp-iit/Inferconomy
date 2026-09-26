"""Price table loading, versioning, override, and cost arithmetic.

Two things are being defended here. First, that a wrong price cannot masquerade
as a right one: nothing derived from an unverified rate is ever reported as an
exact cost. Second, that a missing price fails loudly rather than quietly
returning zero, which would read as "this model is free".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inferconomy import (
    InvalidPriceTable,
    ModelPrice,
    PriceTable,
    UnknownModel,
    Usage,
    UsageBasis,
    load_price_table,
)
from inferconomy.costs import (
    PRICE_TABLE_SCHEMA_VERSION,
    bundled_price_table,
    price_usage,
)

VERIFIED_RATE: dict[str, Any] = {
    "input_per_million": 3.0,
    "output_per_million": 15.0,
    "source": "invoice, 2026-09",
    "as_of": "2026-09-01",
    "verified": True,
}


def write_table(tmp_path: Path, models: dict[str, Any], **extra: Any) -> Path:
    payload: dict[str, Any] = {
        "schema_version": PRICE_TABLE_SCHEMA_VERSION,
        "version": "test-1",
        "models": models,
        **extra,
    }
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def a_table(
    cached_input_per_million: float | None = None,
    reasoning_per_million: float | None = None,
) -> PriceTable:
    return PriceTable(
        version="test-1",
        models={
            "m": ModelPrice(
                **VERIFIED_RATE,
                cached_input_per_million=cached_input_per_million,
                reasoning_per_million=reasoning_per_million,
            )
        },
    )


class TestBundledTable:
    def test_loads_from_the_bundled_file(self) -> None:
        assert len(load_price_table()) > 0

    def test_declares_its_own_version(self) -> None:
        assert load_price_table().version

    def test_schema_version_is_independent_of_the_package(self) -> None:
        """A data correction must never require a release."""
        assert load_price_table().schema_version == PRICE_TABLE_SCHEMA_VERSION

    def test_every_entry_records_its_provenance(self) -> None:
        for model in load_price_table():
            price = load_price_table()[model]
            assert price.source, f"{model} has no source"
            assert price.as_of, f"{model} has no as_of date"

    def test_no_bundled_price_claims_verification(self) -> None:
        """The library cannot check a vendor invoice, so it must not imply it did."""
        assert all(not price.verified for price in load_price_table().models.values())

    def test_cached_without_a_published_discount(self) -> None:
        table = a_table()
        assert table["m"].cached_input_rate == table["m"].input_per_million

    def test_origin_is_recorded_for_audit(self) -> None:
        assert load_price_table().origin.startswith("bundled:")

    def test_is_cached_per_process(self) -> None:
        assert bundled_price_table() is bundled_price_table()

    def test_iteration_is_sorted(self) -> None:
        assert list(bundled_price_table()) == sorted(bundled_price_table().models)

    def test_a_missing_bundled_file_is_reported(self, monkeypatch: Any) -> None:
        def missing(resource: Any) -> Any:
            raise OSError("prices.json is not there")

        monkeypatch.setattr("importlib.resources.as_file", missing)
        with pytest.raises(InvalidPriceTable, match="missing from this installation"):
            load_price_table()


class TestUnknownModel:
    def test_raises_rather_than_returning_zero(self) -> None:
        with pytest.raises(UnknownModel, match="No price for 'mystery'"):
            a_table().get("mystery")

    def test_error_lists_the_models_it_does_know(self) -> None:
        with pytest.raises(UnknownModel, match="Known models: m"):
            a_table().get("mystery")

    def test_error_names_the_table_and_its_version(self) -> None:
        with pytest.raises(UnknownModel, match="version test-1"):
            a_table().get("mystery")

    def test_empty_table_says_so(self) -> None:
        with pytest.raises(UnknownModel, match="Known models: none"):
            PriceTable(version="v", models={}).get("mystery")

    def test_default_is_returned_when_one_is_given(self) -> None:
        fallback = ModelPrice(1.0, 2.0)
        assert a_table().get("mystery", fallback) is fallback

    def test_getitem_goes_through_get(self) -> None:
        assert a_table()["m"] == a_table().get("m")


class TestContainerBehaviour:
    def test_membership(self) -> None:
        assert "m" in a_table()
        assert "mystery" not in a_table()

    def test_length(self) -> None:
        assert len(a_table()) == 1

    def test_missing_key_raises_lookup_error(self) -> None:
        with pytest.raises(UnknownModel):
            a_table()["mystery"]

    def test_models_are_copied_so_the_table_cannot_be_mutated(self) -> None:
        table = a_table()
        dict(table.models)["m"] = ModelPrice(999.0, 999.0)
        assert table["m"].input_per_million == 3.0

    def test_rejects_an_empty_model_name(self) -> None:
        with pytest.raises(InvalidPriceTable, match="non-empty"):
            PriceTable(version="v", models={"": ModelPrice(1.0, 1.0)})

    def test_rejects_a_price_that_is_not_a_model_price(self) -> None:
        entry: dict[str, float] = {"input_per_million": 1.0}
        with pytest.raises(InvalidPriceTable, match="not a ModelPrice"):
            PriceTable(version="v", models={"m": entry})  # type: ignore[dict-item]

    def test_round_trips_through_dict(self) -> None:
        assert PriceTable.from_dict(a_table().to_dict()) == replace_origin(a_table())


def replace_origin(table: PriceTable) -> PriceTable:
    from dataclasses import replace

    return replace(table, origin="bundled")


class TestCostArithmetic:
    def test_input_and_output_are_charged_separately(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert a_table().price_usage(usage, "m").cost_usd == 18.0

    def test_a_thousand_tokens_cost_a_thousandth_of_a_million(self) -> None:
        assert a_table().price_usage(Usage(input_tokens=1_000), "m").cost_usd == 0.003

    def test_zero_usage_costs_nothing(self) -> None:
        assert a_table().price_usage(Usage(), "m").cost_usd == 0.0

    def test_cached_tokens_are_billed_at_the_cache_rate(self) -> None:
        table = a_table(cached_input_per_million=1.0)
        usage = Usage(input_tokens=1_000_000, cached_input_tokens=1_000_000)
        assert table.price_usage(usage, "m").cost_usd == 1.0

    def test_cached_tokens_are_a_subset_not_an_addition(self) -> None:
        """1M total input of which 1M was cached is 1M input, not 2M."""
        table = a_table(cached_input_per_million=0.0)
        usage = Usage(input_tokens=1_000_000, cached_input_tokens=1_000_000)
        assert table.price_usage(usage, "m").cost_usd == 0.0

    def test_reasoning_is_billed_at_its_own_rate(self) -> None:
        table = a_table(reasoning_per_million=30.0)
        usage = Usage(output_tokens=1_000_000, reasoning_tokens=1_000_000)
        assert table.price_usage(usage, "m").cost_usd == 30.0

    def test_reasoning_falls_back_to_the_output_rate(self) -> None:
        usage = Usage(output_tokens=1_000_000, reasoning_tokens=1_000_000)
        assert a_table().price_usage(usage, "m").cost_usd == 15.0

    def test_pricing_ignores_any_cost_already_on_the_record(self) -> None:
        """Re-pricing must compound token counts, never a previous estimate."""
        already = a_table().price_usage(
            Usage(input_tokens=1_000_000, cost_usd=999.0, cost_exact=True), "m"
        )
        assert already.cost_usd == 3.0

    def test_a_violated_subset_invariant_is_impossible_to_reach(self) -> None:
        """Usage validates cached <= input, so pricing need not clamp defensively."""
        with pytest.raises(ValueError, match="cached_input_tokens"):
            Usage(input_tokens=100, cached_input_tokens=250)

    def test_result_is_stable_across_sums(self) -> None:
        """Unrounded per-million division would drift in the low digits."""
        table = a_table()
        usage = Usage(input_tokens=333_333, output_tokens=1)
        assert (
            table.price_usage(usage, "m").cost_usd
            == table.price_usage(usage, "m").cost_usd
        )


class TestProvenance:
    def test_a_verified_rate_makes_a_reported_cost_exact(self) -> None:
        usage = a_table().price_usage(Usage(input_tokens=10, tokens_exact=True), "m")
        assert usage.cost_exact is True
        assert usage.basis is UsageBasis.REPORTED

    def test_an_unverified_rate_never_yields_an_exact_cost(self) -> None:
        table = PriceTable(version="v", models={"m": ModelPrice(3.0, 15.0)})
        usage = table.price_usage(Usage(input_tokens=10, tokens_exact=True), "m")
        assert usage.cost_usd is not None
        assert usage.cost_exact is False

    def test_an_unverified_rate_says_so_in_the_basis(self) -> None:
        table = PriceTable(version="v", models={"m": ModelPrice(3.0, 15.0)})
        usage = table.price_usage(Usage(input_tokens=10, tokens_exact=True), "m")
        assert usage.basis is UsageBasis.PRICED_UNVERIFIED

    def test_a_verified_rate_gives_an_estimated_cost_an_exact_rate(self) -> None:
        """cost_exact describes the rate; exact and basis describe the answer."""
        usage = a_table().price_usage(Usage(input_tokens=10), "m")
        assert usage.cost_exact is True
        assert usage.exact is False
        assert usage.basis is UsageBasis.PRICED_FROM_ESTIMATE

    def test_token_counts_and_their_exactness_survive_pricing(self) -> None:
        usage = a_table().price_usage(
            Usage(input_tokens=10, output_tokens=5, llm_calls=2, tokens_exact=True), "m"
        )
        assert (usage.input_tokens, usage.output_tokens, usage.llm_calls) == (10, 5, 2)
        assert usage.tokens_exact is True

    def test_bundled_prices_are_never_citable(self) -> None:
        usage = bundled_price_table().price_usage(
            Usage(input_tokens=1_000_000, tokens_exact=True), "gpt-4o"
        )
        assert usage.cost_usd == 2.5
        assert usage.basis is UsageBasis.PRICED_UNVERIFIED


class TestUserOverride:
    def test_a_user_file_replaces_the_bundled_table_entirely(
        self, tmp_path: Path
    ) -> None:
        path = write_table(tmp_path, {"my-model": VERIFIED_RATE})
        table = load_price_table(path)
        assert list(table) == ["my-model"]
        assert "gpt-4o" not in table

    def test_a_user_table_can_be_verified(self, tmp_path: Path) -> None:
        path = write_table(tmp_path, {"my-model": VERIFIED_RATE})
        usage = load_price_table(path).price_usage(
            Usage(input_tokens=1_000_000, tokens_exact=True), "my-model"
        )
        assert usage.cost_exact is True
        assert usage.basis is UsageBasis.REPORTED

    def test_origin_is_the_supplied_path(self, tmp_path: Path) -> None:
        path = write_table(tmp_path, {"my-model": VERIFIED_RATE})
        assert load_price_table(path).origin == str(path)

    def test_a_user_table_may_be_empty(self, tmp_path: Path) -> None:
        """Refusing an empty table would block "charge nothing, report nothing"."""
        assert load_price_table(write_table(tmp_path, {})) is not None

    def test_currency_is_carried_through(self, tmp_path: Path) -> None:
        path = write_table(tmp_path, {"m": VERIFIED_RATE}, currency="EUR")
        assert load_price_table(path).currency == "EUR"

    def test_currency_defaults_to_usd(self, tmp_path: Path) -> None:
        assert (
            load_price_table(write_table(tmp_path, {"m": VERIFIED_RATE})).currency
            == "USD"
        )

    def test_a_missing_file_is_reported_not_ignored(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidPriceTable, match="Cannot read price table"):
            load_price_table(tmp_path / "nope.json")

    def test_a_relative_path_works(self, tmp_path: Path, monkeypatch: Any) -> None:
        write_table(tmp_path, {"m": VERIFIED_RATE})
        monkeypatch.chdir(tmp_path)
        assert load_price_table("prices.json") is not None


class TestValidation:
    def test_rejects_a_negative_rate(self) -> None:
        with pytest.raises(ValueError, match="input_per_million must be non-negative"):
            ModelPrice(-1.0, 1.0)

    def test_rejects_a_negative_optional_rate(self) -> None:
        with pytest.raises(ValueError, match="cached_input_per_million must be"):
            ModelPrice(1.0, 1.0, cached_input_per_million=-1.0)

    def test_allows_a_free_model(self) -> None:
        assert ModelPrice(0.0, 0.0).cost_of(Usage(input_tokens=10)) == 0.0

    def test_rejects_a_newer_schema(self) -> None:
        with pytest.raises(InvalidPriceTable, match="newer than this release"):
            PriceTable.from_dict(
                {
                    "schema_version": PRICE_TABLE_SCHEMA_VERSION + 1,
                    "version": "v",
                    "models": {},
                }
            )

    def test_rejects_an_unknown_schema(self) -> None:
        with pytest.raises(InvalidPriceTable, match="Unsupported price table schema"):
            PriceTable.from_dict({"schema_version": 0, "version": "v", "models": {}})

    def test_rejects_a_missing_version(self) -> None:
        with pytest.raises(
            InvalidPriceTable, match="missing required field: 'version'"
        ):
            PriceTable.from_dict({"schema_version": 1, "models": {}})

    def test_rejects_a_missing_models_object(self) -> None:
        with pytest.raises(InvalidPriceTable, match="missing a 'models' object"):
            PriceTable.from_dict({"schema_version": 1, "version": "v", "models": []})

    def test_rejects_a_non_object_table(self) -> None:
        with pytest.raises(InvalidPriceTable, match="must be a JSON object"):
            PriceTable.from_dict([1, 2])  # type: ignore[arg-type]

    def test_rejects_a_non_object_entry(self) -> None:
        with pytest.raises(InvalidPriceTable, match="must be an object"):
            PriceTable.from_dict(
                {"schema_version": 1, "version": "v", "models": {"m": ["cheap"]}}
            )

    def test_rejects_an_entry_missing_a_rate(self) -> None:
        with pytest.raises(InvalidPriceTable, match="missing required field"):
            PriceTable.from_dict(
                {
                    "schema_version": 1,
                    "version": "v",
                    "models": {"m": {"input_per_million": 1}},
                }
            )

    def test_rejects_a_non_numeric_rate(self) -> None:
        with pytest.raises(InvalidPriceTable, match="malformed"):
            PriceTable.from_dict(
                {
                    "schema_version": 1,
                    "version": "v",
                    "models": {
                        "m": {"input_per_million": "free", "output_per_million": 1}
                    },
                }
            )

    def test_rejects_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(InvalidPriceTable, match="not valid JSON"):
            load_price_table(path)

    def test_provenance_defaults_to_unverified(self) -> None:
        """Omitting the field must not imply the price was checked."""
        assert ModelPrice(1.0, 2.0).verified is False
        assert (
            ModelPrice.from_dict(
                {"input_per_million": 1.0, "output_per_million": 2}
            ).verified
            is False
        )

    def test_optional_rates_default_to_absent(self) -> None:
        price = ModelPrice.from_dict(
            {"input_per_million": 1.0, "output_per_million": 2}
        )
        assert price.cached_input_per_million is None
        assert price.reasoning_per_million is None

    def test_provenance_fields_default_to_empty(self) -> None:
        price = ModelPrice(1.0, 2.0)
        assert (price.source, price.as_of) == ("", "")


class TestConvenienceWrapper:
    def test_prices_with_the_bundled_table_by_default(self) -> None:
        assert price_usage(Usage(input_tokens=1_000_000), "gpt-4o").cost_usd == 2.5

    def test_accepts_an_explicit_table(self) -> None:
        assert (
            price_usage(Usage(input_tokens=1_000_000), "m", a_table()).cost_usd == 3.0
        )

    def test_propagates_unknown_models(self) -> None:
        with pytest.raises(UnknownModel):
            price_usage(Usage(), "mystery")
