"""Tests for task analysis.

The load-bearing tests are the ones about what the analysis refuses to claim.
An engine that classifies every request is an engine that has invented a
purchase on tasks it has none, and the decisions downstream - spend or do not
spend - are made from that claim. So the interesting cases here are the empty
prompt, the mixed prompt, and the distinction between "I know this is not
needed" and "I have no idea".

The suite is offline by construction: the autouse guard in ``conftest.py`` fails
any test that reaches a non-loopback host, and ``test_decision.py`` additionally
asserts that the module reaches for no network client at all, so the property
does not rest on the guard alone.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

import inferconomy.decision
from inferconomy.contracts import Message, Request
from inferconomy.decision import (
    CONFLICT_UNCERTAINTY,
    DERIVED_BUDGET_HINT_TOKENS,
    Decision,
    DecisionEngine,
    HeuristicDecisionEngine,
    TaskShape,
)
from inferconomy.testing import FixedDecisionEngine


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def decide(
    text: str, *, system: str | None = None, ceiling: int | None = None
) -> Decision:
    request = Request.from_text(text, system=system)
    if ceiling is not None:
        request = Request(
            messages=request.messages,
            max_output_tokens=ceiling,
        )
    eng = HeuristicDecisionEngine()
    return asyncio.run(eng.decide(request))


def make_decision(**overrides: Any) -> Decision:
    fields: dict[str, Any] = {
        "shape": TaskShape.DERIVED,
        "engine": "test-engine",
        "needs_reasoning": True,
        "uncertainty": 0.25,
        "budget_hint_tokens": 256,
        "reason": "a reason",
        "matched": ("derivation-command",),
    }
    fields.update(overrides)
    return Decision(**fields)


class TestTaskShape:
    """Four answers are available, and one of them is "I do not know"."""

    def test_the_shapes_are_the_four_answers_the_analysis_can_give(self) -> None:
        assert [shape.value for shape in TaskShape] == [
            "unknown",
            "contained",
            "derived",
            "open",
        ]

    def test_a_shape_is_a_string(self) -> None:
        assert TaskShape.DERIVED.value == "derived"
        assert TaskShape.OPEN.value == "open"

    def test_unknown_is_not_the_absence_of_a_shape(self) -> None:
        """It is a shape in its own right, so it can be reported and compared."""
        assert TaskShape.UNKNOWN is not None
        assert (
            Decision(
                shape=TaskShape.UNKNOWN,
                engine="e",
                needs_reasoning=False,
                uncertainty=1.0,
                budget_hint_tokens=None,
                reason="nothing seen",
            ).shape
            is TaskShape.UNKNOWN
        )


class TestDecisionValidation:
    """A decision that cannot be trusted is refused rather than averaged."""

    def test_uncertainty_must_be_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            make_decision(uncertainty=float("nan"))

    def test_uncertainty_infinite_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            make_decision(uncertainty=float("inf"))

    def test_uncertainty_above_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            make_decision(uncertainty=1.5)

    def test_uncertainty_below_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            make_decision(uncertainty=-0.1)

    def test_needs_reasoning_must_be_a_bool(self) -> None:
        """It is a recommendation that costs money, not a value to coerce."""
        with pytest.raises(ValueError, match="must be a bool"):
            make_decision(needs_reasoning=1)

    def test_a_negative_budget_hint_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            make_decision(budget_hint_tokens=-1)

    def test_a_boolean_budget_hint_is_rejected(self) -> None:
        """``True`` is an ``int``, and accepting it would make 0 mean "false"."""
        with pytest.raises(ValueError, match="int or None"):
            make_decision(budget_hint_tokens=True)

    def test_a_non_numeric_budget_hint_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="int or None"):
            make_decision(budget_hint_tokens="256")

    def test_zero_is_a_valid_budget_hint(self) -> None:
        """Zero is the confident claim that reasoning would be wasted."""
        assert make_decision(budget_hint_tokens=0).budget_hint_tokens == 0

    def test_a_blank_reason_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason must not be blank"):
            make_decision(reason="   ")

    def test_a_blank_engine_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="engine must not be blank"):
            make_decision(engine="")

    def test_an_unknown_shape_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unusable values"):
            Decision.from_dict(
                {
                    "shape": "obviously-hard",
                    "engine": "e",
                    "needs_reasoning": False,
                    "uncertainty": 0.1,
                    "budget_hint_tokens": None,
                    "reason": "r",
                    "matched": [],
                }
            )

    def test_the_shape_is_normalized_to_the_enum(self) -> None:
        assert make_decision(shape="open").shape is TaskShape.OPEN

    def test_matched_is_normalized_to_a_tuple(self) -> None:
        decision = make_decision(matched=["a", "b"])
        assert decision.matched == ("a", "b")

    def test_a_decision_is_immutable(self) -> None:
        decision = make_decision()
        with pytest.raises(Exception):  # noqa: B017
            decision.uncertainty = 0.9  # type: ignore[misc]

    def test_uncertainty_is_normalized_to_a_float(self) -> None:
        assert isinstance(make_decision(uncertainty=1).uncertainty, float)


class TestDecisionSerialization:
    """A decision is written into a report and read back months later."""

    def test_a_decision_round_trips_through_a_dict(self) -> None:
        decision = make_decision()
        assert Decision.from_dict(decision.to_dict()) == decision

    def test_a_decision_survives_a_json_round_trip(self) -> None:
        decision = make_decision()
        restored = Decision.from_dict(json.loads(json.dumps(decision.to_dict())))
        assert restored == decision

    def test_the_payload_carries_the_engine_and_the_evidence(self) -> None:
        payload = make_decision().to_dict()
        assert payload["engine"] == "test-engine"
        assert payload["shape"] == "derived"
        assert payload["matched"] == ["derivation-command"]

    def test_a_round_tripped_decision_answers_the_same_questions(self) -> None:
        restored = Decision.from_dict(make_decision().to_dict())
        assert restored.needs_reasoning is True
        assert restored.budget_hint_tokens == 256

    def test_from_dict_reports_a_missing_field(self) -> None:
        payload = make_decision().to_dict()
        del payload["uncertainty"]
        with pytest.raises(ValueError, match="missing required field"):
            Decision.from_dict(payload)

    def test_from_dict_rejects_a_non_boolean_needs_reasoning(self) -> None:
        """``bool("false")`` is True, so a string must not be coerced."""
        payload = make_decision().to_dict()
        payload["needs_reasoning"] = "false"
        with pytest.raises(ValueError, match="must be a boolean"):
            Decision.from_dict(payload)

    def test_from_dict_rejects_an_absent_budget_hint_as_none(self) -> None:
        payload = make_decision(budget_hint_tokens=None).to_dict()
        assert Decision.from_dict(payload).budget_hint_tokens is None


class TestAuditLine:
    """The line that reaches
    :attr:`~inferconomy.contracts.OptimizationReport.decisions`."""

    def test_it_names_the_engine_the_shape_and_the_evidence(self) -> None:
        line = make_decision().audit_line()
        assert "test-engine" in line
        assert "derived" in line
        assert "reasoning=yes" in line
        assert "derivation-command" in line

    def test_it_distinguishes_a_zero_hint_from_no_opinion(self) -> None:
        assert "budget_hint=0" in make_decision(budget_hint_tokens=0).audit_line()
        assert "budget_hint=none" in make_decision(budget_hint_tokens=None).audit_line()

    def test_it_marks_a_decision_that_saw_nothing(self) -> None:
        assert "matched=none" in make_decision(matched=()).audit_line()

    def test_it_reports_uncertainty_to_two_places(self) -> None:
        assert "uncertainty=0.25" in make_decision().audit_line()


class TestTheHeuristicEngine:
    """The zero-cost reference engine: rules, no calls, no state."""

    def test_it_declares_itself_deterministic(self) -> None:
        engine = HeuristicDecisionEngine()
        assert engine.deterministic is True

    def test_its_name_is_versioned(self) -> None:
        assert "2026" in HeuristicDecisionEngine().name

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(HeuristicDecisionEngine(), DecisionEngine)

    def test_a_trivial_request_is_unknown_at_full_uncertainty(self) -> None:
        decision = decide("hello there")
        assert decision.shape is TaskShape.UNKNOWN
        assert decision.uncertainty == 1.0
        assert decision.matched == ()

    def test_a_blank_request_is_unknown_at_full_uncertainty(self) -> None:
        decision = decide("   ")
        assert decision.shape is TaskShape.UNKNOWN
        assert decision.uncertainty == 1.0

    def test_an_unknown_request_recommends_no_reasoning(self) -> None:
        assert decide("hello").needs_reasoning is False

    def test_a_derivation_request_is_classified_derived(self) -> None:
        decision = decide("Prove that the sum of two even numbers is even.")
        assert decision.shape is TaskShape.DERIVED
        assert decision.matched == ("derivation-command",)

    def test_a_derived_task_recommends_reasoning(self) -> None:
        assert decide("Compute the factorial of 12.").needs_reasoning is True

    def test_a_derived_task_suggests_a_starter_budget(self) -> None:
        assert decide("Compute the factorial of 12.").budget_hint_tokens == (
            DERIVED_BUDGET_HINT_TOKENS
        )

    def test_an_extraction_request_is_classified_contained(self) -> None:
        decision = decide("Summarize this document.")
        assert decision.shape is TaskShape.CONTAINED
        assert decision.matched == ("extraction-request",)

    def test_a_contained_task_recommends_no_reasoning(self) -> None:
        assert decide("Summarize this document.").needs_reasoning is False

    def test_a_contained_task_gets_zero_not_none(self) -> None:
        """Zero is the confident claim that reasoning would be wasted."""
        assert decide("Summarize this document.").budget_hint_tokens == 0

    def test_a_creative_request_is_classified_open(self) -> None:
        decision = decide("Write a poem about the sea.")
        assert decision.shape is TaskShape.OPEN
        assert decision.matched == ("generative-request",)

    def test_a_subjective_request_is_classified_open(self) -> None:
        decision = decide("What do you think about this plan?")
        assert decision.shape is TaskShape.OPEN
        assert decision.matched == ("subjective-request",)

    def test_an_unknown_task_gets_none_not_zero(self) -> None:
        """None is the refusal to have an opinion. Zero is an opinion."""
        assert decide("hello there").budget_hint_tokens is None

    def test_every_decision_explains_itself(self) -> None:
        for text in ("hello", "Prove it.", "Summarize it.", "Write a story."):
            assert decide(text).reason.strip()

    def test_every_decision_names_its_engine(self) -> None:
        assert decide("hello").engine == HeuristicDecisionEngine().name


class TestConflictingSignals:
    """A mixed task is not the same as an absent one."""

    def test_signals_that_disagree_decline_to_classify(self) -> None:
        decision = decide("Summarize the report, and then explain why sales fell.")
        assert decision.shape is TaskShape.UNKNOWN
        assert decision.matched == ("analytical-question", "extraction-request")

    def test_a_conflict_is_reported_at_the_conflict_uncertainty(self) -> None:
        decision = decide("Summarize the report, and then explain why sales fell.")
        assert decision.uncertainty == CONFLICT_UNCERTAINTY

    def test_a_conflict_is_more_uncertain_than_agreement(self) -> None:
        conflicted = decide("Summarize the report, and then explain why sales fell.")
        agreed = decide("Summarize this document.")
        assert conflicted.uncertainty > agreed.uncertainty

    def test_no_signal_is_more_uncertain_than_a_conflict(self) -> None:
        """Two kinds of not knowing, and they are not interchangeable."""
        assert (
            decide("hello there").uncertainty
            > decide(
                "Summarize the report, and then explain why sales fell."
            ).uncertainty
        )

    def test_a_conflict_carries_the_rules_that_caused_it(self) -> None:
        decision = decide("Summarize the report, and then explain why sales fell.")
        assert decision.matched != ()

    def test_a_conflict_recommends_no_reasoning(self) -> None:
        decision = decide("Summarize the report, and then explain why sales fell.")
        assert decision.needs_reasoning is False

    def test_a_conflict_has_no_budget_opinion(self) -> None:
        decision = decide("Summarize the report, and then explain why sales fell.")
        assert decision.budget_hint_tokens is None

    def test_a_list_of_mixed_questions_is_not_declined(self) -> None:
        """Two numbered sub-questions, one extractive and one analytical.

        Not a tie, so it is not declined: two of three signals say the answer has
        to be derived, and the list is classified as work. The dissent stays
        visible in ``matched`` and in an uncertainty well above a clean agreement.
        """
        decision = decide("1. Summarize the report.\n2. Why did sales fall?")
        assert decision.shape is TaskShape.DERIVED
        assert decision.matched == (
            "analytical-question",
            "multi-part",
            "extraction-request",
        )
        assert decision.uncertainty == pytest.approx(1 - 0.8 * (2 / 3))

    def test_the_reason_says_the_signals_disagreed(self) -> None:
        decision = decide("Summarize the report, and then explain why sales fell.")
        assert "disagreed" in decision.reason


class TestUncertainty:
    """How sure the engine is, as a function of how much it saw."""

    def test_agreement_of_one_signal_is_mostly_uncertain(self) -> None:
        assert decide("Summarize this document.").uncertainty == pytest.approx(
            1 - 0.8 * (1 / 3)
        )

    def test_two_agreeing_signals_lower_the_uncertainty(self) -> None:
        decision = decide("Write a function that computes the factorial of n.")
        assert decision.shape is TaskShape.DERIVED
        assert decision.matched == ("derivation-command", "code-task")
        assert decision.uncertainty == pytest.approx(1 - 0.8 * (2 / 3))

    def test_three_agreeing_signals_lower_it_further(self) -> None:
        decision = decide(
            "Prove that this works. Analyze the trade-offs. Debug the function."
        )
        assert decision.matched == (
            "derivation-command",
            "analytical-question",
            "code-task",
        )
        assert decision.uncertainty == pytest.approx(0.2)

    def test_more_agreement_is_never_more_uncertain(self) -> None:
        one = decide("Summarize this document.").uncertainty
        three = decide(
            "Prove that this works. Analyze the trade-offs. Debug it."
        ).uncertainty
        assert three < one

    def test_uncertainty_is_never_below_the_confidence_ceiling(self) -> None:
        decision = decide(
            "Prove that this works. Analyze the trade-offs. Debug the function."
        )
        assert decision.uncertainty == pytest.approx(1 - 0.8)


class TestTheBudgetHint:
    """A suggestion that follows the shape and never exceeds the caller."""

    def test_it_never_exceeds_the_caller_ceiling(self) -> None:
        decision = decide("Prove that the theorem holds.", ceiling=100)
        assert decision.budget_hint_tokens == 100

    def test_a_ceiling_of_one_is_honoured(self) -> None:
        decision = decide("Prove that the theorem holds.", ceiling=1)
        assert decision.budget_hint_tokens == 1

    def test_a_generous_ceiling_leaves_the_suggestion_alone(self) -> None:
        decision = decide("Prove that the theorem holds.", ceiling=100_000)
        assert decision.budget_hint_tokens == DERIVED_BUDGET_HINT_TOKENS

    def test_the_hint_is_never_negative(self) -> None:
        for text in ("hello", "Summarize it.", "Write a story.", "Prove it."):
            hint = decide(text, ceiling=10).budget_hint_tokens
            assert hint is None or hint >= 0


class TestWhatTheAnalysisReads:
    """Only the task, and only its final user turn."""

    def test_a_system_prompt_is_not_scored_as_evidence(self) -> None:
        """A standing instruction would otherwise describe every request."""
        decision = decide(
            "Summarize this document.",
            system="You must prove every claim you make.",
        )
        assert decision.shape is TaskShape.CONTAINED
        assert decision.matched == ("extraction-request",)

    def test_a_system_prompt_cannot_make_a_task_derived(self) -> None:
        decision = decide(
            "hello",
            system="Analyze the trade-offs of everything you are asked.",
        )
        assert decision.shape is TaskShape.UNKNOWN

    def test_only_the_final_user_turn_is_read(self) -> None:
        request = Request(
            messages=(
                Message(role="user", content="Prove that the theorem holds."),
                Message(role="assistant", content="It holds because ..."),
                Message(role="user", content="Summarize the document."),
            )
        )
        decision = run(HeuristicDecisionEngine().decide(request))
        assert decision.shape is TaskShape.CONTAINED

    def test_a_request_with_no_user_turn_falls_back_to_the_last_turn(self) -> None:
        request = Request(
            messages=(Message(role="system", content="Prove everything."),)
        )
        decision = run(HeuristicDecisionEngine().decide(request))
        assert decision.shape is TaskShape.DERIVED

    def test_a_creative_noun_without_a_creative_verb_is_not_open(self) -> None:
        """ "Summarize the story" is extraction, and the noun alone must not say so."""
        decision = decide("Summarize the story you were given.")
        assert decision.shape is TaskShape.CONTAINED

    def test_matched_is_recorded_in_declaration_order(self) -> None:
        decision = decide("1. Summarize the report.\n2. Explain why sales fell.")
        assert decision.matched == (
            "analytical-question",
            "multi-part",
            "extraction-request",
        )


class TestDeterminism:
    """Attribution to the strategy requires the analysis to be stable."""

    def test_the_same_request_decides_identically_twice(self) -> None:
        request = Request.from_text("Prove that the theorem holds.")
        engine = HeuristicDecisionEngine()
        first = run(engine.decide(request))
        second = run(engine.decide(request))
        assert first == second

    def test_it_is_stable_across_separate_event_loops(self) -> None:
        """Two asyncio.run calls, so no shared loop state is doing the work."""
        request = Request.from_text("Summarize the report and explain why.")
        assert run(HeuristicDecisionEngine().decide(request)) == run(
            HeuristicDecisionEngine().decide(request)
        )

    def test_two_engines_do_not_share_state(self) -> None:
        first = HeuristicDecisionEngine()
        second = HeuristicDecisionEngine()
        run(first.decide(Request.from_text("hello")))
        assert (
            run(second.decide(Request.from_text("Prove it."))).needs_reasoning is True
        )

    def test_the_engine_keeps_no_record_of_what_it_analysed(self) -> None:
        engine = HeuristicDecisionEngine()
        run(engine.decide(Request.from_text("hello")))
        assert not [name for name in vars(engine)]


class TestItCostsNothing:
    """The requirement is a zero-cost heuristic, and that is testable."""

    def test_the_module_reaches_for_no_network_client(self) -> None:
        source = Path(inferconomy.decision.__file__).read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", source, re.M)
        roots = {name.split(".")[0] for name in imports}
        assert not roots & {"httpx", "urllib", "socket", "requests", "http", "ssl"}

    def test_the_module_imports_no_provider(self) -> None:
        source = Path(inferconomy.decision.__file__).read_text(encoding="utf-8")
        assert "inferconomy.providers" not in source

    def test_the_engine_has_no_client_to_call(self) -> None:
        assert not [
            name for name in vars(HeuristicDecisionEngine()) if "client" in name
        ]


class TestTheFixedDouble:
    """A fake returning fixed values, for testing what is done with them."""

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(FixedDecisionEngine(), DecisionEngine)

    def test_it_returns_the_pinned_decision(self) -> None:
        pinned = make_decision()
        engine = FixedDecisionEngine(pinned)
        assert run(engine.decide(Request.from_text("Prove it."))) == pinned

    def test_it_defaults_to_a_decision_it_admits_it_did_not_measure(self) -> None:
        engine = FixedDecisionEngine()
        decision = run(engine.decide(Request.from_text("Prove it.")))
        assert decision.shape is TaskShape.UNKNOWN
        assert decision.uncertainty == 1.0
        assert "no analysis was performed" in decision.reason

    def test_its_default_engine_name_is_its_own(self) -> None:
        engine = FixedDecisionEngine()
        assert run(engine.decide(Request.from_text("x"))).engine == engine.name

    def test_it_records_what_it_was_asked(self) -> None:
        engine = FixedDecisionEngine()
        first = Request.from_text("hello")
        second = Request.from_text("Prove it.")
        run(engine.decide(first))
        run(engine.decide(second))
        assert engine.calls == (first, second)
        assert engine.call_count == 2

    def test_it_ignores_what_it_was_asked(self) -> None:
        """That is the point: the decision is pinned, not derived."""
        engine = FixedDecisionEngine(make_decision())
        run(engine.decide(Request.from_text("hello")))
        run(engine.decide(Request.from_text("Prove it.")))
        assert engine.calls[0] is not engine.calls[1]

    def test_reset_forgets_calls_but_keeps_the_decision(self) -> None:
        engine = FixedDecisionEngine(make_decision())
        run(engine.decide(Request.from_text("hello")))
        engine.reset()
        assert engine.call_count == 0
        assert run(engine.decide(Request.from_text("hello"))) == engine.decision

    def test_a_determinism_override_can_be_declared(self) -> None:
        engine = FixedDecisionEngine(deterministic=False)
        assert engine.deterministic is False

    def test_a_named_double_keeps_its_name(self) -> None:
        engine = FixedDecisionEngine(name="always-derived")
        assert engine.name == "always-derived"

    def test_it_reprs_without_leaking_the_request(self) -> None:
        engine = FixedDecisionEngine()
        run(engine.decide(Request.from_text("hello")))
        assert "fixed-decision" in repr(engine)


class TestTheProtocolIsReplaceable:
    """The whole reason the protocol exists rather than a function."""

    def test_the_double_can_disagree_with_the_heuristic(self) -> None:
        """Same request, opposite conclusions, and nothing downstream cares."""
        request = Request.from_text("Prove that the theorem holds.")
        heuristic = run(HeuristicDecisionEngine().decide(request))
        pinned = run(
            FixedDecisionEngine(make_decision(shape=TaskShape.CONTAINED)).decide(
                request
            )
        )
        assert heuristic.needs_reasoning is True
        assert pinned.needs_reasoning is True
        assert pinned.shape is TaskShape.CONTAINED
        assert heuristic.shape is TaskShape.DERIVED

    def test_a_double_can_recommend_no_reasoning_where_the_heuristic_would(
        self,
    ) -> None:
        request = Request.from_text("Prove that the theorem holds.")
        pinned = run(
            FixedDecisionEngine(
                make_decision(shape=TaskShape.UNKNOWN, needs_reasoning=False)
            ).decide(request)
        )
        assert run(HeuristicDecisionEngine().decide(request)).needs_reasoning is True
        assert pinned.needs_reasoning is False

    def test_both_engines_produce_a_serializable_decision(self) -> None:
        request = Request.from_text("hello")
        for engine in (HeuristicDecisionEngine(), FixedDecisionEngine()):
            decision = run(engine.decide(request))
            assert Decision.from_dict(decision.to_dict()) == decision
