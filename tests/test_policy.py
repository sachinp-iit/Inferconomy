"""Tests for budget allocation policy.

The load-bearing tests are the ones about what the policy refuses to do. A
policy that escalates on every request or that truncates on a hint is a policy
that will destroy quality or overspend, and neither is auditable after the fact.
So the interesting cases are the conservative defaults, the ceiling, and the
boundary between ``additional_budget`` (an increment) and ``escalated``.

The suite is offline by construction: the autouse guard in ``conftest.py`` fails
any test that reaches a non-loopback host, and ``test_policy.py`` additionally
asserts that the module reaches for no network client at all.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

import inferconomy.policy
from inferconomy.capabilities import CapabilityReport
from inferconomy.contracts import (
    OptimizationReport,
    Request,
    Response,
    StopReason,
)
from inferconomy.decision import Decision, TaskShape
from inferconomy.policy import (
    Allocation,
    ConservativeReferencePolicy,
    Policy,
)
from inferconomy.testing import FixedPolicy


def run(coro: Any) -> Any:
    """Drive one coroutine to completion.

    asyncio.run in a synchronous test keeps the async API under test without
    adding a pytest plugin and its configuration to the dev dependencies.
    """
    return asyncio.run(coro)


def make_decision(
    shape: TaskShape = TaskShape.DERIVED,
    *,
    uncertainty: float = 0.25,
    budget_hint_tokens: int | None = 256,
    needs_reasoning: bool = True,
    matched: tuple[str, ...] = ("derivation-command",),
    reason: str = "a reason",
) -> Decision:
    return Decision(
        shape=shape,
        engine="test-engine",
        needs_reasoning=needs_reasoning,
        uncertainty=uncertainty,
        budget_hint_tokens=budget_hint_tokens,
        reason=reason,
        matched=matched,
    )


def make_request(ceiling: int | None = None, text: str = "x") -> Request:
    request = Request.from_text(text)
    if ceiling is not None:
        return Request(
            messages=request.messages,
            max_output_tokens=ceiling,
        )
    return request


def allocate(
    decision: Decision,
    ceiling: int | None = None,
    *,
    escalations_so_far: int = 0,
    capabilities: CapabilityReport | None = None,
    history: tuple[Response, ...] = (),
    prior_reports: tuple[OptimizationReport, ...] = (),
) -> Allocation:
    policy = ConservativeReferencePolicy()
    request = make_request(ceiling)
    result: Allocation = run(
        policy.allocate(
            decision,
            request,
            capabilities,
            history=history,
            prior_reports=prior_reports,
            escalations_so_far=escalations_so_far,
        )
    )
    return result


class TestAllocationContract:
    """An allocation is a small, auditable object."""

    def test_it_carries_the_three_questions_a_policy_controls(self) -> None:
        decision = make_decision()
        allocation = Allocation(
            initial_budget=256,
            additional_budget=512,
            should_stop=False,
            stop_reason=StopReason.STRATEGY_COMPLETE,
            decision=decision,
            reason="derived task",
        )
        assert allocation.initial_budget == 256
        assert allocation.additional_budget == 512
        assert allocation.should_stop is False
        assert allocation.decision is decision

    def test_a_negative_initial_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="initial_budget must be non-negative"):
            Allocation(
                initial_budget=-1,
                additional_budget=0,
                should_stop=True,
                stop_reason=StopReason.STRATEGY_COMPLETE,
                decision=make_decision(),
                reason="x",
            )

    def test_a_negative_additional_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="additional_budget must be non-negative"):
            Allocation(
                initial_budget=0,
                additional_budget=-1,
                should_stop=True,
                stop_reason=StopReason.STRATEGY_COMPLETE,
                decision=make_decision(),
                reason="x",
            )

    def test_a_blank_reason_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason must not be blank"):
            Allocation(
                initial_budget=0,
                additional_budget=0,
                should_stop=True,
                stop_reason=StopReason.STRATEGY_COMPLETE,
                decision=make_decision(),
                reason="   ",
            )

    def test_a_non_decision_is_rejected(self) -> None:
        """The policy is a function of the analysis, not a replacement for it."""
        with pytest.raises(ValueError, match="must be a Decision instance"):
            Allocation(
                initial_budget=0,
                additional_budget=0,
                should_stop=True,
                stop_reason=StopReason.STRATEGY_COMPLETE,
                decision="not-a-decision",  # type: ignore[arg-type]
                reason="x",
            )

    def test_sufficiency_without_stop_is_rejected(self) -> None:
        """Sufficiency is a stop reason, not a state to carry forward."""
        with pytest.raises(ValueError, match="cannot be SUFFICIENCY unless"):
            Allocation(
                initial_budget=0,
                additional_budget=0,
                should_stop=False,
                stop_reason=StopReason.SUFFICIENCY,
                decision=make_decision(),
                reason="x",
            )

    def test_it_serializes(self) -> None:
        decision = make_decision()
        allocation = Allocation(
            initial_budget=128,
            additional_budget=0,
            should_stop=True,
            stop_reason=StopReason.STRATEGY_COMPLETE,
            decision=decision,
            reason="contained task",
        )
        payload = allocation.to_dict()
        assert payload["initial_budget"] == 128
        assert payload["additional_budget"] == 0
        assert payload["should_stop"] is True
        assert payload["stop_reason"] == "strategy_complete"
        assert payload["decision"]["shape"] == "derived"
        assert payload["reason"] == "contained task"

    def test_it_is_immutable(self) -> None:
        allocation = Allocation(
            initial_budget=0,
            additional_budget=0,
            should_stop=True,
            stop_reason=StopReason.STRATEGY_COMPLETE,
            decision=make_decision(),
            reason="x",
        )
        with pytest.raises(Exception):  # noqa: B017
            allocation.initial_budget = 1  # type: ignore[misc]


class TestTheProtocolIsReplaceable:
    """The whole reason the protocol exists rather than a function."""

    def test_the_reference_policy_satisfies_the_protocol(self) -> None:
        assert isinstance(ConservativeReferencePolicy(), Policy)

    def test_the_fixed_double_satisfies_the_protocol(self) -> None:
        assert isinstance(FixedPolicy(), Policy)

    def test_a_user_supplied_policy_fully_replaces_the_reference(self) -> None:
        """The reference is a default, not a special case."""
        request = make_request()
        decision = make_decision()
        reference = run(ConservativeReferencePolicy().allocate(decision, request))
        fixed = run(FixedPolicy().allocate(decision, request))
        assert reference.initial_budget == 256
        assert fixed.initial_budget == 0
        assert fixed.should_stop is True
        assert fixed.reason == "fixed test double; no allocation was computed"

    def test_both_policies_can_be_called_on_the_same_inputs(self) -> None:
        request = make_request()
        decision = make_decision()
        for policy in (ConservativeReferencePolicy(), FixedPolicy()):
            run(policy.allocate(decision, request))


class TestConservativeDefaults:
    """The reference policy errs on the side of quality."""

    def test_a_derived_task_starts_from_the_decision_hint(self) -> None:
        allocation = allocate(make_decision(budget_hint_tokens=200))
        assert allocation.initial_budget == 200

    def test_a_derived_task_without_a_hint_uses_the_default(self) -> None:
        allocation = allocate(make_decision(budget_hint_tokens=None))
        assert allocation.initial_budget == 512

    def test_a_derived_task_with_a_zero_hint_uses_the_default(self) -> None:
        """A zero hint is a confident claim reasoning is wasted; a derived task
        still gets the default, because the shape says there is work to do."""
        allocation = allocate(make_decision(budget_hint_tokens=0))
        assert allocation.initial_budget == 512

    def test_a_contained_task_is_stopped_immediately(self) -> None:
        allocation = allocate(
            make_decision(shape=TaskShape.CONTAINED, budget_hint_tokens=0)
        )
        assert allocation.initial_budget == 0
        assert allocation.additional_budget == 0
        assert allocation.should_stop is True
        assert allocation.stop_reason is StopReason.STRATEGY_COMPLETE

    def test_an_open_task_gets_a_modest_budget_and_keeps_going(self) -> None:
        allocation = allocate(
            make_decision(
                shape=TaskShape.OPEN, needs_reasoning=False, budget_hint_tokens=0
            )
        )
        assert allocation.initial_budget == 128
        assert allocation.should_stop is False

    def test_an_unknown_task_gets_a_small_non_zero_budget(self) -> None:
        """Conservative means not assuming reasoning is wasted."""
        allocation = allocate(
            make_decision(
                shape=TaskShape.UNKNOWN, uncertainty=1.0, budget_hint_tokens=None
            )
        )
        assert allocation.initial_budget == 128
        assert allocation.should_stop is False

    def test_a_conflicted_task_gets_the_same_as_an_unknown(self) -> None:
        """A conflict is also a refusal to claim a shape."""
        conflicted = allocate(
            make_decision(
                shape=TaskShape.UNKNOWN, uncertainty=0.9, budget_hint_tokens=None
            )
        )
        unknown = allocate(
            make_decision(
                shape=TaskShape.UNKNOWN, uncertainty=1.0, budget_hint_tokens=None
            )
        )
        assert conflicted.initial_budget == unknown.initial_budget


class TestTheCallerCeiling:
    """The policy allocates within the caller's ceiling, never above it."""

    def test_a_derived_task_is_clamped_to_the_ceiling(self) -> None:
        allocation = allocate(make_decision(budget_hint_tokens=2000), ceiling=100)
        assert allocation.initial_budget == 100

    def test_a_ceiling_of_one_is_honoured(self) -> None:
        allocation = allocate(make_decision(budget_hint_tokens=2000), ceiling=1)
        assert allocation.initial_budget == 1

    def test_an_unknown_task_is_clamped_to_a_small_ceiling(self) -> None:
        allocation = allocate(
            make_decision(
                shape=TaskShape.UNKNOWN, uncertainty=1.0, budget_hint_tokens=None
            ),
            ceiling=32,
        )
        assert allocation.initial_budget == 32

    def test_an_open_task_is_clamped_to_the_ceiling(self) -> None:
        allocation = allocate(
            make_decision(
                shape=TaskShape.OPEN, needs_reasoning=False, budget_hint_tokens=0
            ),
            ceiling=64,
        )
        assert allocation.initial_budget == 64

    def test_the_ceiling_is_never_exceeded(self) -> None:
        for ceiling in (1, 10, 64, 256, 1024):
            allocation = allocate(
                make_decision(budget_hint_tokens=4096), ceiling=ceiling
            )
            assert allocation.initial_budget <= ceiling


class TestEscalation:
    """Escalation is an increment, not a rewrite of the initial budget."""

    def test_a_derived_task_offers_an_escalation(self) -> None:
        allocation = allocate(make_decision(budget_hint_tokens=100))
        assert allocation.additional_budget > 0
        assert allocation.should_stop is False

    def test_the_first_escalation_is_a_multiple_of_the_initial(self) -> None:
        allocation = allocate(
            make_decision(budget_hint_tokens=100), escalations_so_far=0
        )
        assert allocation.additional_budget == 200

    def test_escalation_stops_after_the_cap(self) -> None:
        allocation = allocate(
            make_decision(budget_hint_tokens=100), escalations_so_far=3
        )
        assert allocation.additional_budget == 0

    def test_escalation_is_clamped_to_the_remaining_ceiling(self) -> None:
        allocation = allocate(make_decision(budget_hint_tokens=100), ceiling=250)
        assert allocation.initial_budget == 100
        assert allocation.additional_budget == 150

    def test_escalation_never_exceeds_the_ceiling(self) -> None:
        for escalations in range(5):
            allocation = allocate(
                make_decision(budget_hint_tokens=100),
                ceiling=250,
                escalations_so_far=escalations,
            )
            assert allocation.initial_budget + allocation.additional_budget <= 250

    def test_a_contained_task_never_escalates(self) -> None:
        allocation = allocate(
            make_decision(shape=TaskShape.CONTAINED, budget_hint_tokens=0)
        )
        assert allocation.additional_budget == 0

    def test_an_open_task_never_escalates(self) -> None:
        allocation = allocate(
            make_decision(
                shape=TaskShape.OPEN, needs_reasoning=False, budget_hint_tokens=0
            )
        )
        assert allocation.additional_budget == 0

    def test_an_unknown_task_never_escalates(self) -> None:
        allocation = allocate(
            make_decision(
                shape=TaskShape.UNKNOWN, uncertainty=1.0, budget_hint_tokens=None
            )
        )
        assert allocation.additional_budget == 0


class TestDeterminism:
    """Attribution to the policy requires the allocation to be stable."""

    def test_the_same_inputs_allocate_identically_twice(self) -> None:
        decision = make_decision()
        first = allocate(decision)
        second = allocate(decision)
        assert first == second

    def test_it_is_stable_across_separate_event_loops(self) -> None:
        decision = make_decision()
        assert run(
            ConservativeReferencePolicy().allocate(decision, make_request())
        ) == run(ConservativeReferencePolicy().allocate(decision, make_request()))

    def test_the_policy_keeps_no_state(self) -> None:
        policy = ConservativeReferencePolicy()
        run(policy.allocate(make_decision(), make_request()))
        assert not [name for name in vars(policy)]


class TestTheAuditTrail:
    """Every allocation explains itself."""

    def test_every_allocation_carries_a_reason(self) -> None:
        for shape in TaskShape:
            decision = make_decision(shape=shape)
            allocation = allocate(decision)
            assert allocation.reason.strip()

    def test_the_reason_names_the_decision(self) -> None:
        decision = make_decision(
            uncertainty=0.4, budget_hint_tokens=None, matched=("a", "b")
        )
        allocation = allocate(decision)
        assert "unc=0.40" in allocation.reason
        assert "hint=none" in allocation.reason
        assert "matched=a, b" in allocation.reason
        assert decision.reason in allocation.reason

    def test_the_reason_distinguishes_a_zero_hint_from_none(self) -> None:
        zero_hint = allocate(make_decision(budget_hint_tokens=0))
        no_hint = allocate(make_decision(budget_hint_tokens=None))
        assert "hint=0" in zero_hint.reason
        assert "hint=none" in no_hint.reason


class TestItCostsNothing:
    """The reference policy must be free to run."""

    def test_the_module_reaches_for_no_network_client(self) -> None:
        source = Path(inferconomy.policy.__file__).read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", source, re.M)
        roots = {name.split(".")[0] for name in imports}
        assert not roots & {"httpx", "urllib", "socket", "requests", "http", "ssl"}

    def test_the_module_imports_no_provider(self) -> None:
        source = Path(inferconomy.policy.__file__).read_text(encoding="utf-8")
        assert "inferconomy.providers" not in source

    def test_the_policy_has_no_client_to_call(self) -> None:
        assert not [
            name for name in vars(ConservativeReferencePolicy()) if "client" in name
        ]


class TestTheFixedDouble:
    """A fake returning fixed values, for testing what is done with them."""

    def test_it_returns_the_pinned_allocation(self) -> None:
        decision = make_decision()
        request = make_request()
        pinned = Allocation(
            initial_budget=128,
            additional_budget=0,
            should_stop=True,
            stop_reason=StopReason.STRATEGY_COMPLETE,
            decision=decision,
            reason="pinned",
        )
        policy = FixedPolicy(pinned)
        assert run(policy.allocate(decision, request)) == pinned

    def test_it_defaults_to_an_allocation_it_admits_it_did_not_compute(self) -> None:
        policy = FixedPolicy()
        allocation = run(policy.allocate(make_decision(), make_request()))
        assert allocation.initial_budget == 0
        assert allocation.additional_budget == 0
        assert allocation.should_stop is True
        assert "no allocation was computed" in allocation.reason

    def test_its_default_engine_name_is_its_own(self) -> None:
        policy = FixedPolicy()
        allocation = run(policy.allocate(make_decision(), make_request()))
        assert allocation.decision.engine == policy.name

    def test_the_allocation_property_exposes_the_pinned_object(self) -> None:
        policy = FixedPolicy()
        assert policy.allocation.initial_budget == 0
        assert policy.allocation.should_stop is True

    def test_it_records_what_it_was_asked(self) -> None:
        policy = FixedPolicy()
        first = make_decision(shape=TaskShape.CONTAINED, budget_hint_tokens=0)
        second = make_decision()
        run(policy.allocate(first, make_request()))
        run(policy.allocate(second, make_request()))
        assert policy.call_count == 2
        assert policy.calls[0][0] is first
        assert policy.calls[1][0] is second

    def test_reset_forgets_calls_but_keeps_the_allocation(self) -> None:
        policy = FixedPolicy()
        run(policy.allocate(make_decision(), make_request()))
        policy.reset()
        assert policy.call_count == 0
        assert run(policy.allocate(make_decision(), make_request())).initial_budget == 0

    def test_a_determinism_override_can_be_declared(self) -> None:
        policy = FixedPolicy(deterministic=False)
        assert policy.deterministic is False

    def test_a_named_double_keeps_its_name(self) -> None:
        policy = FixedPolicy(name="always-stop")
        assert policy.name == "always-stop"

    def test_it_reprs_without_leaking_the_request(self) -> None:
        policy = FixedPolicy()
        run(policy.allocate(make_decision(), make_request()))
        assert "fixed-policy" in repr(policy)
        assert "test-engine" not in repr(policy)
