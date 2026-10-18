"""Tests for strategy execution.

The load-bearing tests are the ones about how a decision reaches the wire and
how it is recorded afterwards. A strategy that quietly raised the budget,
dropped the caller's ceiling, or reported a stop it never observed would make
every number above it in the stack unverifiable, so those are the cases pinned
here.

The suite is offline by construction: the autouse guard in ``conftest.py`` fails
any test that reaches a non-loopback host, and this module additionally asserts
that the strategy code reaches for no network stack at all.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import inferconomy.strategy
from inferconomy.benchmark import BaselineConfig, Task, run_baseline
from inferconomy.capabilities import Adaptation, CapabilityReport
from inferconomy.client import Client, CompletionOptions
from inferconomy.contracts import (
    Capability,
    OptimizationReport,
    Request,
    Response,
    StopReason,
    Usage,
    UsageBasis,
)
from inferconomy.costs import ModelPrice, PriceTable
from inferconomy.decision import Decision, HeuristicDecisionEngine, TaskShape
from inferconomy.judge import JudgeRequest
from inferconomy.policy import Allocation, ConservativeReferencePolicy
from inferconomy.strategy import (
    DirectStrategy,
    Strategy,
    StrategyResult,
    select_strategy,
)
from inferconomy.testing import FakeClient, FixedStrategy, FunctionJudge


def run(coro: Any) -> Any:
    """Drive one coroutine to completion.

    asyncio.run in a synchronous test keeps the async API under test without
    adding a pytest plugin and its configuration to the dev dependencies.
    """
    return asyncio.run(coro)


def make_decision() -> Decision:
    return Decision(
        shape=TaskShape.UNKNOWN,
        engine="test-engine",
        needs_reasoning=False,
        uncertainty=1.0,
        budget_hint_tokens=None,
        reason="a reason",
    )


def make_allocation(
    initial_budget: int = 128,
    additional_budget: int = 0,
    *,
    should_stop: bool = False,
    stop_reason: StopReason = StopReason.STRATEGY_COMPLETE,
) -> Allocation:
    return Allocation(
        initial_budget=initial_budget,
        additional_budget=additional_budget,
        should_stop=should_stop,
        stop_reason=stop_reason,
        decision=make_decision(),
        reason="test allocation",
    )


def make_request(*, text: str = "hello", ceiling: int | None = None) -> Request:
    request = Request.from_text(text)
    if ceiling is None:
        return request
    return Request(messages=request.messages, max_output_tokens=ceiling)


def run_direct(
    allocation: Allocation | None = None,
    *,
    request: Request | None = None,
    client: FakeClient | None = None,
    capabilities: CapabilityReport | None = None,
) -> tuple[StrategyResult, FakeClient]:
    """Execute one direct attempt and hand back the result and the client."""
    return _run_strategy(
        DirectStrategy(),
        allocation,
        request=request,
        client=client,
        capabilities=capabilities,
    )


def _run_strategy(
    strategy: Strategy,
    allocation: Allocation | None,
    *,
    request: Request | None,
    client: FakeClient | None,
    capabilities: CapabilityReport | None,
) -> tuple[StrategyResult, FakeClient]:
    request = make_request() if request is None else request
    client = FakeClient([Response(text="answer")]) if client is None else client
    allocation = make_allocation() if allocation is None else allocation
    result: StrategyResult = run(
        strategy.execute(request, allocation, client, capabilities)
    )
    return result, client


class TestStrategyResult:
    """A result is a response and its report, and nothing else."""

    def test_it_pairs_the_response_with_its_report(self) -> None:
        response = Response(text="hello")
        report = OptimizationReport(strategy="direct")
        result = StrategyResult(response=response, report=report)
        assert result.response is response
        assert result.report is report

    def test_it_is_frozen(self) -> None:
        result = StrategyResult(
            response=Response(text="hello"),
            report=OptimizationReport(strategy="direct"),
        )
        with pytest.raises(FrozenInstanceError):
            result.response = Response(text="x")  # type: ignore[misc]

    def test_a_non_response_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="response must be a Response"):
            StrategyResult(
                response="not-a-response",  # type: ignore[arg-type]
                report=OptimizationReport(strategy="direct"),
            )

    def test_a_non_report_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="report must be an OptimizationReport"):
            StrategyResult(
                response=Response(text="hello"),
                report="not-a-report",  # type: ignore[arg-type]
            )


class TestTheProtocolIsReplaceable:
    """Any object with the three members is a strategy; no subclassing required."""

    def test_direct_satisfies_the_protocol(self) -> None:
        assert isinstance(DirectStrategy(), Strategy)

    def test_the_fixed_double_satisfies_the_protocol(self) -> None:
        assert isinstance(FixedStrategy(), Strategy)

    def test_the_protocol_is_narrow(self) -> None:
        """A protocol that grows to fit a strategy stops being honest."""
        public = {name for name in vars(Strategy) if not name.startswith("_")}
        assert public == {"name", "deterministic", "execute"}

    def test_an_object_missing_a_member_does_not_satisfy_it(self) -> None:
        class NotAStrategy:
            name = "no-execute"
            deterministic = True

        assert not isinstance(NotAStrategy(), Strategy)

    def test_a_user_defined_class_without_subclassing_is_accepted(self) -> None:
        class Inline:
            name = "inline"
            deterministic = True

            async def execute(
                self,
                request: Request,
                allocation: Allocation,
                client: Client,
                capabilities: CapabilityReport | None = None,
            ) -> StrategyResult:
                return StrategyResult(
                    response=Response(text="inline"),
                    report=OptimizationReport(strategy=self.name),
                )

        inline: Strategy = Inline()
        assert inline.name == "inline"
        result, _ = _run_strategy(
            inline, None, request=None, client=None, capabilities=None
        )
        assert result.response.text == "inline"


class RewritingReport(CapabilityReport):
    """A probed report whose ``adapt`` rewrites options, to tell consultation."""

    def adapt(self, options: CompletionOptions) -> Adaptation:
        return Adaptation(options=CompletionOptions(max_output_tokens=1))


class TestDirectStrategy:
    """One call, the allocation on the wire, and an honest report back."""

    def test_it_makes_exactly_one_call(self) -> None:
        result, client = run_direct()
        assert client.call_count == 1
        assert result.response.text == "answer"

    def test_the_allocated_budget_becomes_the_output_cap(self) -> None:
        _, client = run_direct(make_allocation(initial_budget=256))
        assert client.last_options.max_output_tokens == 256

    def test_it_sets_no_reasoning_levers_and_no_model(self) -> None:
        _, client = run_direct(make_allocation(initial_budget=256))
        options = client.last_options
        assert options.reasoning_budget is None
        assert options.effort is None
        assert options.model is None

    def test_a_zero_allocation_sends_the_callers_ceiling(self) -> None:
        result, client = run_direct(
            make_allocation(initial_budget=0),
            request=make_request(ceiling=4096),
        )
        assert client.last_options.max_output_tokens == 4096
        assert "caller ceiling 4096 sent as the cap" in result.report.decisions[2]

    def test_a_zero_allocation_without_a_ceiling_sends_no_cap(self) -> None:
        result, client = run_direct(make_allocation(initial_budget=0))
        assert client.last_options.max_output_tokens is None
        assert "provider default cap applies" in result.report.decisions[2]

    def test_the_callers_stop_sequences_are_passed_through(self) -> None:
        request = Request(messages=make_request().messages, stop=("END",))
        _, client = run_direct(request=request)
        assert client.last_options.stop == ("END",)

    def test_the_request_reaches_the_client_unmodified(self) -> None:
        request = make_request(text="solve this")
        _, client = run_direct(request=request)
        assert client.calls[0][0] is request

    def test_exact_usage_earns_the_usage_reporting_capability(self) -> None:
        client = FakeClient(
            [
                Response(
                    text="answer",
                    usage=Usage(
                        input_tokens=4,
                        output_tokens=2,
                        llm_calls=1,
                        tokens_exact=True,
                    ),
                )
            ]
        )
        result, _ = run_direct(client=client)
        assert result.report.capabilities_used == (Capability.USAGE_REPORTING,)

    def test_estimated_usage_claims_no_capability(self) -> None:
        result, _ = run_direct(client=FakeClient([Response(text="answer")]))
        assert result.report.capabilities_used == ()

    def test_the_report_records_the_allocation_and_the_response_usage(self) -> None:
        allocation = make_allocation(initial_budget=256, additional_budget=512)
        result, _ = run_direct(allocation)
        report = result.report
        assert report.strategy == "direct"
        assert report.initial_budget == 256
        assert report.additional_budget == 0
        assert report.escalated is False
        assert report.usage is result.response.usage
        assert report.stopped_on is StopReason.STRATEGY_COMPLETE

    def test_the_audit_trail_is_decision_then_policy_then_strategy(self) -> None:
        allocation = make_allocation(initial_budget=128)
        result, _ = run_direct(allocation)
        decisions = result.report.decisions
        assert decisions[0] == allocation.decision.audit_line()
        assert decisions[1] == allocation.reason
        assert decisions[2].startswith("direct: one call, no reasoning; output cap 128")

    def test_an_offered_escalation_is_recorded_as_unused(self) -> None:
        result, _ = run_direct(
            make_allocation(initial_budget=128, additional_budget=256)
        )
        decisions = result.report.decisions
        assert len(decisions) == 4
        assert "escalation was offered" in decisions[3]

    def test_no_escalation_note_when_none_was_offered(self) -> None:
        result, _ = run_direct()
        assert len(result.report.decisions) == 3

    def test_the_attempt_is_made_even_when_the_policy_says_stop(self) -> None:
        allocation = make_allocation(
            0, should_stop=True, stop_reason=StopReason.SUFFICIENCY
        )
        result, client = run_direct(allocation)
        assert client.call_count == 1
        assert result.report.stopped_on is StopReason.STRATEGY_COMPLETE

    def test_the_probed_capabilities_are_consulted_before_the_call(self) -> None:
        report = RewritingReport(client_name="c", model="m", evidence=())
        _, client = run_direct(make_allocation(initial_budget=256), capabilities=report)
        assert client.last_options.max_output_tokens == 1

    def test_without_a_report_the_options_are_sent_as_built(self) -> None:
        _, client = run_direct(make_allocation(initial_budget=256))
        assert client.last_options.max_output_tokens == 256

    def test_client_failures_propagate(self) -> None:
        def boom(request: Request, options: CompletionOptions) -> Response:
            raise RuntimeError("provider down")

        with pytest.raises(RuntimeError, match="provider down"):
            run_direct(client=FakeClient(handler=boom))

    def test_it_holds_no_state_between_calls(self) -> None:
        strategy = DirectStrategy()
        assert vars(strategy) == {}
        run(strategy.execute(make_request(), make_allocation(), FakeClient()))
        assert vars(strategy) == {}
        assert not hasattr(strategy, "client")


class TestSelectStrategy:
    """Selection is by name, and every selection is a fresh instance."""

    def test_it_returns_a_direct_strategy(self) -> None:
        chosen = select_strategy("direct")
        assert isinstance(chosen, DirectStrategy)
        assert chosen.name == "direct"

    def test_each_call_returns_a_fresh_instance(self) -> None:
        assert select_strategy("direct") is not select_strategy("direct")

    def test_an_unknown_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy 'reason'"):
            select_strategy("reason")

    def test_the_error_lists_the_valid_strategies(self) -> None:
        with pytest.raises(ValueError, match="Valid strategies: direct"):
            select_strategy("nope")


class TestTheFixedDouble:
    """The shipped double that lets a user replace a strategy without writing one."""

    def test_with_no_result_it_admits_no_strategy_ran(self) -> None:
        strategy = FixedStrategy()
        result, _ = _run_strategy(
            strategy, None, request=None, client=None, capabilities=None
        )
        assert result.response.text == ""
        assert result.report.strategy == "fixed-strategy"
        assert result.report.usage is None
        assert result.report.usage_basis is UsageBasis.ABSENT
        assert result.report.stopped_on is StopReason.STRATEGY_COMPLETE
        assert result.report.decisions == (
            "fixed test double; no strategy was executed",
        )

    def test_a_pinned_result_is_returned_verbatim(self) -> None:
        pinned = StrategyResult(
            response=Response(text="pinned"),
            report=OptimizationReport(strategy="pinned"),
        )
        strategy = FixedStrategy(pinned)
        result, _ = _run_strategy(
            strategy, None, request=None, client=None, capabilities=None
        )
        assert result is pinned

    def test_it_records_the_request_and_allocation_it_was_given(self) -> None:
        strategy = FixedStrategy()
        request = make_request(text="recorded")
        allocation = make_allocation(initial_budget=64)
        run(strategy.execute(request, allocation, FakeClient()))
        assert strategy.call_count == 1
        assert strategy.calls == ((request, allocation),)

    def test_reset_forgets_the_calls_but_keeps_the_result(self) -> None:
        pinned = StrategyResult(
            response=Response(text="pinned"),
            report=OptimizationReport(strategy="pinned"),
        )
        strategy = FixedStrategy(pinned)
        run(strategy.execute(make_request(), make_allocation(), FakeClient()))
        strategy.reset()
        assert strategy.call_count == 0
        assert strategy.calls == ()
        assert strategy.result is pinned

    def test_its_name_labels_the_default_report(self) -> None:
        strategy = FixedStrategy(name="spy")
        result, _ = _run_strategy(
            strategy, None, request=None, client=None, capabilities=None
        )
        assert strategy.name == "spy"
        assert result.report.strategy == "spy"

    def test_its_determinism_is_configurable(self) -> None:
        assert FixedStrategy().deterministic is True
        assert FixedStrategy(deterministic=False).deterministic is False

    def test_it_ignores_the_capabilities_argument(self) -> None:
        strategy = FixedStrategy()
        report = CapabilityReport(client_name="c", model="m", evidence=())
        run(strategy.execute(make_request(), make_allocation(), FakeClient(), report))
        assert strategy.call_count == 1

    def test_its_repr_names_it_without_leaking_the_request(self) -> None:
        strategy = FixedStrategy(name="spy")
        request = make_request(text="a secret prompt")
        run(strategy.execute(request, make_allocation(), FakeClient()))
        text = repr(strategy)
        assert "FixedStrategy" in text
        assert "spy" in text
        assert "calls=1" in text
        assert "secret" not in text


class TestItCostsNothing:
    """The strategy module reaches for nothing outside the process."""

    def test_the_module_reaches_for_no_network_stack(self) -> None:
        module_file = inferconomy.strategy.__file__
        assert module_file is not None
        source = Path(module_file).read_text(encoding="utf-8")
        roots = {
            name.split(".")[0]
            for name in re.findall(
                r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", source, re.M
            )
        }
        assert not roots & {"httpx", "urllib", "socket", "requests", "http", "ssl"}

    def test_the_module_does_not_import_provider_adapters(self) -> None:
        module_file = inferconomy.strategy.__file__
        assert module_file is not None
        source = Path(module_file).read_text(encoding="utf-8")
        assert "inferconomy.providers" not in source


class TestDeterminism:
    """Replays must be comparable, or no report above this layer can be trusted."""

    def test_a_fresh_instance_reports_itself_deterministic(self) -> None:
        assert DirectStrategy().deterministic is True

    def test_identical_inputs_produce_identical_results(self) -> None:
        request = make_request(text="same")
        allocation = make_allocation(initial_budget=128)
        first, _ = run_direct(
            allocation, request=request, client=FakeClient([Response(text="answer")])
        )
        second, _ = run_direct(
            allocation, request=request, client=FakeClient([Response(text="answer")])
        )
        assert first == second


TABLE = PriceTable(
    version="test-us014",
    models={
        "fake-model": ModelPrice(
            input_per_million=3.0,
            output_per_million=15.0,
            verified=True,
        )
    },
)

PROMPT = "What is the capital of France?"
ANSWER = "The capital of France is Paris."


def verbose_response(request: Request, options: CompletionOptions) -> Response:
    """A fake provider that writes as much as it is allowed to.

    Deliberate: the baseline is capped high and the direct strategy is capped
    low, so a run that respects the cap costs less. A terse fake would make the
    saving emerge from the text rather than from the budget, which is not the
    claim under test.
    """
    budget = options.max_output_tokens or 2048
    text = ((ANSWER + " ") * (budget + 1))[: budget * 4]
    return Response(
        text=text,
        usage=Usage(
            input_tokens=10,
            output_tokens=budget,
            llm_calls=1,
            tokens_exact=True,
        ),
        model="fake-model",
    )


class TestTheDoneWhen:
    """US-014: ``direct`` answers a trivial request well within the baseline cost.

    A mechanism test with fixed fakes, not a published frontier: no real
    provider, no credentials, and no real judge take part. The publish gate in
    :mod:`inferconomy.frontier` exists because a real saving has to be earned.
    """

    def test_direct_is_cheaper_than_the_baseline_at_equal_quality(self) -> None:
        task = Task(task_id="t1", prompt=PROMPT, category="trivia")
        baseline_texts: list[str] = []

        def baseline_handler(request: Request, options: CompletionOptions) -> Response:
            response = verbose_response(request, options)
            baseline_texts.append(response.text)
            return response

        baseline = run(
            run_baseline(
                BaselineConfig(tasks=(task,), budget=2048, model="fake-model"),
                FakeClient(handler=baseline_handler),
                table=TABLE,
            )
        )
        baseline_cost = baseline.rows[0].cost_usd
        assert baseline_cost is not None

        request = task.to_request()
        decision = run(HeuristicDecisionEngine().decide(request))
        allocation = run(ConservativeReferencePolicy().allocate(decision, request))
        # The budget the whole comparison rests on: the engine declines to
        # classify this prompt, and the policy funds it modestly rather than
        # assuming reasoning is wasted.
        assert decision.shape is TaskShape.UNKNOWN
        assert allocation.initial_budget == 128

        client = FakeClient(handler=verbose_response)
        result = run(DirectStrategy().execute(request, allocation, client))
        assert client.call_count == 1
        assert ANSWER in result.response.text
        assert result.report.usage is result.response.usage
        assert result.report.capabilities_used == (Capability.USAGE_REPORTING,)

        direct_cost = TABLE.price_usage(result.response.usage, "fake-model").cost_usd
        assert direct_cost is not None
        assert direct_cost * 4 < baseline_cost

        judge = FunctionJudge(lambda judged: 1.0 if "Paris" in judged.response else 0.0)
        baseline_quality = run(
            judge.score(
                JudgeRequest(
                    task_id="t1",
                    category="trivia",
                    prompt=PROMPT,
                    response=baseline_texts[0],
                )
            )
        )
        direct_quality = run(
            judge.score(
                JudgeRequest(
                    task_id="t1",
                    category="trivia",
                    prompt=PROMPT,
                    response=result.response.text,
                )
            )
        )
        assert baseline_quality.score == 1.0
        assert direct_quality.score == 1.0
