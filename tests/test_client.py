"""Client protocol and fake-client tests.

Two things are being verified here. First, that a fake satisfies the protocol a
provider adapter will have to satisfy. Second, and more important, that the
protocol is narrow and behaviourally specific enough that a fake can implement
it *honestly* — if the fake needs special-casing, the protocol is wrong.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from inferconomy import Capability
from inferconomy.client import Client, CompletionOptions
from inferconomy.contracts import Request, Response, Usage
from inferconomy.testing import (
    FakeClient,
    echo_response,
    estimate_tokens,
    scripted,
    user_request,
)


def run(coro: Any) -> Any:
    """Drive one coroutine to completion.

    Using asyncio.run in synchronous tests keeps the async public API without
    adding a pytest plugin and its configuration to the dev dependencies.
    """
    return asyncio.run(coro)


class TestProtocolConformance:
    def test_fake_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeClient(), Client)

    def test_protocol_is_narrow(self) -> None:
        """A protocol that grows to fit a provider stops being honest."""
        public = {name for name in vars(Client) if not name.startswith("_")}
        assert public == {"capabilities", "complete"}

    def test_an_incomplete_adapter_does_not_satisfy_the_protocol(self) -> None:
        class NotAClient:
            pass

        assert not isinstance(NotAClient(), Client)


class TestCompletionOptions:
    def test_defaults_are_unset(self) -> None:
        options = CompletionOptions()
        assert options.model is None
        assert options.max_output_tokens is None
        assert options.reasoning_budget is None
        assert options.effort is None
        assert options.stop == ()

    def test_rejects_non_positive_max_output_tokens(self) -> None:
        with pytest.raises(ValueError, match="must be positive when set"):
            CompletionOptions(max_output_tokens=0)

    @pytest.mark.parametrize("value", [0, -1], ids=["zero", "negative"])
    def test_rejects_non_positive_reasoning_budget(self, value: int) -> None:
        with pytest.raises(ValueError, match="must be positive when set"):
            CompletionOptions(reasoning_budget=value)

    @pytest.mark.parametrize("effort", ["", "   "], ids=["empty", "blank"])
    def test_rejects_blank_effort(self, effort: str) -> None:
        with pytest.raises(ValueError, match="must not be blank"):
            CompletionOptions(effort=effort)

    def test_normalizes_stop_to_a_tuple(self) -> None:
        assert isinstance(CompletionOptions(stop=["a", "b"]).stop, tuple)  # type: ignore[arg-type]


class TestFakeClientScripting:
    def test_returns_scripted_responses_in_order(self) -> None:
        client = scripted("first", "second")
        assert (
            run(client.complete(user_request("a"), CompletionOptions())).text == "first"
        )
        assert (
            run(client.complete(user_request("a"), CompletionOptions())).text
            == "second"
        )

    def test_records_every_call(self) -> None:
        client = scripted("a", "b")
        run(client.complete(user_request("one"), CompletionOptions(model="m1")))
        run(client.complete(user_request("two"), CompletionOptions(model="m2")))
        assert client.call_count == 2
        assert [request.messages[-1].content for request, _ in client.calls] == [
            "one",
            "two",
        ]
        assert [options.model for _, options in client.calls] == ["m1", "m2"]

    def test_exhausted_script_raises_rather_than_repeating(self) -> None:
        """Silently repeating would hide a test that called more than it intended."""
        client = scripted("only")
        run(client.complete(user_request("a"), CompletionOptions()))
        with pytest.raises(AssertionError, match="script exhausted after 1 call"):
            run(client.complete(user_request("a"), CompletionOptions()))

    def test_repeat_last_is_opt_in(self) -> None:
        client = scripted("only", repeat_last=True)
        run(client.complete(user_request("a"), CompletionOptions()))
        assert (
            run(client.complete(user_request("a"), CompletionOptions())).text == "only"
        )

    def test_handler_takes_precedence_and_may_be_unbounded(self) -> None:
        client = FakeClient(
            handler=lambda request, options: Response(text=f"n={len(request.messages)}")
        )
        for _ in range(5):
            assert (
                run(client.complete(user_request("a"), CompletionOptions())).text
                == "n=1"
            )

    def test_rejects_handler_and_responses_together(self) -> None:
        with pytest.raises(ValueError, match="either handler or responses"):
            FakeClient([Response(text="x")], handler=lambda r, o: Response(text="y"))

    def test_reset_rewinds_the_script(self) -> None:
        client = scripted("first", "second")
        run(client.complete(user_request("a"), CompletionOptions()))
        client.reset()
        assert client.call_count == 0
        assert (
            run(client.complete(user_request("a"), CompletionOptions())).text == "first"
        )

    def test_last_options_raises_before_any_call(self) -> None:
        with pytest.raises(AssertionError, match="No calls have been made yet"):
            _ = FakeClient().last_options

    def test_last_options_returns_the_most_recent_call(self) -> None:
        client = scripted("a", "b")
        run(client.complete(user_request("x"), CompletionOptions(model="first")))
        run(client.complete(user_request("x"), CompletionOptions(model="second")))
        assert client.last_options.model == "second"

    def test_repr_is_informative(self) -> None:
        assert "calls=0" in repr(FakeClient())


class TestFakeClientDefaultBehaviour:
    def test_echoes_the_final_user_turn(self) -> None:
        client = FakeClient()
        response = run(client.complete(user_request("hello"), CompletionOptions()))
        assert response.text == "echo: hello"

    def test_echoes_the_user_turn_not_the_last_turn(self) -> None:
        request = Request.from_text("the question", system="be terse")
        response = run(FakeClient().complete(request, CompletionOptions()))
        assert response.text == "echo: the question"

    def test_honours_the_output_budget(self) -> None:
        """A fake that ignored the budget could not test that budgets land."""
        client = FakeClient()
        response = run(
            client.complete(
                user_request("a fairly long prompt indeed"),
                CompletionOptions(max_output_tokens=4),
            )
        )
        assert len(response.text) == 16

    def test_reports_a_call_in_usage(self) -> None:
        response = run(FakeClient().complete(user_request("hi"), CompletionOptions()))
        assert response.usage.llm_calls == 1

    def test_does_not_claim_exact_tokens(self) -> None:
        response = run(FakeClient().complete(user_request("hi"), CompletionOptions()))
        assert response.usage.cost_exact is False

    def test_is_deterministic(self) -> None:
        request, options = user_request("stable"), CompletionOptions()
        first = run(FakeClient().complete(request, options))
        second = run(FakeClient().complete(request, options))
        assert first == second

    def test_applies_configured_latency(self) -> None:
        client = FakeClient(latency_ms=25.0)
        assert (
            run(client.complete(user_request("x"), CompletionOptions())).latency_ms
            == 25.0
        )

    def test_does_not_overwrite_a_response_latency(self) -> None:
        client = FakeClient([Response(text="x", latency_ms=7.0)], latency_ms=25.0)
        assert (
            run(client.complete(user_request("x"), CompletionOptions())).latency_ms
            == 7.0
        )


class TestCapabilities:
    def test_declares_nothing_by_default(self) -> None:
        """Default to the degraded path so tests exercise it rather than bypass it."""
        assert FakeClient().capabilities == frozenset()

    def test_declares_configured_capabilities(self) -> None:
        client = FakeClient(capabilities=frozenset({Capability.USAGE_REPORTING}))
        assert Capability.USAGE_REPORTING in client.capabilities

    def test_capabilities_are_immutable(self) -> None:
        client = FakeClient()
        assert isinstance(client.capabilities, frozenset)


class TestTokenEstimation:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("", 0), ("abcd", 1), ("abcde", 2)],
        ids=["empty", "exact-multiple", "rounds-up"],
    )
    def test_estimates_by_character_count(self, text: str, expected: int) -> None:
        assert estimate_tokens(text) == expected

    def test_echo_response_is_pure(self) -> None:
        """Determinism is the property the whole fake rests on."""
        request, options = user_request("abcde"), CompletionOptions()
        assert echo_response(request, options) == echo_response(request, options)


class TestNoNetwork:
    """The US-003 acceptance criterion, as executable assertions."""

    def test_the_guard_blocks_external_hosts(self) -> None:
        """Prove the conftest guard is live, or it guards nothing."""
        with pytest.raises(RuntimeError, match="External network access attempted"):
            socket.create_connection(("example.invalid", 80))

    def test_the_guard_permits_loopback(self) -> None:
        """A guard that also breaks loopback would break asyncio, not the tests."""
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            client = socket.create_connection(("127.0.0.1", port), timeout=5)
            client.close()
        finally:
            server.close()

    def test_suite_runs_offline(self) -> None:
        client = scripted("offline works")
        response = run(client.complete(user_request("hi"), CompletionOptions()))
        assert response.usage == Usage(
            output_tokens=estimate_tokens("offline works"), llm_calls=1
        )
