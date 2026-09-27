"""Tests for the OpenAI-compatible provider adapter.

The suite runs offline by construction: every test drives an injected
:class:`RecordingTransport` or a mocked ``httpx`` client, so normalization is
proven against recorded payloads with no socket involved. That is a property of
the design, not a limitation of the tests - the transport is an interface precisely
so this is possible.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from typing import Any

import pytest

from inferconomy.capabilities import ProbeOutcome, probe_capabilities
from inferconomy.client import CompletionOptions
from inferconomy.contracts import Capability, Request, Response, UsageBasis
from inferconomy.providers import (
    AuthenticationError,
    ContextLengthExceeded,
    HttpResponse,
    HttpxTransport,
    ModelNotFoundError,
    OpenAICompatibleClient,
    ProviderError,
    RateLimitError,
    RequestRejectedError,
    ServerError,
    Transport,
    TransportError,
    httpx_available,
    normalize_base_url,
    require_httpx,
)

CHAT = "/chat/completions"


def run(coro: Any) -> Any:
    """Drive one coroutine to completion.

    asyncio.run in a synchronous test keeps the async API under test without
    adding a pytest plugin and its configuration to the dev dependencies.
    """
    return asyncio.run(coro)


class RecordingTransport:
    """Records what was sent and replies with whatever the test scripted."""

    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float = 60.0,
    ) -> HttpResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout": timeout,
            }
        )
        if not self.responses:
            return HttpResponse(status=200, body="{}")
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)

    @property
    def payload(self) -> dict[str, Any]:
        sent: dict[str, Any] = self.calls[0]["payload"]
        return sent

    @property
    def last_payload(self) -> dict[str, Any]:
        """The most recent payload, for tests that probe before calling."""
        sent: dict[str, Any] = self.calls[-1]["payload"]
        return sent

    @property
    def headers(self) -> dict[str, str]:
        sent: dict[str, str] = self.calls[0]["headers"]
        return sent


def reply(body: Mapping[str, Any], status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, body=json.dumps(body))


def completion(
    content: str | None = "answer",
    *,
    finish: str = "stop",
    usage: Mapping[str, Any] | None = None,
    reasoning: str | None = None,
    model: str = "served-model",
) -> HttpResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    body: dict[str, Any] = {
        "id": "x",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return reply(body)


def client(transport: Transport, **kwargs: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://api.openai.com/v1", transport=transport, **kwargs
    )


REQUEST = Request.from_text("What is 17 * 23?")


class TestUrlNormalization:
    """All eight named providers work from the base URL people actually type."""

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("https://api.openai.com/v1", f"https://api.openai.com/v1{CHAT}"),
            ("https://api.openai.com", f"https://api.openai.com/v1{CHAT}"),
            ("https://api.openai.com/v1/", f"https://api.openai.com/v1{CHAT}"),
            (
                "https://api.openai.com/v1/chat/completions",
                f"https://api.openai.com/v1{CHAT}",
            ),
            (
                "https://openrouter.ai/api/v1",
                f"https://openrouter.ai/api/v1{CHAT}",
            ),
            ("https://api.groq.com/openai/v1", f"https://api.groq.com/openai/v1{CHAT}"),
            ("https://api.deepseek.com", f"https://api.deepseek.com/v1{CHAT}"),
            ("https://api.mistral.ai/v1", f"https://api.mistral.ai/v1{CHAT}"),
            ("http://localhost:8000/v1", f"http://localhost:8000/v1{CHAT}"),
            ("http://localhost:11434", f"http://localhost:11434/v1{CHAT}"),
        ],
    )
    def test_normalizes_to_a_chat_completions_endpoint(
        self, given: str, expected: str
    ) -> None:
        assert normalize_base_url(given) == expected

    def test_a_local_server_without_a_version_segment_gets_one(self) -> None:
        """Ollama's OpenAI-compatible route lives under /v1, and a bare host
        pointed at /chat/completions is a 404 that looks like a broken server."""
        assert normalize_base_url("http://localhost:11434").endswith(f"/v1{CHAT}")

    def test_a_full_path_is_not_doubled(self) -> None:
        once = normalize_base_url("https://x.dev/v1/chat/completions")
        assert normalize_base_url(once) == once

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert normalize_base_url("  https://x.dev/v1  ").endswith(CHAT)

    def test_a_blank_base_url_is_refused(self) -> None:
        with pytest.raises(ValueError, match="blank"):
            normalize_base_url("   ")

    def test_a_relative_base_url_is_refused_with_guidance(self) -> None:
        with pytest.raises(ValueError, match="localhost:8000"):
            normalize_base_url("api.openai.com/v1")

    def test_a_query_string_does_not_leak_into_the_path(self) -> None:
        assert "?" not in normalize_base_url("https://x.dev/v1?key=abc")

    def test_the_port_survives(self) -> None:
        assert ":8000" in normalize_base_url("http://localhost:8000/v1")

    def test_the_client_stores_the_normalized_endpoint(self) -> None:
        assert client(RecordingTransport()).endpoint.endswith(CHAT)


class TestRequestPayload:
    """Only what Infereconomy's own types can express reaches the wire."""

    def test_sends_the_model_from_options(self) -> None:
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions(model="m1")))
        assert transport.payload["model"] == "m1"

    def test_falls_back_to_the_configured_default_model(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport, default_model="default").complete(
                REQUEST, CompletionOptions()
            )
        )
        assert transport.payload["model"] == "default"

    def test_options_win_over_the_default_model(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport, default_model="default").complete(
                REQUEST, CompletionOptions(model="chosen")
            )
        )
        assert transport.payload["model"] == "chosen"

    def test_omits_the_model_entirely_when_none_is_known(self) -> None:
        """An explicit null is not the same as an absent field, and endpoints
        disagree about which they accept for "choose the default"."""
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions()))
        assert "model" not in transport.payload

    def test_sends_every_turn_in_order(self) -> None:
        transport = RecordingTransport(completion())
        request = Request.from_text("second", system="first")
        run(client(transport).complete(request, CompletionOptions(model="m")))
        assert transport.payload["messages"] == [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "second"},
        ]

    def test_maps_the_output_limit_to_max_tokens_by_default(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport).complete(
                REQUEST, CompletionOptions(model="m", max_output_tokens=256)
            )
        )
        assert transport.payload["max_tokens"] == 256

    def test_can_be_told_to_use_the_newer_spelling(self) -> None:
        """OpenAI's reasoning models reject max_tokens outright."""
        transport = RecordingTransport(completion())
        run(
            client(transport, max_tokens_field="max_completion_tokens").complete(
                REQUEST, CompletionOptions(model="m", max_output_tokens=256)
            )
        )
        assert transport.payload["max_completion_tokens"] == 256
        assert "max_tokens" not in transport.payload

    def test_omits_the_output_limit_when_unset(self) -> None:
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert "max_tokens" not in transport.payload

    def test_sends_stop_sequences_as_a_list(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport).complete(
                REQUEST, CompletionOptions(model="m", stop=("A", "B"))
            )
        )
        assert transport.payload["stop"] == ["A", "B"]

    def test_omits_stop_when_there_are_none(self) -> None:
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert "stop" not in transport.payload

    def test_sends_effort_under_the_openai_field_name(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport).complete(
                REQUEST, CompletionOptions(model="m", effort="high")
            )
        )
        assert transport.payload["reasoning_effort"] == "high"

    def test_omits_effort_when_the_endpoint_has_no_such_field(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport, effort_field=None).complete(
                REQUEST, CompletionOptions(model="m", effort="high")
            )
        )
        assert "reasoning_effort" not in transport.payload

    def test_omits_effort_when_unset(self) -> None:
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert "reasoning_effort" not in transport.payload

    def test_does_not_guess_a_reasoning_budget_field(self) -> None:
        """The format has no standard one, so inventing a name would either be
        ignored (silent pretence) or draw a 400 on every call."""
        transport = RecordingTransport(completion())
        run(
            client(transport).complete(
                REQUEST, CompletionOptions(model="m", reasoning_budget=4096)
            )
        )
        assert not any("reasoning" in key for key in transport.payload)

    def test_sends_a_reasoning_budget_when_a_dialect_is_named(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport, reasoning_budget_field="reasoning_tokens").complete(
                REQUEST, CompletionOptions(model="m", reasoning_budget=4096)
            )
        )
        assert transport.payload["reasoning_tokens"] == 4096

    def test_sends_nothing_the_caller_did_not_ask_for(self) -> None:
        """No temperature, no top_p, no defaults invented on the caller's behalf."""
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert set(transport.payload) == {"model", "messages"}


class TestHeaders:
    def test_sends_the_api_key_as_a_bearer_token(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport, api_key="sk-test").complete(
                REQUEST, CompletionOptions(model="m")
            )
        )
        assert transport.headers["authorization"] == "Bearer sk-test"

    def test_sends_no_authorization_header_without_a_key(self) -> None:
        """vLLM and Ollama usually run keyless, and an empty bearer header makes
        some servers reject an otherwise valid request."""
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert "authorization" not in transport.headers

    def test_merges_extra_headers(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport, extra_headers={"x-title": "inferconomy"}).complete(
                REQUEST, CompletionOptions(model="m")
            )
        )
        assert transport.headers["x-title"] == "inferconomy"

    def test_extra_headers_cannot_override_the_content_type(self) -> None:
        """The body is always JSON, and a wrong content type produces a parse
        error at the endpoint that reads like a provider fault."""
        transport = RecordingTransport(completion())
        run(
            client(transport, extra_headers={"content-type": "text/plain"}).complete(
                REQUEST, CompletionOptions(model="m")
            )
        )
        assert transport.headers["content-type"] == "application/json"

    def test_extra_headers_may_override_accept(self) -> None:
        """A gateway that wants something else is a real case, and there is no
        such thing as an invalid accept header."""
        transport = RecordingTransport(completion())
        run(
            client(
                transport, extra_headers={"accept": "application/x-ndjson"}
            ).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert transport.headers["accept"] == "application/x-ndjson"

    def test_extra_headers_cannot_replace_the_bearer_token(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(
                transport,
                api_key="real",
                extra_headers={"authorization": "Bearer other"},
            ).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert transport.headers["authorization"] == "Bearer real"

    def test_asks_for_json(self) -> None:
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert transport.headers["accept"] == "application/json"

    def test_a_blank_api_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            client(RecordingTransport(), api_key="   ")

    def test_a_nonpositive_timeout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            client(RecordingTransport(), timeout=0)


class TestResponseNormalization:
    def test_returns_the_text(self) -> None:
        transport = RecordingTransport(completion("the answer is 391"))
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.text == "the answer is 391"

    def test_a_null_content_is_an_empty_answer_not_a_crash(self) -> None:
        """Reasoning models emit null content when they return only a trace."""
        transport = RecordingTransport(completion(None, reasoning="thinking"))
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.text == ""

    def test_a_missing_content_is_an_empty_answer(self) -> None:
        body = {
            "choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]
        }
        transport = RecordingTransport(reply(body))
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.text == ""

    def test_reads_openrouter_style_reasoning(self) -> None:
        transport = RecordingTransport(completion(reasoning="step by step"))
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.reasoning_text == "step by step"

    def test_reads_deepseek_style_reasoning_content(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "391",
                        "reasoning_content": "17 times 23",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        transport = RecordingTransport(reply(body))
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.reasoning_text == "17 times 23"

    def test_both_reasoning_spellings_land_in_the_same_field(self) -> None:
        """The dialect is absorbed here, so no caller ever sees which it was."""
        for key in ("reasoning", "reasoning_content"):
            body = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "x", key: "trace"},
                        "finish_reason": "stop",
                    }
                ]
            }
            response = run(
                client(RecordingTransport(reply(body))).complete(
                    REQUEST, CompletionOptions(model="m")
                )
            )
            assert response.reasoning_text == "trace"

    def test_no_trace_reads_as_absent_not_empty(self) -> None:
        transport = RecordingTransport(completion())
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.reasoning_text is None

    def test_preserves_the_provider_finish_reason_verbatim(self) -> None:
        """Normalizing it would discard the only ground truth about what the
        provider actually did."""
        transport = RecordingTransport(completion(finish="length"))
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.provider_finish_reason == "length"

    def test_keeps_an_unfamiliar_finish_reason_rather_than_guessing(self) -> None:
        transport = RecordingTransport(completion(finish="content_filter"))
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.provider_finish_reason == "content_filter"

    def test_no_finish_reason_reads_as_absent(self) -> None:
        body = {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        response = run(
            client(RecordingTransport(reply(body))).complete(
                REQUEST, CompletionOptions(model="m")
            )
        )
        assert response.provider_finish_reason is None

    def test_reads_back_the_model_the_endpoint_served(self) -> None:
        transport = RecordingTransport(completion(model="gpt-4o-2024-11-20"))
        response = run(
            client(transport, default_model="requested").complete(
                REQUEST, CompletionOptions(model="requested")
            )
        )
        assert response.model == "gpt-4o-2024-11-20"

    def test_falls_back_to_the_requested_model_when_the_endpoint_omits_it(self) -> None:
        body = {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        response = run(
            client(RecordingTransport(reply(body))).complete(
                REQUEST, CompletionOptions(model="m1")
            )
        )
        assert response.model == "m1"

    def test_measures_a_latency(self) -> None:
        transport = RecordingTransport(completion())
        response = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        )
        assert response.latency_ms > 0

    def test_takes_the_first_choice(self) -> None:
        body = {
            "choices": [
                {"message": {"role": "assistant", "content": "first"}},
                {"message": {"role": "assistant", "content": "second"}},
            ]
        }
        response = run(
            client(RecordingTransport(reply(body))).complete(
                REQUEST, CompletionOptions(model="m")
            )
        )
        assert response.text == "first"

    def test_a_success_with_no_choices_is_a_failure(self) -> None:
        transport = RecordingTransport(reply({"choices": []}))
        with pytest.raises(ProviderError, match="no choices"):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_a_choice_without_a_message_is_a_failure(self) -> None:
        transport = RecordingTransport(reply({"choices": [{"finish_reason": "stop"}]}))
        with pytest.raises(ProviderError, match="without a message"):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_a_non_json_success_is_a_failure_mentioning_a_proxy(self) -> None:
        transport = RecordingTransport(HttpResponse(status=200, body="<html>hi</html>"))
        with pytest.raises(ProviderError, match="not JSON"):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_json_that_is_not_an_object_is_a_failure(self) -> None:
        transport = RecordingTransport(HttpResponse(status=200, body="[1, 2]"))
        with pytest.raises(ProviderError, match="not an object"):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_returns_a_response_object_so_callers_see_one_type(self) -> None:
        transport = RecordingTransport(completion())
        assert isinstance(
            run(client(transport).complete(REQUEST, CompletionOptions(model="m"))),
            Response,
        )


class TestUsage:
    """The provider's counts are used as given; its silence becomes an estimate."""

    def test_uses_the_reported_counts_as_measured(self) -> None:
        transport = RecordingTransport(
            completion(usage={"prompt_tokens": 12, "completion_tokens": 3})
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert (usage.input_tokens, usage.output_tokens) == (12, 3)
        assert usage.tokens_exact is True

    def test_reads_reasoning_tokens_from_the_details_block(self) -> None:
        transport = RecordingTransport(
            completion(
                usage={
                    "prompt_tokens": 12,
                    "completion_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 24},
                }
            )
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.reasoning_tokens == 24

    def test_keeps_reasoning_a_subset_of_output(self) -> None:
        transport = RecordingTransport(
            completion(
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 99},
                }
            )
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.reasoning_tokens == usage.output_tokens

    def test_survives_a_usage_block_with_no_details(self) -> None:
        transport = RecordingTransport(
            completion(usage={"prompt_tokens": 5, "completion_tokens": 5})
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.reasoning_tokens == 0

    def test_counts_the_call(self) -> None:
        transport = RecordingTransport(completion(usage={"prompt_tokens": 1}))
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.llm_calls == 1

    def test_never_invents_a_cost(self) -> None:
        """Cost needs a verified rate; the endpoint sent tokens, not money."""
        transport = RecordingTransport(completion(usage={"prompt_tokens": 12}))
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.cost_usd is None
        assert usage.cost_exact is False

    def test_estimates_when_the_endpoint_reports_nothing(self) -> None:
        transport = RecordingTransport(completion())
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.tokens_exact is False
        assert usage.input_tokens > 0

    def test_an_estimate_is_marked_estimated_even_with_a_reported_block(self) -> None:
        transport = RecordingTransport(
            HttpResponse(
                status=200, body='{"choices": [{"message": {"content": "x"}}]}'
            )
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.tokens_exact is False

    def test_estimates_the_reasoning_trace_too(self) -> None:
        transport = RecordingTransport(completion(reasoning="a fairly long trace here"))
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.reasoning_tokens > 0
        assert usage.tokens_exact is False

    def test_a_float_token_count_is_coerced(self) -> None:
        transport = RecordingTransport(
            completion(usage={"prompt_tokens": 12.0, "completion_tokens": 3.9})
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert (usage.input_tokens, usage.output_tokens) == (12, 3)

    def test_a_nonsense_token_count_does_not_become_a_number(self) -> None:
        transport = RecordingTransport(
            completion(usage={"prompt_tokens": "many", "completion_tokens": None})
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert (usage.input_tokens, usage.output_tokens) == (0, 0)

    def test_a_boolean_is_not_a_token_count(self) -> None:
        transport = RecordingTransport(
            completion(usage={"prompt_tokens": True, "completion_tokens": 2})
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.input_tokens == 0

    def test_cached_prompt_tokens_are_counted_as_input(self) -> None:
        """Endpoints fold cache hits into prompt_tokens, and Usage has nowhere to
        put them. Counting them as input matches the provider's own arithmetic,
        which means a flat price table over-states the cost rather than losing it.
        """
        transport = RecordingTransport(
            completion(
                usage={
                    "prompt_tokens": 1000,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 900},
                }
            )
        )
        usage = run(
            client(transport).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        assert usage.input_tokens == 1000

    def test_a_price_table_can_tell_a_measurement_from_a_guess(self) -> None:
        """The whole point of tokens_exact: two calls, one reported and one
        estimated, must land on different sides of the provenance boundary."""
        reported = run(
            client(
                RecordingTransport(
                    completion(usage={"prompt_tokens": 5, "completion_tokens": 2})
                )
            ).complete(REQUEST, CompletionOptions(model="m"))
        ).usage
        estimated = run(
            client(RecordingTransport(completion())).complete(
                REQUEST, CompletionOptions(model="m")
            )
        ).usage
        assert reported.basis is not estimated.basis
        assert estimated.basis is UsageBasis.ESTIMATED
        assert reported.tokens_exact is True


class TestErrorClassification:
    """A policy needs to know whether to retry, and what to change."""

    def test_a_rejected_key_is_not_retryable(self) -> None:
        transport = RecordingTransport(
            reply({"error": {"message": "Incorrect API key"}}, status=401)
        )
        with pytest.raises(AuthenticationError) as info:
            run(
                client(transport, api_key="bad").complete(
                    REQUEST, CompletionOptions(model="m")
                )
            )
        assert info.value.retryable is False
        assert info.value.status == 401

    def test_a_forbidden_key_is_not_retryable(self) -> None:
        transport = RecordingTransport(reply({"error": {"message": "no"}}, status=403))
        with pytest.raises(AuthenticationError):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_a_rate_limit_is_retryable(self) -> None:
        transport = RecordingTransport(
            reply({"error": {"message": "slow down"}}, status=429)
        )
        with pytest.raises(RateLimitError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.retryable is True

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_a_server_failure_is_retryable(self, status: int) -> None:
        transport = RecordingTransport(
            reply({"error": {"message": "oops"}}, status=status)
        )
        with pytest.raises(ServerError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.retryable is True

    def test_a_missing_model_is_named_as_such(self) -> None:
        transport = RecordingTransport(
            reply({"error": {"message": "model not found"}}, status=404)
        )
        with pytest.raises(ModelNotFoundError):
            run(client(transport).complete(REQUEST, CompletionOptions(model="ghost")))

    def test_a_model_error_is_a_rejected_request(self) -> None:
        assert issubclass(ModelNotFoundError, RequestRejectedError)

    def test_a_plain_rejection_is_not_retryable(self) -> None:
        transport = RecordingTransport(
            reply({"error": {"message": "invalid request"}}, status=400)
        )
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.retryable is False

    @pytest.mark.parametrize(
        "message",
        [
            "This model's maximum context length is 8192 tokens",
            "context_length_exceeded",
            "prompt is too long",
            "reduce the length of the messages",
        ],
    )
    def test_a_context_window_overflow_is_its_own_error(self, message: str) -> None:
        """One of the few provider errors a policy can act on by shrinking."""
        transport = RecordingTransport(
            reply({"error": {"message": message}}, status=400)
        )
        with pytest.raises(ContextLengthExceeded):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_a_context_error_is_still_a_rejected_request(self) -> None:
        assert issubclass(ContextLengthExceeded, RequestRejectedError)

    def test_a_long_prompt_is_not_mistaken_for_a_context_error(self) -> None:
        transport = RecordingTransport(
            reply({"error": {"message": "invalid value for temperature"}}, status=400)
        )
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert not isinstance(info.value, ContextLengthExceeded)

    def test_keeps_the_providers_own_words(self) -> None:
        transport = RecordingTransport(
            reply({"error": {"message": "Rate limit reached for gpt-4o"}}, status=429)
        )
        with pytest.raises(RateLimitError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.provider_message == "Rate limit reached for gpt-4o"

    def test_reads_an_error_sent_as_a_bare_string(self) -> None:
        """Some endpoints put the message straight in `error` rather than nesting
        an object, so both spellings have to be read."""
        transport = RecordingTransport(
            HttpResponse(status=400, body='{"error": "context length exceeded"}')
        )
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.provider_message == "context length exceeded"

    def test_reads_a_top_level_message(self) -> None:
        transport = RecordingTransport(
            HttpResponse(status=400, body='{"message": "nope"}')
        )
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.provider_message == "nope"

    def test_falls_back_to_the_body_when_the_error_object_has_no_message(self) -> None:
        transport = RecordingTransport(
            HttpResponse(status=400, body='{"error": {"code": "bad_thing"}}')
        )
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.provider_message is not None
        assert "bad_thing" in info.value.provider_message

    def test_falls_back_when_the_error_object_has_an_empty_message(self) -> None:
        transport = RecordingTransport(
            HttpResponse(status=400, body='{"error": {"message": ""}}')
        )
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.provider_message is not None

    def test_a_json_body_with_no_error_anywhere_still_says_something(self) -> None:
        transport = RecordingTransport(HttpResponse(status=400, body='{"foo": 1}'))
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.provider_message is None
        assert "foo" in str(info.value)

    def test_copes_with_an_empty_error_body(self) -> None:
        transport = RecordingTransport(HttpResponse(status=400, body=""))
        with pytest.raises(RequestRejectedError, match="no error body"):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_copes_with_an_error_body_that_is_not_json(self) -> None:
        transport = RecordingTransport(
            HttpResponse(status=502, body="<html>bad gateway</html>")
        )
        with pytest.raises(ServerError, match="bad gateway"):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_copes_with_an_error_body_that_is_json_but_not_an_object(self) -> None:
        transport = RecordingTransport(HttpResponse(status=400, body='"just a string"'))
        with pytest.raises(RequestRejectedError, match="just a string"):
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))

    def test_does_not_copy_an_entire_prompt_into_the_exception(self) -> None:
        """Error text ends up in logs; a body echoed back verbatim is how a
        prompt ends up in somebody's terminal history."""
        secret = "S" * 5000
        transport = RecordingTransport(HttpResponse(status=400, body=secret))
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert len(str(info.value)) < 500
        assert secret not in str(info.value)

    def test_an_unclassified_failure_defaults_to_not_retryable(self) -> None:
        """Retrying an unclassifiable error forever is how a bug becomes an outage."""
        transport = RecordingTransport(HttpResponse(status=418, body="teapot"))
        with pytest.raises(RequestRejectedError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.retryable is False

    def test_a_network_failure_is_a_transport_error(self) -> None:
        class Broken(RecordingTransport):
            async def post_json(self, url: str, **kwargs: Any) -> HttpResponse:
                raise TransportError("connection reset")

        with pytest.raises(TransportError) as info:
            run(client(Broken()).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.retryable is True

    def test_every_error_carries_its_status_when_the_endpoint_gave_one(self) -> None:
        transport = RecordingTransport(reply({"error": {"message": "x"}}, status=503))
        with pytest.raises(ServerError) as info:
            run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert info.value.status == 503

    def test_an_error_reprs_without_blowing_up(self) -> None:
        assert "429" in repr(RateLimitError("x", status=429))

    def test_the_retryable_flag_is_readable_off_the_class(self) -> None:
        assert ProviderError.retryable is False
        assert TransportError.retryable is True


class TestProbeIntegration:
    """The adapter answers US-010's probe, honestly and without guessing."""

    def test_reasoning_budget_is_unknown_without_spending_a_call(self) -> None:
        transport = RecordingTransport()
        supported, detail = run(
            client(transport).probe_capability(Capability.REASONING_BUDGET, "m")
        )
        assert supported is None
        assert "no standard reasoning-budget field" in detail
        assert not transport.calls

    def test_a_named_dialect_makes_the_budget_probeable(self) -> None:
        transport = RecordingTransport(completion())
        supported, _ = run(
            client(
                transport, reasoning_budget_field="reasoning_tokens"
            ).probe_capability(Capability.REASONING_BUDGET, "m")
        )
        assert supported is None
        assert len(transport.calls) == 1
        assert transport.payload["reasoning_tokens"] == 16

    def test_an_endpoint_naming_the_field_in_its_refusal_refutes_it(self) -> None:
        transport = RecordingTransport(
            reply(
                {"error": {"message": "reasoning_effort is not supported"}}, status=400
            )
        )
        supported, detail = run(
            client(transport).probe_capability(Capability.EFFORT_CONTROL, "m")
        )
        assert supported is False
        assert "reasoning_effort" in detail

    def test_an_endpoint_that_accepts_the_field_leaves_it_unknown(self) -> None:
        """Accepting a field is not proof it did anything, which is the same
        reason US-010 refuses to read absence as refusal."""
        transport = RecordingTransport(completion())
        supported, detail = run(
            client(transport).probe_capability(Capability.EFFORT_CONTROL, "m")
        )
        assert supported is None
        assert "not proof" in detail

    def test_a_refusal_that_does_not_name_the_field_explains_itself(self) -> None:
        transport = RecordingTransport(
            reply({"error": {"message": "bad model"}}, status=400)
        )
        supported, detail = run(
            client(transport).probe_capability(Capability.LOGPROBS, "m")
        )
        assert supported is None
        assert "without mentioning" in detail

    def test_an_endpoint_error_leaves_the_capability_unknown(self) -> None:
        class Broken(RecordingTransport):
            async def post_json(self, url: str, **kwargs: Any) -> HttpResponse:
                raise TransportError("timed out")

        supported, detail = run(
            client(Broken()).probe_capability(Capability.EFFORT_CONTROL, "m")
        )
        assert supported is None
        assert "did not answer" in detail

    def test_a_client_that_omits_effort_never_observes_the_capability(self) -> None:
        transport = RecordingTransport()
        supported, detail = run(
            client(transport, effort_field=None).probe_capability(
                Capability.EFFORT_CONTROL, "m"
            )
        )
        assert supported is None
        assert "never observes" in detail

    @pytest.mark.parametrize(
        "capability",
        [Capability.USAGE_REPORTING, Capability.VISIBLE_COT],
    )
    def test_response_settled_capabilities_cost_nothing(
        self, capability: Capability
    ) -> None:
        transport = RecordingTransport()
        supported, detail = run(client(transport).probe_capability(capability, "m"))
        assert supported is None
        assert "does not spend a call" in detail
        assert not transport.calls

    def test_an_endpoint_without_the_field_reports_it_rather_than_ignoring_it(
        self,
    ) -> None:
        """Sending a lever and being refused is the intended behaviour.

        The alternative is dropping the value quietly, and a caller who set
        effort="high" and watched the request succeed would believe the model had
        been told to think harder when it had not. A loud 400 that names the
        field is the honest outcome; the supported way to avoid it is to probe
        first, which is the composition tested above.
        """
        transport = RecordingTransport(
            reply(
                {
                    "error": {
                        "message": "reasoning_effort is not supported by this model"
                    }
                },
                status=400,
            )
        )
        adapter = client(transport)
        with pytest.raises(RequestRejectedError) as info:
            run(adapter.complete(REQUEST, CompletionOptions(model="m", effort="high")))
        assert transport.payload["reasoning_effort"] == "high"
        assert "reasoning_effort" in str(info.value.provider_message)

    def test_the_end_to_end_probe_reports_the_adapter_against_itself(self) -> None:

        transport = RecordingTransport(
            completion(
                usage={"prompt_tokens": 9, "completion_tokens": 2}, reasoning="hmm"
            )
        )
        report = run(probe_capabilities(client(transport)))
        assert report.supports(Capability.USAGE_REPORTING)
        assert report.supports(Capability.VISIBLE_COT)
        assert report.outcome(Capability.EFFORT_CONTROL) is ProbeOutcome.UNKNOWN
        assert report.outcome(Capability.REASONING_BUDGET) is ProbeOutcome.UNKNOWN

    def test_the_end_to_end_probe_agrees_with_the_static_declaration(self) -> None:
        """The format guarantees usage reporting, so a confirmed usage block
        must not read as a disagreement."""
        transport = RecordingTransport(completion(usage={"prompt_tokens": 9}))
        report = run(probe_capabilities(client(transport)))
        assert not [
            d
            for d in report.disagreements
            if d.capability is Capability.USAGE_REPORTING
        ]

    def test_the_probe_sends_the_probe_prompt_not_the_callers(self) -> None:
        transport = RecordingTransport(completion(usage={"prompt_tokens": 1}))
        run(probe_capabilities(client(transport, default_model="dm")))
        assert transport.payload["model"] == "dm"
        assert transport.payload["max_tokens"] == 16

    def test_the_probe_and_the_adapted_call_compose(self) -> None:
        """US-010 strips the lever upstream so the adapter never sends a field
        the endpoint would reject with a 400."""
        transport = RecordingTransport(
            completion(usage={"prompt_tokens": 9, "completion_tokens": 1})
        )
        adapter = client(transport, default_model="m")
        report = run(probe_capabilities(adapter))
        adapted = report.adapt(CompletionOptions(effort="high", max_output_tokens=64))
        run(adapter.complete(REQUEST, adapted.options))
        assert "reasoning_effort" not in transport.last_payload
        assert transport.last_payload["max_tokens"] == 64


class TestDeclaredCapabilities:
    def test_claims_only_what_the_format_guarantees(self) -> None:
        """Reasoning content and an effort knob are vendor extensions, so
        claiming them would be an overclaim on Mistral."""
        assert client(RecordingTransport()).capabilities == frozenset(
            {Capability.USAGE_REPORTING}
        )

    def test_does_not_claim_a_reasoning_budget(self) -> None:
        assert (
            Capability.REASONING_BUDGET not in client(RecordingTransport()).capabilities
        )

    def test_a_known_endpoint_can_declare_more(self) -> None:
        adapter = client(
            RecordingTransport(),
            capabilities=frozenset(
                {Capability.USAGE_REPORTING, Capability.EFFORT_CONTROL}
            ),
        )
        assert adapter.capabilities == frozenset(
            {Capability.USAGE_REPORTING, Capability.EFFORT_CONTROL}
        )

    def test_has_a_name_for_reports_and_logs(self) -> None:
        assert client(RecordingTransport()).name


class TestTransportContract:
    def test_a_recording_transport_satisfies_the_protocol(self) -> None:
        assert isinstance(RecordingTransport(), Transport)

    def test_the_adapter_satisfies_the_client_protocol(self) -> None:
        from inferconomy.client import Client

        assert isinstance(client(RecordingTransport()), Client)

    def test_the_timeout_is_passed_through(self) -> None:
        transport = RecordingTransport(completion())
        run(
            client(transport, timeout=1.5).complete(
                REQUEST, CompletionOptions(model="m")
            )
        )
        assert transport.calls[0]["timeout"] == 1.5

    def test_the_request_goes_to_the_normalized_endpoint(self) -> None:
        transport = RecordingTransport(completion())
        run(client(transport).complete(REQUEST, CompletionOptions(model="m")))
        assert transport.calls[0]["url"].endswith(CHAT)


class TestHttpResponse:
    @pytest.mark.parametrize(
        ("status", "ok"),
        [
            (200, True),
            (201, True),
            (204, True),
            (299, True),
            (300, False),
            (400, False),
        ],
    )
    def test_ok_covers_the_2xx_range(self, status: int, ok: bool) -> None:
        assert HttpResponse(status=status, body="").ok is ok


class TestHttpxTransport:
    """The shipped default, exercised against a mocked httpx client."""

    def test_returns_the_status_and_body(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        async def check() -> HttpResponse:
            http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            transport = HttpxTransport(client=http_client)
            try:
                return await transport.post_json(
                    "https://x.dev/v1/chat/completions",
                    headers={"a": "b"},
                    payload={"model": "m"},
                )
            finally:
                await transport.aclose()

        response = run(check())
        assert response.status == 200
        assert json.loads(response.body) == {"choices": []}

    def test_sends_the_payload_as_json(self) -> None:
        import httpx

        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["content_type"] = request.headers.get("content-type")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        async def check() -> None:
            http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            transport = HttpxTransport(client=http_client)
            try:
                await transport.post_json(
                    "https://x.dev/v1" + CHAT, headers={}, payload={"model": "m"}
                )
            finally:
                await transport.aclose()

        run(check())
        assert "json" in seen["content_type"]
        assert seen["body"] == {"model": "m"}

    def test_lowercases_response_header_names(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={}, headers={"X-RateLimit-Remaining": "9"})

        async def check() -> HttpResponse:
            http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            transport = HttpxTransport(client=http_client)
            try:
                return await transport.post_json(
                    "https://x.dev", headers={}, payload={}
                )
            finally:
                await transport.aclose()

        assert "x-ratelimit-remaining" in run(check()).headers

    def test_a_network_failure_becomes_a_transport_error(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        async def check() -> None:
            http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            transport = HttpxTransport(client=http_client)
            try:
                await transport.post_json("https://x.dev", headers={}, payload={})
            finally:
                await transport.aclose()

        with pytest.raises(TransportError, match="no route to host"):
            run(check())

    def test_creates_its_own_client_when_none_is_given(self) -> None:
        transport = HttpxTransport()
        run(transport.aclose())

    def test_httpx_is_detected(self) -> None:
        assert httpx_available() is True

    def test_a_missing_optional_dependency_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point of the extra is that its absence is a clear instruction
        rather than an ImportError from inside adapter internals."""
        monkeypatch.setitem(sys.modules, "httpx", None)
        assert httpx_available() is False
        with pytest.raises(RuntimeError, match="openai"):
            require_httpx()

    def test_a_missing_optional_dependency_names_the_injected_alternative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "httpx", None)
        with pytest.raises(RuntimeError, match="Transport"):
            require_httpx()

    def test_require_httpx_returns_the_module(self) -> None:
        import httpx

        assert require_httpx() is httpx

    def test_the_adapter_can_close_a_transport_that_can(self) -> None:
        closed: list[bool] = []

        class Closable(RecordingTransport):
            async def aclose(self) -> None:
                closed.append(True)

        run(client(Closable()).aclose())
        assert closed == [True]

    def test_closing_a_transport_without_a_closer_is_harmless(self) -> None:
        run(client(RecordingTransport()).aclose())
