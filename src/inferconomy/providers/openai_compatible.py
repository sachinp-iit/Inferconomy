"""One adapter for every OpenAI-compatible endpoint.

OpenAI's chat-completions request and response shape has been adopted by
OpenRouter, Groq, Together, Fireworks, DeepSeek, Mistral, and the local servers
behind vLLM and Ollama. That is a genuine standard, and this adapter speaks it
once instead of eight times.

**Normalizing means absorbing the dialect, not pretending the dialect does not
exist.** These endpoints agree on the envelope and disagree on almost everything
inside it, so the disagreements are the work:

- ``max_tokens`` versus ``max_completion_tokens``. OpenAI deprecated the first for
  reasoning models and rejects it; DeepSeek, Mistral, and the local servers accept
  only the first. There is no value that works everywhere, so the field is
  configurable rather than guessed, defaulting to the one with the wider reach.
- Reasoning content in the response arrives under ``reasoning`` (OpenRouter) or
  ``reasoning_content`` (DeepSeek). Both land in
  :attr:`~inferconomy.contracts.Response.reasoning_text`.
- There is **no standard field for a separate reasoning-token budget** in this
  format. OpenAI folds reasoning into ``max_completion_tokens``; OpenRouter wants
  a nested ``reasoning`` object; DeepSeek has neither. So the adapter refuses to
  invent a field name for it, which means it does not claim
  :attr:`~inferconomy.contracts.Capability.REASONING_BUDGET` and drops the value
  unless an endpoint dialect is named. A guessed field would either be ignored,
  producing exactly the silent pretence this library exists to prevent, or draw a
  400 on every call.

**Send what was asked for, and let a bad lever fail loudly.** When a caller sets
:attr:`CompletionOptions.effort`, this adapter sends it. On an endpoint without
that field - most of them; only OpenAI and OpenRouter have ``reasoning_effort`` -
the endpoint answers 400, and the message names the field. That is the intended
behaviour, not an oversight: the alternative is to drop the value quietly, and a
caller who set ``effort="high"`` and watched the request succeed would believe the
model had been told to think harder when it had not. The composition that avoids
the 400 is
:mod:`inferconomy.capabilities` doing its job first::

    report = await probe_capabilities(adapter)
    options = report.adapt(CompletionOptions(effort="high"))   # drops it if unconfirmed
    await adapter.complete(request, options.options)

That ordering is the supported flow, and the alternative - sending a lever and
discovering afterwards that it was ignored - is the one this library exists to
make unnecessary.

**What does not leak upward.** Callers only ever pass
:class:`~inferconomy.client.CompletionOptions` and read
:class:`~inferconomy.contracts.Response`. Which wire field carried the effort
level, which spelling of the reasoning block was used, whether usage was reported
at all, and whether the price of this call is a measurement or an estimate are
all settled here or carried as provenance. No caller of this module should ever
need to know whether it is talking to OpenAI or to a laptop.

**Usage is reported as the provider stated it, and never improved on.** A response
carrying a usage block gives :attr:`~inferconomy.contracts.Usage.tokens_exact` as
``True``, because those are the endpoint's own counts. A response without one
falls back to
:func:`~inferconomy.tokens.estimate_usage`, which is marked
``tokens_exact=False``, and an estimator can never confer exactness. A caller
computing cost from this can therefore always tell which of the two it has, which
is the difference between a bill and a guess.

One known limitation, stated rather than hidden: endpoints report prompt cache
hits inside ``prompt_tokens``, and :class:`~inferconomy.contracts.Usage` has
nowhere to put a cached-token count. They are therefore counted as ordinary
input tokens, exactly as the provider counted them, which means a flat price
table **over-states** the cost of a cached prompt. It is an over-estimate rather
than a wrong-direction one, and the price table cannot yet express the
difference. Fixing it properly means a cached-token field on ``Usage`` and
tiered rates in the price table, which is a change to those contracts and not
something to smuggle in here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from inferconomy.client import CompletionOptions
from inferconomy.contracts import Capability, Request, Response, Usage
from inferconomy.providers.errors import (
    AuthenticationError,
    ContextLengthExceeded,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    RequestRejectedError,
    ServerError,
    TransportError,
)
from inferconomy.providers.transport import (
    DEFAULT_TIMEOUT,
    HttpxTransport,
    Transport,
    require_httpx,
)
from inferconomy.tokens import DEFAULT_ESTIMATOR, TokenEstimator, estimate_usage

__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "DEFAULT_MAX_TOKENS_FIELD",
    "OpenAICompatibleClient",
    "normalize_base_url",
]

CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_MAX_TOKENS_FIELD: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
"""The output-limit spelling with the wider reach.

``max_tokens`` is what DeepSeek, Mistral, vLLM, Ollama, Groq, Together, and
Fireworks accept. OpenAI accepts it too, except for its reasoning models, which
require ``max_completion_tokens``. Since one of the two has to be the default and
this one works for more of the named endpoints, it wins, and an OpenAI reasoning
model needs ``max_tokens_field="max_completion_tokens"``."""

_FORMAT_CAPABILITIES = frozenset({Capability.USAGE_REPORTING})
"""What the *format* guarantees, which is deliberately less than what any one
endpoint offers.

Usage reporting is standardized. Reasoning content and an effort knob are vendor
extensions, present on some of these endpoints and absent on others, so claiming
them here would be an overclaim that US-010's probe would then flag on every
Mistral request. A caller who knows their endpoint has them passes
``capabilities=``; the probe confirms the rest at runtime."""


def normalize_base_url(base_url: str) -> str:
    """Turn a user's base URL into a full chat-completions endpoint.

    Accepts the shapes people actually type, so that all eight named providers
    work from their documented base and none of them needs the full path:

    ==========================  ==========================================
    Given                       Result
    ==========================  ==========================================
    ``https://api.openai.com``  ``https://api.openai.com/v1/chat/completions``
    ``.../v1``                 ``.../v1/chat/completions``
    ``.../v1/``                ``.../v1/chat/completions``
    ``.../v1/chat/completions`` unchanged
    ``http://localhost:11434`` ``http://localhost:11434/v1/chat/completions``
    ==========================  ==========================================

    A base with no ``/v1`` gets one, because the version segment is where these
    endpoints put their compatibility guarantees and appending the bare path
    would land on the wrong route for a server like Ollama. A base that already
    carries the full path is left alone, so a caller who pasted a curl example
    does not get a doubled path.
    """
    candidate = base_url.strip().rstrip("/")
    if not candidate:
        raise ValueError("base_url must not be blank.")
    if candidate.endswith(CHAT_COMPLETIONS_PATH):
        return candidate

    parts = urlsplit(candidate)
    if not parts.scheme or not parts.netloc:
        raise ValueError(
            f"base_url must be an absolute URL with a scheme and a host, got "
            f"{base_url!r}. For a local server that is usually "
            f"'http://localhost:8000/v1'."
        )
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit(
        (parts.scheme, parts.netloc, f"{path}{CHAT_COMPLETIONS_PATH}", "", "")
    )


def _first_str(payload: Mapping[str, Any], *keys: str) -> str | None:
    """First key present with a non-empty string value.

    Providers spell the same field several ways and sometimes send ``null`` for a
    field they support, so presence is not enough and truthiness of the container
    is not enough either.
    """
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class OpenAICompatibleClient:
    """A :class:`~inferconomy.client.Client` for any OpenAI-compatible endpoint.

    :param base_url: Endpoint root. See :func:`normalize_base_url` for the forms
        accepted; ``https://api.openai.com/v1`` and ``http://localhost:8000/v1``
        are the two you will actually type.
    :param api_key: Sent as a bearer token. Optional, because vLLM and Ollama
        usually run without one and an empty ``Authorization`` header makes some
        servers reject an otherwise valid request.
    :param default_model: Used when a call does not name a model. When neither is
        set the field is omitted, letting the endpoint pick, which is the correct
        behaviour and also means the response's model must be read back.
    :param max_tokens_field: ``"max_tokens"`` or ``"max_completion_tokens"``. See
        :data:`DEFAULT_MAX_TOKENS_FIELD`.
    :param effort_field: Wire field for :attr:`CompletionOptions.effort`. ``None``
        omits the value entirely. Defaults to ``reasoning_effort``, which only
        OpenAI and OpenRouter have; see the module docstring on probing first.
    :param reasoning_budget_field: Wire field for
        :attr:`CompletionOptions.reasoning_budget`. ``None`` by default, because
        the format has no standard one; see the module docstring.
    :param extra_headers: Sent on every call. Where an OpenRouter attribution
        header or a gateway's tenant token belongs.
    :param capabilities: What this endpoint declares. Defaults to what the
        *format* guarantees, which is less than any single endpoint offers; see
        :data:`_FORMAT_CAPABILITIES`.
    :param transport: Injected HTTP. Defaults to ``httpx`` if it is installed.
    :param timeout: Seconds per request.
    :param estimator: Token counting for responses that report no usage.
    """

    base_url: str
    api_key: str | None = None
    default_model: str | None = None
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = (
        DEFAULT_MAX_TOKENS_FIELD
    )
    effort_field: str | None = "reasoning_effort"
    reasoning_budget_field: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    capabilities: frozenset[Capability] = _FORMAT_CAPABILITIES
    transport: Transport = field(default_factory=HttpxTransport)
    timeout: float = DEFAULT_TIMEOUT
    estimator: TokenEstimator = DEFAULT_ESTIMATOR
    name: str = "openai-compatible"

    def __post_init__(self) -> None:
        self.endpoint = normalize_base_url(self.base_url)
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}.")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("api_key must not be blank when provided.")

    def _headers(self) -> dict[str, str]:
        """Caller headers first, then the ones the protocol depends on.

        ``content-type`` is authoritative because the body is always JSON and an
        endpoint handed the wrong content type answers with a parse error that
        looks like a provider fault. ``accept`` is the caller's, because a gateway
        that wants something else is a real case and there is no such thing as an
        invalid one.
        """
        headers = dict(self.extra_headers)
        headers["content-type"] = "application/json"
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        headers.setdefault("accept", "application/json")
        return headers

    def _payload(self, request: Request, options: CompletionOptions) -> dict[str, Any]:
        """Build the wire payload, dropping only what this endpoint cannot express."""
        model = options.model or self.default_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        if model is None:
            # An explicit null is not the same as an absent field: endpoints differ
            # on whether they accept a null model, and omitting is what "let the
            # endpoint choose" actually means.
            del payload["model"]

        limit = options.max_output_tokens
        if limit is not None:
            payload[self.max_tokens_field] = limit
        if options.stop:
            payload["stop"] = list(options.stop)
        if options.effort is not None and self.effort_field:
            payload[self.effort_field] = options.effort
        if options.reasoning_budget is not None and self.reasoning_budget_field:
            payload[self.reasoning_budget_field] = options.reasoning_budget
        return payload

    async def complete(self, request: Request, options: CompletionOptions) -> Response:
        """Issue one chat-completions call and normalize the answer."""
        payload = self._payload(request, options)
        started = time.perf_counter()
        http = await self.transport.post_json(
            self.endpoint,
            headers=self._headers(),
            payload=payload,
            timeout=self.timeout,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        if not http.ok:
            raise self._error_for(http)
        body = _parse_json(http.body, self.endpoint)
        return self._to_response(body, request, options, latency_ms)

    def _to_response(
        self,
        body: Mapping[str, Any],
        request: Request,
        options: CompletionOptions,
        latency_ms: float,
    ) -> Response:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(
                f"{self.endpoint} returned a success status with no choices: "
                f"{_clip(body)!r}. A truncated body that parses is still a failure."
            )
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise ProviderError(
                f"{self.endpoint} returned a choice without a message object: "
                f"{_clip(choices[0])!r}."
            )

        # content is null on some reasoning models when they emit only reasoning,
        # so a missing text is not a missing message.
        text = _first_str(message, "content") or ""
        reasoning = _first_str(message, "reasoning", "reasoning_content")

        finish = choices[0].get("finish_reason")
        return Response(
            text=text,
            usage=self._usage(body, request, text, reasoning or ""),
            model=_first_str(body, "model") or options.model or self.default_model,
            provider_finish_reason=None if finish is None else str(finish),
            reasoning_text=reasoning,
            latency_ms=latency_ms,
        )

    def _usage(
        self,
        body: Mapping[str, Any],
        request: Request,
        text: str,
        reasoning: str,
    ) -> Usage:
        """Use the provider's counts if it sent any, and estimate if it did not."""
        raw = body.get("usage")
        if not isinstance(raw, Mapping):
            return estimate_usage(
                request, text, estimator=self.estimator, reasoning_text=reasoning
            )

        prompt = _as_int(raw.get("prompt_tokens"))
        completion = _as_int(raw.get("completion_tokens"))
        details = raw.get("completion_tokens_details")
        reported_reasoning = (
            _as_int(details.get("reasoning_tokens"))
            if isinstance(details, Mapping)
            else 0
        )
        reasoning_tokens = reported_reasoning or 0
        if reasoning_tokens > completion:
            # Preserve the subset invariant rather than raising on a real response.
            reasoning_tokens = completion

        return Usage(
            input_tokens=prompt,
            output_tokens=completion,
            reasoning_tokens=reasoning_tokens,
            llm_calls=1,
            tokens_exact=True,
            cost_usd=None,
            cost_exact=False,
        )

    def _error_for(self, http: Any) -> ProviderError:
        """Classify a non-2xx response, keeping the provider's own words."""
        provider_message = _error_message(http.body)
        detail = provider_message or _clip(http.body) or "no error body"
        status = http.status

        if status in (401, 403):
            return AuthenticationError(
                f"{self.endpoint} rejected the credentials (HTTP {status}): {detail}",
                status=status,
                provider_message=provider_message,
            )
        if status == 429:
            return RateLimitError(
                f"{self.endpoint} rate limited the request (HTTP {status}): {detail}",
                status=status,
                provider_message=provider_message,
            )
        if status >= 500:
            return ServerError(
                f"{self.endpoint} failed (HTTP {status}): {detail}",
                status=status,
                provider_message=provider_message,
            )
        if status == 404:
            return ModelNotFoundError(
                f"{self.endpoint} has no such model or route (HTTP 404): {detail}",
                status=status,
                provider_message=provider_message,
            )
        if status == 400 and _looks_like_context_length(detail):
            return ContextLengthExceeded(
                f"{self.endpoint} says the request is past the context window "
                f"(HTTP 400): {detail}",
                status=status,
                provider_message=provider_message,
            )
        return RequestRejectedError(
            f"{self.endpoint} rejected the request (HTTP {status}): {detail}",
            status=status,
            provider_message=provider_message,
        )

    async def probe_capability(
        self, capability: Capability, model: str | None
    ) -> tuple[bool | None, str]:
        """Test the endpoint about a capability, for :mod:`inferconomy.capabilities`.

        Returns ``None`` whenever the endpoint did not give a definitive answer,
        which US-010 records as ``unknown`` rather than ``unsupported``.

        Reasoning budget is reported as ``None`` without a call, and that is the
        point rather than a limitation: the OpenAI-compatible format has no
        standard field for it, so asking the endpoint would not answer the
        question. It is a fact about the format, and no endpoint's behaviour can
        change it.
        """
        if capability is Capability.REASONING_BUDGET:
            if self.reasoning_budget_field is None:
                return None, (
                    "The OpenAI-compatible format has no standard reasoning-budget "
                    "field, so this is unknown for any endpoint unless a dialect "
                    "is named. Inferconomy will not guess a field name."
                )
            return await self._probe_field(
                capability, self.reasoning_budget_field, 16, model
            )
        if capability is Capability.EFFORT_CONTROL:
            if self.effort_field is None:
                return None, (
                    "This client is configured to omit the effort field, so it "
                    "never observes the capability."
                )
            return await self._probe_field(capability, self.effort_field, "low", model)
        if capability is Capability.LOGPROBS:
            return await self._probe_field(capability, "logprobs", True, model)
        if capability is Capability.USAGE_REPORTING:
            return None, (
                "Usage reporting is settled from any ordinary response, so the "
                "probe does not spend a call on it."
            )
        if capability is Capability.VISIBLE_COT:
            return None, (
                "Visible chain-of-thought is settled from any ordinary response, "
                "so the probe does not spend a call on it."
            )
        # Every member of the closed Capability enum is handled above, so this is
        # unreachable today. It stays because a capability added later should get
        # an honest "this adapter cannot test it" rather than fall through
        # silently, and it cannot be covered by a test without passing something
        # that is not a Capability.
        return (
            None,
            f"{capability.value} is not a capability this adapter can test.",
        )  # pragma: no cover

    async def _probe_field(
        self, capability: Capability, field_name: str, value: Any, model: str | None
    ) -> tuple[bool | None, str]:
        """Send a request carrying one field and see whether the endpoint objects.

        A rejection naming the field means the endpoint does not have it. Anything
        else, including a success, is reported as ``None`` for the reasoning-shaped
        capabilities, because accepting a field is not proof it did anything - the
        same reason US-010 refuses to read absence as refusal.
        """
        payload = self._payload(
            Request.from_text("hi"),
            CompletionOptions(
                model=model or self.default_model,
                max_output_tokens=8,
            ),
        )
        payload[field_name] = value
        try:
            http = await self.transport.post_json(
                self.endpoint,
                headers=self._headers(),
                payload=payload,
                timeout=self.timeout,
            )
        except TransportError as exc:
            return None, f"Endpoint did not answer: {exc}"

        if http.ok:
            return None, (
                f"Endpoint accepted {field_name!r} but accepted is not proof it "
                f"acted on it, so this stays unknown."
            )
        if field_name in http.body:
            return False, (
                f"Endpoint rejected {field_name!r} (HTTP {http.status}): "
                f"{_error_message(http.body) or _clip(http.body)}"
            )
        return None, (
            f"Endpoint returned HTTP {http.status} without mentioning "
            f"{field_name!r}, so it did not identify the cause."
        )

    async def aclose(self) -> None:
        """Release the transport's connections, if it has any."""
        closer = getattr(self.transport, "aclose", None)
        if closer is not None:
            await closer()


def _parse_json(body: str, endpoint: str) -> Mapping[str, Any]:
    """Parse a response body, or fail loudly.

    A 200 carrying an HTML error page is a failure, not an empty completion, and
    returning a blank response for it would let a misconfigured gateway look like
    a model that had nothing to say.
    """
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise ProviderError(
            f"{endpoint} returned a success status with a body that is not JSON: "
            f"{_clip(body)!r}. This is usually a proxy or gateway in the way."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ProviderError(
            f"{endpoint} returned JSON that is not an object: {_clip(body)!r}."
        )
    return parsed


def _error_message(body: str) -> str | None:
    """Pull the message out of an OpenAI-shaped error envelope, if present."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    error = parsed.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
        return _clip(body)
    if isinstance(error, str) and error:
        return error
    message = parsed.get("message")
    if isinstance(message, str) and message:
        return message
    return None


_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "reduce the length",
    "prompt is too long",
    "request too large",
)
"""Substrings that mean "past the context window" across these endpoints.

Every one of these is a guess about wording. The exception carries the provider's
full message, so a caller that disagrees with the guess is never stuck with it -
see the module docstring on classification resting on text."""


def _looks_like_context_length(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in _CONTEXT_MARKERS)


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _clip(value: Any, limit: int = 200) -> str:
    """Shorten a payload for an error message.

    Error text ends up in logs. An unbounded response body or provider payload
    copied into an exception is an easy way to put a prompt, or a fragment of a
    dataset, into somebody's terminal history.
    """
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


# Re-exported so ``from inferconomy.providers.openai_compatible import ...`` is a
# complete import surface, and so a caller comparing types does not have to know
# that the transport module happens to be where the timeout default lives.
__all__ += ["DEFAULT_TIMEOUT", "Transport", "TransportError", "require_httpx"]
