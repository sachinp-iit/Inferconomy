"""Deterministic test doubles for :class:`~inferconomy.client.Client`.

Shipped inside the package rather than confined to the test suite on purpose.
Anyone integrating Inferconomy needs to test their own code without spending
tokens or hitting a rate limit, and that need does not stop at our repository.

Every response here is deterministic: the same inputs produce the same output on
every run and every machine. No sleeps, no randomness, no clock reads.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from inferconomy.client import CompletionOptions
from inferconomy.contracts import Capability, Request, Response, Usage
from inferconomy.tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    HeuristicTokenEstimator,
    estimate_usage,
)

__all__ = [
    "FakeClient",
    "ResponseFactory",
    "echo_response",
    "estimate_tokens",
    "scripted",
    "user_request",
]

_ESTIMATOR = HeuristicTokenEstimator()
"""Shared so the fake and the shipped estimator can never drift apart."""


def estimate_tokens(text: str) -> int:
    """Estimate token count without a tokenizer.

    A thin alias for the shipped estimator. It is not a tokenizer, and the usage
    records built from it carry ``tokens_exact=False``, so nothing derived from
    it can be mistaken for a provider-reported figure.
    """
    return _ESTIMATOR.count(text)


def echo_response(
    request: Request, options: CompletionOptions, *, model: str = "fake-model"
) -> Response:
    """Default factory: echo the final user turn, honouring the output budget.

    Honouring ``max_output_tokens`` is what makes this useful for budget tests.
    A fake that ignored the budget could not demonstrate that a policy's
    allocation actually reaches the provider.
    """
    last_user = next(
        (message for message in reversed(request.messages) if message.role == "user"),
        request.messages[-1],
    )
    text = f"echo: {last_user.content}"
    if options.max_output_tokens is not None:
        text = text[: options.max_output_tokens * DEFAULT_CHARS_PER_TOKEN]

    return Response(
        text=text,
        usage=estimate_usage(request, text),
        model=options.model or model,
        provider_finish_reason="stop",
    )


ResponseFactory = Callable[[Request, CompletionOptions], Response]
"""Builds a response. Must be deterministic and free of side effects."""


class FakeClient:
    """A :class:`Client` that returns canned answers and records what it was asked.

    Answers come from the first of these that is supplied:

    1. ``handler`` — a callable, for computed responses
    2. ``responses`` — a script, consumed one entry per call
    3. neither — :func:`echo_response`

    When a script runs out the fake raises. Silently repeating the last response
    is friendlier, and is not the default for exactly that reason: it hides a
    test that made more calls than it intended, which in a library whose whole
    purpose is counting calls is the bug most worth catching. Pass
    ``repeat_last=True`` to opt in.
    """

    def __init__(
        self,
        responses: Sequence[Response] = (),
        *,
        handler: ResponseFactory | None = None,
        capabilities: frozenset[Capability] = frozenset(),
        repeat_last: bool = False,
        latency_ms: float = 0.0,
    ) -> None:
        if handler is not None and responses:
            raise ValueError("Pass either handler or responses, not both.")
        self._handler = handler
        self._responses = list(responses)
        self._repeat_last = repeat_last
        self._latency_ms = latency_ms
        self._capabilities = frozenset(capabilities)
        self._calls: list[tuple[Request, CompletionOptions]] = []
        self._index = 0

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    @property
    def calls(self) -> tuple[tuple[Request, CompletionOptions], ...]:
        """Every call made, in order, for assertions about what was requested."""
        return tuple(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_options(self) -> CompletionOptions:
        """Options from the most recent call. Raises if there has not been one."""
        if not self._calls:
            raise AssertionError("No calls have been made yet.")
        return self._calls[-1][1]

    def reset(self) -> None:
        """Forget recorded calls and rewind the script. Supplied answers are kept."""
        self._calls.clear()
        self._index = 0

    async def complete(self, request: Request, options: CompletionOptions) -> Response:
        self._calls.append((request, options))
        response = self._next_response(request, options)
        if self._latency_ms and response.latency_ms == 0.0:
            response = replace(response, latency_ms=self._latency_ms)
        return response

    def _next_response(self, request: Request, options: CompletionOptions) -> Response:
        if self._handler is not None:
            return self._handler(request, options)

        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response

        if self._responses and self._repeat_last:
            return self._responses[-1]

        if self._responses:
            raise AssertionError(
                f"FakeClient script exhausted after {self._index} call(s), "
                f"{len(self._responses)} supplied. Supply more responses, pass "
                f"repeat_last=True, or use a handler."
            )

        return echo_response(request, options)

    def __repr__(self) -> str:
        return (
            f"FakeClient(calls={self.call_count}, "
            f"scripted={len(self._responses)}, handler={self._handler is not None})"
        )


def scripted(
    *texts: str,
    capabilities: frozenset[Capability] = frozenset(),
    repeat_last: bool = False,
) -> FakeClient:
    """Convenience: a fake that returns the given texts in order."""
    return FakeClient(
        [
            Response(
                text=text,
                usage=Usage(
                    output_tokens=estimate_tokens(text),
                    llm_calls=1,
                    tokens_exact=False,
                ),
            )
            for text in texts
        ],
        capabilities=capabilities,
        repeat_last=repeat_last,
    )


def user_request(text: str) -> Request:
    """Convenience: a single-turn user request."""
    return Request.from_text(text)
