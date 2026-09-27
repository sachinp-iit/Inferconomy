"""HTTP transport for provider adapters, as something the caller can replace.

The standard library has no async HTTP client, and the project has no runtime
dependencies. Those two facts together mean the adapter cannot both speak HTTP
asynchronously and stay dependency-free unless the HTTP client is either optional
or injected.

So it is injected. A :class:`Transport` here is four fields wide, and the adapter
holds one. The shipped default wraps ``httpx`` and is imported lazily, so
importing this package costs nothing and breaks for nobody who is not calling a
provider. A caller who already has an HTTP client, a proxy, a retry policy, or
corporate TLS interception passes it in and Infereconomy's opinion stops.

This is also what makes the adapter testable under this repository's
network-isolation guarantee: the suite proves normalization against recorded
payloads, with no socket anywhere near it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "HttpResponse",
    "HttpxTransport",
    "Transport",
    "httpx_available",
    "require_httpx",
]

DEFAULT_TIMEOUT = 60.0
"""Seconds before a request is abandoned.

Long enough for a slow reasoning model to finish, since that is the case this
library exists for. A timeout is a :class:`~inferconomy.providers.errors.
TransportError` and is retryable, but a timed-out call may still have been
billed."""


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP response, reduced to what an adapter actually reads."""

    status: int
    body: str
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@runtime_checkable
class Transport(Protocol):
    """Sends one JSON POST and returns the response.

    Deliberately not a general HTTP client. It has no verbs beyond POST, no
    streaming, and no connection management, because an adapter needs none of
    those and a wider interface would be harder to satisfy from something the
    caller already has.
    """

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> HttpResponse:
        """POST ``payload`` as JSON and return the response, whatever its status.

        Raising is reserved for the request never producing a response at all.
        A 4xx or 5xx is a perfectly good answer from a server and is returned
        rather than raised, so the adapter can classify it with the provider's own
        status and message in hand instead of guessing from a transport exception.
        """
        ...


def httpx_available() -> bool:
    """Whether the optional default transport can be constructed."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


def require_httpx() -> Any:
    """Import ``httpx`` or explain precisely what is missing.

    Named so the failure arrives at the point of use with an instruction, rather
    than as an ``ImportError`` from a line of adapter internals that mentions a
    package name the user may not recognise as required.
    """
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "The default transport needs the optional 'httpx' dependency, which "
            "is not installed. Install Inferconomy with the 'openai' extra, or "
            "pass your own Transport to the adapter. Inferconomy has no runtime "
            "dependencies by design, so nothing was installed for you."
        ) from exc
    return httpx


@dataclass
class HttpxTransport:
    """The shipped default, on ``httpx``.

    A thin wrapper with no retry, no pooling, and no logging, because those are
    the transport's business and not this library's. Retries in particular are
    left to the caller: whether a given call is worth repeating depends on what
    the policy is trying to achieve, and a library that silently retried would
    make its own token accounting wrong.

    :param client: An existing ``httpx.AsyncClient``, for the cases where the
        caller's client is the one that has the proxy configured, the corporate
        CA bundle loaded, or a pool that should be shared. Typed as ``Any`` so
        that merely importing this module does not require ``httpx``.
    """

    client: Any = None

    def __post_init__(self) -> None:
        httpx = require_httpx()
        self._client = self.client if self.client is not None else httpx.AsyncClient()

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> HttpResponse:
        try:
            response = await self._client.post(
                url, headers=dict(headers), json=dict(payload), timeout=timeout
            )
        except Exception as exc:  # re-raised as a typed failure below
            from inferconomy.providers.errors import TransportError

            raise TransportError(
                f"Request to {url} did not complete: {type(exc).__name__}: {exc}"
            ) from exc
        return HttpResponse(
            status=response.status_code,
            body=response.text,
            headers={k.lower(): v for k, v in response.headers.items()},
        )

    async def aclose(self) -> None:
        await self._client.aclose()
