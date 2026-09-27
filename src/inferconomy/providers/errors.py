"""Provider failures, classified by what a caller can do about them.

An inference library sitting between a policy and a provider lives or dies on the
distinction between "this call failed" and "this will never work". A policy that
treats a bad API key as a transient error retries a hundred times and reports a
timeout; a policy that treats a rate limit as permanent gives up on a request that
would have succeeded in a second. Collapsing both into one ``ProviderError``
pushes that judgement onto every caller, which is exactly the judgement a library
should be making.

So each class carries :attr:`ProviderError.retryable` and, where the endpoint
supplied one, :attr:`ProviderError.status` and the provider's own message.

**The provider's message is never discarded.** A 400 can mean a malformed request,
a model that does not exist, or a prompt past the context window, and the three
call for completely different responses. Classifying them means reading the
provider's own words, so the words travel with the exception. Where classification
rests on message text rather than a status code, it is best-effort and says so:
an endpoint is free to reword its errors tomorrow, and a library that presented
text matching as certainty would be overstating what it knows.
"""

from __future__ import annotations

__all__ = [
    "AuthenticationError",
    "ContextLengthExceeded",
    "ModelNotFoundError",
    "ProviderError",
    "RateLimitError",
    "RequestRejectedError",
    "ServerError",
    "TransportError",
]


class ProviderError(Exception):
    """Base for every failure that came from the provider side.

    :attr:`retryable` is the field a policy should branch on, and it defaults to
    ``False``: a failure nobody has classified is treated as permanent, because
    retrying an unclassifiable error forever is how a bug becomes an outage.
    """

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.provider_message = provider_message
        """The endpoint's own words, when it supplied any.

        Never dropped. A caller that wants to branch on wording rather than on
        this library's classification can, and the classification itself is
        better for having something to point at when it is wrong."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, status={self.status!r})"


class TransportError(ProviderError):
    """The request never got an answer: DNS, connection reset, timeout.

    Retryable, because none of these say anything about whether the request was
    valid. Note what that implies: a timeout may have been billed. A caller
    retrying should not assume the first attempt was free.
    """

    retryable = True


class AuthenticationError(ProviderError):
    """The endpoint rejected the credentials, HTTP 401 or 403.

    Never retryable. A missing, wrong, or expired key will still be wrong in a
    second, and retrying a rejected key can lock an account on some providers.
    """

    retryable = False


class RateLimitError(ProviderError):
    """HTTP 429, or a provider-specific quota message.

    Retryable, and the only error here where waiting is the entire remedy.
    """

    retryable = True


class RequestRejectedError(ProviderError):
    """The endpoint understood the request and refused it, usually HTTP 400.

    Never retryable as sent, though a policy may well retry a *different* request
    after shrinking one. That is a policy decision, which is why the subclasses
    below exist: they name the reasons a smaller or different request would work.
    """

    retryable = False


class ModelNotFoundError(RequestRejectedError):
    """The named model is not available on this endpoint.

    Distinct from a malformed request because the fix is a different model, not a
    different payload, and because a budget sweep that silently kept using the
    default model instead of the one it asked for would be a measurement
    fabricated from a substitution.
    """


class ContextLengthExceeded(RequestRejectedError):
    """The prompt or the requested output is past the model's context window.

    Worth its own class because it is one of the few provider errors a policy can
    *act* on: shrinking the request, or decomposing it, is a valid response.
    Classified from message text, so it is best-effort.
    """


class ServerError(ProviderError):
    """The endpoint failed, HTTP 5xx.

    Retryable, since 5xx says nothing about the request.
    """

    retryable = True
