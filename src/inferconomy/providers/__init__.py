"""Provider adapters.

Kept separate from :mod:`inferconomy.client` because importing an adapter is not
the same as wanting one. The protocol, the contracts, and the fakes stay free of
any HTTP, and nothing here is imported until a caller asks for a specific
endpoint.
"""

from __future__ import annotations

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
from inferconomy.providers.openai_compatible import (
    OpenAICompatibleClient,
    normalize_base_url,
)
from inferconomy.providers.transport import (
    HttpResponse,
    HttpxTransport,
    Transport,
    httpx_available,
    require_httpx,
)

__all__ = [
    "AuthenticationError",
    "ContextLengthExceeded",
    "HttpResponse",
    "HttpxTransport",
    "ModelNotFoundError",
    "OpenAICompatibleClient",
    "ProviderError",
    "RateLimitError",
    "RequestRejectedError",
    "ServerError",
    "Transport",
    "TransportError",
    "httpx_available",
    "normalize_base_url",
    "require_httpx",
]
