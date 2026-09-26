"""Token counting and usage estimation.

Not every provider reports usage. Some omit it, some report it only on request,
some run behind a gateway that strips it. When the numbers are missing, a
library has three honest choices: report nothing, report an estimate clearly
labelled as one, or report an estimate pretending to be a measurement. Only the
first two are implemented here, and the second is what this module is for.

**An estimator can never confer exactness.** :attr:`Usage.tokens_exact` may be
set only by a provider-reported figure. A better tokenizer narrows the error; it
does not remove the possibility that the provider's tokenizer, its handling of
special tokens, and its accounting of cached or reasoning tokens all differ from
yours. Codifying that here means no future adapter or pluggable estimator can
quietly upgrade an estimate into a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from inferconomy.contracts import Request, Usage

__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "HeuristicTokenEstimator",
    "TokenEstimator",
    "estimate_input_tokens",
    "estimate_usage",
]

DEFAULT_CHARS_PER_TOKEN = 4
"""Characters per token for the default heuristic.

Chosen because it needs no dependency, not because it is accurate. English prose
averages near four characters per token across common BPE vocabularies; code,
tables, and non-Latin scripts diverge substantially in both directions.
"""


@runtime_checkable
class TokenEstimator(Protocol):
    """Counts tokens in a string.

    Implement this to plug in a real tokenizer. Doing so improves the estimate
    but does not make it a measurement, for the reason in the module docstring.
    """

    def count(self, text: str) -> int:
        """Number of tokens in ``text``. Must be non-negative and deterministic."""
        ...


@dataclass(frozen=True)
class HeuristicTokenEstimator:
    """Dependency-free estimator based on text length.

    The default, and the floor. Deliberately boring: it exists so that a
    missing usage figure degrades to an obviously approximate number rather than
    to nothing.
    """

    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN

    def __post_init__(self) -> None:
        if self.chars_per_token <= 0:
            raise ValueError(
                f"chars_per_token must be positive, got {self.chars_per_token}."
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token)


DEFAULT_ESTIMATOR: TokenEstimator = HeuristicTokenEstimator()
"""Module-level default, so the common case needs no configuration."""


def estimate_input_tokens(
    request: Request, estimator: TokenEstimator = DEFAULT_ESTIMATOR
) -> int:
    """Estimated prompt tokens across every turn in the request.

    Includes all turns, not just the last. Ignoring prior turns is a common
    source of silent under-reporting in multi-turn conversations, and it is
    exactly the kind of error that makes a savings claim look better than it is.
    """
    return sum(estimator.count(message.content) for message in request.messages)


def estimate_usage(
    request: Request,
    output_text: str = "",
    *,
    estimator: TokenEstimator = DEFAULT_ESTIMATOR,
    reasoning_text: str = "",
) -> Usage:
    """Build an estimated :class:`Usage` for a call whose provider reported nothing.

    The result always carries ``tokens_exact=False`` and no cost. That is the
    entire point: a caller can display the numbers, and cannot mistake them for
    billing figures.
    """
    output_tokens = estimator.count(output_text)
    reasoning_tokens = estimator.count(reasoning_text) if reasoning_text else 0
    if reasoning_tokens > output_tokens:
        # A provider that separates reasoning from output but billed them
        # together would produce this; clamping keeps the subset invariant
        # rather than raising on a real response.
        reasoning_tokens = output_tokens
    return Usage(
        input_tokens=estimate_input_tokens(request, estimator),
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        llm_calls=1,
        tokens_exact=False,
        cost_usd=None,
        cost_exact=False,
    )
