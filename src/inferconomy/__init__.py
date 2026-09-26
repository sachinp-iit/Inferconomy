"""Inferconomy: adaptive LLM inference optimization.

Inferconomy decides how a model should infer and how much computation each
request deserves, rather than applying one fixed strategy and budget to
everything. It is a library, not a framework: it wraps an LLM client the caller
already has and leaves the surrounding architecture alone.

The public entry point is :func:`optimize`, added in a later release. This
package currently exports its version and its data contracts.
"""

from __future__ import annotations

from inferconomy.contracts import (
    Capability,
    Message,
    OptimizationReport,
    Request,
    Response,
    StopReason,
    Usage,
    UsageBasis,
)
from inferconomy.costs import (
    CostTableError,
    InvalidPriceTable,
    ModelPrice,
    PriceTable,
    UnknownModel,
    load_price_table,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "Capability",
    "CostTableError",
    "InvalidPriceTable",
    "Message",
    "ModelPrice",
    "OptimizationReport",
    "PriceTable",
    "Request",
    "Response",
    "StopReason",
    "UnknownModel",
    "Usage",
    "UsageBasis",
    "__version__",
    "load_price_table",
]
