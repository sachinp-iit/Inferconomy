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
)

__version__ = "0.1.0.dev0"

__all__ = [
    "Capability",
    "Message",
    "OptimizationReport",
    "Request",
    "Response",
    "StopReason",
    "Usage",
    "__version__",
]
