"""Package-level invariants.

These are the tests that would catch a broken install, so they are written to
run against the installed distribution rather than the source tree.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import inferconomy


def test_package_is_importable() -> None:
    assert inferconomy.__name__ == "inferconomy"


def test_version_is_exposed() -> None:
    assert isinstance(inferconomy.__version__, str)
    assert inferconomy.__version__


def test_version_matches_installed_distribution() -> None:
    """The dynamic version in pyproject must track __init__, not drift from it."""
    assert importlib.metadata.version("inferconomy") == inferconomy.__version__


def test_exports_are_importable() -> None:
    for name in inferconomy.__all__:
        assert hasattr(inferconomy, name), f"__all__ advertises missing name: {name}"


def test_type_marker_ships_with_package() -> None:
    """A `Typing :: Typed` classifier is a lie without the marker file present."""
    assert (Path(inferconomy.__path__[0]) / "py.typed").is_file()


def _requires_dist() -> list[str]:
    """What the installed distribution says it needs.

    Read from the installed metadata rather than from pyproject, so the claim is
    checked against what a pip install would actually pull in, and so it works on
    every supported Python without a TOML parser.
    """
    return importlib.metadata.requires("inferconomy") or []


def test_the_library_has_no_runtime_dependencies() -> None:
    """A design commitment, not an accident.

    Inferconomy wraps a client the caller already has, so it must never force a
    provider SDK or a framework on anyone. An adapter that needs HTTP takes an
    injected transport and offers its default as an extra, which keeps this true
    even now that one exists.
    """
    runtime = [req for req in _requires_dist() if "extra ==" not in req]
    assert runtime == [], f"unexpected runtime dependencies: {runtime}"


def test_the_http_transport_is_an_extra_and_not_a_dependency() -> None:
    """An extra is a choice; a dependency is a bill.

    Anything needed to merely import the library would be a runtime dependency.
    """
    extras = [req for req in _requires_dist() if "extra ==" in req]
    assert any("httpx" in req for req in extras)
    assert not any("httpx" in req for req in _requires_dist() if "extra ==" not in req)


def test_the_core_imports_without_any_third_party_package() -> None:
    """Importing the contracts, the protocol, and the fakes must not pull in a
    provider SDK, so that the zero-dependency claim is about what actually
    happens on import rather than about what pyproject promises."""
    import subprocess
    import sys

    script = (
        "import sys, inferconomy, inferconomy.contracts, inferconomy.client, "
        "inferconomy.tokens, inferconomy.testing, inferconomy.costs, "
        "inferconomy.benchmark, inferconomy.judge, inferconomy.frontier, "
        "inferconomy.capabilities; "
        "leaked = [m for m in sys.modules "
        "if m.split('.')[0] in {'httpx', 'openai', 'anthropic', 'requests', "
        "'numpy', 'pydantic'}]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
