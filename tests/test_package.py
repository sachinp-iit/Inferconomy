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
