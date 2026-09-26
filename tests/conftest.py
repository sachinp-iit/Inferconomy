"""Enforce the no-network property across the whole suite.

US-003 is only satisfied if every test in this repository runs offline. Asserting
that inside one test would prove nothing about the rest, so the guard is
autouse: any attempt to reach a non-loopback host anywhere in the suite fails
loudly and says so.

Loopback is deliberately permitted. CPython's asyncio event loop builds a local
socketpair for its self-pipe on Windows, so blocking every connection would
break the event loop rather than test anything. The property worth enforcing is
"no traffic leaves the machine", not "no sockets are ever opened".
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest

_LOOPBACK_NAMES = frozenset({"127.0.0.1", "::1", "localhost", ""})


def _is_external(address: object) -> bool:
    """True only when the address positively identifies a remote host.

    The default is *allow*, deliberately. Blocking unrecognised address forms
    produces false positives that break unrelated machinery rather than catching
    anything: CPython's ``socket.socketpair()`` passes a socket object straight
    to ``connect``, which is how the Windows event loop builds its self-pipe.
    Accidental network use in a test is always a hostname or IP, so a form this
    cannot read is not the thing we are guarding against.
    """
    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, str):
            return host not in _LOOPBACK_NAMES and not host.startswith("127.")
    return False


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make any outbound connection to a non-loopback host an obvious failure."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def guard(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(address: Any, *args: Any, **kwargs: Any) -> Any:
            if not _is_external(address):
                return original(address, *args, **kwargs)
            raise RuntimeError(
                f"External network access attempted during a test (target: "
                f"{address!r}). The suite must run offline; any provider "
                f"behaviour belongs behind a Client fake."
            )

        return wrapper

    monkeypatch.setattr(socket.socket, "connect", guard(original_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(original_connect_ex))
    monkeypatch.setattr(socket, "create_connection", guard(original_create_connection))
    yield
