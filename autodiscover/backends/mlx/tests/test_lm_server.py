"""Tests for the local sampler's ``/reload_adapter`` endpoint.

These tests exercise the ``ReloadingAPIHandler`` factory without actually
loading mlx-lm. We stub:
- ``api_handler_cls``: a minimal ``BaseHTTPRequestHandler``-compatible
  base whose ``do_POST`` we can detect being called for non-reload
  paths.
- ``model_provider``: a lightweight namespace whose ``.model`` is an
  object with a ``load_weights`` callable. We assert it was invoked with
  the safetensors path.

The end-to-end integration test described in
``docs/impl-steps/15-mlx-adapter-reload.md`` (launch the sampler, sample,
train, reload, sample again, observe a token diff) requires mlx-lm and a
real GPU, so it lives outside the unit-test suite.
"""
from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import pytest

from autodiscover.backends.mlx.lm_server import _make_reload_handler


class _StubBaseHandler:
    """Minimal stand-in for ``mlx_lm.server.APIHandler``.

    We bypass the real ``BaseHTTPRequestHandler.__init__`` (which would
    parse a socket) and instead manually populate the attributes our
    handler reads/writes: ``rfile``, ``wfile``, ``headers``, ``path``,
    plus the response helpers.
    """

    # Shared counter that subclasses inherit via the same mutable list
    # reference -- we can't just reassign a class attribute on the base
    # from inside a subclass method because ``type(self)`` resolves to
    # the subclass.
    super_do_POST_calls: list[int] = []  # noqa: RUF012

    def __init__(self) -> None:  # no-op, we drive attrs manually
        self.rfile: BytesIO
        self.wfile: BytesIO
        self.headers: dict[str, str]
        self.path: str
        self._responses: list[tuple[int, dict[str, str], bytes]] = []
        self._current_status: int = 0
        self._current_headers: dict[str, str] = {}

    def do_POST(self) -> None:  # noqa: N802
        # Append to the shared list so the test for non-reload paths can
        # confirm we delegated to the parent regardless of which class
        # in the MRO ``type(self)`` resolves to.
        _StubBaseHandler.super_do_POST_calls.append(1)

    # Response helpers used by ``_reply_json``.
    def send_response(self, status: int) -> None:
        self._current_status = status
        self._current_headers = {}

    def send_header(self, name: str, value: str) -> None:
        self._current_headers[name] = value

    def end_headers(self) -> None:
        # Snapshot the response so the test can read it back.
        self._responses.append(
            (self._current_status, dict(self._current_headers), b""),
        )


def _make_handler_instance(
    *,
    model_provider: Any,
    path: str,
    body: bytes,
) -> _StubBaseHandler:
    cls = _make_reload_handler(_StubBaseHandler, model_provider)
    inst = cls()  # type: ignore[call-arg]
    inst.path = path
    inst.headers = {"Content-Length": str(len(body))}
    inst.rfile = BytesIO(body)
    inst.wfile = BytesIO()
    return inst


def _last_response_body(inst: _StubBaseHandler) -> dict[str, Any]:
    inst.wfile.seek(0)
    raw = inst.wfile.read()
    return json.loads(raw.decode()) if raw else {}


def _last_status(inst: _StubBaseHandler) -> int:
    assert inst._responses, "no response sent"
    return inst._responses[-1][0]


def test_reload_adapter_success(tmp_path) -> None:
    adapter = tmp_path / "current.safetensors"
    adapter.write_bytes(b"\x00" * 16)
    model = MagicMock()
    provider = MagicMock(model=model)

    inst = _make_handler_instance(
        model_provider=provider,
        path="/reload_adapter",
        body=json.dumps({"path": str(adapter)}).encode(),
    )
    inst.do_POST()

    assert _last_status(inst) == 200
    payload = _last_response_body(inst)
    assert payload == {"ok": True, "path": str(adapter)}
    model.load_weights.assert_called_once_with(str(adapter), strict=False)


def test_reload_adapter_missing_path_returns_400(tmp_path) -> None:
    provider = MagicMock(model=MagicMock())
    missing = tmp_path / "does-not-exist.safetensors"

    inst = _make_handler_instance(
        model_provider=provider,
        path="/reload_adapter",
        body=json.dumps({"path": str(missing)}).encode(),
    )
    inst.do_POST()

    assert _last_status(inst) == 400
    assert _last_response_body(inst)["ok"] is False
    provider.model.load_weights.assert_not_called()


def test_reload_adapter_missing_path_field_returns_400() -> None:
    provider = MagicMock(model=MagicMock())

    inst = _make_handler_instance(
        model_provider=provider,
        path="/reload_adapter",
        body=b"{}",
    )
    inst.do_POST()

    assert _last_status(inst) == 400
    provider.model.load_weights.assert_not_called()


def test_reload_adapter_before_model_loaded_returns_503(tmp_path) -> None:
    adapter = tmp_path / "current.safetensors"
    adapter.write_bytes(b"\x00")
    provider = MagicMock(model=None)

    inst = _make_handler_instance(
        model_provider=provider,
        path="/reload_adapter",
        body=json.dumps({"path": str(adapter)}).encode(),
    )
    inst.do_POST()

    assert _last_status(inst) == 503


def test_non_reload_paths_delegate_to_parent() -> None:
    _StubBaseHandler.super_do_POST_calls.clear()
    provider = MagicMock(model=MagicMock())

    inst = _make_handler_instance(
        model_provider=provider,
        path="/v1/completions",
        body=b"{}",
    )
    inst.do_POST()

    assert len(_StubBaseHandler.super_do_POST_calls) == 1
    provider.model.load_weights.assert_not_called()


def test_reload_adapter_load_weights_failure_returns_500(tmp_path) -> None:
    adapter = tmp_path / "current.safetensors"
    adapter.write_bytes(b"\x00")
    model = MagicMock()
    model.load_weights.side_effect = RuntimeError("corrupt safetensors")
    provider = MagicMock(model=model)

    inst = _make_handler_instance(
        model_provider=provider,
        path="/reload_adapter",
        body=json.dumps({"path": str(adapter)}).encode(),
    )
    inst.do_POST()

    assert _last_status(inst) == 500
    assert "corrupt safetensors" in _last_response_body(inst)["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
