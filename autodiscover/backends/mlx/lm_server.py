"""Local sampler process: ``mlx_lm.server`` plus a ``/reload_adapter`` endpoint.

This is the Step-15 *Path B* implementation: the installed ``mlx-lm`` (see
notes below) does not expose a live LoRA reload endpoint, so we ship a
thin wrapper around ``mlx_lm.server`` that:

1. Boots a ``ModelProvider`` in-process so we hold a reference to the
   live ``model`` object (the same one mlx-lm uses to serve OpenAI-style
   completions). The initial LoRA adapter is loaded via the standard
   ``--adapter-path`` argument so the same path that ``mlx_lm.server``
   exposes is reused.
2. Subclasses ``mlx_lm.server.APIHandler`` to add a ``POST
   /reload_adapter`` handler. The handler atomically loads the new LoRA
   weights into the existing model via ``model.load_weights(path,
   strict=False)`` -- the LoRA modules are already wired by the initial
   load, so only the adapter tensors are touched.

Why Path B
----------

Investigation against the installed ``mlx_lm.server`` source (HTTP path
table at the top of ``APIHandler.do_POST``) showed only
``/v1/completions`` and ``/v1/chat/completions`` -- no
``/v1/load_lora_adapter`` or ``/reload`` endpoint, and no ``model``-field
mapping that we could repurpose without re-launching the server.
``mlx_lm.tuner.utils.load_adapters`` is also not usable as a hot-swap
primitive because it expects a *directory* layout (``adapter_config.json``
+ ``adapters.safetensors``) whereas Step 14 writes a single
``current.safetensors`` file. The right primitive is therefore the lower
level ``model.load_weights(path, strict=False)`` -- LoRA tensors share
the same parameter names across snapshots, so loading non-strictly
overwrites only the LoRA weight arrays.

Run with::

    uv run python -m autodiscover.backends.mlx.lm_server \\
        --model mlx-community/gpt-oss-120b-MXFP4 \\
        --adapter-path /run/discover/adapters/current.safetensors \\
        --port 8081

Concurrency note
----------------

``model.load_weights`` mutates module weight arrays in place. The
``threading.Lock`` below serialises reload against itself but does *not*
block already-running ``mlx_lm.server`` forward passes. In the
autodidakt loop reloads happen *between* sampling phases (sample G plans
-> wait for rewards -> train -> reload before sampling next group), so
this is acceptable. Do not invoke ``/reload_adapter`` mid-sampling.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path
from typing import Any


# A single module-level lock protects the in-place adapter swap. The
# ``APIHandler`` subclass below acquires it for the duration of the swap;
# nothing else acquires it today (forward passes happen on the
# ``ResponseGenerator`` thread which we don't synchronise with -- see the
# concurrency note in the module docstring).
_RELOAD_LOCK = threading.Lock()


def main() -> None:
    p = argparse.ArgumentParser(description="autodiscover local sampler")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--adapter-path",
        default=None,
        help="Initial LoRA adapter file or directory. Optional.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    _serve(
        model=args.model,
        adapter_path=args.adapter_path,
        host=args.host,
        port=args.port,
    )


def _serve(
    *,
    model: str,
    adapter_path: str | None,
    host: str,
    port: int,
) -> None:
    """Boot ``mlx_lm.server`` in-process with a ``/reload_adapter`` patch.

    Importing ``mlx_lm`` is deferred so the surrounding module is still
    importable in environments without mlx (e.g. CI on Linux).
    """
    # Local imports: ``mlx_lm`` is a heavy, Mac-only dep. Keep it lazy so
    # the module is importable everywhere.
    from mlx_lm import server as mlx_server  # type: ignore[import-not-found]

    cli_args = _make_cli_args(model=model, adapter_path=adapter_path)
    model_provider = mlx_server.ModelProvider(cli_args)

    handler_class = _make_reload_handler(mlx_server.APIHandler, model_provider)
    mlx_server.run(
        host=host,
        port=port,
        model_provider=model_provider,
        handler_class=handler_class,
    )


def _make_cli_args(
    *,
    model: str,
    adapter_path: str | None,
) -> argparse.Namespace:
    """Build the ``argparse.Namespace`` that ``ModelProvider`` expects.

    ``ModelProvider`` reads many attributes off ``cli_args``; we mirror
    the defaults from ``mlx_lm.server.main`` so unspecified knobs behave
    the same as launching ``mlx_lm.server`` directly.
    """
    return argparse.Namespace(
        model=model,
        adapter_path=adapter_path,
        draft_model=None,
        num_draft_tokens=3,
        trust_remote_code=False,
        chat_template=None,
        use_default_chat_template=False,
        temp=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=512,
        prompt_cache_size=10,
        pipeline=False,
    )


def _make_reload_handler(api_handler_cls: type, model_provider: Any) -> type:
    """Return a subclass of ``mlx_lm.server.APIHandler`` that adds
    ``POST /reload_adapter``.

    Closing over ``model_provider`` (rather than reaching for it via the
    ``ResponseGenerator`` on each request) keeps the contract explicit
    and lets us short-circuit if someone reload-bombs us before the
    initial model load finishes.
    """

    class ReloadingAPIHandler(api_handler_cls):  # type: ignore[misc, valid-type]
        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path == "/reload_adapter":
                self._handle_reload_adapter()
                return
            super().do_POST()

        def _handle_reload_adapter(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length) if content_length else b"{}"
                body = json.loads(raw_body.decode() or "{}")
            except (ValueError, json.JSONDecodeError) as exc:
                self._reply_json(400, {"ok": False, "error": f"bad request: {exc}"})
                return

            path = body.get("path")
            if not isinstance(path, str) or not path:
                self._reply_json(400, {"ok": False, "error": "missing 'path'"})
                return
            adapter_path = Path(path)
            if not adapter_path.exists():
                self._reply_json(
                    400, {"ok": False, "error": f"adapter not found: {path!r}"},
                )
                return

            model = model_provider.model
            if model is None:
                self._reply_json(
                    503, {"ok": False, "error": "model not loaded yet"},
                )
                return

            # ``load_weights(..., strict=False)`` overwrites just the
            # LoRA tensors (which share names with the saved adapter).
            # Base / frozen params are untouched because they're absent
            # from the safetensors file.
            try:
                with _RELOAD_LOCK:
                    model.load_weights(str(adapter_path), strict=False)
            except Exception as exc:  # noqa: BLE001 (surface as 500)
                self._reply_json(
                    500, {"ok": False, "error": f"load_weights failed: {exc}"},
                )
                return

            self._reply_json(200, {"ok": True, "path": str(adapter_path)})

        def _reply_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

    return ReloadingAPIHandler


if __name__ == "__main__":
    main()
