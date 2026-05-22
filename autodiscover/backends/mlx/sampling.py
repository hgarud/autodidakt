"""Local sampling client backed by mlx_lm.server (OpenAI-compatible HTTP)."""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from autodiscover.backends.types import (
    SampledSequence,
    SamplingParams,
    TokenSequence,
)


class MlxSamplingClient:
    """Thin async HTTP client over mlx_lm.server.

    mlx_lm.server's /v1/completions endpoint accepts a string prompt and
    re-tokenizes it server-side. We decode the caller's token IDs to text
    before sending; the server's tokenizer round-trips gpt-oss harmony
    special tokens correctly.

    Args:
        base_url: e.g. "http://127.0.0.1:8081"
        tokenizer: HF tokenizer used to decode the caller's token IDs back
            to text before POSTing. Required because mlx_lm.server's
            CompletionRequest expects a string prompt.
        adapter_id: name of the LoRA adapter currently loaded into the
            sampler. mlx_lm.server selects the adapter via the model field
            in OpenAI-compatible requests; for a single-adapter server we
            don't need to send it. If your version supports per-request
            adapter selection, pass it via `extra_body`.
        timeout_s: per-request timeout. Long because phase-1 sampling at
            ~26k tokens on 120B can be slow.
    """

    def __init__(
        self,
        base_url: str,
        tokenizer: Any | None = None,
        adapter_id: str | None = None,
        timeout_s: float = 600.0,
        max_concurrency: int = 1,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._tokenizer = tokenizer
        self._adapter_id = adapter_id
        self._client = httpx.AsyncClient(timeout=timeout_s)
        # mlx_lm.server multiplexes connections but the underlying MLX
        # model runs one forward pass at a time, so N concurrent POSTs
        # serialize at the server and the last N-1 sit waiting for the
        # response with no bytes flowing — which is exactly when httpx's
        # read timeout fires. Gate concurrency client-side so each request
        # only opens its read window when the GPU is about to serve it.
        self._gpu_sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def sample(
        self,
        prompt: TokenSequence,
        params: SamplingParams,
        num_samples: int = 1,
    ) -> list[SampledSequence]:
        # If num_samples > 1, mlx_lm.server may not honor the `n` field —
        # if not, fall back to launching N parallel requests and gather.
        if num_samples == 1:
            return [await self._sample_one(prompt, params)]
        results = await asyncio.gather(*[
            self._sample_one(prompt, params) for _ in range(num_samples)
        ])
        return list(results)

    async def _sample_one(
        self,
        prompt: TokenSequence,
        params: SamplingParams,
    ) -> SampledSequence:
        if self._tokenizer is None:
            raise RuntimeError(
                "MlxSamplingClient requires a tokenizer to decode token IDs "
                "before POSTing to mlx_lm.server.",
            )
        prompt_text = self._tokenizer.decode(
            prompt.tokens, skip_special_tokens=False,
        )
        body: dict = {
            # mlx_lm.server uses "model" for selecting which loaded model.
            # For a single-model server, any string is accepted; we pass
            # the adapter id when set so future per-request routing works.
            "model": self._adapter_id or "default_model",
            "prompt": prompt_text,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "logprobs": True,
        }
        if params.stop:
            # mlx_lm.server expects string stop sequences; drop integer ids
            # (the server tokenizes stops differently than the caller would).
            string_stops = [s for s in params.stop if isinstance(s, str)]
            if string_stops:
                body["stop"] = string_stops
        async with self._gpu_sem:
            resp = await self._client.post(
                f"{self._base_url}/v1/completions", json=body,
            )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        # mlx_lm.server returns per-token logprobs as
        #   logprobs.content = [{"id": <int>, "logprob": <float>}, ...]
        lp_payload = choice.get("logprobs") or {}
        content = lp_payload.get("content") or []
        if not content:
            raise RuntimeError(
                "mlx_lm.server returned no per-token logprobs; "
                "pass logprobs=True and check your mlx-lm version.",
            )
        tokens = [int(c["id"]) for c in content]
        logprobs = [float(c["logprob"]) for c in content]
        return SampledSequence(
            tokens=tokens,
            logprobs=logprobs,
            stop_reason=choice.get("finish_reason") or "unknown",
        )

    async def compute_logprobs(
        self,
        sequences: list[TokenSequence],
    ) -> list[list[float]]:
        # max_tokens=0 + echo=True is the OpenAI-compatible way to ask
        # for prompt logprobs only.
        async def _one(seq: TokenSequence) -> list[float]:
            body = {
                "model": self._adapter_id or "default",
                "prompt": seq.tokens,
                "max_tokens": 0,
                "echo": True,
                "logprobs": True,
            }
            async with self._gpu_sem:
                resp = await self._client.post(
                    f"{self._base_url}/v1/completions", json=body,
                )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            lp_payload = choice.get("logprobs") or {}
            token_lp = lp_payload.get("token_logprobs")
            if token_lp is None:
                raise RuntimeError(
                    "compute_logprobs: server returned no token_logprobs",
                )
            # First-token logprob is conventionally null/None or 0.
            return [float(x) if x is not None else 0.0 for x in token_lp]

        return list(await asyncio.gather(*[_one(s) for s in sequences]))
