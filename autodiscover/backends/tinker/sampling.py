"""TinkerSamplingClient: adapter over ``tinker.SamplingClient``.

Conforms structurally to ``autodiscover.backends.protocol.SamplingClient``.
"""
from __future__ import annotations

import asyncio

import tinker

from autodiscover.backends.types import (
    SampledSequence,
    SamplingParams,
    TokenSequence,
)


# ---------------------------------------------------------------------------
# Conversions (sampling side)
# ---------------------------------------------------------------------------


def _tokens_to_model_input(seq: TokenSequence) -> tinker.ModelInput:
    return tinker.ModelInput(
        chunks=[tinker.types.EncodedTextChunk(tokens=seq.tokens)],
    )


def _params_to_tinker(p: SamplingParams) -> tinker.SamplingParams:
    return tinker.SamplingParams(
        stop=p.stop,
        max_tokens=p.max_tokens,
        temperature=p.temperature,
    )


# ---------------------------------------------------------------------------
# Sampling client
# ---------------------------------------------------------------------------


class TinkerSamplingClient:
    def __init__(self, inner: tinker.SamplingClient) -> None:
        self._inner = inner

    async def sample(
        self,
        prompt: TokenSequence,
        params: SamplingParams,
        num_samples: int = 1,
    ) -> list[SampledSequence]:
        result = await self._inner.sample_async(
            prompt=_tokens_to_model_input(prompt),
            num_samples=num_samples,
            sampling_params=_params_to_tinker(params),
        )
        out: list[SampledSequence] = []
        for seq in result.sequences:
            assert seq.logprobs is not None, "tinker returned no logprobs"
            out.append(
                SampledSequence(
                    tokens=list(seq.tokens),
                    logprobs=list(seq.logprobs),
                    stop_reason=getattr(seq, "stop_reason", "unknown") or "unknown",
                )
            )
        return out

    async def compute_logprobs(
        self,
        sequences: list[TokenSequence],
    ) -> list[list[float]]:
        results = await asyncio.gather(*[
            self._inner.compute_logprobs_async(_tokens_to_model_input(s))
            for s in sequences
        ])
        # tinker's compute_logprobs_async returns list[float | None]; coerce
        # missing values to 0.0 (the first token's logprob is conventionally
        # 0, matching the protocol's convention).
        return [[0.0 if x is None else float(x) for x in r] for r in results]
