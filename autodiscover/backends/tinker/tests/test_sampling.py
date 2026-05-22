"""Tests for TinkerSamplingClient.

Tinker's SDK is too heavy to spin up in CI. We use ``unittest.mock`` to
fake the SDK surface and verify the adapter's conversion + control flow.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import tinker

from autodiscover.backends.tinker.sampling import TinkerSamplingClient
from autodiscover.backends.types import SamplingParams, TokenSequence


def _fake_sampling_result(
    tokens: list[int], logprobs: list[float], stop_reason: str = "stop",
) -> object:
    """Build a SamplingResult-like object Tinker's SDK returns."""
    seq = SimpleNamespace(
        tokens=tokens, logprobs=logprobs, stop_reason=stop_reason,
    )
    return SimpleNamespace(sequences=[seq])


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_sampling_client_has_protocol_surface() -> None:
    for name in ("sample", "compute_logprobs"):
        assert callable(getattr(TinkerSamplingClient, name, None)), (
            f"TinkerSamplingClient missing {name}"
        )


# ---------------------------------------------------------------------------
# Behavioral
# ---------------------------------------------------------------------------


async def test_tinker_sampling_client_sample_returns_expected_shape() -> None:
    inner = MagicMock(spec=tinker.SamplingClient)
    inner.sample_async = AsyncMock(
        return_value=_fake_sampling_result([10, 11, 12], [-0.1, -0.2, -0.3], "stop"),
    )

    client = TinkerSamplingClient(inner)
    out = await client.sample(
        TokenSequence(tokens=[1, 2]),
        SamplingParams(max_tokens=8, temperature=0.7, stop=["END"]),
    )

    assert len(out) == 1
    assert out[0].tokens == [10, 11, 12]
    assert out[0].logprobs == [-0.1, -0.2, -0.3]
    assert out[0].stop_reason == "stop"

    # Verify it forwarded a tinker.ModelInput + tinker.SamplingParams.
    inner.sample_async.assert_awaited_once()
    kwargs = inner.sample_async.call_args.kwargs
    assert isinstance(kwargs["prompt"], tinker.ModelInput)
    assert isinstance(kwargs["sampling_params"], tinker.SamplingParams)
    assert kwargs["sampling_params"].max_tokens == 8
    assert kwargs["sampling_params"].temperature == 0.7
    assert kwargs["num_samples"] == 1


async def test_tinker_sampling_client_compute_logprobs_shape() -> None:
    inner = MagicMock(spec=tinker.SamplingClient)
    # tinker's compute_logprobs_async returns list[float | None]. The
    # adapter should coerce None -> 0.0.
    inner.compute_logprobs_async = AsyncMock(
        side_effect=[
            [None, -0.1, -0.2],
            [None, -0.3],
        ],
    )

    client = TinkerSamplingClient(inner)
    out = await client.compute_logprobs([
        TokenSequence(tokens=[1, 2, 3]),
        TokenSequence(tokens=[4, 5]),
    ])

    assert out == [[0.0, -0.1, -0.2], [0.0, -0.3]]
    assert inner.compute_logprobs_async.await_count == 2
