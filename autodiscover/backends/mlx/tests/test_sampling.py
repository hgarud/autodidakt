"""Tests for MlxSamplingClient.

We mock httpx.AsyncClient.post on the instance so we don't need respx or
pytest-httpx as test dependencies.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from autodiscover.backends.mlx.sampling import MlxSamplingClient
from autodiscover.backends.protocol import SamplingClient
from autodiscover.backends.types import SamplingParams, TokenSequence


def _mock_response(payload: dict[str, Any]) -> Any:
    resp = AsyncMock()
    # raise_for_status is a sync no-op in httpx but we use AsyncMock for
    # simplicity; configure as a regular MagicMock-style method.
    resp.raise_for_status = lambda: None
    resp.json = lambda: payload
    return resp


def _completion_payload(
    tokens: list[int], token_logprobs: list[float], finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "tokens": tokens,
                "logprobs": {"token_logprobs": token_logprobs},
                "finish_reason": finish_reason,
            },
        ],
    }


def test_class_satisfies_sampling_client_protocol() -> None:
    assert issubclass(MlxSamplingClient, object)
    for name in ("sample", "compute_logprobs"):
        assert callable(getattr(MlxSamplingClient, name, None))


async def test_sample_single_posts_and_parses() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081/", adapter_id="adapter-x")
    post = AsyncMock(
        return_value=_mock_response(
            _completion_payload([1, 2, 3], [-0.1, -0.2, -0.3], "length"),
        ),
    )
    client._client.post = post  # type: ignore[method-assign]

    out = await client.sample(
        TokenSequence(tokens=[200000, 5]),
        SamplingParams(max_tokens=8, temperature=0.7, stop=["END"]),
    )

    assert len(out) == 1
    sampled = out[0]
    assert sampled.tokens == [1, 2, 3]
    assert sampled.logprobs == [-0.1, -0.2, -0.3]
    assert sampled.stop_reason == "length"

    assert post.await_count == 1
    args, kwargs = post.call_args
    assert args[0] == "http://127.0.0.1:8081/v1/completions"
    body = kwargs["json"]
    assert body["prompt"] == [200000, 5]
    assert body["max_tokens"] == 8
    assert body["temperature"] == 0.7
    assert body["logprobs"] is True
    assert body["model"] == "adapter-x"
    assert body["stop"] == ["END"]

    await client.aclose()


async def test_sample_num_samples_issues_n_parallel_posts() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    post = AsyncMock(
        return_value=_mock_response(_completion_payload([7], [-0.5])),
    )
    client._client.post = post  # type: ignore[method-assign]

    out = await client.sample(
        TokenSequence(tokens=[200000]),
        SamplingParams(max_tokens=4, temperature=1.0, stop=[]),
        num_samples=4,
    )

    assert len(out) == 4
    assert post.await_count == 4
    # No stop key when params.stop is empty.
    body = post.call_args.kwargs["json"]
    assert "stop" not in body

    await client.aclose()


async def test_sample_stop_strings_pass_through() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    post = AsyncMock(
        return_value=_mock_response(_completion_payload([1], [-0.1])),
    )
    client._client.post = post  # type: ignore[method-assign]

    await client.sample(
        TokenSequence(tokens=[1]),
        SamplingParams(max_tokens=8, temperature=0.0, stop=["</s>", "</plan>"]),
    )
    body = post.call_args.kwargs["json"]
    assert body["stop"] == ["</s>", "</plan>"]

    await client.aclose()


async def test_sample_stop_ints_pass_through_unchanged() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    post = AsyncMock(
        return_value=_mock_response(_completion_payload([1], [-0.1])),
    )
    client._client.post = post  # type: ignore[method-assign]

    # Regression: integer token-ID stops must pass through, not be filtered.
    await client.sample(
        TokenSequence(tokens=[1]),
        SamplingParams(max_tokens=1, temperature=0.0, stop=[200002, 200003]),
    )
    body = post.call_args.kwargs["json"]
    assert body["stop"] == [200002, 200003]

    await client.aclose()


async def test_sample_stop_mixed_pass_through_unchanged() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    post = AsyncMock(
        return_value=_mock_response(_completion_payload([1], [-0.1])),
    )
    client._client.post = post  # type: ignore[method-assign]

    await client.sample(
        TokenSequence(tokens=[1]),
        SamplingParams(max_tokens=1, temperature=0.0, stop=["</s>", 200002]),
    )
    body = post.call_args.kwargs["json"]
    assert body["stop"] == ["</s>", 200002]

    await client.aclose()


async def test_sample_empty_stop_omits_field() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    post = AsyncMock(
        return_value=_mock_response(_completion_payload([1], [-0.1])),
    )
    client._client.post = post  # type: ignore[method-assign]

    await client.sample(
        TokenSequence(tokens=[1]),
        SamplingParams(max_tokens=1, temperature=0.0, stop=[]),
    )
    body = post.call_args.kwargs["json"]
    assert "stop" not in body

    await client.aclose()


async def test_sample_raises_when_tokens_missing() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    payload = {
        "choices": [
            {
                "text": "hello",
                "logprobs": {"token_logprobs": [-0.1]},
                "finish_reason": "stop",
            },
        ],
    }
    client._client.post = AsyncMock(return_value=_mock_response(payload))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="missing 'tokens'"):
        await client.sample(
            TokenSequence(tokens=[1]),
            SamplingParams(max_tokens=1, temperature=0.0, stop=[]),
        )
    await client.aclose()


async def test_sample_raises_when_logprobs_missing() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    payload = {
        "choices": [
            {"tokens": [1, 2], "logprobs": {}, "finish_reason": "stop"},
        ],
    }
    client._client.post = AsyncMock(return_value=_mock_response(payload))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="no token_logprobs"):
        await client.sample(
            TokenSequence(tokens=[1]),
            SamplingParams(max_tokens=1, temperature=0.0, stop=[]),
        )
    await client.aclose()


async def test_compute_logprobs_posts_and_returns_shape() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    payload_a = {
        "choices": [
            {"logprobs": {"token_logprobs": [None, -0.1, -0.2]}},
        ],
    }
    payload_b = {
        "choices": [
            {"logprobs": {"token_logprobs": [None, -0.3]}},
        ],
    }
    post = AsyncMock(
        side_effect=[_mock_response(payload_a), _mock_response(payload_b)],
    )
    client._client.post = post  # type: ignore[method-assign]

    out = await client.compute_logprobs([
        TokenSequence(tokens=[1, 2, 3]),
        TokenSequence(tokens=[4, 5]),
    ])

    assert out == [[0.0, -0.1, -0.2], [0.0, -0.3]]
    assert post.await_count == 2
    # Verify the body shape on at least one call.
    first_body = post.call_args_list[0].kwargs["json"]
    assert first_body["max_tokens"] == 0
    assert first_body["echo"] is True
    assert first_body["logprobs"] is True

    await client.aclose()


async def test_compute_logprobs_raises_when_missing() -> None:
    client = MlxSamplingClient("http://127.0.0.1:8081")
    payload = {"choices": [{"logprobs": {}}]}
    client._client.post = AsyncMock(return_value=_mock_response(payload))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="no token_logprobs"):
        await client.compute_logprobs([TokenSequence(tokens=[1, 2])])
    await client.aclose()


def test_runtime_protocol_check() -> None:
    # MlxSamplingClient should structurally satisfy SamplingClient.
    client = MlxSamplingClient("http://127.0.0.1:8081")
    assert isinstance(client, SamplingClient)
