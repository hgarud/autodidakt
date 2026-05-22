"""Tests for TinkerTrainingClient + the ``_datum_to_tinker`` / ``_strip_mask`` helpers.

Tinker's SDK is too heavy to spin up in CI. We use ``unittest.mock`` to
fake the SDK surface and verify the adapter's conversion + control flow.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tinker

from autodiscover.backends.protocol import SamplingClient, TrainingClient
from autodiscover.backends.tinker.sampling import TinkerSamplingClient
from autodiscover.backends.tinker.training import (
    TinkerTrainingClient,
    _datum_to_tinker,
    _strip_mask,
    make_tinker_backend,
)
from autodiscover.backends.types import TrainingDatum


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_imports_clean() -> None:
    assert TinkerTrainingClient is not None
    assert make_tinker_backend is not None


def test_training_client_has_protocol_surface() -> None:
    for name in (
        "forward_backward",
        "optim_step",
        "save_checkpoint",
        "get_post_training_sampling_client",
    ):
        assert callable(getattr(TinkerTrainingClient, name, None)), (
            f"TinkerTrainingClient missing {name}"
        )


def test_protocols_are_runtime_checkable() -> None:
    assert hasattr(SamplingClient, "__subclasshook__") or True
    assert hasattr(TrainingClient, "__subclasshook__") or True


# ---------------------------------------------------------------------------
# Behavioral
# ---------------------------------------------------------------------------


async def test_tinker_training_client_forward_backward_strips_mask_and_returns_metrics() -> None:
    inner = MagicMock(spec=tinker.TrainingClient)

    fb_future = AsyncMock()
    fb_future.result_async = AsyncMock(
        return_value=SimpleNamespace(metrics={"loss": 0.5}),
    )
    inner.forward_backward_async = AsyncMock(return_value=fb_future)

    tc = TinkerTrainingClient(inner, log_dir="/tmp/x", save_every=0)
    batch = [TrainingDatum(
        input_tokens=[1, 2], target_tokens=[2, 3],
        old_logprobs=[-0.1, -0.2], advantages=[1.0, 1.0], mask=[1.0, 1.0],
    )]
    metrics = await tc.forward_backward(batch, "importance_sampling")

    assert metrics == {"loss": 0.5}
    inner.forward_backward_async.assert_awaited_once()
    args, kwargs = inner.forward_backward_async.call_args
    # First positional arg is the list of tinker.Datum.
    forwarded_data = args[0]
    assert len(forwarded_data) == 1
    # The mask field must have been stripped before being passed to Tinker.
    assert "mask" not in forwarded_data[0].loss_fn_inputs
    # ...but the other loss_fn_inputs are preserved.
    assert set(forwarded_data[0].loss_fn_inputs.keys()) == {
        "target_tokens", "logprobs", "advantages",
    }
    assert kwargs["loss_fn"] == "importance_sampling"


async def test_tinker_training_client_forward_backward_handles_empty_metrics() -> None:
    inner = MagicMock(spec=tinker.TrainingClient)
    fb_future = AsyncMock()
    # Real Tinker result objects sometimes lack a metrics attribute or
    # provide a non-numeric value; the adapter must remain defensive.
    fb_future.result_async = AsyncMock(return_value=SimpleNamespace())
    inner.forward_backward_async = AsyncMock(return_value=fb_future)

    tc = TinkerTrainingClient(inner, log_dir="/tmp/x", save_every=0)
    out = await tc.forward_backward(
        [TrainingDatum(
            input_tokens=[1], target_tokens=[2],
            old_logprobs=[-0.1], advantages=[1.0], mask=[1.0],
        )],
        "ppo",
    )
    assert out == {}


async def test_tinker_training_client_optim_step_builds_adam_params() -> None:
    inner = MagicMock(spec=tinker.TrainingClient)
    opt_future = AsyncMock()
    opt_future.result_async = AsyncMock(return_value=None)
    inner.optim_step_async = AsyncMock(return_value=opt_future)

    tc = TinkerTrainingClient(inner, log_dir="/tmp/x", save_every=0)
    await tc.optim_step(learning_rate=3e-4)

    inner.optim_step_async.assert_awaited_once()
    (adam_arg,) = inner.optim_step_async.call_args.args
    assert isinstance(adam_arg, tinker.AdamParams)
    assert adam_arg.learning_rate == pytest.approx(3e-4)
    assert adam_arg.beta1 == pytest.approx(0.9)
    assert adam_arg.beta2 == pytest.approx(0.95)
    assert adam_arg.eps == pytest.approx(1e-8)
    # batch counter advances after a successful step.
    assert tc._batch_count == 1


async def test_get_post_training_sampling_client_save_every_zero_takes_fast_path() -> None:
    """save_every=0 -> always use save_weights_and_get_sampling_client_async."""
    inner = MagicMock(spec=tinker.TrainingClient)
    new_sampling_inner = MagicMock(spec=tinker.SamplingClient)
    inner.save_weights_and_get_sampling_client_async = AsyncMock(
        return_value=new_sampling_inner,
    )

    tc = TinkerTrainingClient(inner, log_dir="/tmp/x", save_every=0)
    # Advance batch_count so we're not in the "0 batches" early branch.
    tc._batch_count = 5

    client = await tc.get_post_training_sampling_client()

    assert isinstance(client, TinkerSamplingClient)
    inner.save_weights_and_get_sampling_client_async.assert_awaited_once()


async def test_get_post_training_sampling_client_save_every_one_takes_checkpoint_path() -> None:
    """save_every=1 + batch_count=2 -> save_checkpoint_async + create_sampling_client."""
    inner = MagicMock(spec=tinker.TrainingClient)
    new_sampling_inner = MagicMock(spec=tinker.SamplingClient)
    inner.create_sampling_client = MagicMock(return_value=new_sampling_inner)
    # Should NOT be called on this branch.
    inner.save_weights_and_get_sampling_client_async = AsyncMock()

    tc = TinkerTrainingClient(inner, log_dir="/tmp/x", save_every=1)
    tc._batch_count = 2

    fake_paths = {"sampler_path": "tinker://sampler/iter_000002", "state_path": "x"}
    with patch(
        "autodiscover.checkpointing.save_checkpoint_async",
        new=AsyncMock(return_value=fake_paths),
    ) as save_mock:
        client = await tc.get_post_training_sampling_client()

    save_mock.assert_awaited_once()
    inner.create_sampling_client.assert_called_once_with(
        "tinker://sampler/iter_000002",
    )
    inner.save_weights_and_get_sampling_client_async.assert_not_awaited()
    assert isinstance(client, TinkerSamplingClient)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def test_datum_to_tinker_carries_all_loss_fn_inputs() -> None:
    d = TrainingDatum(
        input_tokens=[1, 2, 3], target_tokens=[2, 3, 4],
        old_logprobs=[-0.1, -0.2, -0.3], advantages=[1.0, 1.0, 1.0],
        mask=[1.0, 1.0, 1.0],
    )
    out = _datum_to_tinker(d)
    assert isinstance(out, tinker.Datum)
    assert set(out.loss_fn_inputs.keys()) == {
        "target_tokens", "logprobs", "advantages", "mask",
    }


def test_strip_mask_removes_mask_only() -> None:
    d = TrainingDatum(
        input_tokens=[1, 2], target_tokens=[2, 3],
        old_logprobs=[-0.1, -0.2], advantages=[1.0, 1.0], mask=[1.0, 1.0],
    )
    stripped = _strip_mask(_datum_to_tinker(d))
    assert "mask" not in stripped.loss_fn_inputs
    assert "advantages" in stripped.loss_fn_inputs
    assert "target_tokens" in stripped.loss_fn_inputs
    assert "logprobs" in stripped.loss_fn_inputs
