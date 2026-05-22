"""Tests for the MLX in-process training client.

These tests instantiate a real MLX model and are therefore:
  * Skipped if ``mlx`` / ``mlx_lm`` are not importable.
  * Skipped unless the ``MLX_TEST_MODEL`` env var is set, naming a tiny
    pre-staged MLX model identifier (e.g.
    ``mlx-community/Qwen2.5-0.5B-MXFP4``). The 120B production model is
    far too large to run in CI.
  * Marked ``slow`` so the default ``pytest -m "not slow"`` invocation
    skips them.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

mlx = pytest.importorskip("mlx", reason="mlx extra not installed")
mx = pytest.importorskip("mlx.core", reason="mlx extra not installed")
pytest.importorskip("mlx_lm", reason="mlx-lm not installed")

# Only import the MlxTrainingClient lazily after the importorskip calls
# above, otherwise the module-import side effects break collection in
# environments without the extra.
from autodiscover.backends.mlx.training import (  # noqa: E402
    MlxTrainingClient,
    _InProcessBaseSampling,
)
from autodiscover.backends.types import TokenSequence, TrainingDatum  # noqa: E402


_MODEL_ENV = "MLX_TEST_MODEL"
_MODEL_NAME = os.environ.get(_MODEL_ENV)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _MODEL_NAME is None,
        reason=f"{_MODEL_ENV} not set; skipping MLX training tests",
    ),
]


_TEST_LR = 1e-5


@pytest.fixture(scope="module")
def trainer(tmp_path_factory) -> MlxTrainingClient:
    adapter_dir = tmp_path_factory.mktemp("adapters")
    return MlxTrainingClient(
        model_name=_MODEL_NAME,  # type: ignore[arg-type]
        lora_rank=4,
        adapter_dir=adapter_dir,
        save_every=1,
        log_dir=str(adapter_dir),
        learning_rate=_TEST_LR,
    )


def test_construction_reports_trainable_params(trainer: MlxTrainingClient) -> None:
    params = trainer.trainable_parameters()
    # Sum across the (nested) parameter tree.
    def _count(p) -> int:
        if hasattr(p, "size"):
            return int(p.size)
        if isinstance(p, dict):
            return sum(_count(v) for v in p.values())
        if isinstance(p, (list, tuple)):
            return sum(_count(v) for v in p)
        return 0
    assert _count(params) > 0, "LoRA should introduce trainable parameters"


def _flatten_arrays(p) -> list:
    """Flatten nested parameter dicts/lists down to mx.array leaves."""
    if hasattr(p, "shape") and hasattr(p, "dtype"):
        return [p]
    if isinstance(p, dict):
        out = []
        for v in p.values():
            out.extend(_flatten_arrays(v))
        return out
    if isinstance(p, (list, tuple)):
        out = []
        for v in p:
            out.extend(_flatten_arrays(v))
        return out
    return []


async def test_forward_backward_and_optim_step_changes_params(
    trainer: MlxTrainingClient,
) -> None:
    # Snapshot trainable params before.
    before = [mx.array(a) for a in _flatten_arrays(trainer.trainable_parameters())]

    T = 8
    batch = [
        TrainingDatum(
            input_tokens=[1] * T, target_tokens=[2] * T,
            old_logprobs=[0.0] * T, advantages=[1.0] * T, mask=[1.0] * T,
        ),
        TrainingDatum(
            input_tokens=[3] * T, target_tokens=[4] * T,
            old_logprobs=[0.0] * T, advantages=[1.0] * T, mask=[1.0] * T,
        ),
    ]
    metrics = await trainer.forward_backward(batch, "importance_sampling")
    assert "loss" in metrics
    await trainer.optim_step(learning_rate=_TEST_LR)

    after = _flatten_arrays(trainer.trainable_parameters())
    # At least one trainable array must have changed.
    assert len(before) == len(after)
    changed = False
    for b, a in zip(before, after):
        if not mx.array_equal(b, a).item():
            changed = True
            break
    assert changed, "optim_step should mutate at least one trainable parameter"


async def test_optim_step_noop_when_no_pending_grads(
    tmp_path_factory,
) -> None:
    # Use a fresh trainer so we know nothing has been backpropped yet.
    adapter_dir = tmp_path_factory.mktemp("adapters_noop")
    fresh = MlxTrainingClient(
        model_name=_MODEL_NAME,  # type: ignore[arg-type]
        lora_rank=4,
        adapter_dir=adapter_dir,
        save_every=1,
        log_dir=str(adapter_dir),
        learning_rate=_TEST_LR,
    )
    before = [mx.array(a) for a in _flatten_arrays(fresh.trainable_parameters())]
    # Should not raise even though no forward_backward has been called.
    await fresh.optim_step(learning_rate=_TEST_LR)
    after = _flatten_arrays(fresh.trainable_parameters())
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert mx.array_equal(b, a).item(), (
            "optim_step with no pending grads must not change parameters"
        )


async def test_optim_step_rejects_nondefault_beta(
    trainer: MlxTrainingClient,
) -> None:
    with pytest.raises(NotImplementedError):
        await trainer.optim_step(learning_rate=_TEST_LR, beta1=0.8)


async def test_save_checkpoint_writes_versioned_and_current(
    trainer: MlxTrainingClient, tmp_path: Path,
) -> None:
    path = await trainer.save_checkpoint("test")
    snapshot = Path(path)
    assert snapshot.exists()
    assert snapshot.name == "iter_test.safetensors"
    current = snapshot.parent / "current.safetensors"
    assert current.exists()
    # current.safetensors should mirror the versioned snapshot byte-for-byte
    # (we hardlink + rename in mlx_training.save_checkpoint).
    assert current.stat().st_size == snapshot.stat().st_size


async def test_in_process_base_sampling_compute_logprobs_shape(
    trainer: MlxTrainingClient,
) -> None:
    base = _InProcessBaseSampling(trainer)
    seq = TokenSequence(tokens=[1, 2, 3, 4])
    out = await base.compute_logprobs([seq])
    assert len(out) == 1
    assert len(out[0]) == len(seq.tokens)
    # First token's logprob is conventionally 0.0.
    assert out[0][0] == 0.0


def test_with_base_only_toggles_lora_active(trainer: MlxTrainingClient) -> None:
    # Find at least one LoRA module to inspect its 'active' flag.
    from mlx_lm.tuner.lora import LoRALinear

    lora_mods = [
        m for _, m in trainer.model().named_modules() if isinstance(m, LoRALinear)
    ]
    if not lora_mods:
        pytest.skip("model has no LoRALinear modules with the configured targets")

    # Default state: active.
    assert all(getattr(m, "active", True) for m in lora_mods)
    with trainer.with_base_only():
        assert all(getattr(m, "active", True) is False for m in lora_mods)
    assert all(getattr(m, "active", True) for m in lora_mods)
