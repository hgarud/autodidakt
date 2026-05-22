"""Backend abstractions.

Two backends are provided:

* ``autodiscover.backends.tinker`` — wraps the Tinker hosted service.
* ``autodiscover.backends.mlx``    — local-only, talks to mlx_lm.server
  + in-process MLX.

Each subpackage exposes ``{Sampling,Training}Client`` adapters and a
``make_{tinker,mlx}_backend`` factory. Pick at runtime via ``--backend``
on ``autodiscover.cli.trainer_server``.
"""
from autodiscover.backends.protocol import SamplingClient, TrainingClient
from autodiscover.backends.types import (
    SampledSequence,
    SamplingParams,
    TokenSequence,
    TrainingDatum,
)

__all__ = [
    "SamplingClient",
    "TrainingClient",
    "SampledSequence",
    "SamplingParams",
    "TokenSequence",
    "TrainingDatum",
]
