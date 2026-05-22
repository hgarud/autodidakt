"""MLX local backend (Apple Silicon, air-gapped).

In-process MLX training + HTTP sampling against a separately-launched
``mlx_lm.server``. Requires the ``[mlx]`` extra to be installed.

Submodules are imported lazily via ``__getattr__`` so the package is
importable in environments without ``mlx`` / ``mlx-lm`` installed
(e.g. CI without the ``[mlx]`` extra). The heavy imports happen the
moment a consumer actually touches the symbol.
"""
from __future__ import annotations

__all__ = [
    "MlxSamplingClient",
    "MlxTrainingClient",
    "make_mlx_backend",
]


def __getattr__(name: str):
    if name == "MlxSamplingClient":
        from autodiscover.backends.mlx.sampling import MlxSamplingClient
        return MlxSamplingClient
    if name == "MlxTrainingClient":
        from autodiscover.backends.mlx.training import MlxTrainingClient
        return MlxTrainingClient
    if name == "make_mlx_backend":
        from autodiscover.backends.mlx.training import make_mlx_backend
        return make_mlx_backend
    raise AttributeError(f"module 'autodiscover.backends.mlx' has no attribute {name!r}")
