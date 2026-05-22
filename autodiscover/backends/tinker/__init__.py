"""Tinker backend (hosted Thinking Machines service).

Wraps the Tinker SDK so it conforms to ``autodiscover.backends.protocol``.
Requires the ``[tinker]`` extra to be installed.

Submodules are imported lazily via ``__getattr__`` so the package is
importable in environments without ``tinker`` installed (e.g. an
air-gapped install using only the ``[mlx]`` extra). The heavy imports
happen the moment a consumer actually touches the symbol.
"""
from __future__ import annotations

__all__ = [
    "TinkerSamplingClient",
    "TinkerTrainingClient",
    "make_tinker_backend",
]


def __getattr__(name: str):
    if name == "TinkerSamplingClient":
        from autodiscover.backends.tinker.sampling import TinkerSamplingClient
        return TinkerSamplingClient
    if name == "TinkerTrainingClient":
        from autodiscover.backends.tinker.training import TinkerTrainingClient
        return TinkerTrainingClient
    if name == "make_tinker_backend":
        from autodiscover.backends.tinker.training import make_tinker_backend
        return make_tinker_backend
    raise AttributeError(f"module 'autodiscover.backends.tinker' has no attribute {name!r}")
