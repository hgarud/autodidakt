"""Backend-neutral data types.

These mirror the shape of Tinker's wire types but contain no
Tinker-specific machinery. Each concrete backend (Tinker, MLX-local)
adapts at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenSequence:
    """A prompt or completion as raw token IDs.

    Replaces ``tinker.ModelInput`` for code that only ever uses
    single-text-chunk inputs (the entire codebase, today). If multimodal
    support is ever needed, extend with an optional chunks field.
    """
    tokens: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)

    def append(self, token: int) -> "TokenSequence":
        return TokenSequence(tokens=self.tokens + [token])

    def extend(self, more: list[int]) -> "TokenSequence":
        return TokenSequence(tokens=self.tokens + more)


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float
    stop: list[str] | list[int] = field(default_factory=list)


@dataclass(frozen=True)
class SampledSequence:
    tokens: list[int]
    logprobs: list[float]
    # Why the sampler stopped: "stop" (hit a stop sequence), "length"
    # (max_tokens), or "eos" (model emitted EOS). Optional; backends that
    # cannot report it should set "unknown".
    stop_reason: str = "unknown"


@dataclass
class TrainingDatum:
    """One training example for forward/backward.

    Layout is consistent with how data.py currently builds tinker.Datum:
    ``input_tokens`` is the right-shifted model input (last token dropped);
    ``target_tokens`` is the left-shifted target (first token dropped).
    They are the same length T. ``old_logprobs`` are the logprobs of the
    sampled action under the policy at sampling time. ``advantages`` and
    ``mask`` are per-token (length T).
    """
    input_tokens: list[int]
    target_tokens: list[int]
    old_logprobs: list[float]
    advantages: list[float]
    mask: list[float]

    def __post_init__(self) -> None:
        T = len(self.input_tokens)
        assert (
            len(self.target_tokens) == T
            and len(self.old_logprobs) == T
            and len(self.advantages) == T
            and len(self.mask) == T
        ), "TrainingDatum fields must all have the same length"
