"""Two-phase token completer (backend-agnostic).

Operates over raw tokens via the SamplingClient protocol. No backend
imports here.
"""
from dataclasses import dataclass
from typing import TypeAlias

from autodiscover.backends.protocol import SamplingClient
from autodiscover.backends.types import SamplingParams, TokenSequence
from autodiscover.tokenizer import Tokenizer

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None
    maybe_mask: list[float] | None = None  # 1.0 = train, 0.0 = don't train

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs

    @property
    def mask(self) -> list[float]:
        if self.maybe_mask is None:
            return [1.0] * len(self.tokens)
        return self.maybe_mask


class TokenCompleter:
    async def __call__(
        self, model_input: TokenSequence, stop: StopCondition,
    ) -> TokensWithLogprobs:
        raise NotImplementedError


@dataclass
class TwoPhaseTokenCompleter(TokenCompleter):
    """Two-phase completer for gpt-oss: if Phase 1 exhausts tokens without
    stop, Phase 2 forces a final answer. Uses the context window fully."""

    sampling_client: SamplingClient
    tokenizer: Tokenizer
    phase1_max_tokens: int          # total budget (prompt + output)
    temperature: float = 1.0
    context_window: int = 32768
    context_buffer: int = 50

    PHASE2_PREFILL = (
        "\n\n... okay, I am out of thinking tokens. "
        "I need to send my final message now."
    )
    GPTOSS_FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
    GPTOSS_FINAL_CHANNEL_INDICATOR = "<|channel|>final<|message|>"

    def _hit_stop_sequence(
        self, tokens: list[int], stop: StopCondition,
    ) -> bool:
        if not tokens:
            return False
        for s in stop:
            if isinstance(s, int):
                if tokens[-1] == s:
                    return True
            else:
                stop_tokens = self.tokenizer.encode(s, add_special_tokens=False)
                if (
                    len(stop_tokens) <= len(tokens)
                    and tokens[-len(stop_tokens):] == stop_tokens
                ):
                    return True
        return False

    def _contains_subsequence(
        self, tokens: list[int], pattern: str,
    ) -> bool:
        pattern_tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
        if len(pattern_tokens) > len(tokens):
            return False
        for i in range(len(tokens) - len(pattern_tokens) + 1):
            if tokens[i:i + len(pattern_tokens)] == pattern_tokens:
                return True
        return False

    async def __call__(
        self, prompt: TokenSequence, stop: StopCondition,
    ) -> TokensWithLogprobs:
        prompt_length = prompt.length
        phase1_max = self.phase1_max_tokens - prompt_length
        if phase1_max <= 0:
            raise ValueError(
                f"Prompt length {prompt_length} exceeds phase1_max_tokens "
                f"{self.phase1_max_tokens}.",
            )

        phase1 = await self.sampling_client.sample(
            prompt=prompt,
            params=SamplingParams(
                stop=stop, max_tokens=phase1_max, temperature=self.temperature,
            ),
            num_samples=1,
        )
        phase1_tokens = phase1[0].tokens
        phase1_logprobs = phase1[0].logprobs

        # Hit stop or naturally short? Done.
        if (
            self._hit_stop_sequence(phase1_tokens, stop)
            or len(phase1_tokens) < phase1_max
        ):
            return TokensWithLogprobs(
                tokens=phase1_tokens, maybe_logprobs=phase1_logprobs,
            )

        # Already in final channel? Continue without prefill.
        if self._contains_subsequence(
            phase1_tokens, self.GPTOSS_FINAL_CHANNEL_INDICATOR,
        ):
            new_prompt = prompt.extend(phase1_tokens)
            phase2_max = (
                self.context_window - prompt_length - len(phase1_tokens)
                - self.context_buffer
            )
            if phase2_max <= 0:
                return TokensWithLogprobs(
                    tokens=phase1_tokens, maybe_logprobs=phase1_logprobs,
                )
            phase2 = await self.sampling_client.sample(
                prompt=new_prompt,
                params=SamplingParams(
                    stop=stop, max_tokens=phase2_max,
                    temperature=self.temperature,
                ),
                num_samples=1,
            )
            return TokensWithLogprobs(
                tokens=phase1_tokens + phase2[0].tokens,
                maybe_logprobs=phase1_logprobs + phase2[0].logprobs,
            )

        # Need a prefill to transition into the final channel.
        end_token_seq = self.tokenizer.encode("<|end|>", add_special_tokens=False)
        ends_with_end = (
            len(end_token_seq) <= len(phase1_tokens)
            and phase1_tokens[-len(end_token_seq):] == end_token_seq
        )
        if ends_with_end:
            prefill_text = (
                self.PHASE2_PREFILL
                + "<|start|>assistant<|channel|>final<|message|>"
            )
        else:
            prefill_text = self.PHASE2_PREFILL + self.GPTOSS_FINAL_MARKER
        prefill_tokens = self.tokenizer.encode(prefill_text, add_special_tokens=False)

        new_prompt = prompt.extend(phase1_tokens).extend(prefill_tokens)
        phase2_max = (
            self.context_window - prompt_length - len(phase1_tokens)
            - len(prefill_tokens) - self.context_buffer
        )
        if phase2_max <= 0:
            return TokensWithLogprobs(
                tokens=phase1_tokens + prefill_tokens,
                maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens),
                maybe_mask=(
                    [1.0] * len(phase1_tokens)
                    + [0.0] * len(prefill_tokens)
                ),
            )

        phase2 = await self.sampling_client.sample(
            prompt=new_prompt,
            params=SamplingParams(
                stop=stop, max_tokens=phase2_max,
                temperature=self.temperature,
            ),
            num_samples=1,
        )
        return TokensWithLogprobs(
            tokens=phase1_tokens + prefill_tokens + phase2[0].tokens,
            maybe_logprobs=(
                phase1_logprobs + [0.0] * len(prefill_tokens)
                + phase2[0].logprobs
            ),
            maybe_mask=(
                [1.0] * len(phase1_tokens)
                + [0.0] * len(prefill_tokens)
                + [1.0] * len(phase2[0].tokens)
            ),
        )
