"""Sample G plans in parallel from a Tinker SamplingClient.

The trainer server calls ``sample_plans(...)`` per ``/rollout/begin``. The
orchestrator never touches Tinker directly.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import tinker

from autodiscover.sampling.completers import StopCondition, TokensWithLogprobs, TwoPhaseTokenCompleter


SYSTEM_PROMPT = (
    "You are an expert at scientific discovery. The user will provide all "
    "the context needed to make progress on a hard scientific problem. "
    "Respond with exactly one self-contained implementation plan, in markdown, "
    "terminated with </plan>. Do not write code; describe the algorithm so "
    "another agent can implement it precisely."
)


def _strip_harmony_markup(text: str) -> str:
    """Strip GPT-OSS harmony channel framing from a plan.

    Raw renderer output for gpt-oss can include the reasoning trace, e.g.
    ``<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...``.
    Only the final-channel content should reach the implementer; the analysis
    trace bloats the prompt and confuses downstream coders.

    Also trims anything after the first ``</plan>`` (inclusive of the tag, so
    callers can detect well-formed plans), and strips ``<|channel|>`` so a
    malformed sample missing the final-channel transition doesn't leak raw
    harmony tokens downstream.
    """
    if not text:
        return text
    final_marker = "<|channel|>final<|message|>"
    idx = text.rfind(final_marker)
    if idx >= 0:
        text = text[idx + len(final_marker):]
    end = text.find("</plan>")
    if end >= 0:
        text = text[: end + len("</plan>")]
    for tok in ("<|return|>", "<|end|>", "<|start|>", "<|message|>", "<|channel|>"):
        text = text.replace(tok, "")
    return text.strip()


@dataclass
class SampledPlan:
    tokens: list[int]
    logprobs: list[float]
    plan_text: str            # already harmony-stripped
    parse_success: float      # 0.0 or 1.0; comes from renderer.parse_response


def _build_user_message(
    context: str,
    parent_plan_text: str | None,
    parent_reward: float | None,
) -> str:
    """Prepend a 'build on this' preamble when a PUCT parent is supplied.

    The seed's plan_text describes the current baseline (see discover.md
    step 2.a), so the preamble is always populated when a parent is
    provided. If the caller does not pass a parent (e.g., legacy code path
    or test), we fall back to the bare context.
    """
    if not parent_plan_text:
        return context
    reward_str = "unknown" if parent_reward is None else f"{parent_reward:.4f}"
    return (
        "## Build on a previously tried solution\n"
        f"The current best plan you should improve upon achieved reward = {reward_str}.\n\n"
        "<previous_plan>\n"
        f"{parent_plan_text}\n"
        "</previous_plan>\n\n"
        "## Original problem context\n"
        f"{context}"
    )


async def sample_plans(
    *,
    sampling_client: tinker.SamplingClient,
    tokenizer,                 # from get_tokenizer
    renderer,                  # from get_renderer
    context: str,
    G: int,
    phase1_max_tokens: int,
    temperature: float,
    stop_condition: StopCondition,
    parent_plan_text: str | None = None,
    parent_reward: float | None = None,
) -> tuple[tinker.ModelInput, list[SampledPlan]]:
    """Build the prompt from ``context`` once, then sample G plans concurrently.

    Returns ``(shared_prompt_input, plans)``. The same ``prompt_input`` is
    returned so the caller can attach it to each ``Transition.ob`` in the
    rollout store.

    If ``parent_plan_text`` is supplied, the user message is prefixed with a
    PUCT-parent preamble so the LM treats the new sample as an improvement
    over that prior plan (paper §3.2: "We select initial states using a
    PUCT-inspired rule … the policy then samples conditioned on s").
    """
    user_content = _build_user_message(context, parent_plan_text, parent_reward)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt_input = renderer.build_generation_prompt(
        messages=messages, role="assistant", prefill=None,
    )

    completer = TwoPhaseTokenCompleter(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        phase1_max_tokens=phase1_max_tokens,
        temperature=temperature,
    )

    async def _one() -> SampledPlan:
        twl: TokensWithLogprobs = await completer(prompt_input, stop_condition)
        message, parse_success = renderer.parse_response(twl.tokens)
        raw_plan = (message.get("content") or "") if isinstance(message, dict) else ""
        plan_text = _strip_harmony_markup(raw_plan)
        return SampledPlan(
            tokens=list(twl.tokens),
            # `.logprobs` raises if None — surface as 500 in caller.
            logprobs=list(twl.logprobs),
            plan_text=plan_text,
            parse_success=float(parse_success),
        )

    plans = await asyncio.gather(*[_one() for _ in range(G)])
    return prompt_input, list(plans)
