"""Sample G plans in parallel via a SamplingClient.

The trainer server calls ``sample_plans(...)`` per ``/rollout/begin``. The
orchestrator never touches the backend directly.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from textwrap import dedent

from autodiscover.backends.protocol import SamplingClient
from autodiscover.backends.types import TokenSequence
from autodiscover.completers import StopCondition, TokensWithLogprobs, TwoPhaseTokenCompleter


SYSTEM_PROMPT = dedent(
    """You are an expert at scientific discovery and are leading a team of junior researchers.
    A junior researcher on your team is working on a problem and needs your instructions on how to proceed.
    They will provide all the context needed to make progress on a hard scientific problem.
    You task is to analyze their query, think hard about the problem, and respond only with precise and concise instructions for them, in markdown, terminated with </plan>.
    Do not summarize the problem, your response should be direct and to the point.
    Do not include any other text, commentary, explanation or summary.
    They already know how to execute and evaluate the code, so you don't need to tell them how to do that.
    They are going to strictly adhere to a single pass read plan -> implement plan -> execute code -> return results so structure your plan accordingly.
    """
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
    sampling_client: SamplingClient,
    tokenizer,                 # from get_tokenizer
    renderer,                  # from get_renderer
    context: str,
    G: int,
    phase1_max_tokens: int,
    temperature: float,
    stop_condition: StopCondition,
    parent_plan_text: str | None = None,
    parent_reward: float | None = None,
) -> tuple[TokenSequence, list[SampledPlan]]:
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
