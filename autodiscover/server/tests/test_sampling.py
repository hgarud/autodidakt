"""Unit tests for autodiscover.server.sampling helpers."""
from __future__ import annotations

from autodiscover.server.sampling import _build_user_message, _strip_harmony_markup


def test_strip_harmony_extracts_final_channel():
    text = (
        "<|channel|>analysis<|message|>think<|end|>"
        "<|start|>assistant<|channel|>final<|message|>plan<|return|>"
    )
    assert _strip_harmony_markup(text) == "plan"


def test_strip_harmony_empty():
    assert _strip_harmony_markup("") == ""


def test_strip_harmony_passthrough():
    assert _strip_harmony_markup("plain text") == "plain text"


def test_strip_harmony_only_strips_residual_markers_when_no_final_channel():
    # No final-channel marker; other markers still scrubbed and trimmed.
    assert _strip_harmony_markup("<|end|>plan<|return|>") == "plan"


def test_strip_harmony_trims_trailing_junk_after_plan_tag():
    # Model kept generating past </plan>; everything after the first </plan>
    # (including the tag's trailing fragment) must be discarded.
    text = (
        "<|channel|>analysis<|message|>think<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "PLAN BODY</plan>\n\nP.S. extra trailing chatter<|return|>"
    )
    assert _strip_harmony_markup(text) == "PLAN BODY</plan>"


def test_strip_harmony_malformed_analysis_only_has_no_channel_token():
    # Reproduces the bug case: stop fired inside the analysis channel, so the
    # final-channel marker never appeared. Must not leak <|channel|> downstream.
    text = (
        "<|channel|>analysisThe user wants us to produce exactly one "
        "self-contained implementation plan in markdown, terminated with "
        "</plan>."
    )
    out = _strip_harmony_markup(text)
    assert "<|channel|>" not in out
    assert "<|message|>" not in out
    # </plan> trim still applies even when the final-channel marker is absent.
    assert out.endswith("</plan>")


def test_build_user_message_no_parent_returns_context():
    assert _build_user_message("CTX", None, None) == "CTX"
    assert _build_user_message("CTX", "", 0.5) == "CTX"


def test_build_user_message_with_parent_includes_preamble():
    msg = _build_user_message("CTX", "prior plan body", 0.7321)
    assert "## Build on a previously tried solution" in msg
    assert "reward = 0.7321" in msg
    assert "<previous_plan>\nprior plan body\n</previous_plan>" in msg
    assert msg.endswith("CTX")


def test_build_user_message_handles_unknown_reward():
    msg = _build_user_message("CTX", "prior", None)
    assert "reward = unknown" in msg
