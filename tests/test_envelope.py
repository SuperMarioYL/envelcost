"""Tests for the EnvelopeProfile primitive + envelope serialization."""

from __future__ import annotations

import json

import pytest

from envelcost.config import load_tasks
from envelcost.envelope import (
    ENVELOPE_NAMES,
    EnvelopeProfile,
    ToolDef,
    all_envelopes,
    get_envelope,
    measure_envelope,
)
from envelcost.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer()


@pytest.fixture(scope="module")
def tasks():
    return load_tasks()


def test_three_envelopes_registered():
    envs = all_envelopes()
    assert set(envs) == set(ENVELOPE_NAMES)
    native = get_envelope("deepseek-native")
    assert native.is_baseline is True
    assert get_envelope("openai-shape").is_baseline is False
    assert get_envelope("claude-code-cliproxy").is_baseline is False


def test_get_envelope_unknown_raises():
    with pytest.raises(ValueError, match="unknown envelope"):
        get_envelope("nope")


def test_tooldef_serialization():
    t = ToolDef("edit", "edit a file", {"type": "object", "properties": {}})
    native = t.native_block()
    openai = t.openai_block()
    assert "[tool:edit]" in native
    block = json.loads(openai)
    assert block["type"] == "function"
    assert block["function"]["name"] == "edit"
    # OpenAI block is materially more verbose than the native inline block.
    assert len(openai) > len(native)


def test_envelope_serialize_grows_with_turns(tasks, tokenizer):
    task = tasks.tasks[0]
    native = get_envelope("deepseek-native")
    openai = get_envelope("openai-shape")
    native_text = native.serialize(task)
    openai_text = openai.serialize(task)
    # The openai-shape envelope must be larger: it repeats the verbose JSON
    # tools array and wraps every turn in a heavy tool_calls message.
    assert len(openai_text) > len(native_text)
    # Serialize is deterministic across calls (reproducible benchmark).
    assert native.serialize(task) == native_text


def test_envelope_sha_is_stable(tasks):
    task = tasks.tasks[0]
    native = get_envelope("deepseek-native")
    sha = native.envelope_sha(task)
    assert len(sha) == 16
    assert native.envelope_sha(task) == sha  # stable hash


def test_measure_native_is_baseline(tasks, tokenizer):
    task = tasks.tasks[0]
    native = get_envelope("deepseek-native")
    prof = measure_envelope(task, native, tokenizer)
    assert prof.harness == "deepseek-native"
    assert prof.multiplier_vs_baseline == 1.0
    assert prof.envelope_overhead_tokens == 0
    assert prof.tool_call_envelope_sha


def test_measure_openai_exceeds_native(tasks, tokenizer):
    task = tasks.tasks[0]
    native = get_envelope("deepseek-native")
    openai = get_envelope("openai-shape")
    native_prof = measure_envelope(task, native, tokenizer)
    openai_prof = measure_envelope(
        task, openai, tokenizer, baseline_input_tokens=native_prof.input_tokens
    )
    assert openai_prof.input_tokens >= native_prof.input_tokens
    assert openai_prof.envelope_overhead_tokens > 0
    # The cross-harness multiplier is the load-bearing m1 number: openai-shape
    # on DeepSeek must burn materially more tokens than native.
    assert openai_prof.multiplier_vs_baseline > 1.5


def test_envelope_profile_to_dict_roundtrip(tasks, tokenizer):
    task = tasks.tasks[0]
    env = get_envelope("openai-shape")
    prof = measure_envelope(
        task, env, tokenizer, baseline_input_tokens=100, quality_score=1.0
    )
    d = prof.to_dict()
    assert d["harness"] == "openai-shape"
    assert d["measured_at"] == prof.measured_at.isoformat()
    assert d["multiplier_vs_baseline"] == prof.multiplier_vs_baseline


def test_envelope_names_order():
    # deepseek-native is the baseline (first) by contract.
    assert ENVELOPE_NAMES[0] == "deepseek-native"
