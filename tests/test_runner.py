"""Tests for the Runner: reproducible cross-envelope variance (m1 gate)."""

from __future__ import annotations

import pytest

from envelcost.runner import (
    DEFAULT_HARNESSES,
    VARIANCE_FLOOR,
    VARIANCE_GATE_MULTIPLIER,
    Runner,
)


@pytest.fixture
def runner(tmp_path):
    # Use a per-test store so the repo never carries benchmark state in git.
    return Runner(store_dir=tmp_path / ".envelcost")


def test_loads_five_canonical_tasks(runner):
    assert len(runner.tasks.tasks) == 5
    ids = [t.task_id for t in runner.tasks.tasks]
    assert all(i.startswith("swe-bench-mini-") for i in ids)


def test_run_task_native_is_baseline(runner):
    prof = runner.run_task("swe-bench-mini-001", "deepseek-native")
    assert prof.harness == "deepseek-native"
    assert prof.multiplier_vs_baseline == 1.0
    assert prof.envelope_overhead_tokens == 0


def test_run_task_openai_exceeds_two_x(runner):
    """The load-bearing m1 assertion: openai-shape > 2x native on this task."""
    prof = runner.run_task("swe-bench-mini-001", "openai-shape")
    assert prof.multiplier_vs_baseline > VARIANCE_GATE_MULTIPLIER


def test_run_benchmark_produces_grid(runner):
    profiles = runner.run_benchmark(harnesses=DEFAULT_HARNESSES)
    # 5 tasks × 2 harnesses.
    assert len(profiles) == 10
    harnesses = {p.harness for p in profiles}
    assert harnesses == set(DEFAULT_HARNESSES)
    tasks = {p.task_id for p in profiles}
    assert len(tasks) == 5


def test_variance_report_m1_gate_passes(runner):
    """mvp_plan §8 kill #1: >2x on >=3/5 tasks (done bar); >=1.5x floor held.

    This is the falsifiable core of the product — if this test goes red, the
    whole thesis is falsified and the build halts (see DRIFT_REPORT protocol).
    """
    profiles = runner.run_benchmark(harnesses=DEFAULT_HARNESSES)
    vr = runner.variance_report(profiles)
    assert vr.task_count == 5
    assert vr.tasks_above_gate >= 3, (
        f"m1 gate FAILED: only {vr.tasks_above_gate}/5 tasks above 2.0x — "
        "core thesis falsified"
    )
    assert vr.floor_passed is True
    assert vr.gate_passed is True


def test_variance_report_per_task_keys(runner):
    profiles = runner.run_benchmark(harnesses=DEFAULT_HARNESSES)
    vr = runner.variance_report(profiles)
    for tid, mults in vr.per_task_multiplier.items():
        assert "deepseek-native" in mults
        assert "openai-shape" in mults
        assert mults["deepseek-native"] == 1.0
        assert mults["openai-shape"] >= VARIANCE_FLOOR


def test_profiles_persist_to_store(runner):
    profiles = runner.run_benchmark(harnesses=DEFAULT_HARNESSES)
    loaded = runner.load_profiles()
    assert len(loaded) == len(profiles)
    assert {p.task_id for p in loaded} == {p.task_id for p in profiles}


def test_three_envelope_benchmark_includes_claude_code(runner):
    harnesses = ("deepseek-native", "openai-shape", "claude-code-cliproxy")
    profiles = runner.run_benchmark(harnesses=harnesses)
    assert len(profiles) == 15
    cc = [p for p in profiles if p.harness == "claude-code-cliproxy"]
    assert cc
    # claude-code-cliproxy is strictly heavier than openai-shape (extra scaffold).
    openai_map = {
        p.task_id: p.multiplier_vs_baseline
        for p in profiles
        if p.harness == "openai-shape"
    }
    for p in cc:
        assert p.multiplier_vs_baseline >= openai_map[p.task_id]
