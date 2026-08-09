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


def test_single_harness_variance_report_skips_kill_gate(runner):
    """fix-single-harness-run-falsifies-thesis: a single-harness (baseline-only)
    run cannot evaluate cross-envelope variance, so the m1 kill gate must be
    SKIPPED — not tripped. Previously peak=1.0 for every task made floor_passed
    False and the CLI falsified the thesis + exit(1)."""
    profiles = runner.run_benchmark(harnesses=("deepseek-native",))
    assert len(profiles) == 5
    vr = runner.variance_report(profiles)
    assert vr.harnesses == ("deepseek-native",)
    assert len(vr.harnesses) < 2
    assert vr.floor_evaluable is False
    # gate is held (skipped), NOT broken — the thesis is not falsified.
    assert vr.floor_passed is True
    # to_dict surfaces the new flag so reports/JSON stay self-describing.
    assert vr.to_dict()["floor_evaluable"] is False


def test_two_harness_variance_report_evaluates_kill_gate(runner):
    """With >=2 harnesses the kill gate is evaluable and the m1 floor holds
    (openai-shape > 1.5x on every task) — guarding against the skip logic
    accidentally neutering the real gate."""
    profiles = runner.run_benchmark(harnesses=DEFAULT_HARNESSES)
    vr = runner.variance_report(profiles)
    assert len(vr.harnesses) == 2
    assert vr.floor_evaluable is True
    assert vr.floor_passed is True
    assert vr.gate_passed is True


def test_store_upsert_no_duplicates_across_runs(runner):
    """fix-store-appends-duplicate-profiles: repeated run_benchmark must NOT
    append duplicate rows. Previously _store opened profiles.jsonl in append
    mode with no dedup, so a second run doubled the row count read by
    report/project (and the file grew without bound)."""
    runner.run_benchmark(harnesses=DEFAULT_HARNESSES)
    runner.run_benchmark(harnesses=DEFAULT_HARNESSES)  # same grid, second time
    loaded = runner.load_profiles()
    # 5 tasks x 2 harnesses = 10, NOT 20 — upsert by (task_id, harness), last-wins.
    assert len(loaded) == 10
    keys = {(p.task_id, p.harness) for p in loaded}
    assert len(keys) == 10  # no duplicate (task_id, harness) keys


def test_store_upsert_replaces_stale_subset(runner):
    """fix-store-appends-duplicate-profiles: a later run with a different
    --harnesses subset must not leave stale profiles from the prior run mixed
    in as duplicates. Upsert keeps the latest measurement per (task_id,
    harness); re-measuring the same (task_id, harness) replaces the old row in
    place rather than appending a second copy."""
    runner.run_benchmark(harnesses=DEFAULT_HARNESSES)  # 10 rows: native + openai
    first_openai = {
        p.task_id: p.multiplier_vs_baseline
        for p in runner.load_profiles()
        if p.harness == "openai-shape"
    }
    # Second run: only openai-shape — its rows are refreshed in place; native
    # rows persist. No duplicate (task_id, harness) keys.
    runner.run_benchmark(harnesses=("openai-shape",))
    loaded = runner.load_profiles()
    keys = {(p.task_id, p.harness) for p in loaded}
    assert len(loaded) == 10  # native retained + openai-shape re-measured, NOT 15
    assert len(keys) == 10
    # openai-shape rows still hold the same deterministic offline values
    # (re-measure replaced the row, did not alter the value).
    for p in loaded:
        if p.harness == "openai-shape":
            assert p.multiplier_vs_baseline == first_openai[p.task_id]
