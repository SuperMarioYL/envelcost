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


def test_two_harness_non_baseline_run_breaks_kill_gate(runner):
    """fix-kill-gate-holds-when-baseline-absent: a valid 2-harness run that
    EXCLUDES the deepseek-native baseline (openai-shape + claude-code-cliproxy,
    both ~3x but within ~1.07x of each other) must report the m1 kill floor
    BROKEN — not HELD. The old gate used ``peak = max(mults.values())``, which is
    only the true cross-harness ratio when the 1.0x baseline is present (then
    min=1.0 and max/min == max). Without the baseline, peak ~3.0 >= 1.5 on every
    task so the gate silently held and the build did not halt — directly
    violating mvp_plan §8 kill #1 (the real spread ~1.07x < 1.5x on all 5 tasks
    => thesis falsified). The ``spread = max/min`` fix makes the gate correct
    regardless of whether the baseline harness was measured."""
    harnesses = ("openai-shape", "claude-code-cliproxy")
    profiles = runner.run_benchmark(harnesses=harnesses)
    assert len(profiles) == 10  # 5 tasks x 2 harnesses
    vr = runner.variance_report(profiles)
    assert set(vr.harnesses) == set(harnesses)
    assert "deepseek-native" not in vr.harnesses  # baseline deliberately absent
    # 2 harnesses => the cross-variance gate IS evaluable (not skipped).
    assert vr.floor_evaluable is True
    # The true cross-harness spread (max/min) is ~1.07x on every task — far
    # below the 1.5x kill floor, so §8 kill #1 trips. (These spread values are
    # computed from the same mults dict pre- and post-fix; the bug was only in
    # how the gate consumed them.)
    for mults in vr.per_task_multiplier.values():
        spread = max(mults.values()) / min(mults.values())
        assert spread < VARIANCE_FLOOR
    # below 1.5x on >=3/5 tasks => kill floor BROKEN, gate (>=3/5 above 2x) NOT
    # passed. On the pre-fix max-only code above_floor=5 and floor_passed=True,
    # so these three assertions are the load-bearing regression checks.
    assert vr.tasks_above_floor < 3
    assert vr.floor_passed is False  # gate BROKEN — halt
    assert vr.gate_passed is False


# --- v0.6.0 grill bug-hunt fixes (amend-envelcost-v0.6.0) ---
# fix-online-usage-shape-silent-zero + fix-tokenizer-trust-remote-code-unpinned.

import httpx  # noqa: E402


class _FakeResponse:
    """Minimal httpx.Response stand-in for the online ``run_online`` path."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _RecordingClient:
    """Context-manager httpx.Client fake returning a fixed JSON payload.

    Records the number of POST calls so a test can assert exactly how many
    (harness, task) pairs were billed through the (mocked) paid API.
    """

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    def __enter__(self) -> "_RecordingClient":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def post(self, url, headers=None, json=None) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(self._payload)


@pytest.mark.parametrize(
    "bad_usage",
    [
        {"output_tokens": 7},   # differently-keyed field
        {},                     # usage present but empty
        None,                   # usage key absent entirely
        "not-a-dict",           # usage is the wrong type
    ],
    ids=["different-key", "empty", "absent", "wrong-type"],
)
def test_online_usage_shape_mismatch_warns_not_silent_zero(
    runner, monkeypatch, bad_usage
):
    """fix-online-usage-shape-silent-zero: a 200 response whose ``usage``
    block is missing or uses a non-standard key must emit a VISIBLE warning
    instead of silently storing ``output_tokens=0``. The module docstring
    promises "a clear warning if the response shape differs"; the old
    ``usage = r.json().get("usage", {})`` /
    ``int(usage.get("completion_tokens", 0))`` did NO shape check, so any 200
    with a missing/differently-keyed usage block silently stored
    ``output_tokens=0`` with no warning — the online run looked successful but
    the output-token data was silently wrong."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    payload = {} if bad_usage is None else {"usage": bad_usage}
    fake = _RecordingClient(payload)
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)

    with pytest.warns(UserWarning, match="non-standard shape"):
        profiles = runner.run_online(
            harnesses=("openai-shape",),
            task_ids=["swe-bench-mini-001"],
        )
    # Exactly one (harness, task) pair was billed through the mocked API.
    assert fake.calls == 1
    assert len(profiles) == 1
    # output_tokens treated as unknown (0) — but WITH a visible warning, NOT a
    # silent zero. The offline measurement is still carried so the profile row
    # is complete.
    assert profiles[0].output_tokens == 0


def test_online_usage_shape_standard_no_warning(runner, monkeypatch):
    """fix-online-usage-shape-silent-zero guard: a 200 response with the
    STANDARD ``usage.completion_tokens`` shape reads ``output_tokens``
    normally and emits NO shape-mismatch warning — the fix must not cry wolf on
    the happy path. The carried online value overrides the offline 0."""
    import warnings as _w

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake = _RecordingClient({"usage": {"completion_tokens": 42}})
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        profiles = runner.run_online(
            harnesses=("openai-shape",),
            task_ids=["swe-bench-mini-001"],
        )
    shape_warnings = [str(w.message) for w in caught]
    assert not any("non-standard shape" in m for m in shape_warnings), shape_warnings
    assert fake.calls == 1
    assert len(profiles) == 1
    # The standard-shape happy path carries the real online output_tokens.
    assert profiles[0].output_tokens == 42


def test_deepseek_tokenizer_pins_revision_and_no_trust_remote_code(monkeypatch):
    """fix-tokenizer-trust-remote-code-unpinned: ``AutoTokenizer.from_pretrained``
    must be called with a PINNED ``revision=`` (a known-good commit SHA, not the
    unpinned ``main`` tip) and ``trust_remote_code=False`` so a later
    compromised / MitM'd repo revision cannot execute arbitrary
    ``tokenization_*.py`` / ``modeling_*.py`` code at tokenizer-load time.
    Mocked — no network / torch needed."""
    import re
    import sys
    from unittest.mock import MagicMock

    fake_transformers = MagicMock()
    captured: dict = {}

    def _from_pretrained(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured["kwargs"] = kwargs
        return MagicMock(name="deepseek_tokenizer")

    fake_transformers.AutoTokenizer.from_pretrained = _from_pretrained
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from envelcost.tokenizer import (
        Tokenizer,
        _DEEPSEEK_TOKENIZER_REPO,
        _DEEPSEEK_TOKENIZER_REVISION,
    )

    tok = Tokenizer()
    tok._ensure_deepseek()

    # The load succeeded against the (mocked) pinned snapshot.
    assert tok._deepseek_ok is True
    # The pinned repo + a 40-char commit SHA (not "main" / None / unpinned).
    assert captured["repo_id"] == _DEEPSEEK_TOKENIZER_REPO
    assert re.fullmatch(r"[0-9a-f]{40}", _DEEPSEEK_TOKENIZER_REVISION)
    assert captured["kwargs"].get("revision") == _DEEPSEEK_TOKENIZER_REVISION
    assert captured["kwargs"].get("revision") not in (None, "main")
    # Custom code in the repo is NOT executed — the standard fast tokenizer
    # needs no trust_remote_code.
    assert captured["kwargs"].get("trust_remote_code") is False


def test_deepseek_tokenizer_falls_back_when_pinned_load_fails(monkeypatch):
    """fix-tokenizer-trust-remote-code-unpinned: if the pinned tokenizer cannot
    be loaded safely (offline / HF down / revision unavailable), the
    deterministic ``_approx_count`` fallback keeps the m1 gate reproducible —
    no crash, no remote-code execution attempt beyond the pinned (non-custom)
    load. Proves the real tokenizer is non-load-bearing for the m1 gate."""
    import sys
    from unittest.mock import MagicMock

    fake_transformers = MagicMock()
    attempted: dict = {}

    def _boom(repo_id, **kwargs):
        attempted["repo_id"] = repo_id
        attempted["kwargs"] = kwargs
        raise RuntimeError("network unavailable / pinned revision not found")

    fake_transformers.AutoTokenizer.from_pretrained = _boom
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from envelcost.tokenizer import (
        Tokenizer,
        _DEEPSEEK_CHARS_PER_TOKEN,
        _DEEPSEEK_TOKENIZER_REPO,
        _DEEPSEEK_TOKENIZER_REVISION,
    )

    tok = Tokenizer()
    tok._ensure_deepseek()

    # The safe (pinned, no-trust) load was attempted before the fallback engaged.
    assert attempted["repo_id"] == _DEEPSEEK_TOKENIZER_REPO
    assert attempted["kwargs"].get("revision") == _DEEPSEEK_TOKENIZER_REVISION
    assert attempted["kwargs"].get("trust_remote_code") is False
    # Load failed safely -> fallback engaged, no crash, no remote code run.
    assert tok._deepseek_ok is False
    assert tok._deepseek_enc is None
    assert tok.deepseek_available is False
    # The deterministic char-ratio approximation is used instead.
    text = "hello world " * 10
    expected = max(int(round(len(text) / _DEEPSEEK_CHARS_PER_TOKEN)), 1)
    assert tok._count_deepseek(text) == expected
