"""Tests for the Runner: reproducible cross-envelope variance (m1 gate)."""

from __future__ import annotations

import json

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


# --- v0.7.0 grill bug-hunt fix (amend-envelcost-v0.7.0) ---
# fix-corrupt-store-line-bricks-all-commands: a single malformed profiles.jsonl
# line (legacy pre-atomic-append partial write / hand edit / external
# corruption) must be skipped with a warning instead of aborting the whole load
# and bricking report/project. Both store read loops are guarded.

def test_load_profiles_skips_corrupt_store_line(runner):
    """fix-corrupt-store-line-bricks-all-commands: ``load_profiles`` does an
    unguarded ``json.loads(line)`` (plus ``datetime.fromisoformat`` +
    ``EnvelopeProfile(**d)``) per line, so a single malformed row raised
    JSONDecodeError/TypeError and aborted the whole load — bricking ``report``
    and ``project``. The bad row is now skipped with a visible warning; the
    good row survives. On the pre-fix code this raises instead of warning."""
    prof = runner.run_task("swe-bench-mini-001", "deepseek-native")
    runner.store_dir.mkdir(parents=True, exist_ok=True)
    store = runner.store_dir / "profiles.jsonl"
    good_line = json.dumps(prof.to_dict())
    # Un-JSON-parseable: a partial write left by a killed pre-v0.3.0 append run.
    corrupt_line = "{this is not valid json"
    store.write_text(good_line + "\n" + corrupt_line + "\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="skipping corrupt profiles.jsonl line"):
        loaded = runner.load_profiles()
    # Only the good row survives — the corrupt row was dropped, not fatal.
    assert len(loaded) == 1
    assert loaded[0].task_id == prof.task_id
    assert loaded[0].harness == prof.harness


def test_store_reread_skips_corrupt_line_and_self_heals(runner):
    """fix-corrupt-store-line-bricks-all-commands: ``_store`` re-reads the
    existing ``profiles.jsonl`` before upserting (the self-heal path at
    runner.py:194-199), so a corrupt line there crashed ``run`` on the SAME bad
    row instead of overwriting it — the user had to manually delete the store.
    The re-read now skips the corrupt line (with a warning) and the subsequent
    atomic rewrite drops it, so a fresh ``envelcost run`` self-heals the store."""
    prof = runner.run_task("swe-bench-mini-001", "deepseek-native")
    runner.store_dir.mkdir(parents=True, exist_ok=True)
    store = runner.store_dir / "profiles.jsonl"
    good_line = json.dumps(prof.to_dict())
    corrupt_line = "<<<corrupt-legacy-row>>>"
    store.write_text(good_line + "\n" + corrupt_line + "\n", encoding="utf-8")

    with pytest.warns(
        UserWarning, match="skipping corrupt profiles.jsonl line during store"
    ):
        runner.run_benchmark(
            harnesses=("deepseek-native",), task_ids=["swe-bench-mini-001"]
        )
    # The store self-healed: no corrupt line remains after the atomic rewrite.
    lines = store.read_text(encoding="utf-8").splitlines()
    assert corrupt_line not in lines
    assert all(json.loads(line) for line in lines if line.strip())
    loaded = runner.load_profiles()
    assert len(loaded) == 1
    assert loaded[0].task_id == "swe-bench-mini-001"
    assert loaded[0].harness == "deepseek-native"


# --- v0.8.0 grill bug-hunt fixes (amend-envelcost-v0.8.0) ---
# fix-package-version-constant-drift: the package __version__ constant must stay
# in lockstep with VERSION + pyproject [project].version, or `envelcost --version`
# (and the rendered demo gif) report a stale version.

def test_version_constant_in_lockstep_with_version_file_and_pyproject():
    """fix-package-version-constant-drift: envelcost.__version__ must equal the
    VERSION file string and the pyproject.toml [project].version field. The
    v0.7.0 ship bumped VERSION/pyproject/CHANGELOG to 0.7.0 but left
    envelcost/__init__.py:__version__ at 0.6.0, so `envelcost --version` printed
    0.6.0 for the whole v0.7.0 line (and assets/demo.gif showed 0.6.0 because
    docs/demo.tape runs `envelcost --version` first). This pins the three
    surfaces together so a future bump cannot silently desync the package
    constant from VERSION / pyproject again."""
    import re
    import envelcost
    from pathlib import Path

    repo_root = Path(envelcost.__file__).resolve().parent.parent
    version_file = (repo_root / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert m is not None, "pyproject.toml [project].version not found"
    pyproject_version = m.group(1)

    assert envelcost.__version__ == version_file, (
        f"__version__ ({envelcost.__version__}) != VERSION ({version_file})"
    )
    assert envelcost.__version__ == pyproject_version, (
        f"__version__ ({envelcost.__version__}) != pyproject version "
        f"({pyproject_version})"
    )


# --- v0.9.0 grill bug-hunt fixes (amend-envelcost-v0.9.0) ---

def _repo_root() -> "Path":
    import envelcost
    from pathlib import Path
    return Path(envelcost.__file__).resolve().parent.parent


def test_wheel_builds_and_contains_tasks_yaml(tmp_path):
    """fix-wheel-build-duplicate-tasks-include: the wheel target must actually
    build. pyproject declared `packages = ["envelcost"]` (which already carries
    envelcost/tasks/swe-bench-mini.yaml) AND a force-include mapping
    "envelcost/tasks" -> "envelcost/tasks", so hatchling added the same file to
    the archive twice and hard-failed every wheel build ("A second file is
    being added to the wheel archive at the same path") — release.yml's
    `python -m build` failed on EVERY tag from v0.1.0 through v0.8.0, and
    `pip install git+...` was broken for every user. CI never caught it
    because it smoke-tests via an editable install. This test drives the same
    WheelBuilder `python -m build` invokes, so the artifact-producing path is
    exercised on every run."""
    import zipfile

    hatchling = pytest.importorskip("hatchling")
    from hatchling.builders.wheel import WheelBuilder

    builder = WheelBuilder(str(_repo_root()))
    artifacts = list(builder.build(directory=str(tmp_path), versions=["standard"]))
    assert artifacts, "wheel builder produced no artifact"
    names = zipfile.ZipFile(artifacts[0]).namelist()
    # The tasks YAML — the very file the duplicate force-include collided on —
    # must be carried by the wheel via `packages = ["envelcost"]`.
    assert "envelcost/tasks/swe-bench-mini.yaml" in names


def test_docs_install_commands_resolve():
    """fix-readme-install-commands-404: no shipped doc may instruct a bare
    PyPI install (`uv tool install envelcost` / `pipx install envelcost` /
    `uvx envelcost`) — the package is not published to PyPI, so those commands
    fail at minute zero of the documented happy path. The v0.8.0 READMEs led
    with `uv tool install envelcost`; the README rewrite on main replaced the
    install sections, and this pins every doc surface (both READMEs +
    examples/quickstart.sh) against regressing to a bare PyPI instruction."""
    for rel in ("README.md", "README.en.md", "examples/quickstart.sh"):
        text = (_repo_root() / rel).read_text(encoding="utf-8")
        assert "uv tool install envelcost" not in text, f"{rel} instructs a bare PyPI install"
        assert "pipx install envelcost" not in text, f"{rel} instructs a bare PyPI install"
        assert "uvx envelcost" not in text, f"{rel} instructs a bare `uvx envelcost`"


def test_readme_recorded_demo_counts_match_computed():
    """fix-readme-roadmap-claims-drift (theme guard): the original defect —
    README claims drifting from shipped reality (an unchecked m2 box, a stale
    2.83–3.27x multiplier range) — was removed upstream by the post-v0.8.0
    README rewrite before this iteration. This guard pins the SAME honesty
    contract on the rewritten READMEs: the recorded presentation-demo token
    counts they embed must equal what the shipped ToolDef/Tokenizer actually
    compute, so a future tokenizer/template change that invalidates the
    recorded numbers fails here instead of silently misleading readers."""
    from envelcost.envelope import ToolDef
    from envelcost.tokenizer import Tokenizer

    tool = ToolDef(
        name="read",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    tok = Tokenizer()
    compact = tok.count(tool.native_block(), "openai")
    schema = tok.count(tool.openai_block(), "openai")

    for rel in ("README.md", "README.en.md"):
        text = (_repo_root() / rel).read_text(encoding="utf-8")
        assert f'"tokens_same_cl100k": {compact}' in text, (
            f"{rel} recorded compact count drifted from the computed {compact}"
        )
        assert f'"tokens_same_cl100k": {schema}' in text, (
            f"{rel} recorded json-schema count drifted from the computed {schema}"
        )
