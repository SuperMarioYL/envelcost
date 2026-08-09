"""Regression tests for the v0.2.0 CLI fixes (amend-envelcost-v0.2.0).

Covers, end-to-end through the typer ``run`` command:

* fix-online-run-ignores-harnesses-filter — ``--online --harnesses <X>`` must
  bill only the selected harness(es) (and ``--task <Y>`` only the selected
  tasks), never the full 5-task x 3-envelope grid.
* fix-single-harness-run-falsifies-thesis — a single-harness run must NOT print
  "core thesis falsified" / exit 1; it surfaces a "skipping kill gate" note.

The DeepSeek API is replaced by a recording fake so no network call is made.
"""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from envelcost.cli import app
from envelcost.runner import Runner


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


class _FakeResponse:
    """Minimal stand-in for the slice of httpx.Response envelcost reads."""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"usage": {"completion_tokens": 7}}


class _FakeClient:
    """Context-manager httpx.Client fake that records every POST body.

    Recording the body lets a test assert which (harness, task) pairs were
    actually billed through the (mocked) paid DeepSeek API.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.posts: list[dict] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def post(self, url, headers=None, json=None) -> _FakeResponse:
        self.posts.append(json or {})
        return _FakeResponse()


def test_online_run_respects_harnesses_filter(cli_runner, tmp_path, monkeypatch):
    """--online --harnesses openai-shape bills ONLY openai-shape (5 tasks x 1
    harness = 5 API calls), not the full 5x3 grid (15 calls = 3x expected spend).
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake = _FakeClient()
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)

    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app,
        ["run", "--online", "--harnesses", "openai-shape", "--store", str(store)],
    )
    assert result.exit_code == 0, result.output

    # Only the selected harness was billed — 5 tasks x 1 harness.
    assert len(fake.posts) == 5
    profiles = Runner(store_dir=store).load_profiles()
    assert {p.harness for p in profiles} == {"openai-shape"}
    assert len(profiles) == 5
    # Single-harness online run -> kill gate skipped, no falsification/halt.
    assert "skipping kill gate" in result.output
    assert "falsified" not in result.output


def test_online_run_respects_task_filter(cli_runner, tmp_path, monkeypatch):
    """--task must reach run_online: --task swe-bench-mini-001 bills a single
    (task, harness) pair, proving --task is no longer a dead option online.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake = _FakeClient()
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)

    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app,
        [
            "run", "--online",
            "--harnesses", "openai-shape",
            "--task", "swe-bench-mini-001",
            "--store", str(store),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(fake.posts) == 1  # 1 task x 1 harness
    profiles = Runner(store_dir=store).load_profiles()
    assert {p.task_id for p in profiles} == {"swe-bench-mini-001"}
    assert {p.harness for p in profiles} == {"openai-shape"}


def test_online_run_respects_two_harness_filter(cli_runner, tmp_path, monkeypatch):
    """A multi-harness online filter bills exactly tasks x |harnesses| pairs,
    guarding against a regression to 'all 3 envelopes'."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake = _FakeClient()
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)

    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app,
        [
            "run", "--online",
            "--harnesses", "deepseek-native,openai-shape",
            "--store", str(store),
        ],
    )
    assert result.exit_code == 0, result.output
    # 5 tasks x 2 harnesses — NOT 5 x 3.
    assert len(fake.posts) == 10
    profiles = Runner(store_dir=store).load_profiles()
    assert {p.harness for p in profiles} == {"deepseek-native", "openai-shape"}


def test_single_harness_offline_run_does_not_halt(cli_runner, tmp_path):
    """The exact regression: `envelcost run --harnesses deepseek-native`
    previously printed 'core thesis falsified' and exit(1); it must now exit 0
    with the 'skipping kill gate' note (cross-variance gate is unevaluable)."""
    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app, ["run", "--harnesses", "deepseek-native", "--store", str(store)]
    )
    assert result.exit_code == 0, result.output
    assert "skipping kill gate" in result.output
    assert "falsified" not in result.output
    assert "SKIPPED" in result.output


def test_online_bad_task_surfaces_before_billing(cli_runner, tmp_path, monkeypatch):
    """A typoed --task raises BadParameter (exit 2) BEFORE any API call is
    billed — the resolver runs before run_online touches the network."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake = _FakeClient()
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)

    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app,
        ["run", "--online", "--task", "nope-not-a-task", "--store", str(store)],
    )
    assert result.exit_code == 2  # typer.BadParameter -> usage error
    assert len(fake.posts) == 0  # nothing billed


def test_offline_run_respects_task_filter(cli_runner, tmp_path):
    """fix-offline-run-ignores-task-filter: the DEFAULT (offline) `run` must
    honor --task. `envelcost run --task swe-bench-mini-001` (no --online) must
    measure only that one task — not all 5. Previously the offline branch called
    run_benchmark with no task_ids, so --task was a dead option in offline mode
    (the v0.2.0 fix only closed the online half)."""
    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app,
        [
            "run",
            "--harnesses", "openai-shape",
            "--task", "swe-bench-mini-001",
            "--store", str(store),
        ],
    )
    assert result.exit_code == 0, result.output
    profiles = Runner(store_dir=store).load_profiles()
    # 1 task x 1 harness — NOT 5 x 1.
    assert {p.task_id for p in profiles} == {"swe-bench-mini-001"}
    assert {p.harness for p in profiles} == {"openai-shape"}
    assert len(profiles) == 1
    # single-harness offline run -> kill gate skipped, no falsification/halt.
    assert "skipping kill gate" in result.output
    assert "falsified" not in result.output


def test_offline_run_default_task_runs_full_set(cli_runner, tmp_path):
    """Guard: the default --task (swe-bench-mini, the shared prefix of all 5
    canonical tasks) still resolves to the full 5-task set offline — the
    fix-offline-run-ignores-task-filter change must not narrow the default."""
    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app, ["run", "--harnesses", "openai-shape", "--store", str(store)]
    )
    assert result.exit_code == 0, result.output
    profiles = Runner(store_dir=store).load_profiles()
    assert {p.harness for p in profiles} == {"openai-shape"}
    assert len({p.task_id for p in profiles}) == 5


def test_offline_bad_task_surfaces_before_run(cli_runner, tmp_path):
    """A typoed --task raises BadParameter (exit 2) in offline mode too — the
    resolver now runs in the offline path, so a bad --task fails fast."""
    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app,
        ["run", "--task", "nope-not-a-task", "--store", str(store)],
    )
    assert result.exit_code == 2  # typer.BadParameter -> usage error
