"""Regression tests for the v0.2.0 + v0.4.0 CLI fixes
(amend-envelcost-v0.2.0, amend-envelcost-v0.4.0).

Covers, end-to-end through the typer ``run`` command:

* fix-online-run-ignores-harnesses-filter — ``--online --harnesses <X>`` must
  bill only the selected harness(es) (and ``--task <Y>`` only the selected
  tasks), never the full 5-task x 3-envelope grid.
* fix-single-harness-run-falsifies-thesis — a single-harness run must NOT print
  "core thesis falsified" / exit 1; it surfaces a "skipping kill gate" note.
* fix-task-resolver-silently-drops-nonmatching-list-parts — a ``--task``
  comma-list part that matches no task id must raise ``BadParameter`` before
  any measurement or paid API call, instead of silently running the matching
  subset.

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


def test_resolve_task_ids_raises_on_nonmatching_list_part():
    """fix-task-resolver-silently-drops-nonmatching-list-parts: a comma-list
    --task part that matches no task id must raise BadParameter — not be
    silently dropped, running only the matching subset. ``--task
    swe-bench-mini-001,002`` previously resolved to just
    ``["swe-bench-mini-001"]`` because ``"002"`` is not a prefix of any shipped
    id (they are ``swe-bench-mini-002``, …), and the aggregate-only ``if not
    selected`` guard never fired (the first part DID match)."""
    import typer

    from envelcost.cli import _resolve_task_ids

    all_ids = [f"swe-bench-mini-{i:03d}" for i in range(1, 6)]
    # First part matches; "002" matches nothing (ids are swe-bench-mini-002, not 002).
    with pytest.raises(typer.BadParameter) as exc_info:
        _resolve_task_ids(all_ids, "swe-bench-mini-001,002")
    assert "002" in str(exc_info.value)


def test_offline_task_partial_match_surfaces_before_run(cli_runner, tmp_path):
    """fix-task-resolver-silently-drops-nonmatching-list-parts: a comma-list
    --task where one part matches and another matches nothing must raise
    BadParameter (exit 2) before any measurement — not silently run only the
    subset that matched. ``envelcost run --task swe-bench-mini-001,002``
    previously measured only swe-bench-mini-001 and exited 0, so ``--online``
    would bill an incomplete grid with no error."""
    store = tmp_path / ".envelcost"
    result = cli_runner.invoke(
        app,
        ["run", "--task", "swe-bench-mini-001,002", "--store", str(store)],
    )
    assert result.exit_code == 2  # typer.BadParameter -> usage error
    assert "002" in result.output  # the offending part is named in the message
    # The resolver raises before run_benchmark, so nothing was measured/stored.
    assert not (store / "profiles.jsonl").exists()


def test_project_partial_store_does_not_falsify_baseline_fit(cli_runner, tmp_path):
    """fix-project-summary-false-not-fit (end-to-end): the natural single-envelope
    flow `envelcost run --harnesses openai-shape` (the v0.2.0 --harnesses fix
    made it first-class) leaves the deepseek-native baseline unmeasured. The
    `project` summary previously derived harnesses only from measured profiles,
    so `fits_harness('deepseek-native')` returned None and the summary printed
    'deepseek-native does NOT fit' — a FALSE verdict for the 1.0x baseline that
    always fits first. It must now project the baseline at 1.0x parity and
    report that it fits.
    """
    store = tmp_path / ".envelcost"
    run_result = cli_runner.invoke(
        app,
        ["run", "--harnesses", "openai-shape", "--store", str(store)],
    )
    assert run_result.exit_code == 0, run_result.output
    # Sanity: the store really is partial (only openai-shape was measured).
    profiles = Runner(store_dir=store).load_profiles()
    assert {p.harness for p in profiles} == {"openai-shape"}

    result = cli_runner.invoke(
        app,
        ["project", "--gpus", "8xH100", "--seats", "50", "--store", str(store)],
    )
    assert result.exit_code == 0, result.output
    # The false verdict must be gone — the baseline fits (1.0x parity).
    assert "deepseek-native does NOT fit" not in result.output
    assert "deepseek-native fits" in result.output


def test_project_bad_gpu_count_surfaces_clean_error(cli_runner, tmp_path):
    """fix-parse-gpu-spec-traceback-and-zero-count (end-to-end): a zero/typoed
    --gpus count must exit non-zero with a clean human message — NOT a Python
    traceback and NOT a silently bogus 0/negative-capex projection.
    """
    store = tmp_path / ".envelcost"
    store.mkdir(parents=True, exist_ok=True)
    for spec in ("0xH100", "-2xH100", "H100x8"):
        result = cli_runner.invoke(
            app,
            ["project", "--gpus", spec, "--seats", "50", "--store", str(store)],
        )
        assert result.exit_code != 0, result.output
        # A clean human message, not a raw traceback to stderr.
        assert "Traceback" not in result.output


def test_project_json_machine_readable(cli_runner, tmp_path):
    """feat-project-json-output: `envelcost project --json` prints
    Projection.to_dict() as JSON to stdout and skips the text table + summary
    line. Mirrors the JSON output `report` already ships; closes the
    inconsistency that project (the m3 commercial hook) printed only a text
    table, blocking CI/scripting integration. No new deps — to_dict() already
    exists and is tested for serializability (test_projection_to_dict_serializable).
    """
    store = tmp_path / ".envelcost"
    # Populate a 2-harness store so the projection has the product-story shape:
    # deepseek-native fits 50 seats, openai-shape (4x) does not.
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--harnesses", "deepseek-native,openai-shape",
            "--store", str(store),
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    result = cli_runner.invoke(
        app,
        ["project", "--json", "--gpus", "8xH100", "--seats", "50", "--store", str(store)],
    )
    assert result.exit_code == 0, result.output
    import json as _json
    parsed = _json.loads(result.output)
    assert parsed["gpu_count"] == 8
    assert parsed["target_seats"] == 50
    rows = {r["harness"]: r for r in parsed["rows"]}
    assert "deepseek-native" in rows
    assert rows["deepseek-native"]["fits_target"] is True
    assert "openai-shape" in rows
    assert rows["openai-shape"]["fits_target"] is False
    # --json path skips the human text table + summary line.
    assert "does NOT fit" not in result.output
    assert "total capex" not in result.output


def test_project_default_path_unchanged_without_json(cli_runner, tmp_path):
    """feat-project-json-output guard: the default (no --json) `project` path
    is unchanged — it still prints the human text table + summary line, not
    JSON.
    """
    store = tmp_path / ".envelcost"
    run_result = cli_runner.invoke(
        app,
        ["run", "--harnesses", "deepseek-native,openai-shape", "--store", str(store)],
    )
    assert run_result.exit_code == 0, run_result.output

    result = cli_runner.invoke(
        app,
        ["project", "--gpus", "8xH100", "--seats", "50", "--store", str(store)],
    )
    assert result.exit_code == 0, result.output
    # The human summary line + table are still printed.
    assert "total capex" in result.output
    assert "deepseek-native fits" in result.output
    # Not JSON.
    import json as _json
    with pytest.raises(_json.JSONDecodeError):
        _json.loads(result.output)
