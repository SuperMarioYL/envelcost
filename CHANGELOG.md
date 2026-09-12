# Changelog

All notable changes to **envelcost** are documented here. Versions follow the
grill bug-hunt amendment cadence. Each entry lists the fix-ids from the
amendment findings YAML.

## [0.9.0] — 2026-09-12

The v0.9.0 grill bug-hunt (amend-envelcost-v0.9.0). Headline: the wheel build
has been broken since the first release — this version makes the package
installable from source for the first time.

### Fixed

- **fix-wheel-build-duplicate-tasks-include** (`pyproject.toml`): the wheel
  target declared `packages = ["envelcost"]` (which already carries
  `envelcost/tasks/swe-bench-mini.yaml`) AND a force-include mapping
  `"envelcost/tasks" -> "envelcost/tasks"`, so hatchling added the same file
  to the archive twice and hard-failed with "A second file is being added to
  the wheel archive at the same path: `envelcost/tasks/swe-bench-mini.yaml`".
  The release workflow's `python -m build` step failed on EVERY tag from
  v0.1.0 through v0.8.0 (8/8 red release runs), meaning the repo never
  produced a buildable wheel and `pip install
  git+https://github.com/SuperMarioYL/envelcost` was broken for every user;
  CI never caught it because the smoke test installs editable. The redundant
  force-include is removed — a regression test now drives hatchling's
  WheelBuilder (the same builder `python -m build` invokes) on every run and
  asserts the tasks YAML ships in the wheel.
- **fix-readme-install-commands-404** (`examples/quickstart.sh`): the
  v0.8.0 READMEs led with `uv tool install envelcost` / `pipx install
  envelcost`, but the package has never been published to PyPI (the release
  workflow's PyPI job is opt-in and was never enabled — and could not have
  succeeded while the wheel build failed), so the documented install path
  404'd at minute zero. The README rewrite that landed on main after the
  v0.8.0 tag already replaced both install sections with a working
  clone-and-editable flow; this release fixes the remaining surface
  (`examples/quickstart.sh`'s bare `uvx envelcost` mention → the git-source
  form that now verifiably builds) and adds a docs regression test pinning
  all doc surfaces against regressing to a bare PyPI instruction.
- **fix-run-online-missing-key-traceback** (`envelcost/cli.py`): `envelcost
  run --online` without `DEEPSEEK_API_KEY` dumped a full Python traceback
  (the RuntimeError from `Runner.run_online`) instead of a clean CLI error —
  the only documented main-path error that did so. The CLI now surfaces the
  runner's friendly one-line message with exit code 1 and no traceback.
- **fix-run-command-duplicate-variance-computation** (`envelcost/cli.py`):
  the `run` command computed `variance_report(profiles)` twice per
  invocation (once inside the `Reporter.render(...)` arguments, once for the
  gate echo). It is now computed once and reused; a regression test asserts
  exactly one call per run.
- **fix-readme-roadmap-claims-drift** (no code change — resolved upstream):
  the defect this fix targeted (the v0.8.0 READMEs' roadmap section: an
  unchecked m2 box for long-shipped functionality and a stale "2.83–3.27x"
  multiplier range vs the measured 2.81–2.99x) was removed by the post-tag
  README rewrite that landed on main before this iteration. The rewritten
  READMEs' recorded demo numbers were verified against the shipped
  `presentation-demo` output (12/37 tokens, exact match), and a guard test
  now pins those recorded counts to the computed values so future drift
  fails CI instead of misleading readers.

### Tests

- Wheel-build regression: hatchling `WheelBuilder` build succeeds and
  `envelcost/tasks/swe-bench-mini.yaml` is present in the artifact
  (the artifact-producing path CI never exercised).
- Docs regression: no shipped doc (README.md, README.en.md,
  examples/quickstart.sh) instructs a bare PyPI install command.
- README recorded demo counts equal the computed ToolDef/Tokenizer counts.
- `run --online` without the API key exits 1 with the friendly message and
  no traceback; nothing is measured or stored.
- `Runner.variance_report` is called exactly once per `run` invocation.

## [0.8.0] — 2026-08-31

The v0.8.0 grill bug-hunt (amend-envelcost-v0.8.0). Two honest version/report
drift fixes; no behavior change to the core variance/runner logic (which the
6 prior grill iterations pin with adversarial tests).

### Fixed

- **fix-package-version-constant-drift** (`envelcost/__init__.py`): the package
  `__version__` constant was still `0.6.0` after the v0.7.0 ship — every other
  version surface (VERSION, `pyproject.toml [project].version`, CHANGELOG) had
  been bumped to 0.7.0, but `__init__.py` was missed. The CLI's `--version`
  callback imports and echoes this constant
  (`from . import __version__` / `typer.echo(__version__)`), so
  `envelcost --version` printed `0.6.0` for the entire v0.7.0 line. The stale
  value also leaked into the rendered demo asset: `docs/demo.tape` runs
  `envelcost --version` as its first step, so `assets/demo.gif` showed `0.6.0`.
  Now bumped to `0.8.0` and kept in lockstep with VERSION + pyproject; a
  regression test asserts `envelcost.__version__` equals the VERSION file and
  that the `--version` CLI flag echoes the same string, so a future bump cannot
  silently desync again.
- **fix-report-md-misleads-on-skipped-kill-gate** (`envelcost/report.py`): the
  persisted `envelcost-report.md` variance-gate section printed
  `gate passed: {{ variance.gate_passed }}` and
  `kill floor held: {{ variance.floor_passed }}` but ignored
  `variance.floor_evaluable` (which `VarianceReport.to_dict()` already emits).
  So a single-harness (baseline-only) run — where `floor_evaluable=False` and
  `floor_passed=True` ONLY because the gate is skipped — rendered the written
  report as "kill floor held: True" / "gate passed: False", claiming the kill
  floor HELD when it was actually unevaluable, and making "gate passed: False"
  read like a done-bar failure rather than "cannot be evaluated with one
  harness". The CLI stdout was already honest ("kill floor (1.5x) SKIPPED (<2
  harnesses)"); the persisted report now mirrors it: the gate line says
  "unevaluable (<2 harnesses measured)" / "passed" / "not yet", and the
  kill-floor line says "skipped (<2 harnesses)" / "held" / "BROKEN — halt". No
  data-model change — `to_dict()` already carried the flag; one localized
  template edit.

### Tests

- Added `tests/test_runner.py` coverage: `envelcost.__version__` equals the
  VERSION file string and the `pyproject.toml [project].version` field, and
  the `envelcost --version` CLI flag echoes that same string — guarding against
  a future version bump silently desyncing the package constant from VERSION /
  pyproject (the v0.7.0 regression).
- Added `tests/test_runner.py` coverage: a single-harness
  (`--harnesses deepseek-native`) run's persisted `envelcost-report.md` says
  the gate is "unevaluable" and the kill floor is "skipped" — NOT "held" — so
  the written report is as honest as the CLI stdout; a 2-harness passing run
  still says "held" / "passed".

## [0.7.0] — 2026-08-23

The v0.7.0 grill bug-hunt (amend-envelcost-v0.7.0). One HIGH-severity
read-side robustness fix; no contract change on the standard happy path.

### Fixed

- **fix-corrupt-store-line-bricks-all-commands** (`envelcost/runner.py`): both
  store read loops did an unguarded per-line `json.loads` — `_store`'s
  re-read-before-upsert (the self-heal path) and `load_profiles`' parse
  (`json.loads` + `datetime.fromisoformat` + `EnvelopeProfile(**d)`) — so a
  single malformed line (a partial write left by a killed pre-v0.3.0
  append-mode run, a hand-edited store, or any externally-corrupted row)
  raised `JSONDecodeError`/`TypeError` and aborted the whole load. This bricked
  every store-touching command: `report` and `project` (via `load_profiles`)
  traceback, and `run` could not recover because `_store` itself re-read the
  existing file before upserting, crashing on the same corrupt line instead of
  overwriting it — the user had to manually delete `.envelcost/profiles.jsonl`.
  The per-line parse is now wrapped in `try/except` that skips the bad row with
  a visible `warnings.warn`; the existing atomic tmp+`os.replace` write path is
  untouched, so a subsequent `envelcost run` upserts over the gap and self-heals
  the store. This is a localized read-side robustness guard against
  legacy/externally-corrupted stores, not a producer change.

### Tests

- Added `tests/test_runner.py` coverage: a store containing one good line
  plus one un-JSON-parseable line is loaded by `load_profiles` without raising
  (the good row survives, the bad row is skipped with a warning); a fresh
  `run_benchmark` re-reads the corrupt store via `_store` without crashing and
  the atomic rewrite self-heals it (no corrupt line remains afterward).

## [0.6.0] — 2026-08-19

The v0.6.0 grill bug-hunt (amend-envelcost-v0.6.0). Two correctness/security
fixes; no behavior change on the standard happy path.

### Fixed

- **fix-online-usage-shape-silent-zero** (`envelcost/runner.py`): the online
  `run_online` path did no shape check on the DeepSeek `usage` block —
  `r.json().get("usage", {})` / `int(usage.get("completion_tokens", 0))`
  silently stored `output_tokens=0` for any 200 response whose `usage` block
  was missing or used a differently-keyed field (e.g. `output_tokens` /
  `generated_tokens`), contradicting the module docstring's promise of "a
  clear warning if the response shape differs." A non-standard/missing usage
  shape now emits a visible `warnings.warn` and treats `output_tokens` as
  unknown (0 + warning, not a silent zero), falling back to the offline
  measurement so the profile row stays complete.
- **fix-tokenizer-trust-remote-code-unpinned** (`envelcost/tokenizer.py`):
  `AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct",
  trust_remote_code=True)` fetched the latest HuggingFace `main` revision with
  no `revision=` pin and executed custom `tokenization_*.py`/`modeling_*.py`
  files from the repo — an RCE risk on a compromised/MitM'd revision. The load
  now pins a known-good commit (`revision=<sha>`) and sets
  `trust_remote_code=False` (the 1.3b-instruct tokenizer ships a standard fast
  tokenizer that needs no custom code); if the pinned snapshot is unavailable
  (offline / HF down), the deterministic `_approx_count` fallback keeps the m1
  gate reproducible without attempting remote-code execution.

### Tests

- Added `tests/test_runner.py` coverage: non-standard online `usage` shapes
  (different key / empty / absent / wrong type) warn instead of silently
  zeroing; the standard shape reads `output_tokens` with no false warning; the
  DeepSeek tokenizer load pins a 40-char commit revision with
  `trust_remote_code=False`; a failed pinned load falls back to `_approx_count`.
