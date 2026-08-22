# Changelog

All notable changes to **envelcost** are documented here. Versions follow the
grill bug-hunt amendment cadence. Each entry lists the fix-ids from the
amendment findings YAML.

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
