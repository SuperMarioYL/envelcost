# Changelog

All notable changes to **envelcost** are documented here. Versions follow the
grill bug-hunt amendment cadence. Each entry lists the fix-ids from the
amendment findings YAML.

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
