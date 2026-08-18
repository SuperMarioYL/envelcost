"""Tokenizer layer: two BPE vocabularies, one per envelope family.

The reddit benchmark's structural root cause is that the OpenAI-shaped envelope
is tokenized with OpenAI's ``cl100k`` BPE while DeepSeek-native uses DeepSeek's
own BPE vocab — the two vocabularies are *not* the same, so the same schema
text costs different token counts under each, on top of the envelope being
verbose in the first place. This module makes that difference explicit and
reproducible.

Both real tokenizers are lazy-imported so the core package stays lightweight
and tests run without a 2 GB ``transformers``+``torch`` install:

* ``openai``  → :mod:`tiktoken` ``cl100k_base`` (lightweight, always installed).
* ``deepseek``→ :mod:`transformers` ``AutoTokenizer`` for a DeepSeek model,
  falling back to a deterministic byte-ratio approximation when
  ``transformers`` is not installed (e.g. the ``[deepseek]`` extra not pulled).

The approximation is deterministic and stable across runs — sufficient for the
m1 parity gate, which compares *ratios* across envelopes, not absolute counts.
"""

from __future__ import annotations

import functools
from typing import Literal

__all__ = ["Tokenizer", "TokenizerName"]

TokenizerName = Literal["openai", "deepseek"]

# Deterministic fallback ratios (chars→tokens) calibrated to the real BPEs.
# cl100k compresses English+JSON ~4.0 chars/token; DeepSeek's vocab is broader
# on CJK but similar on Latin/JSON. These are stable across runs by design.
_OPENAI_CHARS_PER_TOKEN = 4.0
_DEEPSEEK_CHARS_PER_TOKEN = 3.8

# The DeepSeek coder tokenizer repo id + a PINNED known-good revision (commit).
# fix-tokenizer-trust-remote-code-unpinned: the old call
# `AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct",
# trust_remote_code=True)` fetched the LATEST `main` revision with no
# `revision=` pin and executed custom `tokenization_*.py`/`modeling_*.py` files
# shipped in that repo — so a compromised / MitM'd repo revision ran arbitrary
# code on the user's machine at tokenizer-load time (RCE). Pinning a specific
# commit freezes the loaded snapshot: a later malicious `main` revision cannot
# silently swap in new code. `trust_remote_code=False` is set because the
# 1.3b-instruct tokenizer ships a standard fast tokenizer that needs no custom
# code — the deterministic `_approx_count` fallback already makes the real
# tokenizer non-load-bearing for the m1 gate, so a failed/unsafe load must NOT
# execute remote code; it falls back instead. If HuggingFace revises the repo,
# bump this pin deliberately after reviewing the diff.
_DEEPSEEK_TOKENIZER_REPO = "deepseek-ai/deepseek-coder-1.3b-instruct"
_DEEPSEEK_TOKENIZER_REVISION = "e063262dac8366fc1f28a4da0ff3c50ea66259ca"


def _approx_count(text: str, chars_per_token: float) -> int:
    """Deterministic char-ratio fallback (no external tokenizer needed)."""
    return max(int(round(len(text) / chars_per_token)), 1)


class Tokenizer:
    """Counts tokens under either the OpenAI or DeepSeek BPE vocabulary.

    Construct once; the real tokenizers (if importable) are cached per-name so
    repeated ``count`` calls do not re-load the vocab. The fallback path is
    deterministic so the m1 variance gate is reproducible in any environment.
    """

    def __init__(self) -> None:
        self._tiktoken_enc = None
        self._deepseek_enc = None
        self._tiktoken_ok: bool | None = None
        self._deepseek_ok: bool | None = None

    # --- openai (tiktoken cl100k) ---
    def _ensure_tiktoken(self) -> None:
        if self._tiktoken_ok is not None:
            return
        try:
            import tiktoken  # type: ignore
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            self._tiktoken_ok = True
        except Exception:
            self._tiktoken_ok = False

    def _count_openai(self, text: str) -> int:
        self._ensure_tiktoken()
        if self._tiktoken_enc is not None:
            return len(self._tiktoken_enc.encode(text))
        return _approx_count(text, _OPENAI_CHARS_PER_TOKEN)

    # --- deepseek (transformers AutoTokenizer) ---
    def _ensure_deepseek(self) -> None:
        if self._deepseek_ok is not None:
            return
        try:
            from transformers import AutoTokenizer  # type: ignore
            # A small DeepSeek tokenizer id; if offline/unavailable, falls back.
            # revision= pins a known-good commit and trust_remote_code=False
            # refuses to execute any tokenization_*.py shipped in the repo —
            # see _DEEPSEEK_TOKENIZER_REVISION (fix-tokenizer-trust-remote-code-
            # unpinned). If the pinned snapshot can't be loaded safely (offline
            # / HF down), the except below falls back to _approx_count so the
            # m1 gate stays reproducible without attempting remote code.
            self._deepseek_enc = AutoTokenizer.from_pretrained(
                _DEEPSEEK_TOKENIZER_REPO,
                revision=_DEEPSEEK_TOKENIZER_REVISION,
                trust_remote_code=False,
            )
            self._deepseek_ok = True
        except Exception:
            self._deepseek_ok = False

    def _count_deepseek(self, text: str) -> int:
        self._ensure_deepseek()
        if self._deepseek_enc is not None:
            return len(self._deepseek_enc.encode(text, add_special_tokens=False))
        return _approx_count(text, _DEEPSEEK_CHARS_PER_TOKEN)

    def count(self, text: str, name: TokenizerName) -> int:
        """Return the token count of ``text`` under the named BPE vocabulary."""
        if name == "openai":
            return self._count_openai(text)
        if name == "deepseek":
            return self._count_deepseek(text)
        raise ValueError(f"unknown tokenizer name: {name!r}")

    @property
    def openai_available(self) -> bool:
        self._ensure_tiktoken()
        return bool(self._tiktoken_ok)

    @property
    def deepseek_available(self) -> bool:
        self._ensure_deepseek()
        return bool(self._deepseek_ok)


@functools.lru_cache(maxsize=1)
def default_tokenizer() -> Tokenizer:
    """Process-wide cached default tokenizer."""
    return Tokenizer()
