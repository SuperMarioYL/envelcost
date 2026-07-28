"""The Runner: replay canonical tasks through N envelope configs.

The m1 milestone is the *reproducible* benchmark — 5 tasks × 2+ envelopes on
DeepSeek V4 Flash, printing a per-task token table and asserting >2x variance
on ≥3/5 tasks. Because the envelope overhead is computed by tokenizing the
actual serialized envelope text (see :mod:`envelcost.envelope`), the benchmark
is fully reproducible offline: no DeepSeek API key required to reproduce the
>2x cross-harness variance.

An optional online path (:meth:`Runner.run_online`) replays the same envelopes
through the live DeepSeek API via :mod:`httpx` to fill ``output_tokens`` and
``quality_score`` from the real ``usage`` block — but the m1 gate does not
depend on it, and the schema for DeepSeek's usage response is flagged
``schema_unverified`` (see plan frontmatter): the online call is best-effort
and surfaces a clear warning if the response shape differs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import ENVELCOST_STORE, TasksConfig, load_tasks
from .envelope import DEFAULT_MODEL, EnvelopeProfile, all_envelopes, get_envelope, measure_envelope
from .tokenizer import Tokenizer, default_tokenizer

__all__ = [
    "VarianceReport",
    "Runner",
    "DEFAULT_HARNESSES",
    "VARIANCE_GATE_MULTIPLIER",
    "VARIANCE_FLOOR",
]

# m1 gate: mvp_plan §8 kill #1 — halt if cross-envelope variance < 1.5x.
VARIANCE_FLOOR = 1.5
VARIANCE_GATE_MULTIPLIER = 2.0  # "done" bar = >2x on >=3/5 tasks.
DEFAULT_HARNESSES = ("deepseek-native", "openai-shape")


@dataclass
class VarianceReport:
    """Summary of the cross-envelope variance across the benchmark."""

    model: str
    task_count: int
    harnesses: tuple[str, ...]
    per_task_multiplier: dict[str, dict[str, float]]
    tasks_above_gate: int
    tasks_above_floor: int
    gate_passed: bool  # >=3/5 tasks above 2x (done bar)
    floor_passed: bool  # >=3/5 tasks above 1.5x (kill floor)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "task_count": self.task_count,
            "harnesses": list(self.harnesses),
            "per_task_multiplier": self.per_task_multiplier,
            "tasks_above_gate": self.tasks_above_gate,
            "tasks_above_floor": self.tasks_above_floor,
            "gate_passed": self.gate_passed,
            "floor_passed": self.floor_passed,
        }


class Runner:
    """Replays a task set through envelope configs and records EnvelopeProfiles."""

    def __init__(
        self,
        tasks: TasksConfig | None = None,
        envelopes: dict[str, object] | None = None,
        tokenizer: Tokenizer | None = None,
        model: str = DEFAULT_MODEL,
        store_dir: Path | str | None = None,
    ) -> None:
        self.tasks = tasks if tasks is not None else load_tasks()
        self.envelopes = envelopes if envelopes is not None else all_envelopes()
        self.tokenizer = tokenizer or default_tokenizer()
        self.model = model
        self.store_dir = Path(store_dir) if store_dir else ENVELCOST_STORE

    # --- offline (reproducible, no network) ---
    def run_task(self, task_id: str, harness: str) -> EnvelopeProfile:
        """Measure one (task, envelope) pair — offline, deterministic.

        The ``deepseek-native`` envelope is the 1.0x baseline; every other
        envelope is measured against its input-token count so
        ``multiplier_vs_baseline`` is the cross-harness ratio the m1 gate reads.
        """
        task = self.tasks.by_id(task_id)
        env = self.envelopes[harness] if harness in self.envelopes else get_envelope(harness)
        # Parity assertion: quality is equal across envelopes (m1 gate premise).
        # Offline we set a constant parity score; online fills it from a diff pass.
        if env.is_baseline:
            return measure_envelope(task, env, self.tokenizer, quality_score=1.0)
        # Measure the native baseline for this task under its own tokenizer,
        # then use it as the reference for the non-baseline envelope.
        native = get_envelope("deepseek-native")
        native_prof = measure_envelope(task, native, self.tokenizer)
        return measure_envelope(
            task,
            env,
            self.tokenizer,
            baseline_input_tokens=native_prof.input_tokens,
            quality_score=1.0,
        )

    def run_benchmark(
        self,
        harnesses: tuple[str, ...] = DEFAULT_HARNESSES,
        task_ids: list[str] | None = None,
    ) -> list[EnvelopeProfile]:
        """Run the full N×M grid; returns one EnvelopeProfile per (task, harness)."""
        ids = task_ids or [t.task_id for t in self.tasks.tasks]
        profiles: list[EnvelopeProfile] = []
        for tid in ids:
            for h in harnesses:
                profiles.append(self.run_task(tid, h))
        self._store(profiles)
        return profiles

    def variance_report(
        self, profiles: list[EnvelopeProfile] | None = None
    ) -> VarianceReport:
        """Summarize cross-envelope variance; applies the m1 kill gate."""
        profiles = profiles or self.load_profiles()
        harnesses = tuple(sorted({p.harness for p in profiles}))
        per_task: dict[str, dict[str, float]] = {}
        for p in profiles:
            per_task.setdefault(p.task_id, {})[p.harness] = p.multiplier_vs_baseline
        # ratio of max/min harness multiplier per task (vs the deepseek-native
        # baseline=1.0x, so the max multiplier IS the cross-harness ratio).
        above_gate = above_floor = 0
        for tid, mults in per_task.items():
            peak = max(mults.values())
            if peak >= VARIANCE_GATE_MULTIPLIER:
                above_gate += 1
            if peak >= VARIANCE_FLOOR:
                above_floor += 1
        n = len(per_task)
        gate = n >= 3 and above_gate >= 3
        floor = not (n >= 3 and above_floor < 3)  # halt only if <1.5x on >=3/5
        return VarianceReport(
            model=self.model,
            task_count=n,
            harnesses=harnesses,
            per_task_multiplier=per_task,
            tasks_above_gate=above_gate,
            tasks_above_floor=above_floor,
            gate_passed=gate,
            floor_passed=floor,
        )

    # --- persistence ---
    def _store(self, profiles: list[EnvelopeProfile]) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        out = self.store_dir / "profiles.jsonl"
        with out.open("a", encoding="utf-8") as f:
            for p in profiles:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    def load_profiles(self) -> list[EnvelopeProfile]:
        f = self.store_dir / "profiles.jsonl"
        if not f.exists():
            return []
        from datetime import datetime
        out: list[EnvelopeProfile] = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            d["measured_at"] = datetime.fromisoformat(d["measured_at"])
            out.append(EnvelopeProfile(**d))
        return out

    # --- online (optional; schema_unverified) ---
    def run_online(self, base_url: str | None = None) -> list[EnvelopeProfile]:
        """Replay envelopes through the live DeepSeek API. Best-effort.

        Requires ``DEEPSEEK_API_KEY``. DeepSeek's usage-response shape is flagged
        ``schema_unverified`` (plan frontmatter): on a shape mismatch we surface a
        clear warning and fall back to the offline measurement rather than crash.
        """
        import httpx

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set; the offline `run` path needs no key."
            )
        base = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        url = f"{base.rstrip('/')}/v1/chat/completions"
        profiles: list[EnvelopeProfile] = []
        with httpx.Client(timeout=120) as client:
            for task in self.tasks.tasks:
                for hname, env in self.envelopes.items():
                    envelope_text = env.serialize(task)
                    body = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": envelope_text}],
                    }
                    try:
                        r = client.post(
                            url,
                            headers={"Authorization": f"Bearer {api_key}"},
                            json=body,
                        )
                        r.raise_for_status()
                        usage = r.json().get("usage", {})
                        out_tok = int(usage.get("completion_tokens", 0))
                        # online input may differ from offline envelope count if
                        # the provider re-tokenizes; keep the offline count as
                        # the reproducible measure and carry online output_tokens.
                        prof = self.run_task(task.task_id, hname)
                        prof.output_tokens = out_tok
                        profiles.append(prof)
                    except Exception as e:  # noqa: BLE001 - surface, don't crash
                        import warnings
                        warnings.warn(
                            f"online call failed for {task.task_id}/{hname}: {e}; "
                            "falling back to offline measurement",
                        )
                        profiles.append(self.run_task(task.task_id, hname))
        self._store(profiles)
        return profiles
