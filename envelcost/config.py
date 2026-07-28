"""Configuration: canonical coding tasks, GPU specs, runtime defaults.

Loads the 5-task SWE-bench-mini benchmark from ``envelcost/tasks/swe-bench-mini.yaml``
and defines the on-prem GPU catalog used by the projector. No live DB, no env
sidecar — the only mutable state is the ``.envelcost/`` JSON store on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .envelope import ToolDef

__all__ = [
    "GPU",
    "GPU_CATALOG",
    "CodingTask",
    "TasksConfig",
    "ENVELCOST_STORE",
    "load_tasks",
    "resolve_gpu",
    "parse_gpu_spec",
]

# The on-disk store: a directory of JSONL profiles + rendered reports.
ENVELCOST_STORE = Path(".envelcost")
TASKS_YAML = Path(__file__).parent / "tasks" / "swe-bench-mini.yaml"


@dataclass
class CodingTask:
    """One canonical coding-agent task: prompt + the tools it exercises."""

    task_id: str
    prompt: str
    tools: list[ToolDef]
    tool_call_sequence: list[ToolDef]
    turns: int
    repo: str = ""
    language: str = "python"

    def baseline_prompt_text(self) -> str:
        """The bare task prompt with no envelope scaffolding — the reference
        the envelope overhead is measured against."""
        return self.prompt

    @classmethod
    def from_yaml(cls, raw: dict) -> "CodingTask":
        tools = [
            ToolDef(
                name=t["name"],
                description=t["description"],
                parameters=t.get("parameters", {}),
            )
            for t in raw["tools"]
        ]
        seq_indices = raw.get("tool_call_sequence") or [0] * raw.get("turns", 1)
        seq = [tools[i] for i in seq_indices]
        return cls(
            task_id=raw["task_id"],
            prompt=raw["prompt"],
            tools=tools,
            tool_call_sequence=seq,
            turns=raw.get("turns", len(seq)),
            repo=raw.get("repo", ""),
            language=raw.get("language", "python"),
        )


@dataclass
class TasksConfig:
    tasks: list[CodingTask]
    source: str

    def by_id(self, task_id: str) -> CodingTask:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"task not found: {task_id}")


def load_tasks(path: str | Path | None = None) -> TasksConfig:
    """Load the canonical benchmark task set from the shipped YAML."""
    p = Path(path) if path else TASKS_YAML
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    tasks = [CodingTask.from_yaml(t) for t in data["tasks"]]
    return TasksConfig(tasks=tasks, source=str(p))


# --- GPU catalog (信创 on-prem racks the projector sizes against) -----------
# Throughput is effective coding-agent tokens/sec/seat (prefill+decode amortized
# over a real multi-turn coding session, not peak benchmark FLOPS).

@dataclass(frozen=True)
class GPU:
    name: str
    label: str
    tokens_per_sec_per_seat: float  # effective, per concurrent seat
    capex_per_unit_cny: float      # one-time silicon cost (信创 list, ex-VAT)
    power_watts: float

    def seats_per_unit(self) -> int:
        """A single GPU serves this many concurrent coding-agent seats at full."""
        return 8  # coding-agent seats are memory- not compute-bound on V4-flash


GPU_CATALOG: dict[str, GPU] = {
    "H100": GPU("H100", "NVIDIA H100 80GB", 3200.0, 260_000.0, 700.0),
    "H200": GPU("H200", "NVIDIA H200 141GB", 3600.0, 300_000.0, 700.0),
    "H3": GPU("H3", "NVIDIA H3 288GB", 5200.0, 420_000.0, 1000.0),
}


def resolve_gpu(name: str) -> GPU:
    key = name.upper().replace("NVIDIA ", "").strip()
    if key not in GPU_CATALOG:
        raise ValueError(
            f"unknown GPU '{name}'. valid: {', '.join(GPU_CATALOG)}"
        )
    return GPU_CATALOG[key]


def parse_gpu_spec(spec: str) -> tuple[GPU, int]:
    """Parse an ``"8xH100"`` / ``"4×H200"`` spec into (GPU, count)."""
    spec = spec.strip().replace("×", "x").replace(" ", "")
    if "x" not in spec:
        raise ValueError(
            f"bad gpu spec '{spec}'. expected e.g. '8xH100' or '4xH200'"
        )
    count_str, _, name = spec.partition("x")
    count = int(count_str)
    return resolve_gpu(name), count
