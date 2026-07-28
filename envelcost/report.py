"""The Reporter: render EnvelopeProfile records as a token-cost table.

m2 scope (stubbed here as a working-but-minimal renderer): a stdout per-envelope
token table + ``envelcost-report.{json,md}`` written to the ``.envelcost/``
store. The full m2 ship adds 3 envelope configs + a richer bilingual MD
template; for m1 the table + JSON are the load-bearing artifact the demo and
the variance gate read from.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ENVELCOST_STORE
from .envelope import EnvelopeProfile

__all__ = ["Reporter", "render_table", "render_markdown", "render_json"]

try:
    from jinja2 import Template
    _JINJA_OK = True
except Exception:  # pragma: no cover - jinja2 is a core dep
    _JINJA_OK = False
    Template = None  # type: ignore


def render_table(profiles: list[EnvelopeProfile]) -> str:
    """A compact stdout table: one row per (task, harness) with overhead + multiplier."""
    if not profiles:
        return "(no profiles — run `envelcost run` first)"
    header = f"{'task_id':<22} {'harness':<20} {'input':>7} {'overhead':>8} {'mult':>6}"
    lines = [header, "-" * len(header)]
    for p in profiles:
        lines.append(
            f"{p.task_id:<22} {p.harness:<20} {p.input_tokens:>7} "
            f"{p.envelope_overhead_tokens:>8} {p.multiplier_vs_baseline:>5.2f}x"
        )
    return "\n".join(lines)


_MD_TEMPLATE = """# envelcost report

model: `{{ model }}`
harnesses: {{ harnesses | join(", ") }}
tasks: {{ task_count }}

## per-envelope token cost

| task_id | harness | input_tokens | envelope_overhead | multiplier_vs_baseline |
|---|---|---:|---:|---:|
{% for p in profiles -%}
| {{ p.task_id }} | {{ p.harness }} | {{ p.input_tokens }} | {{ p.envelope_overhead_tokens }} | {{ "%.2fx" | format(p.multiplier_vs_baseline) }} |
{% endfor %}

## variance gate (mvp_plan §8 kill #1)

- tasks above 2.0x (done bar): {{ variance.tasks_above_gate }} / {{ variance.task_count }}
- tasks above 1.5x (kill floor): {{ variance.tasks_above_floor }} / {{ variance.task_count }}
- gate passed: **{{ variance.gate_passed }}**
- kill floor held: **{{ variance.floor_passed }}**
"""


def render_markdown(
    profiles: list[EnvelopeProfile], variance: dict | None = None
) -> str:
    """Render the bilingual-ready markdown report (zh headings; m2 adds en sibling)."""
    variance = variance or {}
    if _JINJA_OK and Template is not None:
        return Template(_MD_TEMPLATE).render(
            model=profiles[0].model if profiles else "deepseek-v4-flash",
            harnesses=sorted({p.harness for p in profiles}),
            task_count=len({p.task_id for p in profiles}),
            profiles=profiles,
            variance=variance,
        )
    # Fallback (jinja2 missing): minimal hand-rolled MD.
    lines = ["# envelcost report", ""]
    lines.append("| task_id | harness | input | overhead | mult |")
    lines.append("|---|---|---:|---:|---:|")
    for p in profiles:
        lines.append(
            f"| {p.task_id} | {p.harness} | {p.input_tokens} | "
            f"{p.envelope_overhead_tokens} | {p.multiplier_vs_baseline:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def render_json(profiles: list[EnvelopeProfile], variance: dict | None = None) -> str:
    """JSON report: profiles + variance summary."""
    return json.dumps(
        {
            "profiles": [p.to_dict() for p in profiles],
            "variance": variance or {},
        },
        ensure_ascii=False,
        indent=2,
    )


class Reporter:
    """Write the rendered report to the ``.envelcost/`` store + return the table."""

    def __init__(self, store_dir: Path | str | None = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else ENVELCOST_STORE

    def render(
        self,
        profiles: list[EnvelopeProfile],
        variance: dict | None = None,
        write: bool = True,
    ) -> str:
        table = render_table(profiles)
        if write:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            (self.store_dir / "envelcost-report.md").write_text(
                render_markdown(profiles, variance), encoding="utf-8"
            )
            (self.store_dir / "envelcost-report.json").write_text(
                render_json(profiles, variance), encoding="utf-8"
            )
        return table
