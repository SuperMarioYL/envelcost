"""envelcost CLI — typer entrypoint: run / report / project.

Three commands map to the m1/m2/m3 milestones:

* ``envelcost run`` (m1+m2) — replay the 5-task benchmark through N envelope
  configs and store :class:`~envelcost.envelope.EnvelopeProfile` records to
  ``.envelcost/``. Offline + reproducible by default; the m1 variance gate is
  asserted on the printed table.
* ``envelcost report`` (m2) — read the stored profiles and print the per-envelope
  token-cost table + write ``envelcost-report.{json,md}``.
* ``envelcost project`` (m3) — read the measured multipliers and project a fixed
  GPU cluster's per-harness seat-capacity + cost/seat. The commercial hook.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .envelope import ENVELOPE_NAMES
from .projector import AMORTIZATION_YEARS, Projector
from .report import Reporter
from .runner import DEFAULT_HARNESSES, Runner

app = typer.Typer(
    name="envelcost",
    help="Per-harness DeepSeek tool-call-envelope token-cost profiler.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the envelcost version and exit.",
    ),
) -> None:
    """Per-harness DeepSeek tool-call-envelope token-cost profiler."""


def _parse_harnesses(raw: str) -> tuple[str, ...]:
    parts = [h.strip() for h in raw.split(",") if h.strip()]
    bad = [h for h in parts if h not in ENVELOPE_NAMES]
    if bad:
        raise typer.BadParameter(
            f"unknown harness(es): {', '.join(bad)}. valid: {', '.join(ENVELOPE_NAMES)}"
        )
    return tuple(parts) or DEFAULT_HARNESSES


def _resolve_task_ids(all_ids: list[str], task: str) -> list[str]:
    """Resolve a ``--task`` value into concrete task ids for the online path.

    The default ``swe-bench-mini`` is the shared set prefix of all 5 canonical
    tasks, so it resolves to the full set; a specific id like
    ``swe-bench-mini-001`` resolves to just that task. Comma-separated lists are
    accepted. Each part matches either an exact id or a prefix. Raises
    :class:`typer.BadParameter` if nothing matches, so a typoed ``--task`` is
    surfaced before any API call is billed.
    """
    parts = [p.strip() for p in task.split(",") if p.strip()]
    selected: list[str] = []
    for p in parts:
        for tid in all_ids:
            if (tid == p or tid.startswith(p)) and tid not in selected:
                selected.append(tid)
    if not selected:
        raise typer.BadParameter(
            f"no tasks match --task '{task}'. available: {', '.join(all_ids)}"
        )
    return selected


@app.command()
def run(
    task: str = typer.Option(
        "swe-bench-mini", "--task", "-t", help="Benchmark task set id."
    ),
    harnesses: str = typer.Option(
        ",".join(DEFAULT_HARNESSES),
        "--harnesses", "-H",
        help=f"Comma-separated envelope configs. valid: {', '.join(ENVELOPE_NAMES)}",
    ),
    online: bool = typer.Option(
        False, "--online", help="Replay through the live DeepSeek API (needs DEEPSEEK_API_KEY)."
    ),
    store: Optional[Path] = typer.Option(
        None, "--store", help="Override the .envelcost store directory."
    ),
) -> None:
    """Replay the benchmark through N envelope configs and assert the m1 gate."""
    harness_list = _parse_harnesses(harnesses)
    runner = Runner(store_dir=store) if store else Runner()
    if online:
        # Thread --harnesses (and a resolved --task filter) into the online path
        # so it bills only the selected (harness, task) pairs through the paid
        # DeepSeek API — never the full 5-task x 3-envelope grid. The offline
        # path keeps its existing run_benchmark(harnesses=...) forwarding.
        task_ids = _resolve_task_ids(
            [t.task_id for t in runner.tasks.tasks], task
        )
        profiles = runner.run_online(harnesses=harness_list, task_ids=task_ids)
    else:
        profiles = runner.run_benchmark(harnesses=harness_list)
    table = Reporter(store_dir=runner.store_dir).render(
        profiles, runner.variance_report(profiles).to_dict()
    )
    typer.echo(table)
    vr = runner.variance_report(profiles)
    typer.echo("")
    if vr.floor_evaluable:
        floor_status = "HELD" if vr.floor_passed else "BROKEN — halt"
    else:
        floor_status = "SKIPPED (<2 harnesses)"
    typer.echo(
        f"m1 variance gate: {vr.tasks_above_gate}/{vr.task_count} tasks above 2.0x "
        f"(done bar {'PASSED' if vr.gate_passed else 'NOT YET'}); "
        f"kill floor (1.5x) {floor_status}"
    )
    if not vr.floor_evaluable:
        # mvp_plan §8 kill #1 is about cross-envelope variance; a single-harness
        # (baseline-only) run cannot evaluate it — surface a clear note and do
        # NOT falsify the thesis / exit 1.
        typer.echo(
            f"insufficient harnesses for cross-variance evaluation "
            f"(ran {len(vr.harnesses)}; need >=2); skipping kill gate"
        )
    elif not vr.floor_passed:
        typer.echo(
            "WARNING: cross-envelope variance below 1.5x on >=3/5 tasks — "
            "core thesis falsified (mvp_plan §8 kill #1). Halt.",
            err=True,
        )
        sys.exit(1)


@app.command()
def report(
    store: Optional[Path] = typer.Option(
        None, "--store", help="Override the .envelcost store directory."
    ),
) -> None:
    """Render the per-envelope token-cost table + write JSON/MD to the store."""
    runner = Runner(store_dir=store) if store else Runner()
    profiles = runner.load_profiles()
    if not profiles:
        typer.echo(
            "no profiles found — run `envelcost run` first.", err=True
        )
        raise typer.Exit(1)
    table = Reporter(store_dir=runner.store_dir).render(
        profiles, runner.variance_report(profiles).to_dict()
    )
    typer.echo(table)
    typer.echo(
        f"\nwrote: {runner.store_dir / 'envelcost-report.md'} "
        f"+ {runner.store_dir / 'envelcost-report.json'}"
    )


@app.command()
def project(
    gpus: str = typer.Option(
        "8xH100", "--gpus", "-g", help="GPU cluster spec, e.g. 8xH100 / 4xH200 / 2xH3."
    ),
    seats: int = typer.Option(
        50, "--seats", "-s", help="Target concurrent coding-agent seats."
    ),
    store: Optional[Path] = typer.Option(
        None, "--store", help="Override the .envelcost store directory."
    ),
) -> None:
    """Project per-harness seat-capacity + cost/seat onto a fixed GPU cluster."""
    projector = Projector.from_store(store) if store else Projector.from_store()
    projection = projector.project(gpus, seats)
    typer.echo(
        f"cluster: {projection.gpu_count}×{projection.gpu.label}  "
        f"target seats: {projection.target_seats}  "
        f"total capex: ¥{projection.total_capex_cny:,.0f}"
    )
    typer.echo("")
    typer.echo(
        f"{'harness':<20} {'mult':>6} {'raw':>5} {'eff':>5} "
        f"{'fits':>6} {'deficit':>8} {'¥/seat':>12} {'¥/seat/yr':>11}"
    )
    typer.echo("-" * 82)
    for r in projection.rows:
        fits = "yes" if r.fits_target else "NO"
        typer.echo(
            f"{r.harness:<20} {r.multiplier:>5.2f}x {r.raw_seats:>5} "
            f"{r.effective_seats:>5} {fits:>6} {r.deficit:>+8} "
            f"¥{r.capex_per_seat_cny:>10,.0f} ¥{r.annual_cost_per_seat_cny:>9,.0f}"
        )
    typer.echo("")
    fits_native = projection.fits_harness("deepseek-native")
    fits_openai = projection.fits_harness("openai-shape")
    typer.echo(
        f"on {projection.gpu_count}×{projection.gpu.name} / {seats} seats: "
        f"deepseek-native {'fits' if fits_native else 'does NOT fit'}, "
        f"openai-shape {'fits' if fits_openai else 'does NOT fit'} "
        f"(capex amortized over {AMORTIZATION_YEARS}y)"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
