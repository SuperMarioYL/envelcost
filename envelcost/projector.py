"""The Projector: turn measured envelope multipliers into GPU seat-capacity.

The commercial monetization hook. Given a fixed 信创 on-prem GPU cluster
(``8xH100``, ``4xH200``...) and a target seat count, read the measured
per-harness multipliers from :class:`~envelcost.envelope.EnvelopeProfile`
records and project each harness's effective seat-capacity + cost/seat.

The model is intentionally simple and defensible: a coding-agent seat is
memory-bound (one GPU serves a fixed number of concurrent seats), but a
harness that burns ``M``× tokens per task vs the baseline consumes ``M``× the
sustained throughput per seat, so the *effective* seats a fixed cluster can
feed is ``raw_seats / M``. A harness at 4× therefore serves 1/4 the seats on
the same silicon — that is the "8×H100 / 50 座位下 harness A 撑得住、
harness B 撑不住" number a 信创 ML-platform engineer reads off in one line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from .config import ENVELCOST_STORE, GPU, parse_gpu_spec
from .envelope import EnvelopeProfile, all_envelopes

__all__ = [
    "HarnessCapacity",
    "Projection",
    "Projector",
    "AMORTIZATION_YEARS",
]

# Cluster capex amortization window for annualized cost-per-seat.
AMORTIZATION_YEARS = 3


@dataclass
class HarnessCapacity:
    """One row of the per-harness GPU-capacity projection."""

    harness: str
    multiplier: float
    raw_seats: int  # memory-bound seats the cluster physically holds
    effective_seats: int  # raw_seats / multiplier (token-bound floor)
    fits_target: bool
    deficit: int  # target - effective (negative = surplus)
    capex_per_seat_cny: float
    annual_cost_per_seat_cny: float

    def to_dict(self) -> dict:
        return {
            "harness": self.harness,
            "multiplier": round(self.multiplier, 3),
            "raw_seats": self.raw_seats,
            "effective_seats": self.effective_seats,
            "fits_target": self.fits_target,
            "deficit": self.deficit,
            "capex_per_seat_cny": round(self.capex_per_seat_cny, 0),
            "annual_cost_per_seat_cny": round(self.annual_cost_per_seat_cny, 0),
        }


@dataclass
class Projection:
    gpu: GPU
    gpu_count: int
    target_seats: int
    total_capex_cny: float
    rows: list[HarnessCapacity]
    # Harnesses the caller asked about (or the summary names) with NO measurement
    # in the store. They have no row — a multiplier would be invented — but they
    # are reported so machine consumers can see the gap instead of a fit verdict.
    unmeasured: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gpu": self.gpu.label,
            "gpu_count": self.gpu_count,
            "target_seats": self.target_seats,
            "total_capex_cny": round(self.total_capex_cny, 0),
            "rows": [r.to_dict() for r in self.rows],
            "unmeasured": list(self.unmeasured),
        }

    def fits_harness(self, harness: str) -> bool | None:
        for r in self.rows:
            if r.harness == harness:
                return r.fits_target
        return None


class Projector:
    """Project measured envelope multipliers onto a fixed GPU cluster."""

    def __init__(self, profiles: list[EnvelopeProfile] | None = None) -> None:
        self.profiles = profiles if profiles is not None else []

    @classmethod
    def from_store(cls, store_dir: Path | str | None = None) -> "Projector":
        """Load profiles from the ``.envelcost/profiles.jsonl`` store."""
        from .runner import Runner
        runner = Runner(store_dir=store_dir) if store_dir else Runner()
        return cls(profiles=runner.load_profiles())

    def multiplier_for(self, harness: str) -> float | None:
        """Mean cross-task multiplier for one harness, or None when unmeasured.

        The baseline envelope is 1.0x *by definition* — every other multiplier is
        measured against it — so an unmeasured baseline is still a known ratio.
        Any other harness with no measured profile has an UNKNOWN multiplier:
        returning 1.0 (parity) made `project` print a "fits" verdict a measured
        1.0x would produce, i.e. a fabricated measurement on the commercial hook.
        """
        vals = [p.multiplier_vs_baseline for p in self.profiles if p.harness == harness]
        if vals:
            return mean(vals)
        env = all_envelopes().get(harness)
        if env is not None and env.is_baseline:
            return 1.0
        return None

    def project(
        self,
        gpu_spec: str,
        target_seats: int,
        harnesses: tuple[str, ...] | None = None,
    ) -> Projection:
        """Project per-harness seat-capacity + cost for a cluster + seat target."""
        gpu, count = parse_gpu_spec(gpu_spec)
        # The seat target is half the projection's input: a 0/negative value has
        # no meaning and previously rendered a full table (negative deficits,
        # everything "fits") that read as a measured result. Same <= 0 guard the
        # GPU count gets in parse_gpu_spec; the CLI already turns this ValueError
        # into a one-line `error:` + exit 1.
        if target_seats < 1:
            raise ValueError(f"target seats must be >= 1, got {target_seats}")
        raw_seats = count * gpu.seats_per_unit()
        total_capex = count * gpu.capex_per_unit_cny
        # Union the derived harness set with the two harnesses the `project`
        # summary always names (deepseek-native, openai-shape) — the baseline
        # 1.0x and the OpenAI-shape hook — so a PARTIAL single-harness store
        # (e.g. `envelcost run --harnesses openai-shape`, the natural
        # single-envelope flow the v0.2.0 --harnesses fix made first-class)
        # still projects every summary-referenced harness whose multiplier is
        # known. Without this union, a partial store left deepseek-native out of
        # `rows`, fits_harness('deepseek-native') returned None, and the CLI
        # printed a false 'deepseek-native does NOT fit' for the 1.0x baseline
        # that always fits first (fix-project-summary-false-not-fit). A candidate
        # whose multiplier is UNKNOWN (no profile, not the baseline) gets no row:
        # it is listed on `unmeasured` so no ratio/verdict is invented for it
        # (fix-project-fabricates-parity-for-unmeasured-harness).
        measured = tuple(sorted({p.harness for p in self.profiles}))
        if harnesses:
            harness_set = tuple(harnesses)
        elif measured:
            harness_set = tuple(
                dict.fromkeys(measured + ("deepseek-native", "openai-shape"))
            )
        else:
            # Empty store: the two summary-named harnesses are the candidates.
            harness_set = ("deepseek-native", "openai-shape")
        rows: list[HarnessCapacity] = []
        unmeasured: list[str] = []
        for h in harness_set:
            m = self.multiplier_for(h)
            if m is None:
                unmeasured.append(h)
                continue
            effective = max(int(raw_seats / m), 1)
            fits = effective >= target_seats
            capex_per_seat = total_capex / effective if effective else float("inf")
            annual = capex_per_seat / AMORTIZATION_YEARS
            rows.append(
                HarnessCapacity(
                    harness=h,
                    multiplier=m,
                    raw_seats=raw_seats,
                    effective_seats=effective,
                    fits_target=fits,
                    deficit=target_seats - effective,
                    capex_per_seat_cny=capex_per_seat,
                    annual_cost_per_seat_cny=annual,
                )
            )
        return Projection(
            gpu=gpu,
            gpu_count=count,
            target_seats=target_seats,
            total_capex_cny=total_capex,
            rows=rows,
            unmeasured=unmeasured,
        )
