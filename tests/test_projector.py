"""Tests for the Projector: GPU seat-capacity projection (m3 hook)."""

from __future__ import annotations

import pytest

from envelcost.config import GPU_CATALOG, parse_gpu_spec, resolve_gpu
from envelcost.envelope import EnvelopeProfile, get_envelope
from envelcost.projector import AMORTIZATION_YEARS, Projector


def _fake_profiles() -> list[EnvelopeProfile]:
    """A deterministic 2-harness × 2-task profile set for projector tests."""
    from datetime import datetime, timezone

    out: list[EnvelopeProfile] = []
    for tid in ("t1", "t2"):
        out.append(
            EnvelopeProfile(
                harness="deepseek-native",
                model="deepseek-v4-flash",
                task_id=tid,
                tool_call_envelope_sha="abc",
                input_tokens=100,
                output_tokens=0,
                envelope_overhead_tokens=0,
                quality_score=1.0,
                multiplier_vs_baseline=1.0,
                measured_at=datetime.now(timezone.utc),
            )
        )
        out.append(
            EnvelopeProfile(
                harness="openai-shape",
                model="deepseek-v4-flash",
                task_id=tid,
                tool_call_envelope_sha="def",
                input_tokens=400,
                output_tokens=0,
                envelope_overhead_tokens=300,
                quality_score=1.0,
                multiplier_vs_baseline=4.0,
                measured_at=datetime.now(timezone.utc),
            )
        )
    return out


def test_parse_gpu_spec_variants():
    gpu, count = parse_gpu_spec("8xH100")
    assert gpu.name == "H100"
    assert count == 8
    # unicode × and spaces tolerated.
    gpu2, count2 = parse_gpu_spec("4 × H200")
    assert gpu2.name == "H200" and count2 == 4


def test_parse_gpu_spec_bad():
    with pytest.raises(ValueError):
        parse_gpu_spec("H100")  # no count
    with pytest.raises(ValueError):
        parse_gpu_spec("8xTPU")  # unknown chip


def test_parse_gpu_spec_bad_count():
    """fix-parse-gpu-spec-traceback-and-zero-count: a typoed/zero/negative gpu
    count must raise the friendly 'bad gpu spec'/'count' ValueError — NOT a raw
    int() traceback (H100x8) and NOT a silently bogus 0/negative-capex projection
    (0xH100 / -2xH100). Existing test_parse_gpu_spec_bad only covered the
    no-'x' and unknown-chip paths; the bad-count paths were untested."""
    # Reversed/typoed spec: "H100x8" — int("H100") previously raised a raw
    # ValueError traceback instead of the friendly "bad gpu spec" message.
    with pytest.raises(ValueError, match="bad gpu spec"):
        parse_gpu_spec("H100x8")
    # Zero count: "0xH100" previously parsed to count=0 and projected a bogus
    # 0-GPU cluster (1 effective seat, ¥0 capex) with no error.
    with pytest.raises(ValueError, match="count"):
        parse_gpu_spec("0xH100")
    # Negative count: "-2xH100" previously parsed to count=-2 and projected a
    # negative-GPU cluster (negative capex) with no error.
    with pytest.raises(ValueError, match="count"):
        parse_gpu_spec("-2xH100")
    # The well-formed spec still parses.
    gpu, count = parse_gpu_spec("8xH100")
    assert gpu.name == "H100"
    assert count == 8


def test_gpu_catalog_has_three_racks():
    assert set(GPU_CATALOG) == {"H100", "H200", "H3"}
    assert resolve_gpu("nvidia h100").name == "H100"


def test_projector_multiplier_mean():
    proj = Projector(profiles=_fake_profiles())
    assert proj.multiplier_for("deepseek-native") == 1.0
    assert proj.multiplier_for("openai-shape") == 4.0


def test_projector_unmeasured_harness_defaults_parity():
    proj = Projector(profiles=[])
    assert proj.multiplier_for("deepseek-native") == 1.0


def test_project_native_fits_openai_does_not():
    """The product story: 8xH100 / 50 seats — native fits, 4x harness does not."""
    proj = Projector(profiles=_fake_profiles())
    projection = proj.project("8xH100", 50)
    native = next(r for r in projection.rows if r.harness == "deepseek-native")
    openai = next(r for r in projection.rows if r.harness == "openai-shape")
    # 8 H100 × 8 seats = 64 raw seats. native mult 1.0 → 64 eff >= 50 → fits.
    assert native.raw_seats == 64
    assert native.effective_seats == 64
    assert native.fits_target is True
    assert native.deficit == 50 - 64
    # openai mult 4.0 → 64/4 = 16 eff < 50 → does NOT fit.
    assert openai.effective_seats == 16
    assert openai.fits_target is False
    assert openai.deficit == 50 - 16
    # A 4x harness costs 4x the capex per seat on the same silicon.
    assert openai.capex_per_seat_cny == pytest.approx(
        native.capex_per_seat_cny * 4.0
    )


def test_projection_total_capex_and_amortization():
    proj = Projector(profiles=_fake_profiles())
    projection = proj.project("2xH200", 10)
    gpu = resolve_gpu("H200")
    assert projection.total_capex_cny == 2 * gpu.capex_per_unit_cny
    native = next(r for r in projection.rows if r.harness == "deepseek-native")
    assert native.annual_cost_per_seat_cny == pytest.approx(
        native.capex_per_seat_cny / AMORTIZATION_YEARS
    )


def test_projection_to_dict_serializable():
    proj = Projector(profiles=_fake_profiles())
    projection = proj.project("8xH100", 50)
    d = projection.to_dict()
    assert d["gpu_count"] == 8
    assert d["target_seats"] == 50
    assert len(d["rows"]) == 2
    assert {r["harness"] for r in d["rows"]} == {"deepseek-native", "openai-shape"}


def test_projector_from_store_empty(tmp_path):
    """No measurements yet → projection still runs with parity defaults."""
    proj = Projector.from_store(store_dir=tmp_path / ".envelcost")
    projection = proj.project("4xH100", 30)
    # No profiles → all harnesses default to 1.0x.
    for r in projection.rows:
        assert r.multiplier == 1.0


def _write_store_openai_only(store_dir, n_tasks: int = 5) -> None:
    """Persist a PARTIAL single-harness store: ONLY openai-shape profiles.

    Mirrors the natural single-envelope flow `envelcost run --harnesses
    openai-shape` produces (the v0.2.0 --harnesses fix made it first-class) —
    a store where the deepseek-native baseline was never measured.
    """
    import json
    from datetime import datetime, timezone

    store_dir.mkdir(parents=True, exist_ok=True)
    out = store_dir / "profiles.jsonl"
    rows = []
    for i in range(n_tasks):
        prof = EnvelopeProfile(
            harness="openai-shape",
            model="deepseek-v4-flash",
            task_id=f"swe-bench-mini-{i + 1:03d}",
            tool_call_envelope_sha="abc",
            input_tokens=400,
            output_tokens=0,
            envelope_overhead_tokens=300,
            quality_score=1.0,
            multiplier_vs_baseline=4.0,
            measured_at=datetime.now(timezone.utc),
        )
        rows.append(json.dumps(prof.to_dict(), ensure_ascii=False))
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_project_partial_single_harness_store_includes_baseline_row(tmp_path):
    """fix-project-summary-false-not-fit: a store holding ONLY openai-shape
    profiles must still project a deepseek-native baseline row at 1.0x (parity)
    so the project summary never prints a false 'deepseek-native does NOT fit'.

    Previously `project` derived harnesses ONLY from measured profiles
    (envelcost/projector.py:118), so a partial single-harness store left
    deepseek-native out of `rows`; `fits_harness("deepseek-native")` returned
    None (envelcost/projector.py:80-84) and the CLI printed it as
    'does NOT fit' (envelcost/cli.py:233) — a FALSE verdict for the 1.0x
    baseline that always fits first. Existing projector tests covered only a
    2-harness store or an EMPTY store (parity fallback), never a partial
    single-harness store, so the defect was untested.
    """
    store = tmp_path / ".envelcost"
    _write_store_openai_only(store, n_tasks=5)
    proj = Projector.from_store(store_dir=store)
    # Sanity: the store really is partial (only openai-shape was measured).
    assert {p.harness for p in proj.profiles} == {"openai-shape"}
    projection = proj.project("8xH100", 50)
    # The baseline harness must now get a real row projected at 1.0x parity
    # (multiplier_for defaults unmeasured harnesses to 1.0), fitting 50 seats
    # (64 raw seats / 1.0 = 64 >= 50).
    native = next(
        (r for r in projection.rows if r.harness == "deepseek-native"), None
    )
    assert native is not None, (
        "deepseek-native row missing from partial-store projection"
    )
    assert native.multiplier == 1.0
    assert native.fits_target is True
    # fits_harness must return True (NOT None -> printed as 'does NOT fit').
    assert projection.fits_harness("deepseek-native") is True
