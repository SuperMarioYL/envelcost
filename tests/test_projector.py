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
