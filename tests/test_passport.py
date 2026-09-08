"""The sensor passport must identify what it is looking at without a model."""

from __future__ import annotations

import pytest

from satquery.enums import Modality
from satquery.passport import PassportError, build_passport


def test_optical_geotiff_is_identified(paths):
    p = build_passport(paths["single_optical"])
    assert p.modality is Modality.OPTICAL
    assert p.modality_confidence > 0.6
    assert p.band_count == 12
    assert p.georeferenced
    assert p.epsg == 32643
    assert p.gsd_x_m == pytest.approx(10.0)
    assert p.area_km2 == pytest.approx((192 * 10) ** 2 / 1e6)
    assert p.acquisition_date == "2024-01-18"
    assert p.modality_evidence


def test_optical_band_roles_resolve(paths):
    p = build_passport(paths["single_optical"])
    assert p.has_roles("red", "nir", "green", "swir1")
    assert p.band_index("nir") == 7  # B08 is the eighth band of the 12-band layout
    assert p.band_index("red") == 3


def test_sar_geotiff_is_identified(paths):
    p = build_passport(paths["single_sar"])
    assert p.modality is Modality.SAR
    assert p.band_count == 2
    assert p.has_roles("vv", "vh")
    # Backscatter in dB is negative; that is the strongest single SAR cue.
    assert p.band_stats[0].mean < 0


def test_png_is_accepted_but_not_georeferenced(paths):
    p = build_passport(paths["benchmark_png"])
    assert not p.georeferenced
    assert p.area_km2 is None
    assert p.gsd_x_m is None
    assert any("pixel" in w for w in p.warnings)


def test_pixel_area_is_none_without_georeferencing(paths):
    assert build_passport(paths["benchmark_png"]).pixel_area_m2() is None
    assert build_passport(paths["single_optical"]).pixel_area_m2() == pytest.approx(100.0)


def test_unreadable_input_raises(tmp_path):
    bad = tmp_path / "not_an_image.tif"
    bad.write_text("this is not a raster")
    with pytest.raises(PassportError):
        build_passport(str(bad))


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(PassportError, match="unsupported extension"):
        build_passport(str(bad))


def test_summary_line_mentions_extent(paths):
    line = build_passport(paths["single_optical"]).summary_line()
    assert "km2" in line
    assert "12 band(s)" in line
