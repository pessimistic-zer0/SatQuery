"""Incompatible inputs must be rejected before any model is selected."""

from __future__ import annotations

from satquery.compat import order_bitemporal, validate_inputs
from satquery.enums import CheckStatus, InputConfiguration
from satquery.passport import build_passport


def test_single_image_is_always_a_single_configuration(paths):
    report = validate_inputs([build_passport(paths["single_optical"])])
    assert report.configuration is InputConfiguration.SINGLE
    assert report.compatible


def test_optical_plus_sar_is_a_cross_modal_pair(paths):
    report = validate_inputs([
        build_passport(paths["cross_optical"]),
        build_passport(paths["cross_sar"]),
    ])
    assert report.configuration is InputConfiguration.CROSS_MODAL_PAIR
    assert report.compatible
    assert report.metrics["overlap_fraction"] > 0.99


def test_two_optical_dates_form_a_bitemporal_pair(paths):
    report = validate_inputs([
        build_passport(paths["bitemporal_t1"]),
        build_passport(paths["bitemporal_t2"]),
    ])
    assert report.configuration is InputConfiguration.BI_TEMPORAL_PAIR
    assert report.compatible
    assert report.metrics["acquisition_gap_days"] > 700


def test_non_overlapping_scenes_are_rejected(paths):
    report = validate_inputs([
        build_passport(paths["bitemporal_t1"]),
        build_passport(paths["incompatible_far"]),
    ])
    assert not report.compatible
    assert report.configuration is InputConfiguration.INCOMPATIBLE
    assert any(c.name == "co_registration" and c.status is CheckStatus.FAIL
               for c in report.checks)
    assert report.reasons


def test_mixing_georeferenced_and_plain_png_is_rejected(paths):
    report = validate_inputs([
        build_passport(paths["single_optical"]),
        build_passport(paths["benchmark_png"]),
    ])
    assert not report.compatible
    assert any("georeferenced" in r for r in report.reasons)


def test_three_images_are_rejected(paths):
    report = validate_inputs([
        build_passport(paths["single_optical"]),
        build_passport(paths["single_sar"]),
        build_passport(paths["bitemporal_t1"]),
    ])
    assert not report.compatible
    assert any("unsupported input count" in r for r in report.reasons)


def test_bitemporal_ordering_uses_dates(paths):
    t1 = build_passport(paths["bitemporal_t1"])
    t2 = build_passport(paths["bitemporal_t2"])
    assert order_bitemporal(t2, t1)[0].acquisition_date == t1.acquisition_date
