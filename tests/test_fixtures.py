"""The fixtures must know their own ground truth, or the tests prove nothing."""

from __future__ import annotations

import pytest

from satquery.fixtures.synth import CLASS_ID, build_scene, render_optical, render_sar


def test_planted_areas_are_recoverable():
    scene = build_scene(200, 200, gsd_m=10.0)
    total = sum(scene.class_pixels.values())
    assert total == 200 * 200
    assert scene.area_km2("water") > 0
    assert scene.area_km2("built_up") > 0
    assert sum(scene.areas_km2().values()) == pytest.approx(4.0)  # 2 km x 2 km


def test_bitemporal_growth_is_the_planted_amount():
    t1 = build_scene(200, 200, urban_blocks=((0.58, 0.55, 0.27, 0.24),))
    t2 = build_scene(200, 200, urban_blocks=((0.58, 0.55, 0.27, 0.24), (0.30, 0.60, 0.18, 0.16)))
    delta = t2.area_km2("built_up") - t1.area_km2("built_up")
    assert delta > 0
    # The added block is 18% x 16% of a 2 km x 2 km scene, minus any overlap.
    assert delta == pytest.approx(0.115, abs=0.02)


def test_optical_spectra_have_the_right_sign():
    scene = build_scene(128, 128)
    cube = render_optical(scene, seed=1).astype(float) / 10_000.0
    red, nir = cube[3], cube[7]
    ndvi = (nir - red) / (nir + red + 1e-9)
    vegetation = scene.labels == CLASS_ID["vegetation"]
    water = scene.labels == CLASS_ID["water"]
    assert ndvi[vegetation].mean() > 0.6
    assert ndvi[water].mean() < 0.0


def test_sar_water_is_darker_than_built_up():
    scene = build_scene(128, 128)
    sar = render_sar(scene, seed=1)
    vv = sar[0]
    assert vv[scene.labels == CLASS_ID["water"]].mean() < -18.0
    assert vv[scene.labels == CLASS_ID["built_up"]].mean() > -8.0
