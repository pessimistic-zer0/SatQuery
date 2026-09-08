"""The measurement layer: index maths, thresholding and mask hygiene."""

from __future__ import annotations

import numpy as np
import pytest

from satquery.fixtures.synth import CLASS_ID, build_scene, render_optical, render_sar
from satquery.indices import (
    compute_index,
    count_regions,
    despeckle,
    lee_filter,
    normalised_difference,
    otsu_threshold,
    remove_small_regions,
    threshold_confidence,
)


def _bands(scene, seed=1):
    cube = render_optical(scene, seed=seed).astype(np.float32) / 10_000.0
    return {"blue": cube[1], "green": cube[2], "red": cube[3],
            "nir": cube[7], "swir1": cube[10]}


def test_normalised_difference_handles_a_zero_denominator():
    out = normalised_difference(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.0)


def test_indices_have_the_expected_sign_per_class():
    scene = build_scene(160, 160)
    bands = _bands(scene)
    water = scene.labels == CLASS_ID["water"]
    vegetation = scene.labels == CLASS_ID["vegetation"]
    built = scene.labels == CLASS_ID["built_up"]

    ndvi = compute_index("ndvi", bands)
    mndwi = compute_index("mndwi", bands)
    ndbi = compute_index("ndbi", bands)

    assert ndvi[vegetation].mean() > 0.6      # healthy canopy
    assert ndvi[water].mean() < 0.0           # water absorbs NIR
    assert mndwi[water].mean() > 0.3          # water is bright in green, dark in SWIR
    assert mndwi[vegetation].mean() < 0.0
    assert ndbi[built].mean() > 0.0           # impervious surfaces are SWIR-bright
    assert ndbi[vegetation].mean() < 0.0


def test_unknown_index_is_rejected():
    with pytest.raises(KeyError, match="unknown index"):
        compute_index("ndxx", {})


def test_missing_band_is_reported():
    with pytest.raises(KeyError, match="needs the bands"):
        compute_index("ndvi", {"nir": np.zeros((4, 4))})


def test_otsu_finds_the_valley_of_a_bimodal_histogram():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0.2, 0.03, 4000), rng.normal(0.8, 0.03, 4000)])
    threshold = otsu_threshold(values)
    assert 0.35 < threshold < 0.65


def test_otsu_on_a_constant_array_is_that_constant():
    assert otsu_threshold(np.full(100, 0.42)) == pytest.approx(0.42, abs=1e-3)


def test_small_regions_are_removed_and_counted():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:30, 10:30] = True     # one large block
    mask[50, 50] = True           # one speck
    mask[55, 55] = True           # another speck
    cleaned, dropped = remove_small_regions(mask, min_pixels=25)
    assert dropped == 2
    assert count_regions(cleaned) == 1
    assert cleaned.sum() == 400


def test_region_counting_answers_how_many():
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:10, 2:10] = True
    mask[20:30, 20:30] = True
    mask[2:10, 25:35] = True
    assert count_regions(mask) == 3


def test_lee_filter_reduces_speckle_variance():
    scene = build_scene(160, 160)
    vv = render_sar(scene, seed=2)[0]
    water = scene.labels == CLASS_ID["water"]
    filtered = lee_filter(vv, 5)
    # Within a uniform class the filter must cut the spread without moving the mean.
    assert filtered[water].std() < vv[water].std()
    assert filtered[water].mean() == pytest.approx(vv[water].mean(), abs=1.5)


def test_despeckle_reports_which_filter_ran():
    array = np.random.default_rng(0).normal(0, 1, (32, 32)).astype(np.float32)
    _, note = despeckle(array, "none")
    assert "no speckle filter" in note
    _, note = despeckle(array, "median")
    assert "median" in note
    _, note = despeckle(array, "lee")
    assert "Lee" in note


def test_confidence_rises_with_distance_from_the_threshold():
    near = np.full(1000, 0.01, dtype=np.float32)
    far = np.full(1000, 0.60, dtype=np.float32)
    assert threshold_confidence(near, 0.0) < threshold_confidence(far, 0.0)
    assert threshold_confidence(far, 0.0) <= 0.99
