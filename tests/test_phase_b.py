"""Phase B tools, checked against the areas the fixtures actually planted.

The point of the synthetic fixtures is that these assertions compare against a
number that was placed deliberately, not against a picture that looks plausible.
Tolerances are wide because index thresholding is not segmentation - but the
sign, the ordering and the rough magnitude must all be right.
"""

from __future__ import annotations

import os

import pytest

from satquery.controller import Controller
from satquery.enums import Task, ToolStatus
from satquery.passport import build_passport
from satquery.registry import REGISTRY, ToolContext
from satquery.tools.change import run_change_index_diff
from satquery.tools.fusion import run_modality_reliability
from satquery.tools.optical import run_spectral_index
from satquery.tools.sar import run_sar_backscatter
from satquery.tools.verify import run_physics_verifier
from satquery.enums import InputConfiguration


def _ctx(paths, keys, task, configuration, tmp_path, params=None, entities=None, prior=None):
    passports = [build_passport(paths[k]) for k in keys]
    return ToolContext(
        passports=passports,
        query="",
        task=task,
        configuration=configuration,
        params=params or {},
        workdir=str(tmp_path),
        entities=entities or {},
        prior=prior or [],
    )


def _class(metrics, name):
    return next(c for c in metrics["classes"] if c["class"] == name)


# --- spectral index --------------------------------------------------------

def test_spectral_index_recovers_the_planted_water_area(paths, demo_set, tmp_path):
    truth = demo_set["files"]["single_optical"]["areas_km2"]
    ctx = _ctx(paths, ["single_optical"], Task.VQA, InputConfiguration.SINGLE, tmp_path,
               params={"index": "auto", "min_region_px": 25})
    result = run_spectral_index(ctx)
    assert result.status is ToolStatus.OK
    measured = _class(result.metrics, "water")["area_km2"]
    assert measured == pytest.approx(truth["water"], rel=0.35)


def test_spectral_index_recovers_the_planted_builtup_area(paths, demo_set, tmp_path):
    truth = demo_set["files"]["single_optical"]["areas_km2"]
    ctx = _ctx(paths, ["single_optical"], Task.VQA, InputConfiguration.SINGLE, tmp_path)
    result = run_spectral_index(ctx)
    measured = _class(result.metrics, "built_up")["area_km2"]
    assert measured == pytest.approx(truth["built_up"], rel=0.35)


def test_bare_soil_is_not_labelled_built_up(paths, demo_set, tmp_path):
    """NDBI alone confuses the two; the BSI exclusion is what separates them."""
    truth = demo_set["files"]["single_optical"]["areas_km2"]
    ctx = _ctx(paths, ["single_optical"], Task.VQA, InputConfiguration.SINGLE, tmp_path)
    result = run_spectral_index(ctx)
    bare = _class(result.metrics, "bare_soil")["area_km2"]
    assert bare == pytest.approx(truth["bare_soil"], rel=0.35)


def test_water_bodies_are_counted_separately(paths, tmp_path):
    """The fixture has a lake and a river; 'how many water bodies' must see both."""
    ctx = _ctx(paths, ["single_optical"], Task.VQA, InputConfiguration.SINGLE, tmp_path,
               entities={"classes": ["water"]})
    result = run_spectral_index(ctx)
    assert _class(result.metrics, "water")["regions"] >= 2
    assert "water" in result.answer.lower()


def test_spectral_index_writes_visual_evidence(paths, tmp_path):
    ctx = _ctx(paths, ["single_optical"], Task.VQA, InputConfiguration.SINGLE, tmp_path)
    result = run_spectral_index(ctx)
    assert result.artifacts
    for artifact in result.artifacts:
        assert os.path.isfile(artifact["path"])
    assert any(a["format"] == "png" for a in result.artifacts)
    assert any(a["format"] == "geotiff" for a in result.artifacts)


def test_a_named_index_is_honoured(paths, tmp_path):
    ctx = _ctx(paths, ["single_optical"], Task.VQA, InputConfiguration.SINGLE, tmp_path,
               params={"index": "ndvi", "threshold": 0.3, "min_region_px": 25})
    result = run_spectral_index(ctx)
    assert result.metrics["index"] == "ndvi"
    assert result.metrics["threshold"] == 0.3
    assert "NDVI" in result.answer


def test_rgb_png_cannot_support_spectral_indices(paths, tmp_path):
    """A benchmark RGB PNG has no NIR or SWIR band, so the tool must decline.

    This is the honest outcome, not a failure: the registry keeps spectral_index
    off an RGB input in the first place, and scene_statistics answers instead.
    """
    ctx = _ctx(paths, ["benchmark_png"], Task.VQA, InputConfiguration.SINGLE, tmp_path)
    result = run_spectral_index(ctx)
    assert result.status is ToolStatus.SKIPPED
    assert "band" in result.detail or "index" in result.detail


def test_registry_keeps_spectral_index_away_from_rgb_input(paths):
    """The routing layer must not offer a tool the image cannot support."""
    spec = REGISTRY.get("spectral_index")
    ok, reason = spec.applicable_to([build_passport(paths["benchmark_png"])])
    assert not ok
    assert "band" in reason


# --- SAR --------------------------------------------------------------------

def test_sar_finds_water_from_backscatter(paths, demo_set, tmp_path):
    truth = demo_set["files"]["single_sar"]["areas_km2"]
    ctx = _ctx(paths, ["single_sar"], Task.VQA, InputConfiguration.SINGLE, tmp_path,
               params={"water_vv_db": -18.0, "builtup_vv_db": -5.0, "speckle_filter": "lee"})
    result = run_sar_backscatter(ctx)
    assert result.status is ToolStatus.OK
    measured = _class(result.metrics, "water")["area_km2"]
    assert measured == pytest.approx(truth["water"], rel=0.5)
    assert result.metrics["vv_mean_db"] < 0


def test_speckle_filter_choice_is_recorded(paths, tmp_path):
    ctx = _ctx(paths, ["single_sar"], Task.VQA, InputConfiguration.SINGLE, tmp_path,
               params={"speckle_filter": "median", "water_vv_db": -18.0,
                       "builtup_vv_db": -5.0})
    result = run_sar_backscatter(ctx)
    assert result.metrics["speckle_filter"] == "median"
    assert "median" in result.detail


# --- change -----------------------------------------------------------------

def test_change_recovers_the_planted_builtup_growth(paths, demo_set, tmp_path):
    truth = demo_set["bitemporal_truth_km2"]["built_up"]
    ctx = _ctx(paths, ["bitemporal_t1", "bitemporal_t2"], Task.CHANGE_VQA,
               InputConfiguration.BI_TEMPORAL_PAIR, tmp_path,
               entities={"classes": ["built_up"]})
    result = run_change_index_diff(ctx)
    assert result.status is ToolStatus.OK
    delta = next(d for d in result.metrics["class_deltas"] if d["class"] == "built_up")
    assert delta["delta_km2"] > 0, "built-up was planted to grow"
    assert delta["delta_km2"] == pytest.approx(truth, rel=0.5)


def test_change_recovers_the_planted_water_loss(paths, demo_set, tmp_path):
    truth = demo_set["bitemporal_truth_km2"]["water"]
    ctx = _ctx(paths, ["bitemporal_t1", "bitemporal_t2"], Task.CHANGE_VQA,
               InputConfiguration.BI_TEMPORAL_PAIR, tmp_path)
    result = run_change_index_diff(ctx)
    delta = next(d for d in result.metrics["class_deltas"] if d["class"] == "water")
    assert delta["delta_km2"] < 0, "the lake was planted to shrink"
    assert delta["delta_km2"] == pytest.approx(truth, rel=0.5)


def test_change_map_is_written_as_evidence(paths, tmp_path):
    ctx = _ctx(paths, ["bitemporal_t1", "bitemporal_t2"], Task.CHANGE_MAP,
               InputConfiguration.BI_TEMPORAL_PAIR, tmp_path)
    result = run_change_index_diff(ctx)
    assert result.artifacts
    assert all(os.path.isfile(a["path"]) for a in result.artifacts)
    assert result.metrics["changed_area_km2"] > 0


def test_otsu_and_fixed_thresholding_both_run(paths, tmp_path):
    for mode in ("otsu", "fixed"):
        ctx = _ctx(paths, ["bitemporal_t1", "bitemporal_t2"], Task.CHANGE_MAP,
                   InputConfiguration.BI_TEMPORAL_PAIR, tmp_path,
                   params={"thresholding": mode, "change_threshold": 0.4,
                           "index": "auto", "min_region_px": 25})
        result = run_change_index_diff(ctx)
        assert result.status is ToolStatus.OK
        assert result.metrics["thresholding"] == mode


# --- optical-SAR fusion -----------------------------------------------------

def test_reliability_map_finds_the_cloud(paths, demo_set, tmp_path):
    planted = demo_set["files"]["cross_optical"]["cloud_fraction"]
    ctx = _ctx(paths, ["cross_optical", "cross_sar"], Task.CROSS_MODAL_ANALYSIS,
               InputConfiguration.CROSS_MODAL_PAIR, tmp_path)
    result = run_modality_reliability(ctx)
    assert result.status is ToolStatus.OK
    assert result.metrics["cloud_fraction"] == pytest.approx(planted, abs=0.08)


def test_sar_answers_where_optical_is_blocked(paths, tmp_path):
    ctx = _ctx(paths, ["cross_optical", "cross_sar"], Task.CROSS_MODAL_ANALYSIS,
               InputConfiguration.CROSS_MODAL_PAIR, tmp_path)
    result = run_modality_reliability(ctx)
    # The fixture puts cloud over part of the scene; the radar must still find
    # water there, and the fused estimate must exceed the optical-only one.
    assert result.metrics["water_under_cloud_km2"] > 0
    assert result.metrics["fused_water_km2"] >= result.metrics["optical_only_water_km2"]
    assert "radar" in result.answer.lower() or "sar" in result.answer.lower()


def test_sensors_largely_agree_where_the_sky_is_clear(paths, tmp_path):
    ctx = _ctx(paths, ["cross_optical", "cross_sar"], Task.CROSS_MODAL_ANALYSIS,
               InputConfiguration.CROSS_MODAL_PAIR, tmp_path)
    result = run_modality_reliability(ctx)
    assert result.metrics["agreement_rate"] > 0.80
    assert 0.0 < result.confidence <= 0.98


# --- physics verifier -------------------------------------------------------

def test_verifier_supports_a_true_growth_claim(paths, tmp_path):
    ctx = _ctx(paths, ["bitemporal_t1", "bitemporal_t2"], Task.CHANGE_VQA,
               InputConfiguration.BI_TEMPORAL_PAIR, tmp_path,
               params={"strictness": "balanced"})
    change = run_change_index_diff(ctx)
    ctx.prior = [("change_index_diff", change)]
    verdict = run_physics_verifier(ctx)
    assert verdict.status is ToolStatus.OK
    assert verdict.metrics["contradicted"] == 0
    assert verdict.metrics["supported"] >= 1
    assert "supported" in verdict.answer


def test_verifier_contradicts_a_false_claim(paths, tmp_path):
    """A claim the pixels do not support must lower the confidence, not pass."""
    from satquery.registry import ToolResult

    ctx = _ctx(paths, ["bitemporal_t1", "bitemporal_t2"], Task.CHANGE_VQA,
               InputConfiguration.BI_TEMPORAL_PAIR, tmp_path)
    false_claim = ToolResult(
        status=ToolStatus.OK,
        answer="The built-up area decreased between the two dates.",
        confidence=0.9,
    )
    ctx.prior = [("some_model", false_claim)]
    verdict = run_physics_verifier(ctx)
    assert verdict.metrics["contradicted"] >= 1
    assert verdict.confidence < 0.6
    assert "contradicted" in verdict.answer


def test_verifier_skips_when_there_is_nothing_to_check(paths, tmp_path):
    ctx = _ctx(paths, ["single_optical"], Task.VQA, InputConfiguration.SINGLE, tmp_path)
    assert run_physics_verifier(ctx).status is ToolStatus.SKIPPED


# --- end to end through the controller --------------------------------------

def test_controller_answers_a_change_question_with_real_numbers(paths, tmp_path):
    controller = Controller(workdir=str(tmp_path / "runs"), gpu_available=False)
    result = controller.run(
        "Has the built-up area increased, decreased, or remained unchanged?",
        [paths["bitemporal_t1"], paths["bitemporal_t2"]],
    )
    assert not result.trace.rejected
    assert result.trace.task is Task.CHANGE_VQA
    assert "change_index_diff" in result.trace.tools
    assert "physics_verifier" in result.trace.tools
    assert "km2" in result.trace.answer
    assert result.trace.artifacts


def test_controller_answers_a_cross_modal_question(paths, tmp_path):
    controller = Controller(workdir=str(tmp_path / "runs"), gpu_available=False)
    result = controller.run(
        "Use the optical and SAR images together to identify built-up and water regions.",
        [paths["cross_optical"], paths["cross_sar"]],
    )
    assert "modality_reliability" in result.trace.tools
    assert result.trace.answer
    assert result.trace.confidence is not None


def test_every_registered_tool_is_now_implemented_or_neural():
    for spec in REGISTRY.all():
        if spec.backend == "classical":
            assert spec.implemented, f"{spec.name} is classical but still pending"


def test_radar_reports_built_up_through_the_cloud(paths, demo_set, tmp_path):
    """SAR penetrates cloud, so its classes must cover the whole scene.

    The fixture places cloud over the town, so restricting the radar result to
    cloud-free pixels would discard most of it - and discard the entire reason
    for carrying a second sensor.
    """
    truth = demo_set["files"]["cross_optical"]["areas_km2"]["built_up"]
    ctx = _ctx(paths, ["cross_optical", "cross_sar"], Task.CROSS_MODAL_ANALYSIS,
               InputConfiguration.CROSS_MODAL_PAIR, tmp_path)
    result = run_modality_reliability(ctx)
    measured = result.metrics["sar_builtup_km2"]
    assert measured == pytest.approx(truth, rel=0.6)
    assert result.metrics["sar_builtup_under_cloud_km2"] > 0
    assert "invisible to the optical image" in result.answer
