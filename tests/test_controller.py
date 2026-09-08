"""End-to-end controller behaviour, including what the trace must contain."""

from __future__ import annotations

from satquery.controller import Controller
from satquery.enums import InputConfiguration, StepKind, Task, ToolStatus


def controller(tmp_path) -> Controller:
    # Force the no-GPU path so the tests exercise the classical fallback.
    return Controller(workdir=str(tmp_path / "runs"), gpu_available=False)


def test_single_image_query_produces_an_answer(paths, tmp_path):
    result = controller(tmp_path).run(
        "How many water bodies are visible in this image?", [paths["single_optical"]],
    )
    assert not result.trace.rejected
    assert result.trace.configuration is InputConfiguration.SINGLE
    assert result.trace.task is Task.VQA
    assert result.trace.answer
    assert 0.0 < result.trace.confidence <= 1.0
    # spectral_index outranks scene_statistics now that it is implemented.
    assert "spectral_index" in result.trace.tools


def test_trace_records_every_stage(paths, tmp_path):
    result = controller(tmp_path).run("Describe this image.", [paths["single_optical"]])
    kinds = [step.kind for step in result.trace.steps]
    for expected in (StepKind.VALIDATE, StepKind.COMPATIBILITY, StepKind.INTERPRET,
                     StepKind.ROUTE, StepKind.TOOL, StepKind.INTEGRATE):
        assert expected in kinds, f"{expected} missing from the trace"
    assert all(step.index == i + 1 for i, step in enumerate(result.trace.steps))
    assert result.trace.total_ms > 0


def test_incompatible_pair_is_rejected_before_any_tool_runs(paths, tmp_path):
    result = controller(tmp_path).run(
        "What changed between these two dates?",
        [paths["bitemporal_t1"], paths["incompatible_far"]],
    )
    assert result.trace.rejected
    assert result.trace.configuration is InputConfiguration.INCOMPATIBLE
    assert result.trace.rejection_reasons
    assert result.trace.tools == []
    assert not any(step.kind is StepKind.TOOL for step in result.trace.steps)
    assert result.trace.task is None


def test_bitemporal_pair_routes_to_a_change_task(paths, tmp_path):
    result = controller(tmp_path).run(
        "Has the built-up area increased, decreased, or remained unchanged?",
        [paths["bitemporal_t1"], paths["bitemporal_t2"]],
    )
    assert result.trace.configuration is InputConfiguration.BI_TEMPORAL_PAIR
    assert result.trace.task is Task.CHANGE_VQA
    assert result.trace.answer


def test_cross_modal_pair_routes_to_fusion(paths, tmp_path):
    result = controller(tmp_path).run(
        "Use the optical and SAR images together to identify built-up and water regions.",
        [paths["cross_optical"], paths["cross_sar"]],
    )
    assert result.trace.configuration is InputConfiguration.CROSS_MODAL_PAIR
    assert result.trace.task is Task.CROSS_MODAL_ANALYSIS
    # The neural fusion model needs a GPU, so with none available the classical
    # reliability map must take the query and still produce a grounded answer.
    assert "modality_reliability" in result.trace.tools
    assert result.trace.answer


def test_pending_tool_is_reported_not_faked(paths, tmp_path):
    result = Controller(workdir=str(tmp_path / "runs"), gpu_available=True).run(
        "Highlight the water body referred to in the query.", [paths["single_optical"]],
    )
    assert result.trace.task is Task.GROUNDING
    assert "text_grounding" in result.trace.tools
    statuses = {name: r.status for name, r in result.executed}
    assert statuses["text_grounding"] is ToolStatus.NOT_IMPLEMENTED
    assert any("not yet implemented" in w for w in result.trace.warnings)


def test_rejected_parameters_are_recorded_in_the_trace(paths, tmp_path):
    result = controller(tmp_path).run(
        "Describe this image.", [paths["single_optical"]],
        params_by_tool={"spectral_index": {"min_region_px": 50, "not_a_real_param": 3}},
    )
    step = next(s for s in result.trace.steps if s.name == "spectral_index")
    assert step.params["min_region_px"] == 50
    assert "not_a_real_param" in step.rejected_params
    assert any("rejected" in w for w in result.trace.warnings)


def test_task_can_be_overridden(paths, tmp_path):
    result = controller(tmp_path).run(
        "How many water bodies?", [paths["single_optical"]], force_task=Task.CAPTION,
    )
    assert result.trace.task is Task.CAPTION
    assert "overridden" in result.interpretation.note


def test_unreadable_file_is_rejected_not_crashed(tmp_path):
    bad = tmp_path / "broken.tif"
    bad.write_text("not a raster")
    result = controller(tmp_path).run("Describe this.", [str(bad)])
    assert result.trace.rejected
    assert result.trace.rejection_reasons


def test_png_benchmark_input_still_runs(paths, tmp_path):
    result = controller(tmp_path).run("What is in this image?", [paths["benchmark_png"]])
    assert not result.trace.rejected
    assert result.trace.answer
    assert any("pixel" in w for w in result.trace.warnings)


def test_summary_lines_render_the_slide_trace(paths, tmp_path):
    result = controller(tmp_path).run(
        "What changed between these two dates?",
        [paths["bitemporal_t1"], paths["bitemporal_t2"]],
    )
    lines = "\n".join(result.trace.summary_lines())
    assert "task:" in lines
    assert "tools:" in lines
    assert "configuration: bi_temporal_pair" in lines
