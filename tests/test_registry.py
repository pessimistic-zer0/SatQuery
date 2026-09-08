"""Only permitted parameters may reach a tool, and selection must be explainable."""

from __future__ import annotations

from satquery.enums import InputConfiguration, Task
from satquery.passport import build_passport
from satquery.registry import REGISTRY
from satquery.tools import RS_VLM_VQA, SPECTRAL_INDEX


def test_every_declared_tool_is_registered():
    from satquery.tools import BUILTIN_TOOLS

    for spec in BUILTIN_TOOLS:
        assert REGISTRY.get(spec.name) is spec


def test_defaults_are_applied_when_nothing_is_supplied():
    accepted, rejected = RS_VLM_VQA.validate_params(None)
    assert accepted["max_new_tokens"] == 128
    assert accepted["use_adapter"] is True
    assert rejected == {}


def test_unknown_parameters_are_rejected_not_ignored():
    accepted, rejected = RS_VLM_VQA.validate_params({"secret_flag": True, "temperature": 0.5})
    assert "secret_flag" in rejected
    assert "not a permitted parameter" in rejected["secret_flag"]
    assert accepted["temperature"] == 0.5


def test_out_of_range_values_are_rejected_and_fall_back_to_the_default():
    accepted, rejected = RS_VLM_VQA.validate_params({"temperature": 9.0})
    assert "temperature" in rejected
    assert accepted["temperature"] == RS_VLM_VQA.params[1].default


def test_choices_are_enforced():
    accepted, rejected = SPECTRAL_INDEX.validate_params({"index": "nsdi"})
    assert "index" in rejected
    assert accepted["index"] == "auto"
    accepted, rejected = SPECTRAL_INDEX.validate_params({"index": "ndwi"})
    assert rejected == {}
    assert accepted["index"] == "ndwi"


def test_selection_reports_why_each_tool_was_dropped(paths):
    passports = [build_passport(paths["single_optical"])]
    eligible, candidates = REGISTRY.select(
        task=Task.VQA, configuration=InputConfiguration.SINGLE,
        passports=passports, gpu_available=False,
    )
    names = {c.spec.name for c in candidates}
    assert "scene_statistics" in names
    assert all(c.reason for c in candidates)
    # With no GPU, the neural VQA model must be excluded with a stated reason.
    vlm = next(c for c in candidates if c.spec.name == "rs_vlm_vqa")
    assert not vlm.eligible
    assert "GPU" in vlm.reason
    assert any(spec.name == "scene_statistics" for spec in eligible)


def test_sar_tools_are_not_offered_for_an_optical_image(paths):
    passports = [build_passport(paths["single_optical"])]
    _, candidates = REGISTRY.select(
        task=Task.VQA, configuration=InputConfiguration.SINGLE,
        passports=passports, gpu_available=True,
    )
    sar = next(c for c in candidates if c.spec.name == "sar_backscatter")
    assert not sar.eligible
    assert "sar" in sar.reason.lower()


def test_implemented_tools_outrank_pending_ones(paths):
    passports = [build_passport(paths["single_optical"])]
    eligible, _ = REGISTRY.select(
        task=Task.VQA, configuration=InputConfiguration.SINGLE,
        passports=passports, gpu_available=False,
    )
    assert eligible[0].implemented
