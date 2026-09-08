"""The downloadable report is generated from the trace, not written by hand."""

from __future__ import annotations

import json

from satquery.controller import Controller
from satquery.report import PDF_AVAILABLE, build_report_dict, write_json_report, write_pdf_report


def _run(paths, tmp_path):
    return Controller(workdir=str(tmp_path / "runs"), gpu_available=False).run(
        "Describe the land cover in this image.", [paths["single_optical"]],
    )


def test_json_report_round_trips(paths, tmp_path):
    result = _run(paths, tmp_path)
    path = write_json_report(result, str(tmp_path / "report.json"))
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["run_id"] == result.run_id
    assert data["trace"]["steps"]
    assert data["passports"][0]["modality"] == "optical"
    assert data["compatibility"]["configuration"] == "single"


def test_report_contains_permitted_parameters(paths, tmp_path):
    result = _run(paths, tmp_path)
    report = build_report_dict(result)
    tool_steps = [s for s in report["trace"]["steps"] if s["kind"] == "tool"]
    assert tool_steps
    assert "params" in tool_steps[0]


def test_pdf_report_is_written(paths, tmp_path):
    if not PDF_AVAILABLE:
        return
    result = _run(paths, tmp_path)
    path = write_pdf_report(result, str(tmp_path / "report.pdf"))
    with open(path, "rb") as handle:
        assert handle.read(5) == b"%PDF-"


def test_pdf_renders_a_rejected_run(paths, tmp_path):
    if not PDF_AVAILABLE:
        return
    result = Controller(workdir=str(tmp_path / "runs"), gpu_available=False).run(
        "What changed?", [paths["bitemporal_t1"], paths["incompatible_far"]],
    )
    assert result.trace.rejected
    path = write_pdf_report(result, str(tmp_path / "rejected.pdf"))
    with open(path, "rb") as handle:
        assert handle.read(5) == b"%PDF-"
