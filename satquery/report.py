"""Downloadable reports.

The problem statement asks for an auditable execution summary and a downloadable
report. Both come from the same trace object: the JSON export is the machine
record, the PDF is what a reviewer reads.
"""

from __future__ import annotations

import json
import os

from .compat import CompatibilityReport
from .controller import RunResult
from .enums import StepKind

PDF_AVAILABLE = True
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
except ImportError:  # pragma: no cover - PDF export is optional
    PDF_AVAILABLE = False


def build_report_dict(result: RunResult) -> dict:
    """The complete machine-readable record of one run."""
    return {
        "report_type": "satquery_execution_report",
        "generated_from": "SatQuery AI controller",
        **result.to_dict(),
    }


def write_json_report(result: RunResult, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(build_report_dict(result), handle, indent=2)
    return path


def _status_hex(status: str) -> str:
    """Colour for a status, as the '#rrggbb' string reportlab markup expects."""
    return "#" + _status_colour(status).hexval()[2:]


def _status_colour(status: str):
    return {
        "ok": colors.HexColor("#1b7f4b"),
        "pass": colors.HexColor("#1b7f4b"),
        "warn": colors.HexColor("#b26a00"),
        "fail": colors.HexColor("#b3261e"),
        "rejected": colors.HexColor("#b3261e"),
        "failed": colors.HexColor("#b3261e"),
        "not_implemented": colors.HexColor("#5b5b66"),
    }.get(status, colors.HexColor("#2b2b33"))


def _kv_table(rows: list[tuple[str, str]], styles, width_mm=170) -> Table:
    body = [[Paragraph(f"<b>{k}</b>", styles["cell"]), Paragraph(v, styles["cell"])]
            for k, v in rows]
    table = Table(body, colWidths=[45 * mm, (width_mm - 45) * mm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#dcdce4")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def write_pdf_report(result: RunResult, path: str) -> str:
    """Render the trace as a one-or-two page audit document."""
    if not PDF_AVAILABLE:
        raise RuntimeError("reportlab is not installed; PDF export is unavailable")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    trace = result.trace
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=17, spaceAfter=2,
                                textColor=colors.HexColor("#14141c")),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9,
                              textColor=colors.HexColor("#5b5b66"), spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, spaceBefore=12,
                             spaceAfter=5, textColor=colors.HexColor("#14141c")),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=8.5, leading=11.5),
        "mono": ParagraphStyle("m", parent=base["Normal"], fontName="Courier", fontSize=8,
                               leading=10.5),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=13),
    }

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"SatQuery AI execution report {trace.run_id}",
    )
    story = [
        Paragraph("SatQuery AI - Execution Report", styles["title"]),
        Paragraph(f"Run {trace.run_id} &nbsp;·&nbsp; {trace.created_at} &nbsp;·&nbsp; "
                  f"{trace.total_ms:.0f} ms", styles["sub"]),
    ]

    # --- query and outcome -------------------------------------------------
    outcome = ("REJECTED" if trace.rejected
               else (trace.answer or "no answer produced").replace("\n\n", "<br/><br/>"))
    story += [
        Paragraph("Query and outcome", styles["h2"]),
        _kv_table([
            ("Query", trace.query or "(none)"),
            ("Inputs", ", ".join(trace.input_files) or "(none)"),
            ("Configuration", (trace.configuration.value if trace.configuration else "n/a")),
            ("Task", trace.task.value if trace.task else "n/a"),
            ("Tools", ", ".join(trace.tools) or "none selected"),
            ("Confidence", f"{trace.confidence:.2f}" if trace.confidence is not None else "n/a"),
            ("Answer", outcome),
        ], styles),
    ]

    if trace.rejected:
        story += [
            Paragraph("Rejection reasons", styles["h2"]),
            _kv_table([(f"{i + 1}", r) for i, r in enumerate(trace.rejection_reasons)], styles),
        ]

    # --- sensor passports --------------------------------------------------
    if result.passports:
        story.append(Paragraph("Sensor passports", styles["h2"]))
        header = ["File", "Sensor", "Size", "Resolution", "CRS", "Extent", "Date"]
        rows = [[Paragraph(f"<b>{h}</b>", styles["cell"]) for h in header]]
        for p in result.passports:
            rows.append([
                Paragraph(p.filename, styles["cell"]),
                Paragraph(f"{p.sensor_guess or p.modality.value}<br/>"
                          f"<font size=7>conf {p.modality_confidence:.2f}</font>", styles["cell"]),
                Paragraph(f"{p.width}x{p.height}<br/><font size=7>{p.band_count} band(s)</font>",
                          styles["cell"]),
                Paragraph(f"{p.gsd_x_m:.1f} m/px" if p.gsd_x_m else "n/a", styles["cell"]),
                Paragraph(p.crs or "none", styles["cell"]),
                Paragraph(f"{p.area_km2:.2f} km2" if p.area_km2 else "pixel-space", styles["cell"]),
                Paragraph(p.acquisition_date or "unknown", styles["cell"]),
            ])
        table = Table(rows, colWidths=[32 * mm, 30 * mm, 22 * mm, 20 * mm, 26 * mm, 22 * mm, 18 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f5")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dcdce4")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)

    # --- compatibility checks ---------------------------------------------
    report: CompatibilityReport | None = result.compatibility
    if report is not None and report.checks:
        story.append(Paragraph("Input compatibility checks", styles["h2"]))
        rows = [[Paragraph("<b>Check</b>", styles["cell"]),
                 Paragraph("<b>Status</b>", styles["cell"]),
                 Paragraph("<b>Detail</b>", styles["cell"])]]
        for check in report.checks:
            rows.append([
                Paragraph(check.name, styles["cell"]),
                Paragraph(f'<font color="{_status_hex(check.status.value)}">'
                          f"<b>{check.status.value.upper()}</b></font>", styles["cell"]),
                Paragraph(check.detail, styles["cell"]),
            ])
        table = Table(rows, colWidths=[35 * mm, 20 * mm, 115 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f5")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dcdce4")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)

    # --- execution trace ---------------------------------------------------
    story.append(Paragraph("Execution trace", styles["h2"]))
    rows = [[Paragraph(f"<b>{h}</b>", styles["cell"])
             for h in ["#", "Stage", "Name", "Status", "ms", "Parameters / detail"]]]
    for step in trace.steps:
        params = ", ".join(f"{k}={v}" for k, v in step.params.items())
        detail = step.detail
        if params:
            detail = f"<font face='Courier' size=7>{params}</font><br/>{detail}"
        if step.rejected_params:
            rejected = "; ".join(f"{k}: {v}" for k, v in step.rejected_params.items())
            detail += f"<br/><font color='#b3261e' size=7>rejected: {rejected}</font>"
        rows.append([
            Paragraph(str(step.index), styles["cell"]),
            Paragraph(step.kind.value, styles["cell"]),
            Paragraph(step.name, styles["cell"]),
            Paragraph(f'<font color="{_status_hex(step.status)}">'
                      f"{step.status}</font>", styles["cell"]),
            Paragraph(f"{step.duration_ms:.1f}", styles["cell"]),
            Paragraph(detail[:900], styles["cell"]),
        ])
    table = Table(rows, colWidths=[7 * mm, 24 * mm, 30 * mm, 22 * mm, 12 * mm, 75 * mm],
                  repeatRows=1)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f5")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dcdce4")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(table)

    # --- warnings ----------------------------------------------------------
    if trace.warnings:
        story.append(KeepTogether([
            Paragraph("Warnings", styles["h2"]),
            _kv_table([(f"{i + 1}", w) for i, w in enumerate(trace.warnings)], styles),
        ]))

    story += [
        Spacer(1, 8 * mm),
        Paragraph(
            "Generated by the SatQuery AI agentic controller. Every row above is taken from the "
            "recorded execution trace; no part of this report is generated text.",
            styles["sub"],
        ),
    ]
    doc.build(story)
    return path
