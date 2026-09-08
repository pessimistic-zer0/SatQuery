"""Geometry and scene-statistics tools.

These two are fully implemented in Phase A. They answer nothing about land cover
- that is Phase B - but they are real tools over real raster metadata, so the
controller has something genuine to execute end to end in all three input
configurations, and the trace shows a complete run rather than a plan.
"""

from __future__ import annotations

from ..compat import bbox_overlap_fraction, order_bitemporal
from ..enums import InputConfiguration, Task, ToolStatus
from ..registry import ToolContext, ToolParam, ToolResult, ToolSpec


def _extent_phrase(passport) -> str:
    if passport.area_km2:
        return (
            f"{passport.area_km2:.2f} km2 "
            f"({passport.width * passport.gsd_x_m / 1000:.2f} km x "
            f"{passport.height * passport.gsd_y_m / 1000:.2f} km)"
        )
    return f"{passport.width} x {passport.height} pixels (no georeferencing, so no ground extent)"


def run_scene_statistics(ctx: ToolContext) -> ToolResult:
    """Describe the geometry, sensor and radiometry of a single image."""
    p = ctx.primary
    lines = [
        f"The image is a {p.width} x {p.height} pixel {p.driver} raster with "
        f"{p.band_count} band(s) of type {p.dtype}.",
        f"It was identified as {p.sensor_guess or p.modality.value} imagery "
        f"(modality confidence {p.modality_confidence:.2f}).",
        f"It covers {_extent_phrase(p)}.",
    ]
    if p.georeferenced:
        lines.append(
            f"Coordinate reference system: {p.crs}, at {p.gsd_x_m:.2f} m per pixel."
        )
        if p.bounds_wgs84:
            lon = (p.bounds_wgs84[0] + p.bounds_wgs84[2]) / 2
            lat = (p.bounds_wgs84[1] + p.bounds_wgs84[3]) / 2
            lines.append(f"Scene centre: {lat:.4f} N, {lon:.4f} E.")
    if p.acquisition_date:
        lines.append(f"Acquired on {p.acquisition_date} (from {p.acquisition_date_source}).")

    metrics: dict = {
        "width": p.width,
        "height": p.height,
        "band_count": p.band_count,
        "area_km2": round(p.area_km2, 4) if p.area_km2 else None,
        "gsd_m": round(p.gsd_x_m, 3) if p.gsd_x_m else None,
        "georeferenced": p.georeferenced,
    }
    if ctx.params.get("include_band_statistics", True):
        metrics["bands"] = [
            {
                "band": (p.band_profile.names[i] if p.band_profile
                         and i < len(p.band_profile.names) else f"B{i + 1}"),
                **stats.to_dict(),
            }
            for i, stats in enumerate(p.band_stats)
        ]
    valid = p.band_stats[0].valid_fraction if p.band_stats else 1.0
    metrics["valid_pixel_fraction"] = round(valid, 4)

    return ToolResult(
        status=ToolStatus.OK,
        answer=" ".join(lines),
        detail="derived from raster metadata and a decimated statistical sample",
        metrics=metrics,
        confidence=round(min(0.99, 0.6 + 0.4 * p.modality_confidence), 3),
    )


def run_pair_geometry(ctx: ToolContext) -> ToolResult:
    """Report the shared extent and alignment of a two-image submission."""
    a, b = ctx.passports[0], ctx.passports[1]
    if ctx.configuration is InputConfiguration.BI_TEMPORAL_PAIR:
        a, b = order_bitemporal(a, b)
        role_a, role_b = "T1", "T2"
    else:
        role_a = "optical" if a.modality.value == "optical" else a.modality.value
        role_b = "SAR" if b.modality.value == "sar" else b.modality.value

    lines = [
        f"{role_a}: {a.filename} - {a.summary_line()}.",
        f"{role_b}: {b.filename} - {b.summary_line()}.",
    ]
    metrics: dict = {"role_a": role_a, "role_b": role_b}

    if a.georeferenced and b.georeferenced and a.bounds_wgs84 and b.bounds_wgs84:
        overlap = bbox_overlap_fraction(a.bounds_wgs84, b.bounds_wgs84)
        shared = [
            max(a.bounds_wgs84[0], b.bounds_wgs84[0]), max(a.bounds_wgs84[1], b.bounds_wgs84[1]),
            min(a.bounds_wgs84[2], b.bounds_wgs84[2]), min(a.bounds_wgs84[3], b.bounds_wgs84[3]),
        ]
        metrics["overlap_fraction"] = round(overlap, 4)
        metrics["shared_bounds_wgs84"] = [round(v, 6) for v in shared]
        lines.append(f"The two footprints overlap over {overlap:.1%} of the smaller scene.")
        finest = min(v for v in (a.gsd_x_m, b.gsd_x_m) if v)
        coarsest = max(v for v in (a.gsd_x_m, b.gsd_x_m) if v)
        metrics["analysis_gsd_m"] = round(coarsest, 3)
        if coarsest > finest:
            lines.append(
                f"Joint analysis will run at the coarser {coarsest:.1f} m grid."
            )
    else:
        lines.append(
            f"Neither image is georeferenced; they are treated as pixel-aligned at "
            f"{a.width} x {a.height}."
        )
        metrics["pixel_aligned"] = (a.width, a.height) == (b.width, b.height)

    if a.acquisition_date and b.acquisition_date:
        import datetime as dt

        gap = abs((dt.date.fromisoformat(b.acquisition_date)
                   - dt.date.fromisoformat(a.acquisition_date)).days)
        metrics["acquisition_gap_days"] = gap
        lines.append(f"The acquisitions are {gap} days apart "
                     f"({a.acquisition_date} to {b.acquisition_date}).")

    return ToolResult(
        status=ToolStatus.OK,
        answer=" ".join(lines),
        detail="footprint intersection computed in WGS84 from both geotransforms",
        metrics=metrics,
        confidence=0.95 if a.georeferenced and b.georeferenced else 0.6,
    )


SCENE_STATISTICS = ToolSpec(
    name="scene_statistics",
    version="1.0",
    summary="Reports scene geometry, sensor identification, extent in km2 and per-band radiometry.",
    backend="classical",
    tasks=(Task.VQA, Task.CAPTION),
    configurations=(InputConfiguration.SINGLE,),
    params=(
        ToolParam("include_band_statistics", "bool", True,
                  "Include per-band minimum, maximum, mean and standard deviation."),
    ),
    priority=20,  # a real answer, but Phase B tools should outrank it
    implemented=True,
    run=run_scene_statistics,
)

PAIR_GEOMETRY = ToolSpec(
    name="pair_geometry",
    version="1.0",
    summary="Reports the shared extent, alignment and acquisition gap of an image pair.",
    backend="classical",
    tasks=(Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA, Task.CROSS_MODAL_ANALYSIS),
    configurations=(InputConfiguration.BI_TEMPORAL_PAIR, InputConfiguration.CROSS_MODAL_PAIR),
    params=(),
    priority=15,
    implemented=True,
    run=run_pair_geometry,
)
