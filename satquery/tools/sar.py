"""Radar backscatter analysis.

SAR sees structure, not colour. Open water is a mirror that reflects the pulse
away from the sensor, so it is dark; buildings are corner reflectors, so they
are bright. Those two facts alone answer a surprising number of operational
questions, and they work at night and through cloud.
"""

from __future__ import annotations

import numpy as np

from ..enums import Modality, ToolStatus
from ..imaging import rgb_preview, write_evidence
from ..indices import classify_sar, despeckle, threshold_confidence
from ..raster import read_bands
from ..registry import ToolContext, ToolResult


def _sar_passport(ctx: ToolContext):
    for passport in ctx.passports:
        if passport.modality is Modality.SAR:
            return passport
    return ctx.primary


def run_sar_backscatter(ctx: ToolContext) -> ToolResult:
    passport = _sar_passport(ctx)
    roles = tuple(r for r in ("vv", "vh") if passport.band_index(r) is not None)
    if "vv" not in roles:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail=f"{passport.filename} has no VV polarisation band",
        )

    bands, grid, scaling = read_bands(passport, roles)
    method = str(ctx.params.get("speckle_filter", "lee"))
    vv, filter_note = despeckle(bands["vv"], method)
    vh = bands.get("vh")
    if vh is not None and method != "none":
        vh, _ = despeckle(vh, method)

    water_db = float(ctx.params.get("water_vv_db", -18.0))
    built_db = float(ctx.params.get("builtup_vv_db", -5.0))
    results = classify_sar(vv, vh, grid, water_vv_db=water_db, builtup_vv_db=built_db,
                           min_region_px=25)

    finite = vv[np.isfinite(vv)]
    sentences = [
        f"VV backscatter ranges from {finite.min():.1f} to {finite.max():.1f} dB "
        f"(mean {finite.mean():.1f} dB) after the {filter_note}."
    ]
    for result in results:
        if not result.pixels:
            continue
        extent = (f"{result.area_km2:.3f} km2" if result.area_km2 is not None
                  else f"{result.pixels} pixels")
        label = {"water": "Open water", "built_up": "Built-up or strongly scattering surfaces",
                 "land": "Vegetated or bare land"}.get(result.name, result.name)
        sentences.append(
            f"{label}: {extent} ({result.coverage:.0%} of the scene) where {result.rule}."
        )
    if vh is not None:
        ratio = float(np.nanmean(vv - vh))
        sentences.append(
            f"The mean VV/VH difference is {ratio:.1f} dB; a larger difference indicates "
            "surface scattering, a smaller one volume scattering from vegetation."
        )

    base = rgb_preview({"vv": vv})
    artifacts = write_evidence(
        ctx.workdir, "sar_classes", base,
        [(r.name, r.mask) for r in results if r.pixels and r.name != "land"], grid,
        caption=f"SAR classes from VV thresholds ({filter_note})",
    )

    return ToolResult(
        status=ToolStatus.OK,
        answer=" ".join(sentences),
        detail=f"{filter_note}; {scaling}",
        metrics={
            "classes": [r.to_dict() for r in results],
            "vv_mean_db": round(float(finite.mean()), 3),
            "vv_min_db": round(float(finite.min()), 3),
            "vv_max_db": round(float(finite.max()), 3),
            "speckle_filter": method,
            "water_vv_db": water_db,
            "builtup_vv_db": built_db,
            "grid": grid.to_dict(),
        },
        artifacts=artifacts,
        confidence=threshold_confidence(vv, water_db, scale=6.0),
    )
