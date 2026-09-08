"""Optical-SAR joint analysis and the Modality-Reliability Map.

The point of a co-registered pair is not that two images are better than one; it
is that the two sensors fail in different places. Optical carries spectral
detail but stops at the cloud top. SAR sees through cloud and at night but
cannot tell a wet field from a lake by colour.

So this tool does not blend the two into a single number. It reports, per
region, which sensor answered and why - and where cloud blocks the optical
image, it says plainly that the radar carried the answer.
"""

from __future__ import annotations

import numpy as np

from ..enums import Modality, ToolStatus
from ..imaging import rgb_preview, write_evidence
from ..indices import (
    WATER_MNDWI,
    classify_sar,
    cloud_mask,
    compute_index,
    despeckle,
    remove_small_regions,
)
from ..raster import read_pair
from ..registry import ToolContext, ToolResult

OPTICAL_ROLES = ("blue", "green", "red", "nir", "swir1")


def run_modality_reliability(ctx: ToolContext) -> ToolResult:
    optical = next((p for p in ctx.passports if p.modality is Modality.OPTICAL), None)
    sar = next((p for p in ctx.passports if p.modality is Modality.SAR), None)
    if optical is None or sar is None:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail="this tool needs one optical and one SAR image over the same area",
        )

    optical_roles = tuple(r for r in OPTICAL_ROLES if optical.band_index(r) is not None)
    sar_roles = tuple(r for r in ("vv", "vh") if sar.band_index(r) is not None)
    if "vv" not in sar_roles or "green" not in optical_roles:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail="the pair lacks the bands needed for a joint water comparison "
                   "(optical green and SAR VV)",
        )

    opt, rad, grid, notes = read_pair(optical, sar, optical_roles, sar_roles)
    cloud_threshold = float(ctx.params.get("cloud_threshold", 0.28))
    tolerance = float(ctx.params.get("agreement_tolerance", 0.15))

    # --- each sensor's independent answer ----------------------------------
    water_index = "mndwi" if "swir1" in opt else "ndwi"
    optical_water_index = compute_index(water_index, opt)
    optical_water = np.nan_to_num(optical_water_index, nan=-1.0) > WATER_MNDWI

    vv, filter_note = despeckle(rad["vv"], "lee")
    sar_classes = classify_sar(vv, rad.get("vh"), grid, min_region_px=25)
    sar_water = next(c.mask for c in sar_classes if c.name == "water")
    sar_builtup = next(c.mask for c in sar_classes if c.name == "built_up")

    cloud = cloud_mask(opt, cloud_threshold)
    cloud, _ = remove_small_regions(cloud, 25)
    clear = ~cloud

    # --- where do they agree? ----------------------------------------------
    # Pixels whose optical index sits within the tolerance of its threshold are
    # genuinely borderline. Counting them either way would inflate or deflate
    # the agreement figure, so they are excluded and reported separately.
    ambiguous = np.abs(np.nan_to_num(optical_water_index, nan=0.0) - WATER_MNDWI) < tolerance
    decidable = clear & ~ambiguous

    agreement = (optical_water == sar_water) & decidable
    disagreement = (optical_water != sar_water) & decidable
    sar_only = cloud  # optical is unusable here, so SAR is the only source

    decidable_pixels = int(decidable.sum())
    agreement_rate = float(agreement.sum() / decidable_pixels) if decidable_pixels else 0.0

    # --- the fused best estimate -------------------------------------------
    # Optical where it can see, SAR where cloud blocks it. That is the whole
    # argument for carrying both sensors.
    fused_water = np.where(clear, optical_water, sar_water)
    fused_water, _ = remove_small_regions(fused_water, 25)

    cloud_area = grid.area_km2(cloud)
    water_under_cloud = grid.area_km2(sar_water & cloud)
    fused_area = grid.area_km2(fused_water)
    optical_area = grid.area_km2(optical_water & clear)
    # Radar penetrates cloud, so its own classes are reported over the whole
    # scene. Restricting them to the cloud-free part would discard the one
    # thing SAR is carried for.
    builtup_area = grid.area_km2(sar_builtup)
    builtup_under_cloud = grid.area_km2(sar_builtup & cloud)

    def extent(area, mask):
        return f"{area:.3f} km2" if area is not None else f"{int(mask.sum())} pixels"

    sentences = [
        f"Cloud or haze obscures {cloud.mean():.1%} of the optical image "
        f"({extent(cloud_area, cloud)}). Over that area the optical bands carry no "
        "usable information and the SAR image supplies the answer.",
        f"Across the {decidable.mean():.1%} of the scene that is both cloud-free and not "
        f"borderline, the two sensors agree on water for {agreement_rate:.1%} of pixels, "
        f"using {water_index.upper()} for the optical image and a VV backscatter threshold "
        f"for the radar.",
    ]
    if water_under_cloud and water_under_cloud > 0:
        sentences.append(
            f"The radar identifies {water_under_cloud:.3f} km2 of water beneath the cloud "
            "that the optical image cannot see at all."
        )
    sentences.append(
        f"Combining both sensors, water covers {extent(fused_area, fused_water)} of the shared "
        f"extent, against {extent(optical_area, optical_water & clear)} visible to the optical "
        "image alone."
    )
    builtup_sentence = (
        f"Radar-bright surfaces consistent with built-up areas cover "
        f"{extent(builtup_area, sar_builtup)} across the whole scene"
    )
    if builtup_under_cloud and builtup_under_cloud > 0:
        share = builtup_under_cloud / builtup_area if builtup_area else 0.0
        builtup_sentence += (
            f", of which {builtup_under_cloud:.3f} km2 ({share:.0%}) lies beneath the cloud "
            "and is invisible to the optical image"
        )
    sentences.append(builtup_sentence + ".")

    artifacts = write_evidence(
        ctx.workdir, "modality_reliability", rgb_preview(opt),
        [("cloud", cloud), ("agreement", agreement & fused_water),
         ("disagreement", disagreement),
         ("sar_only", sar_only & (sar_water | sar_builtup))],
        grid,
        caption="Which sensor answered: agreement, disagreement, and SAR-only regions",
    )
    artifacts += write_evidence(
        ctx.workdir, "fused_water", rgb_preview(opt), [("water", fused_water)], grid,
        caption="Water extent from optical where clear, from SAR under cloud",
    )

    return ToolResult(
        status=ToolStatus.OK,
        answer=" ".join(sentences),
        detail=f"optical {water_index.upper()} against SAR VV thresholds; {filter_note}; "
               + "; ".join(notes),
        metrics={
            "optical_file": optical.filename,
            "sar_file": sar.filename,
            "cloud_fraction": round(float(cloud.mean()), 4),
            "cloud_area_km2": round(cloud_area, 4) if cloud_area is not None else None,
            "clear_fraction": round(float(clear.mean()), 4),
            "agreement_rate": round(agreement_rate, 4),
            "decidable_fraction": round(float(decidable.mean()), 4),
            "ambiguous_fraction": round(float((ambiguous & clear).mean()), 4),
            "disagreement_fraction": round(float(disagreement.mean()), 4),
            "water_under_cloud_km2": (round(water_under_cloud, 4)
                                      if water_under_cloud is not None else None),
            "optical_only_water_km2": round(optical_area, 4) if optical_area is not None else None,
            "fused_water_km2": round(fused_area, 4) if fused_area is not None else None,
            "sar_builtup_km2": round(builtup_area, 4) if builtup_area is not None else None,
            "sar_builtup_under_cloud_km2": (round(builtup_under_cloud, 4)
                                            if builtup_under_cloud is not None else None),
            "water_index": water_index,
            "cloud_threshold": cloud_threshold,
            "agreement_tolerance": tolerance,
            "grid": grid.to_dict(),
        },
        artifacts=artifacts,
        # Confidence is the measured agreement between two independent sensors,
        # discounted by how much of the scene only one of them could see.
        confidence=round(float(np.clip(agreement_rate * (0.7 + 0.3 * clear.mean()), 0.0, 0.98)), 3),
    )
