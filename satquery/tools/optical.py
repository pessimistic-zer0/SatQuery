"""Spectral index analysis of an optical image.

Answers questions about what is on the ground - how much water, how much
built-up area, how many separate lakes - using published band ratios rather
than a model. It needs no GPU and no weights, which is what makes it the
fallback that keeps the demo alive.
"""

from __future__ import annotations

from ..enums import InputConfiguration, Modality, Task, ToolStatus
from ..imaging import rgb_preview, write_evidence
from ..indices import (
    INDEX_DEFINITIONS,
    classify_optical,
    compute_index,
    count_regions,
    remove_small_regions,
    summarise_classes,
    threshold_confidence,
)
from ..raster import read_bands
from ..registry import ToolContext, ToolResult

# Bands the classifier would like; whatever is present is used.
PREFERRED_ROLES = ("blue", "green", "red", "nir", "swir1")

# Which index answers a question about each land-cover class.
CLASS_INDEX = {
    "water": "mndwi", "vegetation": "ndvi", "built_up": "ndbi", "bare_soil": "bsi",
}


def _optical_passport(ctx: ToolContext):
    """The optical member of the input, whatever the configuration."""
    for passport in ctx.passports:
        if passport.modality is Modality.OPTICAL:
            return passport
    return ctx.primary


def run_spectral_index(ctx: ToolContext) -> ToolResult:
    passport = _optical_passport(ctx)
    roles = tuple(r for r in PREFERRED_ROLES if passport.band_index(r) is not None)
    if not roles:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail=f"{passport.filename} has no recognisable spectral bands",
        )

    bands, grid, scaling = read_bands(passport, roles)
    min_region = int(ctx.params.get("min_region_px", 25))
    requested = str(ctx.params.get("index", "auto")).lower()

    # Which classes the question is actually about; empty means describe it all.
    asked_for = [c for c in ctx.entities.get("classes", []) if c in CLASS_INDEX]

    # --- a single named index, thresholded ---------------------------------
    if requested != "auto":
        definition = INDEX_DEFINITIONS.get(requested)
        missing = [r for r in definition.roles if r not in bands] if definition else ["unknown"]
        if missing:
            return ToolResult(
                status=ToolStatus.SKIPPED,
                detail=f"'{requested}' needs the bands {missing}, which {passport.filename} "
                       "does not provide",
            )
        array = compute_index(requested, bands)
        threshold = float(ctx.params.get("threshold", 0.0))
        mask, dropped = remove_small_regions(array > threshold, min_region)
        area = grid.area_km2(mask)
        extent = f"{area:.3f} km2" if area is not None else f"{int(mask.sum())} pixels"
        answer = (
            f"{requested.upper()} ({definition.formula}) above {threshold} covers {extent}, "
            f"{grid.coverage(mask):.1%} of the analysed scene, in "
            f"{count_regions(mask)} separate region(s)."
        )
        artifacts = write_evidence(
            ctx.workdir, f"spectral_{requested}", rgb_preview(bands),
            [("change", mask)], grid,
            caption=f"{requested.upper()} > {threshold}",
        )
        return ToolResult(
            status=ToolStatus.OK,
            answer=answer,
            detail=f"{definition.description} ({definition.reference}); {scaling}",
            metrics={
                "index": requested, "threshold": threshold,
                "area_km2": round(area, 4) if area is not None else None,
                "coverage": round(grid.coverage(mask), 4),
                "regions": count_regions(mask),
                "small_regions_removed": dropped,
                "grid": grid.to_dict(),
            },
            artifacts=artifacts,
            confidence=threshold_confidence(array, threshold),
        )

    # --- full land-cover breakdown -----------------------------------------
    results, computed = classify_optical(bands, grid, min_region_px=min_region)
    if not results:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail="no index could be computed from the available bands",
        )

    sentences = []
    if grid.total_area_km2 is not None:
        sentences.append(
            f"The analysed scene covers {grid.total_area_km2:.2f} km2 at "
            f"{grid.gsd_x_m:.1f} m per pixel."
        )
    sentences.append(summarise_classes(results, grid))

    # Lead with the class the question asked about.
    for class_name in asked_for:
        match = next((r for r in results if r.name == class_name), None)
        if match is None:
            continue
        extent = (f"{match.area_km2:.3f} km2" if match.area_km2 is not None
                  else f"{match.pixels} pixels")
        sentences.append(
            f"{match.name.replace('_', ' ').capitalize()}: {match.regions} separate region(s) "
            f"totalling {extent}, identified by {match.rule}."
        )

    base = rgb_preview(bands)
    artifacts = write_evidence(
        ctx.workdir, "land_cover", base,
        [(r.name, r.mask) for r in results if r.pixels], grid,
        caption="Land cover from spectral indices",
    )

    primary_index = CLASS_INDEX.get(asked_for[0], "ndvi") if asked_for else "ndvi"
    reference = computed.get(primary_index)
    confidence = threshold_confidence(reference, 0.0) if reference is not None else 0.6

    return ToolResult(
        status=ToolStatus.OK,
        answer=" ".join(s for s in sentences if s),
        detail=f"indices computed: {', '.join(sorted(computed))}; {scaling}",
        metrics={
            "classes": [r.to_dict() for r in results],
            "indices_available": sorted(computed),
            "index_definitions": {k: v.to_dict() for k, v in INDEX_DEFINITIONS.items()
                                  if k in computed},
            "grid": grid.to_dict(),
            "focus_classes": asked_for,
        },
        artifacts=artifacts,
        confidence=confidence,
    )
