"""Bi-temporal change analysis.

Two dates over the same ground, differenced. The tool answers three related
questions with one computation: what changed, where it changed, and by how much
in square kilometres.

Direction matters as much as magnitude. A change map that says 'these pixels
changed' is far less useful than one that says 'built-up grew here and water
retreated there', so gains and losses are reported separately per class.
"""

from __future__ import annotations

import numpy as np

from ..compat import order_bitemporal
from ..enums import ToolStatus
from ..imaging import rgb_preview, write_evidence
from ..indices import (
    INDEX_DEFINITIONS,
    classify_optical,
    compute_index,
    count_regions,
    otsu_threshold,
    remove_small_regions,
)
from ..raster import read_pair
from ..registry import ToolContext, ToolResult

PREFERRED_ROLES = ("blue", "green", "red", "nir", "swir1")

# Which index tracks each land-cover class, for a focused answer.
CLASS_INDEX = {"water": "mndwi", "vegetation": "ndvi", "built_up": "ndbi", "bare_soil": "bsi"}


def _change_vector(t1: dict, t2: dict, roles: list[str]) -> np.ndarray:
    """Change vector analysis: euclidean distance across all shared bands."""
    stack1 = np.stack([t1[r] for r in roles])
    stack2 = np.stack([t2[r] for r in roles])
    return np.sqrt(np.nansum((stack2 - stack1) ** 2, axis=0)).astype(np.float32)


def _normalise(magnitude: np.ndarray) -> np.ndarray:
    """Scale a magnitude image to 0-1 using its own 99th percentile."""
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return np.zeros_like(magnitude)
    peak = float(np.percentile(finite, 99))
    if peak < 1e-9:
        return np.zeros_like(magnitude)
    return np.clip(np.nan_to_num(magnitude, nan=0.0) / peak, 0.0, 1.0)


def run_change_index_diff(ctx: ToolContext) -> ToolResult:
    if len(ctx.passports) != 2:
        return ToolResult(status=ToolStatus.SKIPPED, detail="a bi-temporal pair is required")

    t1_passport, t2_passport = order_bitemporal(ctx.passports[0], ctx.passports[1])
    roles = tuple(r for r in PREFERRED_ROLES
                  if t1_passport.band_index(r) is not None
                  and t2_passport.band_index(r) is not None)
    if len(roles) < 2:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail="the two images do not share enough recognisable spectral bands",
        )

    t1, t2, grid, notes = read_pair(t1_passport, t2_passport, roles, roles)
    min_region = int(ctx.params.get("min_region_px", 25))
    requested = str(ctx.params.get("index", "auto")).lower()
    mode = str(ctx.params.get("thresholding", "otsu"))

    asked_for = [c for c in ctx.entities.get("classes", []) if c in CLASS_INDEX]
    if requested == "auto":
        requested = CLASS_INDEX[asked_for[0]] if asked_for else "cva"

    # --- magnitude of change ------------------------------------------------
    if requested == "cva":
        magnitude = _change_vector(t1, t2, list(roles))
        signed = None
        basis = f"change vector analysis over {len(roles)} bands"
    else:
        definition = INDEX_DEFINITIONS.get(requested)
        if definition is None or not all(r in t1 for r in definition.roles):
            return ToolResult(
                status=ToolStatus.SKIPPED,
                detail=f"'{requested}' cannot be computed from the shared bands {list(roles)}",
            )
        index_1 = compute_index(requested, t1)
        index_2 = compute_index(requested, t2)
        signed = np.nan_to_num(index_2 - index_1, nan=0.0)
        magnitude = np.abs(signed)
        basis = f"{requested.upper()} differenced between the two dates"

    normalised = _normalise(magnitude)
    if mode == "otsu":
        threshold = otsu_threshold(normalised)
        threshold_note = f"Otsu threshold at {threshold:.3f} of the normalised magnitude"
    else:
        threshold = float(ctx.params.get("change_threshold", 0.5))
        threshold_note = f"fixed threshold at {threshold:.3f} of the normalised magnitude"

    changed, dropped = remove_small_regions(normalised > threshold, min_region)
    changed_area = grid.area_km2(changed)

    # --- per-class area accounting -----------------------------------------
    classes_1, _ = classify_optical(t1, grid, min_region_px=min_region)
    classes_2, _ = classify_optical(t2, grid, min_region_px=min_region)
    by_name_1 = {c.name: c for c in classes_1}
    by_name_2 = {c.name: c for c in classes_2}

    deltas: list[dict] = []
    for name in sorted(set(by_name_1) | set(by_name_2)):
        before = by_name_1.get(name)
        after = by_name_2.get(name)
        area_before = before.area_km2 if before else 0.0
        area_after = after.area_km2 if after else 0.0
        px_before = before.pixels if before else 0
        px_after = after.pixels if after else 0
        entry = {
            "class": name,
            "pixels_t1": px_before, "pixels_t2": px_after,
            "area_t1_km2": round(area_before, 4) if area_before is not None else None,
            "area_t2_km2": round(area_after, 4) if area_after is not None else None,
        }
        if area_before is not None and area_after is not None:
            entry["delta_km2"] = round(area_after - area_before, 4)
            entry["percent_change"] = (
                round((area_after - area_before) / area_before * 100.0, 1)
                if area_before > 1e-9 else None
            )
        else:
            entry["delta_pixels"] = px_after - px_before
        deltas.append(entry)

    # --- direction masks for the class in question --------------------------
    focus = asked_for[0] if asked_for else max(
        (d for d in deltas if d.get("delta_km2") is not None),
        key=lambda d: abs(d["delta_km2"]), default={"class": None},
    )["class"]

    # The overlay must always show every changed area - "where did the change
    # occur" is a question about the whole scene. Directional colouring for the
    # class in question is layered on top of that, never instead of it.
    overlay = [("change", changed)]
    gain_area = loss_area = None
    if focus and focus in by_name_1 and focus in by_name_2:
        gain = by_name_2[focus].mask & ~by_name_1[focus].mask & changed
        loss = by_name_1[focus].mask & ~by_name_2[focus].mask & changed
        gain, _ = remove_small_regions(gain, min_region)
        loss, _ = remove_small_regions(loss, min_region)
        gain_area, loss_area = grid.area_km2(gain), grid.area_km2(loss)
        overlay = [("change", changed & ~(gain | loss)),
                   ("increase", gain), ("decrease", loss)]

    # --- the answer ---------------------------------------------------------
    extent = f"{changed_area:.3f} km2" if changed_area is not None \
        else f"{int(changed.sum())} pixels"
    sentences = [
        f"Between {t1_passport.acquisition_date or 'the first date'} and "
        f"{t2_passport.acquisition_date or 'the second date'}, {extent} changed "
        f"({grid.coverage(changed):.1%} of the shared extent), in "
        f"{count_regions(changed)} distinct areas.",
        f"Detected by {basis}, using the {threshold_note}.",
    ]

    ranked = sorted((d for d in deltas if d.get("delta_km2") is not None),
                    key=lambda d: -abs(d["delta_km2"]))
    for entry in ranked[:3]:
        direction = "increased" if entry["delta_km2"] > 0 else "decreased"
        percent = (f" ({entry['percent_change']:+.1f}%)"
                   if entry.get("percent_change") is not None else "")
        sentences.append(
            f"{entry['class'].replace('_', ' ').capitalize()} {direction} by "
            f"{abs(entry['delta_km2']):.3f} km2{percent}, from "
            f"{entry['area_t1_km2']:.3f} to {entry['area_t2_km2']:.3f} km2."
        )
    if focus and gain_area is not None and loss_area is not None:
        sentences.append(
            f"For {focus.replace('_', ' ')}, {gain_area:.3f} km2 was gained and "
            f"{loss_area:.3f} km2 was lost within the changed areas."
        )

    artifacts = write_evidence(
        ctx.workdir, "change_map", rgb_preview(t2), overlay, grid,
        caption=f"Change between {t1_passport.acquisition_date} and "
                f"{t2_passport.acquisition_date}",
    )

    # Confidence rises with how far the changed pixels sit above the threshold:
    # a change map thresholded through the middle of a smooth distribution is
    # not a confident one.
    changed_values = normalised[changed] if changed.any() else np.array([threshold])
    separation = float(np.mean(changed_values) - threshold)
    confidence = float(np.clip(0.5 + separation * 1.6, 0.3, 0.95))

    return ToolResult(
        status=ToolStatus.OK,
        answer=" ".join(sentences),
        detail=f"{basis}; {threshold_note}; {'; '.join(notes)}",
        metrics={
            "t1": t1_passport.filename, "t2": t2_passport.filename,
            "basis": requested, "thresholding": mode,
            "threshold_normalised": round(float(threshold), 4),
            "changed_area_km2": round(changed_area, 4) if changed_area is not None else None,
            "changed_coverage": round(grid.coverage(changed), 4),
            "changed_regions": count_regions(changed),
            "small_regions_removed": dropped,
            "focus_class": focus,
            "gain_km2": round(gain_area, 4) if gain_area is not None else None,
            "loss_km2": round(loss_area, 4) if loss_area is not None else None,
            "class_deltas": deltas,
            "grid": grid.to_dict(),
        },
        artifacts=artifacts,
        confidence=confidence,
    )
