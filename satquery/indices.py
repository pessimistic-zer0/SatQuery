"""Spectral and radar indices, and the land-cover rules built on them.

This is the measurement layer. Nothing here is learned - every number comes from
a band ratio with a published definition and a threshold from the remote-sensing
literature. That matters twice over: it works with no GPU and no training, and
it gives the Physics Verifier something independent to check a model's claim
against.

Thresholds are defaults, not truths. Every one of them is exposed as a permitted
tool parameter, and every result carries the threshold it used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .raster import Grid

# --- Index definitions ------------------------------------------------------


@dataclass(frozen=True)
class IndexDefinition:
    name: str
    roles: tuple[str, ...]
    formula: str
    description: str
    reference: str

    def to_dict(self) -> dict:
        return {"name": self.name, "formula": self.formula,
                "description": self.description, "reference": self.reference}


INDEX_DEFINITIONS: dict[str, IndexDefinition] = {
    "ndvi": IndexDefinition(
        "ndvi", ("nir", "red"), "(NIR - Red) / (NIR + Red)",
        "Vegetation vigour; high over healthy canopy, negative over water.",
        "Rouse et al. 1974",
    ),
    "ndwi": IndexDefinition(
        "ndwi", ("green", "nir"), "(Green - NIR) / (Green + NIR)",
        "Open water; water absorbs strongly in the near infrared.",
        "McFeeters 1996",
    ),
    "mndwi": IndexDefinition(
        "mndwi", ("green", "swir1"), "(Green - SWIR1) / (Green + SWIR1)",
        "Open water, with built-up surfaces suppressed better than NDWI.",
        "Xu 2006",
    ),
    "ndbi": IndexDefinition(
        "ndbi", ("swir1", "nir"), "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "Built-up and impervious surfaces.",
        "Zha et al. 2003",
    ),
    "bsi": IndexDefinition(
        "bsi", ("swir1", "red", "nir", "blue"),
        "((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))",
        "Bare soil and exposed ground.",
        "Rikimaru et al. 2002",
    ),
}

# Default decision thresholds. Literature-typical values, all overridable.
WATER_NDWI = 0.00
WATER_MNDWI = 0.00
VEGETATION_NDVI = 0.30
BUILTUP_NDBI = 0.00
BUILTUP_MAX_NDVI = 0.25
BARE_BSI = 0.10

# SAR backscatter thresholds in dB, for Sentinel-1 GRD VV.
WATER_VV_DB = -18.0
BUILTUP_VV_DB = -5.0

# Class precedence when a pixel satisfies more than one rule. Water first
# because open water is the least ambiguous signature in both modalities.
CLASS_PRIORITY = ("water", "built_up", "vegetation", "bare_soil")


def normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), with a zero denominator returned as NaN."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(denominator) < 1e-6, np.nan, (a - b) / denominator)
    return out.astype(np.float32)


def compute_index(name: str, bands: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate one named index from a dict of semantic bands."""
    key = name.lower()
    if key not in INDEX_DEFINITIONS:
        raise KeyError(f"unknown index '{name}'; known indices are "
                       f"{sorted(INDEX_DEFINITIONS)}")
    missing = [r for r in INDEX_DEFINITIONS[key].roles if r not in bands]
    if missing:
        raise KeyError(f"index '{key}' needs the bands {missing}")

    if key == "ndvi":
        return normalised_difference(bands["nir"], bands["red"])
    if key == "ndwi":
        return normalised_difference(bands["green"], bands["nir"])
    if key == "mndwi":
        return normalised_difference(bands["green"], bands["swir1"])
    if key == "ndbi":
        return normalised_difference(bands["swir1"], bands["nir"])
    # bsi
    return normalised_difference(bands["swir1"] + bands["red"],
                                 bands["nir"] + bands["blue"])


def available_indices(passport_roles: dict[str, int]) -> list[str]:
    """Which indices the bands present actually support."""
    return [name for name, definition in INDEX_DEFINITIONS.items()
            if all(role in passport_roles for role in definition.roles)]


# --- Thresholding and mask hygiene -----------------------------------------


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: the split that minimises within-class variance.

    Used for change magnitude, where the right cut-off depends on the scene and
    a fixed number would be a guess.
    """
    sample = np.asarray(values, dtype=np.float64).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0
    low, high = float(sample.min()), float(sample.max())
    if high - low < 1e-9:
        return low

    histogram, edges = np.histogram(sample, bins=bins, range=(low, high))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return low

    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(histogram)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return low

    cumulative_mean = np.cumsum(histogram * centres)
    total_mean = cumulative_mean[-1]
    mean_bg = np.divide(cumulative_mean, weight_bg, out=np.zeros_like(weight_bg), where=weight_bg > 0)
    mean_fg = np.divide(total_mean - cumulative_mean, weight_fg,
                        out=np.zeros_like(weight_fg), where=weight_fg > 0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -np.inf

    # When the two modes are well separated the criterion is flat right across
    # the empty valley between them, and every bin in that plateau is an equally
    # good threshold. argmax would return its first bin, which sits hard against
    # the lower mode; the middle of the plateau is the robust choice.
    peak = float(between.max())
    plateau = np.flatnonzero(between >= peak - 1e-12)
    return float(centres[int(plateau[len(plateau) // 2])])


def remove_small_regions(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, int]:
    """Drop connected regions below a size, and report how many were dropped.

    Index thresholding produces speckled single pixels that are not real
    features. Removing them is what turns a noisy mask into a countable set of
    objects.
    """
    if min_pixels <= 1 or not mask.any():
        return mask, 0
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = sizes >= min_pixels
    cleaned = keep[labels]
    return cleaned, int(count - max(0, int(keep.sum())))


def filter_regions_by_statistic(
    mask: np.ndarray,
    statistic: np.ndarray,
    threshold: float,
    *,
    keep: str = "below",
) -> tuple[np.ndarray, int]:
    """Keep or drop whole connected regions by their mean value of an index.

    Judging a class per pixel makes the decision as noisy as a single pixel. A
    car park is a region, not a pixel, so averaging the discriminating index
    over each connected region before deciding cuts the noise by roughly the
    square root of the region size - and it is what stops speckle from eating
    a correctly detected object.
    """
    if not mask.any():
        return mask, 0
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask, 0
    indices = np.arange(1, count + 1)
    means = np.array(ndimage.mean(np.nan_to_num(statistic, nan=0.0), labels, indices))
    keep_labels = means < threshold if keep == "below" else means > threshold
    lookup = np.zeros(count + 1, dtype=bool)
    lookup[1:] = keep_labels
    return lookup[labels], int(count - int(keep_labels.sum()))


def count_regions(mask: np.ndarray) -> int:
    """Number of connected regions in a mask - how 'how many lakes?' is answered."""
    if not mask.any():
        return 0
    _, count = ndimage.label(mask)
    return int(count)


def lee_filter(array: np.ndarray, size: int = 5) -> np.ndarray:
    """Lee speckle filter: adaptive smoothing that preserves edges.

    SAR thresholding on raw backscatter is dominated by speckle; this is the
    standard first step before any radar decision rule.
    """
    data = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(data)
    filled = np.where(finite, data, np.nanmean(data) if finite.any() else 0.0)

    mean = ndimage.uniform_filter(filled, size)
    mean_square = ndimage.uniform_filter(filled ** 2, size)
    variance = np.maximum(mean_square - mean ** 2, 0.0)
    overall = float(variance.mean())
    weights = variance / (variance + overall + 1e-9)
    out = mean + weights * (filled - mean)
    return np.where(finite, out, np.nan).astype(np.float32)


def despeckle(array: np.ndarray, method: str, size: int = 5) -> tuple[np.ndarray, str]:
    """Apply the requested speckle filter and say which one ran."""
    if method == "none":
        return array, "no speckle filter applied"
    if method == "median":
        return ndimage.median_filter(array, size=size), f"{size}x{size} median filter"
    return lee_filter(array, size), f"{size}x{size} Lee filter"


# --- Classification ---------------------------------------------------------


@dataclass
class ClassResult:
    """One land-cover class, the rule that produced it, and its extent."""

    name: str
    mask: np.ndarray = field(repr=False)
    rule: str
    index: str
    threshold: float
    pixels: int = 0
    regions: int = 0
    area_km2: float | None = None
    coverage: float = 0.0
    mean_index: float | None = None

    def to_dict(self) -> dict:
        return {
            "class": self.name,
            "rule": self.rule,
            "index": self.index,
            "threshold": round(self.threshold, 4),
            "pixels": self.pixels,
            "regions": self.regions,
            "area_km2": round(self.area_km2, 4) if self.area_km2 is not None else None,
            "coverage": round(self.coverage, 4),
            "mean_index": round(self.mean_index, 4) if self.mean_index is not None else None,
        }


def _finalise(result: ClassResult, grid: Grid, index_array: np.ndarray | None) -> ClassResult:
    result.pixels = int(np.count_nonzero(result.mask))
    result.regions = count_regions(result.mask)
    result.area_km2 = grid.area_km2(result.mask)
    result.coverage = grid.coverage(result.mask)
    if index_array is not None and result.pixels:
        values = index_array[result.mask]
        values = values[np.isfinite(values)]
        result.mean_index = float(values.mean()) if values.size else None
    return result


def classify_optical(
    bands: dict[str, np.ndarray],
    grid: Grid,
    *,
    min_region_px: int = 25,
    thresholds: dict | None = None,
) -> tuple[list[ClassResult], dict[str, np.ndarray]]:
    """Label water, vegetation, built-up and bare soil from spectral indices."""
    t = {
        "water_ndwi": WATER_NDWI, "water_mndwi": WATER_MNDWI,
        "vegetation_ndvi": VEGETATION_NDVI, "builtup_ndbi": BUILTUP_NDBI,
        "builtup_max_ndvi": BUILTUP_MAX_NDVI, "bare_bsi": BARE_BSI,
        **(thresholds or {}),
    }
    computed: dict[str, np.ndarray] = {}
    for name, definition in INDEX_DEFINITIONS.items():
        if all(role in bands for role in definition.roles):
            computed[name] = compute_index(name, bands)

    results: list[ClassResult] = []

    # Water: MNDWI where SWIR is available, NDWI otherwise.
    if "mndwi" in computed:
        water = np.nan_to_num(computed["mndwi"], nan=-1.0) > t["water_mndwi"]
        water_index, water_threshold = "mndwi", t["water_mndwi"]
    elif "ndwi" in computed:
        water = np.nan_to_num(computed["ndwi"], nan=-1.0) > t["water_ndwi"]
        water_index, water_threshold = "ndwi", t["water_ndwi"]
    else:
        water, water_index, water_threshold = None, "", 0.0
    if water is not None:
        water, _ = remove_small_regions(water, min_region_px)
        results.append(ClassResult(
            "water", water, f"{water_index.upper()} > {water_threshold}",
            water_index, water_threshold))

    if "ndvi" in computed:
        vegetation = np.nan_to_num(computed["ndvi"], nan=-1.0) > t["vegetation_ndvi"]
        vegetation, _ = remove_small_regions(vegetation, min_region_px)
        results.append(ClassResult(
            "vegetation", vegetation, f"NDVI > {t['vegetation_ndvi']}",
            "ndvi", t["vegetation_ndvi"]))

    if "ndbi" in computed and "ndvi" in computed:
        builtup = (np.nan_to_num(computed["ndbi"], nan=-1.0) > t["builtup_ndbi"]) & \
                  (np.nan_to_num(computed["ndvi"], nan=1.0) < t["builtup_max_ndvi"])
        rule = f"NDBI > {t['builtup_ndbi']} and NDVI < {t['builtup_max_ndvi']}"
        builtup, _ = remove_small_regions(builtup, min_region_px)
        if "bsi" in computed:
            # NDBI on its own cannot tell an asphalt car park from a dry field.
            # BSI separates them, but only reliably as a region average - per
            # pixel it discards a third of a correctly detected built-up area.
            builtup, _ = filter_regions_by_statistic(
                builtup, computed["bsi"], t["bare_bsi"], keep="below")
            rule += f" and mean BSI < {t['bare_bsi']} per region"
        results.append(ClassResult("built_up", builtup, rule, "ndbi", t["builtup_ndbi"]))

    if "bsi" in computed:
        bare = np.nan_to_num(computed["bsi"], nan=-1.0) > t["bare_bsi"]
        bare, _ = remove_small_regions(bare, min_region_px)
        results.append(ClassResult(
            "bare_soil", bare, f"BSI > {t['bare_bsi']}", "bsi", t["bare_bsi"]))

    # Resolve overlaps by precedence so the classes partition the scene.
    order = {name: i for i, name in enumerate(CLASS_PRIORITY)}
    results.sort(key=lambda r: order.get(r.name, 99))
    taken: np.ndarray | None = None
    for result in results:
        if taken is None:
            taken = np.zeros_like(result.mask, dtype=bool)
        result.mask = result.mask & ~taken
        taken |= result.mask
        _finalise(result, grid, computed.get(result.index))

    return results, computed


def classify_sar(
    vv_db: np.ndarray,
    vh_db: np.ndarray | None,
    grid: Grid,
    *,
    water_vv_db: float = WATER_VV_DB,
    builtup_vv_db: float = BUILTUP_VV_DB,
    min_region_px: int = 25,
) -> list[ClassResult]:
    """Label water, built-up and other land from backscatter thresholds.

    Water is a specular reflector - it sends almost nothing back to the sensor,
    so it is dark. Buildings are corner reflectors and are bright. The middle of
    the range is vegetated or bare land, which SAR alone cannot separate
    reliably, so it is reported as one class rather than guessed at.
    """
    values = np.nan_to_num(vv_db, nan=0.0)
    results: list[ClassResult] = []

    water = values < water_vv_db
    water, _ = remove_small_regions(water, min_region_px)
    results.append(ClassResult("water", water, f"VV < {water_vv_db} dB", "vv", water_vv_db))

    builtup = values > builtup_vv_db
    builtup, _ = remove_small_regions(builtup, min_region_px)
    results.append(ClassResult("built_up", builtup, f"VV > {builtup_vv_db} dB",
                               "vv", builtup_vv_db))

    other = ~(water | builtup)
    results.append(ClassResult(
        "land", other, f"{water_vv_db} dB <= VV <= {builtup_vv_db} dB",
        "vv", builtup_vv_db))

    taken: np.ndarray | None = None
    for result in results:
        if taken is None:
            taken = np.zeros_like(result.mask, dtype=bool)
        result.mask = result.mask & ~taken
        taken |= result.mask
        _finalise(result, grid, vv_db)
    return results


def cloud_mask(bands: dict[str, np.ndarray], threshold: float = 0.28) -> np.ndarray:
    """Bright, spectrally flat pixels - a first-order cloud test.

    Not a replacement for a proper cloud mask such as Sentinel-2's scene
    classification layer, but it needs no extra band and it is enough to say
    which parts of an optical image cannot be trusted.
    """
    visible = [bands[name] for name in ("blue", "green", "red") if name in bands]
    if not visible:
        visible = [next(iter(bands.values()))]
    stack = np.stack(visible)
    brightness = np.nanmean(stack, axis=0)
    flat = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0) < 0.20
    return np.nan_to_num(brightness, nan=0.0) > threshold if len(visible) < 3 \
        else (np.nan_to_num(brightness, nan=0.0) > threshold) & flat


def summarise_classes(results: list[ClassResult], grid: Grid) -> str:
    """A plain sentence describing what the scene is made of."""
    ranked = sorted((r for r in results if r.pixels), key=lambda r: -r.coverage)
    if not ranked:
        return "No land-cover class passed its index threshold."
    parts = []
    for result in ranked:
        extent = (f"{result.area_km2:.2f} km2" if result.area_km2 is not None
                  else f"{result.pixels} pixels")
        parts.append(f"{result.name.replace('_', ' ')} {result.coverage:.0%} ({extent})")
    return "Land cover by area: " + ", ".join(parts) + "."


def threshold_confidence(index_array: np.ndarray, threshold: float,
                         scale: float = 0.15) -> float:
    """How decisively the pixels sit away from a decision threshold.

    A scene whose index values cluster right at the cut-off is a scene where the
    classification is a coin flip, and the reported confidence should say so.
    ``scale`` is the margin, in index units, at which confidence saturates.
    """
    values = np.asarray(index_array, dtype=np.float32).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    margin = float(np.abs(values - threshold).mean())
    return float(np.clip(0.5 + 0.5 * min(1.0, margin / scale), 0.0, 0.99))
