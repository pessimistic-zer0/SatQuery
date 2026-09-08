"""Rendering the visual evidence.

The problem statement asks for visual evidence alongside the text answer, so
every spatial tool writes a PNG a person can look at and, where the input was
georeferenced, a GeoTIFF a GIS can open.

Colours are chosen to stay readable when printed in a report and when projected
in a room, which rules out thin lines and low-contrast fills.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .raster import Grid, write_geotiff_mask

# Class colours, reused across every overlay so the legend means one thing.
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "water": (56, 132, 255),
    "vegetation": (74, 190, 108),
    "built_up": (240, 108, 84),
    "bare_soil": (214, 178, 92),
    "land": (150, 150, 160),
    "change": (255, 92, 160),
    "increase": (255, 92, 92),
    "decrease": (96, 176, 255),
    "cloud": (235, 235, 245),
    "sar_only": (250, 196, 64),
    "agreement": (86, 200, 140),
    "disagreement": (232, 96, 96),
}


# Preview gamma. A linear stretch of a scene containing cloud, snow or bare sand
# pushes everything else into the shadows, because the bright surface owns the
# top of the range. Lifting the mid-tones keeps the land readable without
# clipping the bright end.
PREVIEW_GAMMA = 1.0 / 1.8


def _stretch(band: np.ndarray, low: float = 2.0, high: float = 98.0,
             gamma: float = PREVIEW_GAMMA) -> np.ndarray:
    """Percentile stretch to 0-255, so a scene is visible rather than correct."""
    values = np.asarray(band, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low, high])
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    scaled = np.clip(np.nan_to_num((values - lo) / (hi - lo), nan=0.0), 0.0, 1.0)
    return (np.power(scaled, gamma) * 255).astype(np.uint8)


def rgb_preview(bands: dict[str, np.ndarray]) -> np.ndarray:
    """A true-colour preview when the bands allow it, greyscale otherwise."""
    if all(name in bands for name in ("red", "green", "blue")):
        return np.stack([_stretch(bands["red"]), _stretch(bands["green"]),
                         _stretch(bands["blue"])], axis=-1)
    first = next(iter(bands.values()))
    grey = _stretch(first)
    return np.stack([grey, grey, grey], axis=-1)


def overlay_masks(
    base: np.ndarray,
    masks: list[tuple[str, np.ndarray]],
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """Paint named masks over an RGB preview."""
    out = base.astype(np.float32).copy()
    for name, mask in masks:
        colour = np.array(CLASS_COLOURS.get(name, (255, 255, 255)), dtype=np.float32)
        selection = np.asarray(mask, dtype=bool)
        if not selection.any():
            continue
        out[selection] = (1 - alpha) * out[selection] + alpha * colour
    return np.clip(out, 0, 255).astype(np.uint8)


def save_png(path: str, image: np.ndarray) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    array = np.asarray(image)
    Image.fromarray(array.astype(np.uint8),
                    mode="RGB" if array.ndim == 3 else "L").save(path)
    return path


def write_evidence(
    workdir: str,
    name: str,
    base: np.ndarray,
    masks: list[tuple[str, np.ndarray]],
    grid: Grid,
    *,
    caption: str = "",
) -> list[dict]:
    """Write one overlay PNG, plus a GeoTIFF per mask when georeferenced.

    Returns artifact records for the execution trace.
    """
    artifacts: list[dict] = []
    png = save_png(os.path.join(workdir, f"{name}.png"), overlay_masks(base, masks))
    artifacts.append({
        "kind": "overlay",
        "name": name,
        "path": png,
        "format": "png",
        "caption": caption or f"{name.replace('_', ' ')} overlay",
        "legend": [{"class": label, "colour": CLASS_COLOURS.get(label, (255, 255, 255))}
                   for label, mask in masks if np.asarray(mask, dtype=bool).any()],
    })
    if grid.georeferenced:
        for label, mask in masks:
            selection = np.asarray(mask, dtype=bool)
            if not selection.any():
                continue
            path = write_geotiff_mask(
                os.path.join(workdir, f"{name}_{label}.tif"), selection, grid)
            artifacts.append({
                "kind": "mask",
                "name": f"{name}_{label}",
                "path": path,
                "format": "geotiff",
                "crs": grid.crs,
                "caption": f"{label.replace('_', ' ')} mask, {grid.crs}",
            })
    return artifacts
