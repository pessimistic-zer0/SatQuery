"""Synthetic remote-sensing fixtures with known ground truth.

Dataset download is the long pole of this project, and nothing in the controller
should have to wait for it. These generators build physically plausible
Sentinel-2 and Sentinel-1 rasters from a hand-drawn label map, so every test can
assert against an area that was planted deliberately: if the change tool reports
2.4 km2 of new built-up, the fixture knows whether 2.4 km2 was actually put
there.

Reflectance and backscatter values are representative rather than calibrated -
enough for band ratios, thresholds and modality inference to behave as they do
on real imagery.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from ..sensors import S2_12

CLASS_NAMES = ("water", "vegetation", "built_up", "bare_soil")
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Surface reflectance per class for the 12 BigEarthNet Sentinel-2 bands, in the
# order B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12.
S2_REFLECTANCE: dict[str, list[float]] = {
    "water":      [0.06, 0.05, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],
    "vegetation": [0.04, 0.03, 0.06, 0.04, 0.10, 0.25, 0.32, 0.35, 0.36, 0.30, 0.18, 0.08],
    "built_up":   [0.14, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.17, 0.17, 0.16, 0.20, 0.18],
    "bare_soil":  [0.17, 0.15, 0.19, 0.26, 0.30, 0.33, 0.35, 0.34, 0.35, 0.30, 0.42, 0.38],
}
# Sentinel-1 GRD backscatter in dB, as (VV, VH).
S1_BACKSCATTER_DB: dict[str, tuple[float, float]] = {
    "water":      (-22.0, -28.0),
    "vegetation": (-9.0, -15.0),
    "built_up":   (-4.0, -11.0),
    "bare_soil":  (-12.0, -19.0),
}

REFLECTANCE_SCALE = 10_000  # Sentinel-2 L2A quantisation
DEFAULT_EPSG = 32643        # UTM zone 43N - western India, matching the ISRO context
DEFAULT_GSD = 10.0          # metres, Sentinel-2 10 m bands
DEFAULT_ORIGIN = (700_000.0, 2_500_000.0)


@dataclass
class SyntheticScene:
    """A label map plus the class areas it implies."""

    labels: np.ndarray
    gsd_m: float
    class_pixels: dict[str, int] = field(default_factory=dict)

    @property
    def pixel_area_km2(self) -> float:
        return (self.gsd_m ** 2) / 1e6

    def area_km2(self, class_name: str) -> float:
        return self.class_pixels.get(class_name, 0) * self.pixel_area_km2

    def areas_km2(self) -> dict[str, float]:
        return {name: round(self.area_km2(name), 6) for name in CLASS_NAMES}


def _ellipse(height: int, width: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2 <= 1.0


def build_scene(
    width: int = 512,
    height: int = 512,
    *,
    gsd_m: float = DEFAULT_GSD,
    lake_scale: float = 1.0,
    urban_blocks: tuple[tuple[float, float, float, float], ...] = ((0.58, 0.55, 0.27, 0.24),),
    river: bool = True,
) -> SyntheticScene:
    """Draw a land-cover label map.

    ``urban_blocks`` are (x0, y0, w, h) rectangles in fractional coordinates, so
    a bi-temporal pair can be built by adding one block to the second date.
    """
    labels = np.full((height, width), CLASS_ID["vegetation"], dtype=np.uint8)

    # Bare soil across the lower fifth of the scene.
    labels[int(height * 0.82):, :] = CLASS_ID["bare_soil"]

    # A lake in the upper left; lake_scale shrinks it for the second date.
    lake = _ellipse(height, width, height * 0.32, width * 0.28,
                    height * 0.17 * lake_scale, width * 0.20 * lake_scale)
    labels[lake] = CLASS_ID["water"]

    # A meandering river across the scene.
    if river:
        xs = np.arange(width)
        centre = height * 0.66 + height * 0.06 * np.sin(2 * np.pi * xs / max(width, 1) * 1.5)
        half = max(2, int(width * 0.012))
        for x in xs:
            y0 = int(np.clip(centre[x] - half, 0, height))
            y1 = int(np.clip(centre[x] + half, 0, height))
            labels[y0:y1, x] = CLASS_ID["water"]

    # Built-up blocks.
    for fx, fy, fw, fh in urban_blocks:
        x0, y0 = int(width * fx), int(height * fy)
        x1, y1 = int(width * (fx + fw)), int(height * (fy + fh))
        labels[max(y0, 0):min(y1, height), max(x0, 0):min(x1, width)] = CLASS_ID["built_up"]

    counts = {name: int((labels == idx).sum()) for name, idx in CLASS_ID.items()}
    return SyntheticScene(labels=labels, gsd_m=gsd_m, class_pixels=counts)


def render_optical(
    scene: SyntheticScene,
    *,
    seed: int = 0,
    noise: float = 0.012,
    cloud_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render a 12-band Sentinel-2-like uint16 reflectance cube."""
    rng = np.random.default_rng(seed)
    height, width = scene.labels.shape
    cube = np.zeros((12, height, width), dtype=np.float32)
    for name, idx in CLASS_ID.items():
        mask = scene.labels == idx
        if not mask.any():
            continue
        for band in range(12):
            cube[band][mask] = S2_REFLECTANCE[name][band]
    cube += rng.normal(0.0, noise, cube.shape).astype(np.float32)

    if cloud_mask is not None and cloud_mask.any():
        # Cloud is bright and spectrally flat across the visible and NIR bands.
        cloud_reflectance = np.array(
            [0.62, 0.60, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.50, 0.30, 0.22],
            dtype=np.float32,
        )
        for band in range(12):
            cube[band][cloud_mask] = cloud_reflectance[band] + rng.normal(0, 0.02, int(cloud_mask.sum()))

    cube = np.clip(cube, 0.0, 1.6)
    return (cube * REFLECTANCE_SCALE).astype(np.uint16)


def render_sar(scene: SyntheticScene, *, seed: int = 0, looks: int = 4) -> np.ndarray:
    """Render a 2-band VV/VH backscatter image in dB, with speckle."""
    rng = np.random.default_rng(seed + 991)
    height, width = scene.labels.shape
    out = np.zeros((2, height, width), dtype=np.float32)
    for name, idx in CLASS_ID.items():
        mask = scene.labels == idx
        if not mask.any():
            continue
        for pol in (0, 1):
            mean_db = S1_BACKSCATTER_DB[name][pol]
            linear = 10.0 ** (mean_db / 10.0)
            # Multiplicative speckle: an L-look intensity image is gamma distributed.
            speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=int(mask.sum()))
            out[pol][mask] = 10.0 * np.log10(np.maximum(linear * speckle, 1e-8))
    return out.astype(np.float32)


def make_cloud_mask(height: int, width: int, *, seed: int = 0,
                    fraction: float = 0.18) -> np.ndarray:
    """A soft blob of cloud covering roughly ``fraction`` of the scene."""
    rng = np.random.default_rng(seed + 7)
    mask = np.zeros((height, width), dtype=bool)
    target = fraction * height * width
    while mask.sum() < target:
        cy = rng.uniform(0.15, 0.85) * height
        cx = rng.uniform(0.45, 0.95) * width
        ry = rng.uniform(0.08, 0.18) * height
        rx = rng.uniform(0.08, 0.18) * width
        mask |= _ellipse(height, width, cy, cx, ry, rx)
    return mask


def write_geotiff(
    path: str,
    array: np.ndarray,
    *,
    descriptions: list[str],
    gsd_m: float = DEFAULT_GSD,
    epsg: int = DEFAULT_EPSG,
    origin: tuple[float, float] = DEFAULT_ORIGIN,
    tags: dict | None = None,
    nodata: float | None = None,
) -> str:
    """Write a band-first array as a georeferenced GeoTIFF."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count, height, width = array.shape
    transform = Affine(gsd_m, 0.0, origin[0], 0.0, -gsd_m, origin[1])
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": array.dtype.name,
        "crs": CRS.from_epsg(epsg),
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        dst.descriptions = tuple(descriptions)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def write_rgb_png(path: str, optical: np.ndarray, *, gain: float = 3.2) -> str:
    """Write a benchmark-style RGB PNG with no georeferencing at all."""
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    red, green, blue = optical[3], optical[2], optical[1]  # B04, B03, B02
    stack = np.stack([red, green, blue], axis=-1).astype(np.float32) / REFLECTANCE_SCALE
    stack = np.clip(stack * gain, 0.0, 1.0)
    Image.fromarray((stack * 255).astype(np.uint8), mode="RGB").save(path)
    return path


def make_demo_set(outdir: str = "data/fixtures", *, size: int = 512, seed: int = 7) -> dict:
    """Generate the full fixture set used by the tests and the demo scenarios."""
    os.makedirs(outdir, exist_ok=True)
    manifest: dict = {"outdir": outdir, "gsd_m": DEFAULT_GSD, "epsg": DEFAULT_EPSG, "files": {}}

    def record(key: str, path: str, **extra) -> None:
        manifest["files"][key] = {"path": path, **extra}

    # --- 1. single optical and single SAR of the same scene ----------------
    scene = build_scene(size, size)
    optical = render_optical(scene, seed=seed)
    sar = render_sar(scene, seed=seed)
    record("single_optical", write_geotiff(
        os.path.join(outdir, "S2A_MSIL2A_20240118_demo_optical.tif"), optical,
        descriptions=S2_12, tags={"ACQUISITION_DATE": "2024-01-18", "SENSOR": "Sentinel-2 MSI"},
    ), areas_km2=scene.areas_km2())
    record("single_sar", write_geotiff(
        os.path.join(outdir, "S1A_IW_GRDH_20240118_demo_sar.tif"), sar,
        descriptions=["VV", "VH"],
        tags={"ACQUISITION_DATE": "2024-01-18", "SENSOR": "Sentinel-1 GRD", "UNIT": "dB"},
    ), areas_km2=scene.areas_km2())

    # --- 2. cross-modal pair, with cloud over part of the optical ----------
    cloud = make_cloud_mask(size, size, seed=seed, fraction=0.18)
    optical_cloudy = render_optical(scene, seed=seed + 1, cloud_mask=cloud)
    record("cross_optical", write_geotiff(
        os.path.join(outdir, "S2A_MSIL2A_20240118_cloudy_optical.tif"), optical_cloudy,
        descriptions=S2_12,
        tags={"ACQUISITION_DATE": "2024-01-18", "SENSOR": "Sentinel-2 MSI",
              "CLOUD_FRACTION": round(float(cloud.mean()), 4)},
    ), cloud_fraction=round(float(cloud.mean()), 4), areas_km2=scene.areas_km2())
    record("cross_sar", write_geotiff(
        os.path.join(outdir, "S1A_IW_GRDH_20240118_paired_sar.tif"), sar,
        descriptions=["VV", "VH"],
        tags={"ACQUISITION_DATE": "2024-01-18", "SENSOR": "Sentinel-1 GRD", "UNIT": "dB"},
    ))

    # --- 3. bi-temporal pair: urban growth and a shrinking lake ------------
    t1 = build_scene(size, size, lake_scale=1.0,
                     urban_blocks=((0.58, 0.55, 0.27, 0.24),))
    t2 = build_scene(size, size, lake_scale=0.78,
                     urban_blocks=((0.58, 0.55, 0.27, 0.24), (0.30, 0.60, 0.18, 0.16)))
    record("bitemporal_t1", write_geotiff(
        os.path.join(outdir, "S2A_MSIL2A_20220112_T1.tif"),
        render_optical(t1, seed=seed + 2), descriptions=S2_12,
        tags={"ACQUISITION_DATE": "2022-01-12", "SENSOR": "Sentinel-2 MSI"},
    ), areas_km2=t1.areas_km2())
    record("bitemporal_t2", write_geotiff(
        os.path.join(outdir, "S2A_MSIL2A_20240118_T2.tif"),
        render_optical(t2, seed=seed + 3), descriptions=S2_12,
        tags={"ACQUISITION_DATE": "2024-01-18", "SENSOR": "Sentinel-2 MSI"},
    ), areas_km2=t2.areas_km2())
    manifest["bitemporal_truth_km2"] = {
        name: round(t2.area_km2(name) - t1.area_km2(name), 6) for name in CLASS_NAMES
    }

    # --- 4. a scene 200 km away, to demonstrate rejection ------------------
    far = build_scene(size, size, lake_scale=0.6)
    record("incompatible_far", write_geotiff(
        os.path.join(outdir, "S2A_MSIL2A_20240118_far_scene.tif"),
        render_optical(far, seed=seed + 4), descriptions=S2_12,
        origin=(DEFAULT_ORIGIN[0] + 200_000.0, DEFAULT_ORIGIN[1] + 200_000.0),
        tags={"ACQUISITION_DATE": "2024-01-18", "SENSOR": "Sentinel-2 MSI"},
    ), note="offset 200 km from the other fixtures; a pair with these must be rejected")

    # --- 5. benchmark-style PNG with no georeferencing ---------------------
    record("benchmark_png", write_rgb_png(
        os.path.join(outdir, "benchmark_rgb_no_georeference.png"), optical,
    ), note="stands in for VRSBench / RSVQA / CDVQA imagery: PNG, no CRS, no geotransform")

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/fixtures"
    result = make_demo_set(target)
    print(json.dumps(result, indent=2))
