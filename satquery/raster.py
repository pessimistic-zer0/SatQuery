"""Reading pixels, and putting two images on the same grid.

The passport describes an image; this module opens it. Two jobs matter here:

* **normalisation.** Sentinel-2 arrives as scaled integers and Sentinel-1 as
  either dB or linear power. Every index downstream assumes reflectance in 0-1
  and backscatter in dB, so the conversion happens once, here, and records which
  rule it applied.
* **alignment.** A pair of images is only comparable pixel by pixel once both
  sit on one grid. Real inputs differ in CRS, resolution and extent, so the pair
  is resampled onto the intersection of the two footprints at the coarser of the
  two resolutions - never upsampling one image to flatter the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from .passport import ImagePassport

# Longest side an analysis grid is allowed to reach. A 5-megapixel scene is
# plenty for index thresholding and keeps the demo responsive on a laptop.
DEFAULT_MAX_SIDE = 1536

# Sentinel-2 L2A quantisation. Reflectance is stored as an integer 0-10000.
REFLECTANCE_SCALE = 10_000.0


@dataclass
class Grid:
    """The grid an analysis ran on, and how to convert pixels to ground area."""

    width: int
    height: int
    crs: str | None
    transform: list[float] | None
    gsd_x_m: float | None
    gsd_y_m: float | None
    georeferenced: bool
    note: str = ""

    @property
    def pixel_area_m2(self) -> float | None:
        if self.gsd_x_m is None or self.gsd_y_m is None:
            return None
        return self.gsd_x_m * self.gsd_y_m

    @property
    def total_area_km2(self) -> float | None:
        """Ground area of the whole analysis grid."""
        area = self.pixel_area_m2
        if area is None:
            return None
        return self.width * self.height * area / 1e6

    def area_km2(self, mask: np.ndarray) -> float | None:
        """Ground area of a boolean mask, or None when there is no geometry."""
        area = self.pixel_area_m2
        if area is None:
            return None
        return float(np.count_nonzero(mask)) * area / 1e6

    def coverage(self, mask: np.ndarray) -> float:
        """Fraction of the grid a mask covers."""
        total = self.width * self.height
        return float(np.count_nonzero(mask) / total) if total else 0.0

    def affine(self) -> Affine | None:
        return Affine(*self.transform) if self.transform else None

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "gsd_m": round(self.gsd_x_m, 3) if self.gsd_x_m else None,
            "georeferenced": self.georeferenced,
            "pixel_area_m2": round(self.pixel_area_m2, 3) if self.pixel_area_m2 else None,
            "total_area_km2": round(self.total_area_km2, 4) if self.total_area_km2 else None,
            "note": self.note,
        }


def _decimated_shape(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = max(1.0, max(width, height) / max_side)
    return max(1, int(round(height / scale))), max(1, int(round(width / scale)))


def to_reflectance(array: np.ndarray, dtype: str) -> tuple[np.ndarray, str]:
    """Convert optical pixel values to reflectance in roughly 0-1.

    Integer products are assumed to be scaled by 10000 (Sentinel-2 L2A) unless
    their range says otherwise; 8-bit imagery is assumed to be 0-255.
    """
    values = np.asarray(array, dtype=np.float32)
    if dtype.startswith("uint8"):
        return values / 255.0, "8-bit imagery divided by 255"
    if dtype.startswith(("uint", "int")):
        peak = float(np.nanmax(values)) if values.size else 0.0
        if peak > 1000.0:
            return values / REFLECTANCE_SCALE, "integer reflectance divided by 10000"
        if peak > 1.5:
            return values / 255.0, "integer imagery divided by 255"
        return values, "integer values already in 0-1"
    peak = float(np.nanmax(values)) if values.size else 0.0
    if peak > 1.5:
        return values / REFLECTANCE_SCALE, "float values divided by 10000"
    return values, "float values already in reflectance units"


def to_decibels(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert SAR backscatter to dB, leaving it alone if it is already dB."""
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values, "no finite pixels"
    if float(finite.min()) < 0.0:
        return values, "values are already in dB"
    # Linear power: convert, guarding against zeros.
    return (10.0 * np.log10(np.maximum(values, 1e-8))).astype(np.float32), \
        "linear power converted to dB"


def _grid_from_passport(passport: ImagePassport, height: int, width: int) -> Grid:
    """Grid for a single decimated read, with the geotransform scaled to match."""
    if not passport.georeferenced or passport.transform is None:
        return Grid(width, height, None, None, None, None, False,
                    "not georeferenced; areas are reported in pixels")
    scale_x = passport.width / width
    scale_y = passport.height / height
    src = Affine(*passport.transform)
    transform = src @ Affine.scale(scale_x, scale_y)
    return Grid(
        width, height, passport.crs, list(transform)[:6],
        (passport.gsd_x_m or 0) * scale_x, (passport.gsd_y_m or 0) * scale_y,
        True,
        f"read at 1/{scale_x:.2f} of native resolution" if scale_x > 1 else "native resolution",
    )


def read_bands(
    passport: ImagePassport,
    roles: tuple[str, ...] | None = None,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
) -> tuple[dict[str, np.ndarray], Grid, str]:
    """Read named bands from one image, normalised for its modality.

    ``roles`` are semantic names from the band profile ('red', 'nir', 'vv').
    Passing None reads every band, keyed by its profile name.
    """
    height, width = _decimated_shape(passport.width, passport.height, max_side)

    if roles is None:
        wanted = {name: i for i, name in enumerate(
            passport.band_profile.names if passport.band_profile
            else [f"B{i + 1}" for i in range(passport.band_count)])}
    else:
        wanted = {}
        for role in roles:
            index = passport.band_index(role)
            if index is None:
                raise KeyError(f"{passport.filename} has no '{role}' band")
            wanted[role] = index

    with rasterio.open(passport.path) as dataset:
        indexes = [i + 1 for i in wanted.values()]
        stack = dataset.read(indexes, out_shape=(len(indexes), height, width),
                             resampling=Resampling.average)

    if passport.modality.value == "sar":
        converted, note = to_decibels(stack)
    else:
        converted, note = to_reflectance(stack, passport.dtype)

    if passport.nodata is not None:
        raw = np.asarray(stack, dtype=np.float32)
        converted = np.where(raw == passport.nodata, np.nan, converted)

    bands = {name: converted[i] for i, name in enumerate(wanted)}
    return bands, _grid_from_passport(passport, height, width), note


def _target_grid(a: ImagePassport, b: ImagePassport, max_side: int) -> tuple[Grid, Affine]:
    """Common grid over the intersection of two footprints, at the coarser GSD."""
    crs = a.crs
    bounds_a = a.bounds_native
    bounds_b = transform_bounds(b.crs, a.crs, *b.bounds_native) if b.crs != a.crs \
        else b.bounds_native

    left = max(bounds_a[0], bounds_b[0])
    bottom = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    top = min(bounds_a[3], bounds_b[3])
    if right <= left or top <= bottom:
        raise ValueError("the two footprints do not intersect")

    gsd = max(a.gsd_x_m or 1.0, b.gsd_x_m or 1.0)
    width = max(1, int(round((right - left) / gsd)))
    height = max(1, int(round((top - bottom) / gsd)))
    if max(width, height) > max_side:
        scale = max(width, height) / max_side
        gsd *= scale
        width = max(1, int(round((right - left) / gsd)))
        height = max(1, int(round((top - bottom) / gsd)))

    transform = Affine(gsd, 0.0, left, 0.0, -gsd, top)
    grid = Grid(
        width, height, crs, list(transform)[:6], gsd, gsd, True,
        f"shared extent of both images, resampled to {gsd:.2f} m",
    )
    return grid, transform


def read_pair(
    a: ImagePassport,
    b: ImagePassport,
    roles_a: tuple[str, ...] | None,
    roles_b: tuple[str, ...] | None,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Grid, list[str]]:
    """Read two images onto one grid, ready for pixel-by-pixel comparison."""
    notes: list[str] = []

    if not (a.georeferenced and b.georeferenced):
        # Benchmark imagery: no geometry, so the pair must already be aligned.
        if (a.width, a.height) != (b.width, b.height):
            raise ValueError(
                f"cannot align {a.filename} ({a.width}x{a.height}) with "
                f"{b.filename} ({b.width}x{b.height}) without georeferencing"
            )
        bands_a, grid, note_a = read_bands(a, roles_a, max_side=max_side)
        bands_b, _, note_b = read_bands(b, roles_b, max_side=max_side)
        notes += [f"{a.filename}: {note_a}", f"{b.filename}: {note_b}",
                  "images assumed pixel-aligned; areas reported in pixels"]
        return bands_a, bands_b, grid, notes

    grid, transform = _target_grid(a, b, max_side)
    notes.append(grid.note)

    def warp(passport: ImagePassport, roles: tuple[str, ...] | None) -> dict[str, np.ndarray]:
        if roles is None:
            names = (passport.band_profile.names if passport.band_profile
                     else [f"B{i + 1}" for i in range(passport.band_count)])
            wanted = {name: i for i, name in enumerate(names)}
        else:
            wanted = {}
            for role in roles:
                index = passport.band_index(role)
                if index is None:
                    raise KeyError(f"{passport.filename} has no '{role}' band")
                wanted[role] = index

        out = np.full((len(wanted), grid.height, grid.width), np.nan, dtype=np.float32)
        with rasterio.open(passport.path) as dataset:
            source = dataset.read([i + 1 for i in wanted.values()]).astype(np.float32)
            reproject(
                source=source,
                destination=out,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=transform,
                dst_crs=grid.crs,
                src_nodata=passport.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        if passport.modality.value == "sar":
            converted, note = to_decibels(out)
        else:
            converted, note = to_reflectance(out, passport.dtype)
        notes.append(f"{passport.filename}: {note}")
        return {name: converted[i] for i, name in enumerate(wanted)}

    return warp(a, roles_a), warp(b, roles_b), grid, notes


def write_geotiff_mask(path: str, mask: np.ndarray, grid: Grid) -> str:
    """Write a boolean or label array as a small GeoTIFF, georeferenced if possible."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = np.asarray(mask)
    if data.dtype == bool:
        data = data.astype(np.uint8) * 255
    else:
        data = data.astype(np.uint8)
    profile = {
        "driver": "GTiff", "width": grid.width, "height": grid.height, "count": 1,
        "dtype": "uint8", "compress": "deflate", "nodata": 0,
    }
    if grid.georeferenced and grid.transform:
        profile["crs"] = grid.crs
        profile["transform"] = grid.affine()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path
