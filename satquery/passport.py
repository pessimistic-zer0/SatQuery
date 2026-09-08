"""The Sensor Passport.

One passport is built per uploaded image. It answers, deterministically and
without any model, the questions the controller must settle before it can route
a query: what sensor is this, what does one pixel cover on the ground, where on
Earth is it, and can an answer be given in square kilometres at all.

Benchmark imagery (VRSBench, RSVQA, CDVQA) arrives as PNG or JPEG with no
georeferencing. That is legal input under the problem statement, so it is not an
error - the passport records ``georeferenced: False`` and every downstream area
figure is reported in pixels instead of km2.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import warnings as warns
from dataclasses import dataclass, field

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError

from .enums import Modality
from .sensors import BandProfile, PixelStats, SensorInference, compute_pixel_stats, infer_sensor

# Longest side of the decimated read used for statistics. Keeps passport build
# time flat regardless of how large the source raster is.
STATS_MAX_SIDE = 512

RASTER_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
GEOSPATIAL_EXTENSIONS = {".tif", ".tiff"}

_DATE_PATTERNS = [
    (r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})T?\d{0,6}", "%Y%m%d"),
]
_DATE_TAG_KEYS = [
    "ACQUISITION_DATE", "ACQUISITIONDATE", "DATE_ACQUIRED", "SENSING_TIME",
    "TIFFTAG_DATETIME", "DATETIME", "TIME_START", "DATE",
]


class PassportError(Exception):
    """Raised when a file cannot be opened as a raster at all."""


def _parse_date(text: str) -> str | None:
    for pattern, _fmt in _DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            year, month, day = (int(g) for g in match.groups())
            try:
                return dt.date(year, month, day).isoformat()
            except ValueError:
                continue
    return None


def extract_acquisition_date(filename: str, tags: dict) -> tuple[str | None, str | None]:
    """Return (ISO date, where it came from). Tags win over the filename."""
    for key in _DATE_TAG_KEYS:
        for tag_key, value in tags.items():
            if tag_key.upper() == key and value:
                parsed = _parse_date(str(value))
                if parsed:
                    return parsed, f"metadata tag {tag_key}"
    parsed = _parse_date(filename)
    if parsed:
        return parsed, "filename"
    return None, None


def _gsd_metres(crs, transform, centre_lat: float | None) -> tuple[float | None, float | None, str]:
    """Ground sample distance in metres, converting from degrees when needed."""
    if transform is None:
        return None, None, "no geotransform"
    px = abs(transform.a)
    py = abs(transform.e)
    if px == 0 or py == 0:
        return None, None, "degenerate geotransform"
    if crs is None:
        return None, None, "no CRS"
    if crs.is_projected:
        # Assume metre-based projected CRS; note it when the unit differs.
        unit = (crs.linear_units or "metre").lower()
        if unit.startswith("met"):
            return px, py, "projected CRS in metres"
        return None, None, f"projected CRS in unsupported unit '{unit}'"
    # Geographic CRS: convert degrees to metres at the image centre latitude.
    lat = centre_lat if centre_lat is not None else 0.0
    metres_per_deg_lat = 111_132.0
    metres_per_deg_lon = 111_320.0 * max(math.cos(math.radians(lat)), 1e-6)
    return px * metres_per_deg_lon, py * metres_per_deg_lat, (
        f"geographic CRS converted at latitude {lat:.3f} degrees"
    )


@dataclass
class ImagePassport:
    """Everything the controller knows about one image before any model runs."""

    path: str
    filename: str
    size_bytes: int
    driver: str
    width: int
    height: int
    band_count: int
    dtype: str
    nodata: float | None

    georeferenced: bool
    crs: str | None
    epsg: int | None
    transform: list[float] | None
    bounds_native: list[float] | None
    bounds_wgs84: list[float] | None
    gsd_x_m: float | None
    gsd_y_m: float | None
    gsd_note: str
    area_km2: float | None

    modality: Modality
    modality_confidence: float
    modality_evidence: list[str]
    sensor_guess: str | None
    band_profile: BandProfile | None
    band_stats: list[PixelStats]

    acquisition_date: str | None
    acquisition_date_source: str | None
    tags: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def band_index(self, role: str) -> int | None:
        """Index of a semantic band ('nir', 'vv', ...), or None if absent."""
        if self.band_profile is None:
            return None
        return self.band_profile.roles.get(role)

    def has_roles(self, *roles: str) -> bool:
        return all(self.band_index(r) is not None for r in roles)

    def pixel_area_m2(self) -> float | None:
        if self.gsd_x_m is None or self.gsd_y_m is None:
            return None
        return self.gsd_x_m * self.gsd_y_m

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "driver": self.driver,
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "dtype": self.dtype,
            "nodata": self.nodata,
            "georeferenced": self.georeferenced,
            "crs": self.crs,
            "epsg": self.epsg,
            "transform": self.transform,
            "bounds_native": self.bounds_native,
            "bounds_wgs84": self.bounds_wgs84,
            "gsd_x_m": round(self.gsd_x_m, 4) if self.gsd_x_m else None,
            "gsd_y_m": round(self.gsd_y_m, 4) if self.gsd_y_m else None,
            "gsd_note": self.gsd_note,
            "area_km2": round(self.area_km2, 4) if self.area_km2 else None,
            "modality": self.modality.value,
            "modality_confidence": round(self.modality_confidence, 3),
            "modality_evidence": self.modality_evidence,
            "sensor_guess": self.sensor_guess,
            "band_profile": self.band_profile.to_dict() if self.band_profile else None,
            "band_stats": [s.to_dict() for s in self.band_stats],
            "acquisition_date": self.acquisition_date,
            "acquisition_date_source": self.acquisition_date_source,
            "warnings": self.warnings,
        }

    def summary_line(self) -> str:
        """One-line human summary, used in the trace and the report."""
        parts = [f"{self.width}x{self.height}", f"{self.band_count} band(s)", self.dtype]
        parts.append(self.sensor_guess or f"{self.modality.value} (unidentified sensor)")
        if self.gsd_x_m:
            parts.append(f"{self.gsd_x_m:.1f} m/px")
        if self.area_km2:
            parts.append(f"{self.area_km2:.2f} km2")
        else:
            parts.append("pixel-space only")
        if self.acquisition_date:
            parts.append(self.acquisition_date)
        return " | ".join(parts)


def build_passport(path: str) -> ImagePassport:
    """Open one image and derive its passport. Never runs a model."""
    if not os.path.isfile(path):
        raise PassportError(f"file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in RASTER_EXTENSIONS:
        raise PassportError(
            f"unsupported extension '{ext}'; expected one of {sorted(RASTER_EXTENSIONS)}"
        )

    warnings: list[str] = []
    try:
        # A benchmark PNG legitimately has no geotransform, so rasterio's warning
        # about it is expected here and is reported as a passport warning instead.
        with warns.catch_warnings():
            warns.simplefilter("ignore", NotGeoreferencedWarning)
            dataset = rasterio.open(path)
    except RasterioIOError as exc:
        raise PassportError(f"cannot open {os.path.basename(path)} as a raster: {exc}") from exc

    with dataset:
        width, height = dataset.width, dataset.height
        band_count = dataset.count
        dtype = dataset.dtypes[0] if dataset.dtypes else "unknown"
        nodata = dataset.nodata
        tags = dict(dataset.tags())
        descriptions = dataset.descriptions
        crs = dataset.crs

        transform = dataset.transform
        # A raster with no CRS carries rasterio's default identity transform;
        # that is not a real geotransform, so treat it as absent.
        has_geo = crs is not None and not transform.is_identity
        bounds_native = list(dataset.bounds) if has_geo else None

        bounds_wgs84 = None
        centre_lat = None
        if has_geo:
            try:
                to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                xs = [dataset.bounds.left, dataset.bounds.right]
                ys = [dataset.bounds.bottom, dataset.bounds.top]
                lon0, lat0 = to_wgs84.transform(xs[0], ys[0])
                lon1, lat1 = to_wgs84.transform(xs[1], ys[1])
                bounds_wgs84 = [min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1)]
                centre_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2.0
            except Exception as exc:  # noqa: BLE001 - a bad CRS must not kill the upload
                warnings.append(f"could not reproject bounds to WGS84: {exc}")

        gsd_x, gsd_y, gsd_note = _gsd_metres(crs, transform if has_geo else None, centre_lat)

        # Decimated read for statistics: cap the longest side.
        scale = max(1.0, max(width, height) / STATS_MAX_SIDE)
        out_h = max(1, int(height / scale))
        out_w = max(1, int(width / scale))
        sample = dataset.read(out_shape=(band_count, out_h, out_w), masked=False)

    band_stats = [compute_pixel_stats(sample[i], nodata) for i in range(band_count)]
    overall = compute_pixel_stats(sample, nodata)

    filename = os.path.basename(path)
    inference: SensorInference = infer_sensor(
        filename=filename,
        band_count=band_count,
        dtype=str(dtype),
        descriptions=descriptions,
        tags=tags,
        stats=overall,
        gsd_m=gsd_x,
    )

    area_km2 = None
    if gsd_x and gsd_y:
        area_km2 = (width * gsd_x) * (height * gsd_y) / 1e6

    acq_date, acq_source = extract_acquisition_date(filename, tags)

    # --- warnings the operator should see ---------------------------------
    if not has_geo:
        if ext in GEOSPATIAL_EXTENSIONS:
            warnings.append(
                "GeoTIFF carries no CRS or geotransform; area answers will be in pixels, not km2"
            )
        else:
            warnings.append(
                f"{ext.lstrip('.').upper()} input is not georeferenced; accepted for benchmark "
                "datasets only, and all spatial answers will be in pixel coordinates"
            )
    if crs is not None and transform is not None and transform.is_identity:
        warnings.append("a CRS is set but the geotransform is the identity - georeferencing ignored")
    if inference.modality is Modality.UNKNOWN:
        warnings.append("sensor modality could not be determined; cross-modal routing is unavailable")
    if band_count == 4:
        warnings.append("4-band order assumed to be R,G,B,NIR - verify before trusting spectral indices")
    if overall.valid_fraction < 0.5:
        warnings.append(
            f"only {overall.valid_fraction:.0%} of sampled pixels are valid; the scene may be mostly nodata"
        )
    if acq_date is None:
        warnings.append("no acquisition date found; bi-temporal ordering will rely on upload order")

    return ImagePassport(
        path=path,
        filename=filename,
        size_bytes=os.path.getsize(path),
        driver=dataset.driver,
        width=width,
        height=height,
        band_count=band_count,
        dtype=str(dtype),
        nodata=float(nodata) if nodata is not None else None,
        georeferenced=bool(has_geo),
        crs=str(crs) if crs is not None else None,
        epsg=crs.to_epsg() if crs is not None else None,
        transform=list(transform)[:6] if has_geo else None,
        bounds_native=bounds_native,
        bounds_wgs84=bounds_wgs84,
        gsd_x_m=gsd_x,
        gsd_y_m=gsd_y,
        gsd_note=gsd_note,
        area_km2=area_km2,
        modality=inference.modality,
        modality_confidence=inference.confidence,
        modality_evidence=inference.evidence,
        sensor_guess=inference.sensor_guess,
        band_profile=inference.band_profile,
        band_stats=band_stats,
        acquisition_date=acq_date,
        acquisition_date_source=acq_source,
        tags=tags,
        warnings=warnings,
    )
