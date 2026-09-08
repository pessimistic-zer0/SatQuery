"""Sensor inference: what kind of image is this, really?

The problem statement requires the controller to check the modality of every
input before selecting a model. Filenames lie and metadata is often missing, so
inference combines three independent lines of evidence:

1. band count and layout matched against known sensor profiles,
2. metadata tags and the filename,
3. the pixel statistics themselves - SAR speckle looks nothing like optical
   reflectance.

Every conclusion carries the evidence that produced it, so the trace can show
why an image was called SAR rather than asserting it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .enums import Modality

# --- Known band layouts ----------------------------------------------------
# Sentinel-2 band order as distributed by BigEarthNet (12 bands, no B10) and as
# full L1C (13 bands). Wavelengths are the S2A central values in nanometres.
S2_12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S2_13 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
S2_WAVELENGTHS_NM = {
    "B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740,
    "B07": 783, "B08": 842, "B8A": 865, "B09": 945, "B10": 1375, "B11": 1610,
    "B12": 2190,
}

# Canonical roles used by the Phase B index toolbox. Keeping the mapping here
# means an index only has to ask for "nir", never for a band number.
S2_ROLES = {
    "coastal": "B01", "blue": "B02", "green": "B03", "red": "B04",
    "rededge1": "B05", "rededge2": "B06", "rededge3": "B07", "nir": "B08",
    "narrow_nir": "B8A", "water_vapour": "B09", "cirrus": "B10",
    "swir1": "B11", "swir2": "B12",
}


@dataclass
class BandProfile:
    """A matched band layout, giving each band a name and a role."""

    profile: str
    names: list[str]
    roles: dict[str, int] = field(default_factory=dict)
    wavelengths_nm: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "names": self.names,
            "roles": self.roles,
            "wavelengths_nm": self.wavelengths_nm,
        }


@dataclass
class SensorInference:
    """Result of modality and sensor identification for one image."""

    modality: Modality
    confidence: float
    sensor_guess: str | None
    band_profile: BandProfile | None
    evidence: list[str]

    def to_dict(self) -> dict:
        return {
            "modality": self.modality.value,
            "confidence": round(self.confidence, 3),
            "sensor_guess": self.sensor_guess,
            "band_profile": self.band_profile.to_dict() if self.band_profile else None,
            "evidence": self.evidence,
        }


def _s2_profile(names: list[str]) -> BandProfile:
    roles = {role: names.index(band) for role, band in S2_ROLES.items() if band in names}
    return BandProfile(
        profile=f"sentinel2_{len(names)}band",
        names=list(names),
        roles=roles,
        wavelengths_nm={b: S2_WAVELENGTHS_NM[b] for b in names},
    )


def _s1_profile() -> BandProfile:
    return BandProfile(
        profile="sentinel1_grd_dualpol",
        names=["VV", "VH"],
        roles={"vv": 0, "vh": 1},
    )


def match_band_profile(band_count: int, descriptions: tuple | None = None) -> BandProfile | None:
    """Match a band layout by explicit band descriptions first, then by count."""
    labels = [d.strip().upper() for d in (descriptions or []) if d]

    if labels and {"VV", "VH"} <= set(labels):
        return BandProfile(
            profile="sar_dualpol",
            names=labels,
            roles={"vv": labels.index("VV"), "vh": labels.index("VH")},
        )
    if labels and all(re.fullmatch(r"B\d{1,2}A?", lb) for lb in labels) and len(labels) >= 4:
        roles = {role: labels.index(b) for role, b in S2_ROLES.items() if b in labels}
        return BandProfile(
            profile="sentinel2_labelled",
            names=labels,
            roles=roles,
            wavelengths_nm={b: S2_WAVELENGTHS_NM[b] for b in labels if b in S2_WAVELENGTHS_NM},
        )

    if band_count == 12:
        return _s2_profile(S2_12)
    if band_count == 13:
        return _s2_profile(S2_13)
    if band_count == 2:
        return _s1_profile()
    if band_count == 3:
        return BandProfile("optical_rgb", ["R", "G", "B"], {"red": 0, "green": 1, "blue": 2})
    if band_count == 4:
        # Band order for 4-band products is genuinely ambiguous (RGBN vs BGRN).
        # Assume RGBN and let the passport raise a warning.
        return BandProfile(
            "optical_rgbn", ["R", "G", "B", "NIR"],
            {"red": 0, "green": 1, "blue": 2, "nir": 3},
        )
    if band_count == 1:
        return BandProfile("single_band", ["B1"], {})
    return None


@dataclass
class PixelStats:
    """Cheap statistics sampled from a decimated read of the raster."""

    minimum: float
    maximum: float
    mean: float
    std: float
    negative_fraction: float
    coefficient_of_variation: float
    valid_fraction: float

    def to_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


def compute_pixel_stats(array: np.ndarray, nodata: float | None = None) -> PixelStats:
    """Summarise a sampled array, ignoring nodata and non-finite pixels."""
    flat = np.asarray(array, dtype=np.float64).ravel()
    valid = np.isfinite(flat)
    if nodata is not None:
        valid &= flat != nodata
    total = flat.size or 1
    sample = flat[valid]
    if sample.size == 0:
        return PixelStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = float(sample.mean())
    std = float(sample.std())
    return PixelStats(
        minimum=float(sample.min()),
        maximum=float(sample.max()),
        mean=mean,
        std=std,
        negative_fraction=float((sample < 0).mean()),
        coefficient_of_variation=float(std / abs(mean)) if abs(mean) > 1e-9 else 0.0,
        valid_fraction=float(sample.size / total),
    )


_SENSOR_HINTS: list[tuple[str, str, Modality]] = [
    (r"\bS1[AB]?\b|SENTINEL[-_ ]?1|GRDH?|IW_GRD", "Sentinel-1 SAR", Modality.SAR),
    (r"\bS2[AB]?\b|SENTINEL[-_ ]?2|MSIL[12][CA]", "Sentinel-2 MSI", Modality.OPTICAL),
    (r"RISAT", "RISAT SAR", Modality.SAR),
    (r"CARTOSAT", "Cartosat-2S optical", Modality.OPTICAL),
    (r"TERRASAR|TSX|COSMO|ALOS|PALSAR|RADARSAT", "SAR (other mission)", Modality.SAR),
    (r"LANDSAT|LC0[89]", "Landsat optical", Modality.OPTICAL),
    (r"RESOURCESAT|LISS", "Resourcesat optical", Modality.OPTICAL),
]


def _hint_from_text(text: str) -> tuple[str, Modality] | None:
    upper = text.upper()
    for pattern, name, modality in _SENSOR_HINTS:
        if re.search(pattern, upper):
            return name, modality
    return None


def infer_sensor(
    *,
    filename: str,
    band_count: int,
    dtype: str,
    descriptions: tuple | None,
    tags: dict | None,
    stats: PixelStats | None,
    gsd_m: float | None,
) -> SensorInference:
    """Score optical against SAR from layout, metadata and pixel statistics."""
    evidence: list[str] = []
    optical = 0.0
    sar = 0.0
    sensor_guess: str | None = None

    profile = match_band_profile(band_count, descriptions)

    # --- 1. metadata and filename hints -----------------------------------
    tag_blob = " ".join(f"{k}={v}" for k, v in (tags or {}).items())
    hit = _hint_from_text(filename) or _hint_from_text(tag_blob)
    if hit:
        sensor_guess, hinted = hit
        evidence.append(f"name/metadata matches {sensor_guess}")
        if hinted is Modality.SAR:
            sar += 3.0
        else:
            optical += 3.0

    # --- 2. band layout ----------------------------------------------------
    if profile:
        if profile.profile.startswith("sentinel2"):
            optical += 3.0
            evidence.append(f"{band_count} bands matches the Sentinel-2 layout")
            sensor_guess = sensor_guess or "Sentinel-2 MSI (band count)"
        elif "dualpol" in profile.profile or profile.profile == "sar_dualpol":
            sar += 2.0
            evidence.append("2 bands labelled/ordered as VV+VH dual polarisation")
            sensor_guess = sensor_guess or "dual-pol SAR"
        elif profile.profile in {"optical_rgb", "optical_rgbn"}:
            optical += 1.5
            evidence.append(f"{band_count}-band layout is typical of optical imagery")

    # --- 3. dtype ----------------------------------------------------------
    if dtype.startswith("uint"):
        optical += 1.0
        evidence.append(f"integer dtype ({dtype}) is typical of optical reflectance products")
    elif dtype.startswith("float"):
        sar += 0.5
        evidence.append(f"floating-point dtype ({dtype}) is common for calibrated SAR backscatter")

    # --- 4. pixel statistics ----------------------------------------------
    if stats is not None:
        if stats.negative_fraction > 0.4 and -60.0 <= stats.minimum <= 5.0:
            sar += 3.0
            evidence.append(
                f"{stats.negative_fraction:.0%} of pixels are negative in the range "
                f"[{stats.minimum:.1f}, {stats.maximum:.1f}] - consistent with backscatter in dB"
            )
        if stats.coefficient_of_variation > 0.55 and band_count <= 2:
            sar += 1.5
            evidence.append(
                f"high coefficient of variation ({stats.coefficient_of_variation:.2f}) "
                "is consistent with SAR speckle"
            )
        elif 0.0 < stats.coefficient_of_variation < 0.45 and band_count >= 3:
            optical += 0.5
            evidence.append(
                f"low coefficient of variation ({stats.coefficient_of_variation:.2f}) "
                "is consistent with optical reflectance"
            )

    # --- 5. ground sample distance ----------------------------------------
    if gsd_m is not None:
        if 9.0 <= gsd_m <= 11.0 and band_count >= 10:
            optical += 1.0
            evidence.append("10 m ground sample distance with >=10 bands points to Sentinel-2")
        elif gsd_m < 2.0:
            evidence.append(
                f"sub-2 m ground sample distance ({gsd_m:.2f} m) indicates a very-high-resolution sensor"
            )
            if sensor_guess is None and band_count <= 4:
                sensor_guess = "very-high-resolution optical (Cartosat-2S class)"
                optical += 0.5

    total = optical + sar
    if total < 1.0:
        return SensorInference(Modality.UNKNOWN, 0.0, sensor_guess, profile,
                               evidence or ["no distinguishing evidence found"])

    if sar > optical:
        modality, score = Modality.SAR, sar
    else:
        modality, score = Modality.OPTICAL, optical
    confidence = min(0.99, score / total * min(1.0, total / 5.0) + 0.5 * (score / total))
    confidence = min(0.99, max(0.0, confidence))
    return SensorInference(modality, confidence, sensor_guess, profile, evidence)
