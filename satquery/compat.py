"""Pair compatibility checking.

Slide 3 of the deck promises that incompatible pairs are rejected before any
model runs. This module is that promise. Given one or two passports it decides
which of the three legal input configurations applies, and produces a list of
named checks - each with a status and a human-readable detail - that goes
straight into the execution trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import CheckStatus, InputConfiguration, Modality
from .passport import ImagePassport

# Overlap of the two footprints, as a fraction of the smaller footprint.
OVERLAP_FAIL_BELOW = 0.50
OVERLAP_WARN_BELOW = 0.90
# Ratio between the two ground sample distances.
GSD_WARN_ABOVE = 1.5
GSD_FAIL_ABOVE = 4.0
# Days between acquisitions before a same-modality pair is called bi-temporal.
SAME_DATE_TOLERANCE_DAYS = 2


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    value: object = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "value": self.value,
        }


@dataclass
class CompatibilityReport:
    """Outcome of validating the whole input set."""

    configuration: InputConfiguration
    compatible: bool
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.WARN]

    def to_dict(self) -> dict:
        return {
            "configuration": self.configuration.value,
            "compatible": self.compatible,
            "checks": [c.to_dict() for c in self.checks],
            "metrics": self.metrics,
            "reasons": self.reasons,
        }


def bbox_overlap_fraction(a: list[float], b: list[float]) -> float:
    """Intersection area as a fraction of the smaller of the two boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-12)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)
    return float(inter / min(area_a, area_b))


def _days_between(d1: str | None, d2: str | None) -> int | None:
    if not d1 or not d2:
        return None
    import datetime as dt

    return abs((dt.date.fromisoformat(d1) - dt.date.fromisoformat(d2)).days)


def check_single(passport: ImagePassport) -> CompatibilityReport:
    """Validate a one-image submission."""
    checks = [
        CheckResult(
            "readable", CheckStatus.PASS,
            f"opened as {passport.driver}: {passport.summary_line()}",
        )
    ]
    if passport.modality is Modality.UNKNOWN:
        checks.append(CheckResult(
            "modality", CheckStatus.WARN,
            "modality could not be determined; only sensor-agnostic tools are available",
            "unknown",
        ))
    else:
        checks.append(CheckResult(
            "modality", CheckStatus.PASS,
            f"identified as {passport.modality.value} "
            f"(confidence {passport.modality_confidence:.2f})",
            passport.modality.value,
        ))
    checks.append(CheckResult(
        "georeferencing",
        CheckStatus.PASS if passport.georeferenced else CheckStatus.WARN,
        f"CRS {passport.crs}, {passport.gsd_x_m:.2f} m/px" if passport.georeferenced
        else "not georeferenced; spatial answers will be in pixel coordinates",
        passport.georeferenced,
    ))
    return CompatibilityReport(
        configuration=InputConfiguration.SINGLE,
        compatible=True,
        checks=checks,
        metrics={"area_km2": passport.area_km2},
    )


def check_pair(a: ImagePassport, b: ImagePassport) -> CompatibilityReport:
    """Validate a two-image submission and classify the configuration."""
    checks: list[CheckResult] = []
    metrics: dict = {}
    reasons: list[str] = []

    both_geo = a.georeferenced and b.georeferenced

    # --- coordinate reference system --------------------------------------
    if both_geo:
        if a.epsg is not None and a.epsg == b.epsg:
            checks.append(CheckResult("crs_match", CheckStatus.PASS,
                                      f"both images are in EPSG:{a.epsg}", a.epsg))
        else:
            checks.append(CheckResult(
                "crs_match", CheckStatus.WARN,
                f"different CRS ({a.crs} vs {b.crs}); footprints compared in WGS84 "
                "and reprojection will be required before pixel-level fusion",
                [a.epsg, b.epsg],
            ))
    elif a.georeferenced != b.georeferenced:
        checks.append(CheckResult(
            "crs_match", CheckStatus.FAIL,
            "one image is georeferenced and the other is not, so the pair cannot be "
            "spatially related",
        ))
        reasons.append("mixed georeferenced and non-georeferenced inputs")
    else:
        checks.append(CheckResult(
            "crs_match", CheckStatus.WARN,
            "neither image is georeferenced; treating them as a pixel-aligned benchmark pair",
        ))

    # --- spatial overlap or pixel alignment -------------------------------
    if both_geo and a.bounds_wgs84 and b.bounds_wgs84:
        overlap = bbox_overlap_fraction(a.bounds_wgs84, b.bounds_wgs84)
        metrics["overlap_fraction"] = round(overlap, 4)
        if overlap < OVERLAP_FAIL_BELOW:
            checks.append(CheckResult(
                "co_registration", CheckStatus.FAIL,
                f"footprints overlap by only {overlap:.1%}; the images do not cover the same area",
                round(overlap, 4),
            ))
            reasons.append(f"footprint overlap {overlap:.1%} is below the {OVERLAP_FAIL_BELOW:.0%} minimum")
        elif overlap < OVERLAP_WARN_BELOW:
            checks.append(CheckResult(
                "co_registration", CheckStatus.WARN,
                f"footprints overlap by {overlap:.1%}; analysis is restricted to the shared extent",
                round(overlap, 4),
            ))
        else:
            checks.append(CheckResult(
                "co_registration", CheckStatus.PASS,
                f"footprints overlap by {overlap:.1%}", round(overlap, 4),
            ))
    elif not a.georeferenced and not b.georeferenced:
        if (a.width, a.height) == (b.width, b.height):
            checks.append(CheckResult(
                "co_registration", CheckStatus.PASS,
                f"both images are {a.width}x{a.height}; assumed pixel-aligned",
                [a.width, a.height],
            ))
        else:
            checks.append(CheckResult(
                "co_registration", CheckStatus.FAIL,
                f"raster sizes differ ({a.width}x{a.height} vs {b.width}x{b.height}) and there is "
                "no georeferencing to align them",
            ))
            reasons.append("non-georeferenced images of different sizes cannot be aligned")

    # --- ground sample distance -------------------------------------------
    if a.gsd_x_m and b.gsd_x_m:
        ratio = max(a.gsd_x_m, b.gsd_x_m) / min(a.gsd_x_m, b.gsd_x_m)
        metrics["gsd_ratio"] = round(ratio, 3)
        if ratio > GSD_FAIL_ABOVE:
            checks.append(CheckResult(
                "resolution_ratio", CheckStatus.FAIL,
                f"resolutions differ by {ratio:.1f}x ({a.gsd_x_m:.1f} m vs {b.gsd_x_m:.1f} m), "
                f"beyond the {GSD_FAIL_ABOVE:.0f}x limit for joint analysis",
                round(ratio, 3),
            ))
            reasons.append(f"resolution mismatch of {ratio:.1f}x")
        elif ratio > GSD_WARN_ABOVE:
            checks.append(CheckResult(
                "resolution_ratio", CheckStatus.WARN,
                f"resolutions differ by {ratio:.1f}x; the finer image will be resampled to "
                f"{max(a.gsd_x_m, b.gsd_x_m):.1f} m",
                round(ratio, 3),
            ))
        else:
            checks.append(CheckResult(
                "resolution_ratio", CheckStatus.PASS,
                f"comparable resolutions ({a.gsd_x_m:.1f} m and {b.gsd_x_m:.1f} m)",
                round(ratio, 3),
            ))

    # --- modality and dates decide the configuration ----------------------
    gap_days = _days_between(a.acquisition_date, b.acquisition_date)
    if gap_days is not None:
        metrics["acquisition_gap_days"] = gap_days

    modalities = {a.modality, b.modality}
    cross_modal = modalities == {Modality.OPTICAL, Modality.SAR}
    same_modality = a.modality == b.modality and a.modality is not Modality.UNKNOWN

    if cross_modal:
        configuration = InputConfiguration.CROSS_MODAL_PAIR
        checks.append(CheckResult(
            "configuration", CheckStatus.PASS,
            f"one optical and one SAR image of the same area -> cross-modal pair",
            configuration.value,
        ))
        if gap_days is not None and gap_days > 30:
            checks.append(CheckResult(
                "acquisition_gap", CheckStatus.WARN,
                f"the two acquisitions are {gap_days} days apart; differences may reflect "
                "change over time rather than sensor complementarity",
                gap_days,
            ))
    elif same_modality:
        configuration = InputConfiguration.BI_TEMPORAL_PAIR
        if gap_days is None:
            checks.append(CheckResult(
                "configuration", CheckStatus.WARN,
                f"two {a.modality.value} images with no dates in metadata -> bi-temporal pair "
                "assumed, ordered by upload",
                configuration.value,
            ))
        elif gap_days <= SAME_DATE_TOLERANCE_DAYS:
            checks.append(CheckResult(
                "configuration", CheckStatus.WARN,
                f"two {a.modality.value} images acquired {gap_days} day(s) apart; there may be "
                "no change to detect",
                configuration.value,
            ))
        else:
            checks.append(CheckResult(
                "configuration", CheckStatus.PASS,
                f"two {a.modality.value} images {gap_days} days apart -> bi-temporal pair",
                configuration.value,
            ))
    else:
        configuration = InputConfiguration.BI_TEMPORAL_PAIR
        checks.append(CheckResult(
            "configuration", CheckStatus.WARN,
            f"modalities are {a.modality.value} and {b.modality.value}; falling back to a "
            "bi-temporal reading of the pair",
            configuration.value,
        ))

    compatible = not any(c.status is CheckStatus.FAIL for c in checks)
    if not compatible:
        configuration = InputConfiguration.INCOMPATIBLE

    return CompatibilityReport(
        configuration=configuration,
        compatible=compatible,
        checks=checks,
        metrics=metrics,
        reasons=reasons,
    )


def validate_inputs(passports: list[ImagePassport]) -> CompatibilityReport:
    """Entry point: validate one or two images and classify the configuration."""
    if len(passports) == 1:
        return check_single(passports[0])
    if len(passports) == 2:
        return check_pair(passports[0], passports[1])
    return CompatibilityReport(
        configuration=InputConfiguration.INCOMPATIBLE,
        compatible=False,
        checks=[CheckResult(
            "input_count", CheckStatus.FAIL,
            f"{len(passports)} images supplied; the supported configurations are one image, "
            "a cross-modal pair, or a bi-temporal pair",
            len(passports),
        )],
        reasons=[f"unsupported input count: {len(passports)}"],
    )


def order_bitemporal(a: ImagePassport, b: ImagePassport) -> tuple[ImagePassport, ImagePassport]:
    """Return the pair as (T1, T2), using dates when both are known."""
    if a.acquisition_date and b.acquisition_date and b.acquisition_date < a.acquisition_date:
        return b, a
    return a, b
