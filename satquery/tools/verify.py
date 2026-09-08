"""The Physics Verifier.

Every answer this system produces is checked against an independent physical
measurement before it is reported. A model - or an index threshold - can say
"built-up area increased"; the verifier goes back to the pixels, computes the
mean spectral index at each date, and asks whether the physics agrees.

The check is only worth anything if it is *not* the computation that produced
the claim. Two cases:

* **A direction claim over a bi-temporal pair.** The change tool decides by
  thresholding a normalised magnitude and counting classified pixels; the
  verifier decides by the shift in the scene-wide mean index, tested against its
  own standard error. Different statistics over the same physics - agreement is
  real evidence, disagreement is a real warning.
* **A presence claim from an index tool.** There is nothing to check. Asking
  whether MNDWI is positive, when the claim came from thresholding MNDWI, is
  circular, and reporting it as "supported" would inflate confidence for free.
  Those claims are marked ``not_independent`` and excluded from the score.

When the neural tools arrive in Phase D the second case becomes genuinely
independent - a vision-language model's claim checked against the pixels - and
the same code starts earning its place there.

This is what "confidence is measured, not guessed" has to mean in practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..compat import order_bitemporal
from ..enums import InputConfiguration, Modality, ToolStatus
from ..indices import INDEX_DEFINITIONS, compute_index
from ..raster import read_bands, read_pair
from ..registry import ToolContext, ToolResult

# The index whose sign tracks each class, and which direction means "more of it".
CLASS_INDEX = {
    "water": ("mndwi", +1), "vegetation": ("ndvi", +1),
    "built_up": ("ndbi", +1), "bare_soil": ("bsi", +1),
}

CLASS_WORDS = {
    "water": r"\b(water|lake|river|pond|reservoir|flood)\w*\b",
    "vegetation": r"\b(vegetation|forest|tree|canopy|crop|green)\w*\b",
    "built_up": r"\b(built[- ]?up|urban|building|settlement|impervious)\w*\b",
    "bare_soil": r"\b(bare soil|barren|bare ground)\w*\b",
}
# How each class and direction is written back to the reader, so a verdict is a
# sentence rather than a pair of keywords.
CLASS_LABEL = {
    "water": "water", "vegetation": "vegetation",
    "built_up": "built-up area", "bare_soil": "bare soil",
}
DIRECTION_LABEL = {"increase": "increased", "decrease": "decreased", "present": "is present"}

INCREASE_WORDS = r"\b(increas|grew|grow|expand|gain|rose|risen|more)\w*\b"
DECREASE_WORDS = r"\b(decreas|shrank|shrunk|shrink|reduc|lost|loss|retreat|fell|less)\w*\b"

# Tools whose answers are themselves derived from spectral indices. A presence
# claim from one of these cannot be checked by recomputing the same index.
INDEX_DERIVED_TOOLS = frozenset({
    "spectral_index", "sar_backscatter", "change_index_diff", "modality_reliability",
})

# How much index movement counts as real, per strictness setting. A land-cover
# change affecting a few percent of a scene moves the scene-wide mean index by
# only about 0.01, so these floors are deliberately small - the noise-relative
# test below is what actually rejects a spurious shift.
STRICTNESS_MARGIN = {"lenient": 0.002, "balanced": 0.005, "strict": 0.015}

# A shift must also beat this many standard errors of the difference of means.
NOISE_SIGMAS = 3.0


@dataclass
class Verdict:
    claim: str
    subject: str
    direction: str
    evidence: str
    status: str  # supported | contradicted | inconclusive | unverifiable
    measured: float | None = None

    def to_dict(self) -> dict:
        return {
            "claim": self.claim, "subject": self.subject, "direction": self.direction,
            "status": self.status, "evidence": self.evidence,
            "measured": round(self.measured, 5) if self.measured is not None else None,
        }


def _extract_claims(text: str) -> list[tuple[str, str]]:
    """Find (class, direction) claims in a sentence of generated or composed text."""
    claims: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.;])\s+", text or ""):
        lowered = sentence.lower()
        rising = bool(re.search(INCREASE_WORDS, lowered))
        falling = bool(re.search(DECREASE_WORDS, lowered))
        if rising and falling:
            # A sentence that reports both a gain and a loss - "0.7 km2 was
            # gained and 0.1 km2 was lost" - states no net direction. Reading a
            # direction out of it would invent a claim and then contradict it.
            continue
        for class_name, pattern in CLASS_WORDS.items():
            if not re.search(pattern, lowered):
                continue
            if rising:
                claims.append((class_name, "increase"))
            elif falling:
                claims.append((class_name, "decrease"))
            else:
                claims.append((class_name, "present"))
    # Keep the first claim per (class, direction) pair.
    seen: set[tuple[str, str]] = set()
    unique = []
    for claim in claims:
        if claim not in seen:
            seen.add(claim)
            unique.append(claim)
    return unique


def _phrase(class_name: str, direction: str) -> str:
    """Write a claim the way a person would say it."""
    label = CLASS_LABEL.get(class_name, class_name.replace("_", " "))
    return f"{label} {DIRECTION_LABEL.get(direction, direction)}"


def _mean_index(bands: dict, name: str) -> tuple[float, float] | None:
    """Mean of an index and the standard error of that mean."""
    definition = INDEX_DEFINITIONS.get(name)
    if definition is None or not all(role in bands for role in definition.roles):
        return None
    values = compute_index(name, bands)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean()), float(finite.std() / np.sqrt(finite.size))


def run_physics_verifier(ctx: ToolContext) -> ToolResult:
    claim_text = " ".join(
        result.answer for _, result in ctx.prior
        if result.status is ToolStatus.OK and result.answer
    )
    # Was the claim produced by index maths, or by something independent?
    circular = all(name in INDEX_DERIVED_TOOLS for name, result in ctx.prior
                   if result.status is ToolStatus.OK and result.answer)
    if not claim_text:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail="no prior answer to verify",
        )

    claims = _extract_claims(claim_text)
    if not claims:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail="the prior answer makes no land-cover claim that indices can check",
        )

    margin = STRICTNESS_MARGIN.get(str(ctx.params.get("strictness", "balanced")), 0.015)
    optical = [p for p in ctx.passports if p.modality is Modality.OPTICAL]
    if not optical:
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail="verification needs an optical image; SAR-only claims are not "
                   "checkable by spectral index",
        )

    verdicts: list[Verdict] = []
    measurements: dict = {}

    if ctx.configuration is InputConfiguration.BI_TEMPORAL_PAIR and len(ctx.passports) == 2:
        t1_p, t2_p = order_bitemporal(ctx.passports[0], ctx.passports[1])
        roles = tuple(r for r in ("blue", "green", "red", "nir", "swir1")
                      if t1_p.band_index(r) is not None and t2_p.band_index(r) is not None)
        t1, t2, _grid, _notes = read_pair(t1_p, t2_p, roles, roles)

        for class_name, direction in claims:
            index_name, sign = CLASS_INDEX.get(class_name, (None, 1))
            if index_name is None:
                continue
            measured_1 = _mean_index(t1, index_name)
            measured_2 = _mean_index(t2, index_name)
            if measured_1 is None or measured_2 is None:
                verdicts.append(Verdict(
                    f"{class_name} {direction}", class_name, direction,
                    f"{index_name.upper()} cannot be computed from the available bands",
                    "unverifiable"))
                continue
            before, error_1 = measured_1
            after, error_2 = measured_2
            shift = (after - before) * sign
            noise = float(np.hypot(error_1, error_2))
            effective = max(margin, NOISE_SIGMAS * noise)
            measurements[f"{index_name}_t1"] = round(before, 5)
            measurements[f"{index_name}_t2"] = round(after, 5)
            measurements[f"{index_name}_shift"] = round(shift, 5)
            measurements[f"{index_name}_noise"] = round(noise, 6)

            evidence = (f"mean {index_name.upper()} moved from {before:+.4f} to {after:+.4f} "
                        f"({shift:+.4f}, {abs(shift) / noise:.0f} standard errors)")
            if direction == "present":
                status = "inconclusive"
            elif abs(shift) < effective:
                status = "inconclusive"
                evidence += (f"; below the {effective:.4f} threshold "
                             f"(floor {margin:.3f}, noise {NOISE_SIGMAS:.0f}x{noise:.5f})")
            elif (shift > 0) == (direction == "increase"):
                status = "supported"
            else:
                status = "contradicted"
            verdicts.append(Verdict(_phrase(class_name, direction), class_name, direction,
                                    evidence, status, shift))
    else:
        passport = optical[0]
        roles = tuple(r for r in ("blue", "green", "red", "nir", "swir1")
                      if passport.band_index(r) is not None)
        bands, grid, _note = read_bands(passport, roles)
        for class_name, direction in claims:
            index_name, sign = CLASS_INDEX.get(class_name, (None, 1))
            if index_name is None:
                continue
            definition = INDEX_DEFINITIONS.get(index_name)
            if definition is None or not all(r in bands for r in definition.roles):
                verdicts.append(Verdict(
                    _phrase(class_name, direction), class_name, direction,
                    f"{index_name.upper()} cannot be computed from the available bands",
                    "unverifiable"))
                continue
            values = compute_index(index_name, bands)
            coverage = float(np.nanmean(np.nan_to_num(values, nan=-1.0) > 0.0))
            measurements[f"{index_name}_positive_fraction"] = round(coverage, 5)
            evidence = (f"{index_name.upper()} is positive over {coverage:.1%} of the scene")
            if direction in {"increase", "decrease"}:
                status = "unverifiable"
                evidence += "; a single image cannot show a trend"
            elif circular:
                status = "not_independent"
                evidence += (f"; the claim itself came from {index_name.upper()}, so "
                             "recomputing it proves nothing")
            elif coverage > 0.01:
                status = "supported"
            else:
                status = "contradicted"
                evidence += ", too little to support the claim"
            verdicts.append(Verdict(_phrase(class_name, direction), class_name, direction,
                                    evidence, status, coverage))

    if not verdicts:
        return ToolResult(status=ToolStatus.SKIPPED,
                          detail="no claim could be matched to a computable index")

    if all(v.status == "not_independent" for v in verdicts):
        return ToolResult(
            status=ToolStatus.SKIPPED,
            detail=("every claim was derived from the same spectral indices this tool "
                    "would check it against; no independent verification is possible, "
                    "so no confidence is added"),
            metrics={"verdicts": [v.to_dict() for v in verdicts], "overall": "not_independent"},
        )

    supported = sum(1 for v in verdicts if v.status == "supported")
    contradicted = sum(1 for v in verdicts if v.status == "contradicted")
    checkable = sum(1 for v in verdicts if v.status in {"supported", "contradicted"})

    lines = []
    for verdict in verdicts:
        lines.append(f"The claim that {verdict.claim} is {verdict.status}: "
                     f"{verdict.evidence}.")
    answer = "Checked against the pixels. " + " ".join(lines)

    if checkable == 0:
        confidence = 0.5
        overall = "inconclusive"
    else:
        confidence = round(0.35 + 0.6 * (supported / checkable), 3)
        overall = ("supported" if contradicted == 0 else
                   "contradicted" if supported == 0 else "mixed")

    return ToolResult(
        status=ToolStatus.OK,
        answer=answer,
        detail=f"{supported}/{checkable} checkable claims supported by an independent "
               f"index measurement (strictness margin {margin})",
        metrics={
            "overall": overall,
            "supported": supported,
            "contradicted": contradicted,
            "checkable": checkable,
            "margin": margin,
            "verdicts": [v.to_dict() for v in verdicts],
            "measurements": measurements,
        },
        confidence=confidence,
    )
