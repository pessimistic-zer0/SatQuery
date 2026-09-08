"""Runtime capability detection.

The demo has to run on a 6 GB laptop and it has to run with no GPU at all, so
the controller asks this module what is available rather than assuming. Setting
``SATQUERY_FORCE_CPU=1`` forces the no-GPU path, which is how the classical
fallback is exercised on a machine that does have a card.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GpuInfo:
    available: bool
    name: str | None = None
    total_mb: int | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "name": self.name,
            "total_mb": self.total_mb,
            "reason": self.reason,
        }


@lru_cache(maxsize=1)
def detect_gpu() -> GpuInfo:
    """Report whether a usable CUDA device is present."""
    if os.environ.get("SATQUERY_FORCE_CPU", "").strip() in {"1", "true", "yes"}:
        return GpuInfo(False, reason="SATQUERY_FORCE_CPU is set")
    try:
        import torch  # noqa: PLC0415 - optional dependency, absent until Phase D
    except ImportError:
        return GpuInfo(False, reason="PyTorch is not installed (Phase D dependency)")
    try:
        if not torch.cuda.is_available():
            return GpuInfo(False, reason="PyTorch reports no CUDA device")
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return GpuInfo(True, props.name, int(props.total_memory / (1024 * 1024)))
    except Exception as exc:  # noqa: BLE001 - a broken driver must not stop the app
        return GpuInfo(False, reason=f"CUDA probe failed: {exc}")


def vram_budget_mb() -> int:
    """Usable VRAM, leaving headroom for activations and the display server."""
    gpu = detect_gpu()
    if not gpu.available or not gpu.total_mb:
        return 0
    return max(0, int(gpu.total_mb * 0.80) - 512)
