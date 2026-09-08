"""Shared vocabulary for the SatQuery AI controller.

Every enum here is part of the observable execution trace, so the string values
are stable and safe to render in the UI or the exported report.
"""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """Sensor family of a single input image."""

    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"


class InputConfiguration(str, Enum):
    """How many images were supplied and how they relate to each other.

    The problem statement defines exactly three legal configurations. Anything
    else is rejected before a model runs.
    """

    SINGLE = "single"
    CROSS_MODAL_PAIR = "cross_modal_pair"
    BI_TEMPORAL_PAIR = "bi_temporal_pair"
    INCOMPATIBLE = "incompatible"


class Task(str, Enum):
    """Task types the controller can route a natural-language query to."""

    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_VQA = "change_vqa"
    CHANGE_MAP = "change_map"
    CROSS_MODAL_ANALYSIS = "cross_modal_analysis"


class CheckStatus(str, Enum):
    """Outcome of one compatibility check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class StepKind(str, Enum):
    """Stage of the controller pipeline a trace step belongs to."""

    VALIDATE = "validate"
    COMPATIBILITY = "compatibility"
    INTERPRET = "interpret"
    ROUTE = "route"
    TOOL = "tool"
    INTEGRATE = "integrate"


class ToolStatus(str, Enum):
    """Result state of a single tool execution."""

    OK = "ok"
    NOT_IMPLEMENTED = "not_implemented"
    SKIPPED = "skipped"
    FAILED = "failed"


# Which input configurations each task is legal for. The router uses this to
# discard tasks that the uploaded images cannot support, before scoring.
TASK_CONFIGURATIONS: dict[Task, frozenset[InputConfiguration]] = {
    Task.VQA: frozenset({InputConfiguration.SINGLE}),
    Task.CAPTION: frozenset({InputConfiguration.SINGLE}),
    Task.GROUNDING: frozenset({InputConfiguration.SINGLE}),
    Task.CHANGE_DESCRIPTION: frozenset({InputConfiguration.BI_TEMPORAL_PAIR}),
    Task.CHANGE_VQA: frozenset({InputConfiguration.BI_TEMPORAL_PAIR}),
    Task.CHANGE_MAP: frozenset({InputConfiguration.BI_TEMPORAL_PAIR}),
    Task.CROSS_MODAL_ANALYSIS: frozenset({InputConfiguration.CROSS_MODAL_PAIR}),
}

# Fallback task per configuration, used when the query gives the router nothing
# to go on.
DEFAULT_TASK: dict[InputConfiguration, Task] = {
    InputConfiguration.SINGLE: Task.VQA,
    InputConfiguration.BI_TEMPORAL_PAIR: Task.CHANGE_DESCRIPTION,
    InputConfiguration.CROSS_MODAL_PAIR: Task.CROSS_MODAL_ANALYSIS,
}
