"""The execution trace.

The problem statement is explicit that internal planning text is not evaluated -
only the observable trace: the task chosen, the tools invoked, the permitted
parameters they ran with, and the outputs. So the trace is not logging bolted on
afterwards; it is the primary output of the controller, and every stage writes
to it as it happens.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import asdict, dataclass, field

from .enums import InputConfiguration, StepKind, Task, ToolStatus

TRACE_SCHEMA_VERSION = "1.0"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class TraceStep:
    """One observable action taken by the controller."""

    index: int
    kind: StepKind
    name: str
    started_at: str
    duration_ms: float
    status: str
    detail: str = ""
    params: dict = field(default_factory=dict)
    rejected_params: dict = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    confidence: float | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["kind"] = self.kind.value
        if self.confidence is not None:
            data["confidence"] = round(self.confidence, 3)
        data["duration_ms"] = round(self.duration_ms, 2)
        return data


@dataclass
class ExecutionTrace:
    """The auditable record of one query run."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=_now)
    query: str = ""
    input_files: list[str] = field(default_factory=list)
    configuration: InputConfiguration | None = None
    task: Task | None = None
    tools: list[str] = field(default_factory=list)
    steps: list[TraceStep] = field(default_factory=list)
    answer: str | None = None
    confidence: float | None = None
    artifacts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    total_ms: float = 0.0

    _clock_start: float = field(default_factory=time.perf_counter, repr=False)

    # --- recording ---------------------------------------------------------
    def step(
        self,
        kind: StepKind,
        name: str,
        *,
        status: str = "ok",
        detail: str = "",
        params: dict | None = None,
        rejected_params: dict | None = None,
        inputs: list[str] | None = None,
        outputs: dict | None = None,
        artifacts: list[dict] | None = None,
        confidence: float | None = None,
        duration_ms: float = 0.0,
        started_at: str | None = None,
    ) -> TraceStep:
        entry = TraceStep(
            index=len(self.steps) + 1,
            kind=kind,
            name=name,
            started_at=started_at or _now(),
            duration_ms=duration_ms,
            status=status,
            detail=detail,
            params=params or {},
            rejected_params=rejected_params or {},
            inputs=inputs or [],
            outputs=outputs or {},
            artifacts=artifacts or [],
            confidence=confidence,
        )
        self.steps.append(entry)
        if artifacts:
            self.artifacts.extend(artifacts)
        return entry

    def timed(self, kind: StepKind, name: str, **kwargs) -> "_TimedStep":
        """Context manager that records a step with its measured duration."""
        return _TimedStep(self, kind, name, kwargs)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def reject(self, reasons: list[str]) -> None:
        self.rejected = True
        self.configuration = InputConfiguration.INCOMPATIBLE
        self.rejection_reasons.extend(r for r in reasons if r not in self.rejection_reasons)

    def finish(self, answer: str | None = None, confidence: float | None = None) -> None:
        self.answer = answer
        self.confidence = confidence
        self.total_ms = (time.perf_counter() - self._clock_start) * 1000.0

    # --- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "query": self.query,
            "input_files": self.input_files,
            "configuration": self.configuration.value if self.configuration else None,
            "task": self.task.value if self.task else None,
            "tools": self.tools,
            "answer": self.answer,
            "confidence": round(self.confidence, 3) if self.confidence is not None else None,
            "rejected": self.rejected,
            "rejection_reasons": self.rejection_reasons,
            "warnings": self.warnings,
            "artifacts": self.artifacts,
            "total_ms": round(self.total_ms, 2),
            "steps": [s.to_dict() for s in self.steps],
        }

    def summary_lines(self) -> list[str]:
        """The compact trace shown on slide 3, rendered from real data."""
        lines = [
            f"run: {self.run_id}",
            f"query: {self.query}",
            f"inputs: [{', '.join(self.input_files)}]",
            f"configuration: {self.configuration.value if self.configuration else 'n/a'}",
        ]
        if self.rejected:
            lines.append(f"rejected: {'; '.join(self.rejection_reasons)}")
            return lines
        lines.append(f"task: {self.task.value if self.task else 'n/a'}")
        lines.append(f"tools: [{', '.join(self.tools)}]")
        for step in self.steps:
            if step.kind is StepKind.TOOL and step.params:
                lines.append(f"params({step.name}): {step.params}")
        if self.answer:
            lines.append(f"answer: {self.answer}")
        if self.confidence is not None:
            lines.append(f"confidence: {self.confidence:.2f}")
        lines.append(f"elapsed: {self.total_ms:.0f} ms")
        return lines


class _TimedStep:
    """Helper returned by ExecutionTrace.timed."""

    def __init__(self, trace: ExecutionTrace, kind: StepKind, name: str, kwargs: dict):
        self._trace = trace
        self._kind = kind
        self._name = name
        self._kwargs = kwargs
        self._start = 0.0
        self._started_at = ""
        self.step: TraceStep | None = None

    def __enter__(self) -> "_TimedStep":
        self._start = time.perf_counter()
        self._started_at = _now()
        return self

    def update(self, **kwargs) -> None:
        """Fill in fields discovered while the step was running."""
        self._kwargs.update(kwargs)

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = (time.perf_counter() - self._start) * 1000.0
        if exc is not None:
            self._kwargs["status"] = ToolStatus.FAILED.value
            self._kwargs["detail"] = f"{type(exc).__name__}: {exc}"
        self.step = self._trace.step(
            self._kind, self._name,
            duration_ms=duration, started_at=self._started_at, **self._kwargs,
        )
        return False
