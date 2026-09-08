"""The predefined model and tool registry.

Two rules from the problem statement shape this module:

* the controller selects tools "from a predefined registry" - so tools declare
  themselves here, and the controller never constructs one ad hoc;
* the controller may "configure only permitted task parameters" - so every tool
  declares its parameter schema, and anything outside it is rejected and
  recorded in the trace rather than silently dropped.

Tools declared but not yet implemented are still registered. That keeps routing
honest: the trace shows the tool the controller would have run, and says plainly
that the implementation is pending.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .enums import InputConfiguration, Modality, Task, ToolStatus
from .passport import ImagePassport


@dataclass(frozen=True)
class ToolParam:
    """One permitted task parameter."""

    name: str
    type: str  # "float" | "int" | "bool" | "str"
    default: Any
    description: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple | None = None

    def coerce(self, value: Any) -> tuple[bool, Any, str]:
        """Return (accepted, value, message)."""
        try:
            if self.type == "float":
                value = float(value)
            elif self.type == "int":
                value = int(value)
            elif self.type == "bool":
                if isinstance(value, str):
                    value = value.strip().lower() in {"1", "true", "yes", "on"}
                value = bool(value)
            elif self.type == "str":
                value = str(value)
        except (TypeError, ValueError):
            return False, self.default, f"'{value}' is not a valid {self.type}"

        if self.choices is not None and value not in self.choices:
            return False, self.default, f"must be one of {list(self.choices)}"
        if self.minimum is not None and value < self.minimum:
            return False, self.default, f"below the minimum of {self.minimum}"
        if self.maximum is not None and value > self.maximum:
            return False, self.default, f"above the maximum of {self.maximum}"
        return True, value, ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "type": self.type, "default": self.default,
            "description": self.description, "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": list(self.choices) if self.choices else None,
        }


@dataclass
class ToolContext:
    """Everything a tool is given when it runs."""

    passports: list[ImagePassport]
    query: str
    task: Task
    configuration: InputConfiguration
    params: dict
    workdir: str
    gpu_available: bool = False
    entities: dict = field(default_factory=dict)
    # Results of the tools that already ran in this pipeline, in order. The
    # Physics Verifier reads this to find the claim it has to check.
    prior: list[tuple[str, "ToolResult"]] = field(default_factory=list)

    @property
    def primary(self) -> ImagePassport:
        return self.passports[0]


@dataclass
class ToolResult:
    """What a tool hands back to the controller."""

    status: ToolStatus
    answer: str | None = None
    detail: str = ""
    metrics: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    confidence: float | None = None

    @classmethod
    def pending(cls, tool_name: str, phase: str) -> "ToolResult":
        return cls(
            status=ToolStatus.NOT_IMPLEMENTED,
            detail=f"'{tool_name}' is declared in the registry; implementation lands in {phase}",
        )


@dataclass(frozen=True)
class ToolSpec:
    """A registry entry: one model or algorithm the controller may select."""

    name: str
    version: str
    summary: str
    backend: str  # "classical" | "neural"
    tasks: tuple[Task, ...]
    configurations: tuple[InputConfiguration, ...]
    params: tuple[ToolParam, ...] = ()
    required_modalities: tuple[Modality, ...] = ()
    required_roles: tuple[str, ...] = ()
    requires_georeferencing: bool = False
    requires_gpu: bool = False
    vram_mb: int = 0
    priority: int = 50  # higher wins when several tools serve the same task
    implemented: bool = False
    run: Callable[[ToolContext], ToolResult] | None = None

    def validate_params(self, supplied: dict | None) -> tuple[dict, dict]:
        """Apply defaults, coerce permitted parameters, reject everything else."""
        schema = {p.name: p for p in self.params}
        accepted = {p.name: p.default for p in self.params}
        rejected: dict = {}
        for key, value in (supplied or {}).items():
            param = schema.get(key)
            if param is None:
                rejected[key] = f"'{key}' is not a permitted parameter of {self.name}"
                continue
            ok, coerced, message = param.coerce(value)
            if ok:
                accepted[key] = coerced
            else:
                rejected[key] = message
        return accepted, rejected

    def applicable_to(self, passports: list[ImagePassport]) -> tuple[bool, str]:
        """Check this tool against the actual images, not just the task."""
        if self.requires_georeferencing and not all(p.georeferenced for p in passports):
            return False, "requires georeferenced input"
        if self.required_modalities:
            present = {p.modality for p in passports}
            missing = [m.value for m in self.required_modalities if m not in present]
            if missing:
                return False, f"requires {', '.join(missing)} imagery"
        if self.required_roles and not any(p.has_roles(*self.required_roles) for p in passports):
            return False, f"requires the spectral bands {', '.join(self.required_roles)}"
        return True, ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "summary": self.summary,
            "backend": self.backend,
            "tasks": [t.value for t in self.tasks],
            "configurations": [c.value for c in self.configurations],
            "params": [p.to_dict() for p in self.params],
            "required_modalities": [m.value for m in self.required_modalities],
            "required_roles": list(self.required_roles),
            "requires_georeferencing": self.requires_georeferencing,
            "requires_gpu": self.requires_gpu,
            "vram_mb": self.vram_mb,
            "priority": self.priority,
            "implemented": self.implemented,
        }


@dataclass
class Candidate:
    """A tool the router considered, and why it was kept or dropped."""

    spec: ToolSpec
    eligible: bool
    reason: str

    def to_dict(self) -> dict:
        return {"tool": self.spec.name, "eligible": self.eligible, "reason": self.reason}


class ToolRegistry:
    """Holds the tool specs and answers selection queries."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return sorted(self._tools.values(), key=lambda s: (-s.priority, s.name))

    def tasks_supported(self) -> set[Task]:
        return {t for spec in self._tools.values() for t in spec.tasks}

    def select(
        self,
        *,
        task: Task,
        configuration: InputConfiguration,
        passports: list[ImagePassport],
        gpu_available: bool,
        prefer_backend: str | None = None,
    ) -> tuple[list[ToolSpec], list[Candidate]]:
        """Rank the tools that can serve this task on these images.

        Returns the eligible tools best-first, plus the full candidate list so
        the trace can show what was ruled out and why.
        """
        candidates: list[Candidate] = []
        eligible: list[ToolSpec] = []

        for spec in self.all():
            if task not in spec.tasks:
                candidates.append(Candidate(spec, False, f"does not serve the '{task.value}' task"))
                continue
            if configuration not in spec.configurations:
                candidates.append(
                    Candidate(spec, False, f"not defined for a {configuration.value} input")
                )
                continue
            ok, reason = spec.applicable_to(passports)
            if not ok:
                candidates.append(Candidate(spec, False, reason))
                continue
            if spec.requires_gpu and not gpu_available:
                candidates.append(
                    Candidate(spec, False, "needs a GPU and none is available in this session")
                )
                continue
            candidates.append(Candidate(spec, True, "eligible"))
            eligible.append(spec)

        def rank(spec: ToolSpec) -> tuple:
            backend_bonus = 1 if (prefer_backend and spec.backend == prefer_backend) else 0
            return (-int(spec.implemented), -backend_bonus, -spec.priority, spec.name)

        eligible.sort(key=rank)
        return eligible, candidates


# The process-wide registry. Importing satquery.tools populates it.
REGISTRY = ToolRegistry()
