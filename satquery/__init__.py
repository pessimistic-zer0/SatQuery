"""SatQuery AI - an agentic vision-language assistant for remote-sensing imagery.

Phase A provides the controller spine: the sensor passport, input compatibility
checking, the predefined tool registry, query routing and the execution trace.
The analysis tools themselves are registered but land in later phases.
"""

__version__ = "0.1.0"

from .compat import CompatibilityReport, validate_inputs  # noqa: F401
from .controller import Controller, RunResult  # noqa: F401
from .enums import InputConfiguration, Modality, Task  # noqa: F401
from .passport import ImagePassport, build_passport  # noqa: F401
from .registry import REGISTRY, ToolRegistry, ToolSpec  # noqa: F401
from .trace import ExecutionTrace  # noqa: F401

__all__ = [
    "Controller",
    "RunResult",
    "ImagePassport",
    "build_passport",
    "validate_inputs",
    "CompatibilityReport",
    "ExecutionTrace",
    "REGISTRY",
    "ToolRegistry",
    "ToolSpec",
    "Task",
    "Modality",
    "InputConfiguration",
    "__version__",
]
