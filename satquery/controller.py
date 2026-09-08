"""The agentic controller.

This is the orchestration layer from the technical-approach slide, in order:

    validate -> check compatibility -> interpret the query -> route to a task ->
    select tools from the registry -> bind permitted parameters -> execute ->
    integrate the outputs

Every stage writes to the execution trace as it runs. An incompatible pair is
rejected at stage two, before any model is selected, which is what the deck
promises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .compat import CompatibilityReport, validate_inputs
from .enums import CheckStatus, InputConfiguration, StepKind, Task, ToolStatus
from .passport import ImagePassport, PassportError, build_passport
from .registry import REGISTRY, Candidate, ToolContext, ToolRegistry, ToolResult, ToolSpec
from .router import DEFAULT_INTERPRETER, QueryInterpretation, QueryInterpreter
from .runtime import detect_gpu
from .trace import ExecutionTrace
from . import tools as _tools  # noqa: F401 - importing populates the registry

VERIFIER_TOOL = "physics_verifier"


@dataclass
class RunResult:
    """Everything one query run produced."""

    trace: ExecutionTrace
    passports: list[ImagePassport] = field(default_factory=list)
    compatibility: CompatibilityReport | None = None
    interpretation: QueryInterpretation | None = None
    executed: list[tuple[str, ToolResult]] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.trace.run_id

    @property
    def answer(self) -> str | None:
        return self.trace.answer

    def to_dict(self) -> dict:
        return {
            "run_id": self.trace.run_id,
            "answer": self.trace.answer,
            "confidence": self.trace.confidence,
            "rejected": self.trace.rejected,
            "passports": [p.to_dict() for p in self.passports],
            "compatibility": self.compatibility.to_dict() if self.compatibility else None,
            "interpretation": self.interpretation.to_dict() if self.interpretation else None,
            "trace": self.trace.to_dict(),
        }


class Controller:
    """Routes one natural-language query over one or two images."""

    def __init__(
        self,
        registry: ToolRegistry = REGISTRY,
        interpreter: QueryInterpreter = DEFAULT_INTERPRETER,
        workdir: str = "data/runs",
        gpu_available: bool | None = None,
    ) -> None:
        self.registry = registry
        self.interpreter = interpreter
        self.workdir = workdir
        self.gpu_available = detect_gpu().available if gpu_available is None else gpu_available

    # --- stages ------------------------------------------------------------
    def _stage_validate(self, trace: ExecutionTrace, paths: list[str]) -> list[ImagePassport]:
        passports: list[ImagePassport] = []
        for path in paths:
            with trace.timed(StepKind.VALIDATE, "sensor_passport",
                             inputs=[os.path.basename(path)]) as timer:
                passport = build_passport(path)
                passports.append(passport)
                timer.update(
                    status="ok",
                    detail=passport.summary_line(),
                    outputs=passport.to_dict(),
                    confidence=passport.modality_confidence,
                )
            for warning in passport.warnings:
                trace.warn(f"{passport.filename}: {warning}")
        return passports

    def _stage_compatibility(
        self, trace: ExecutionTrace, passports: list[ImagePassport]
    ) -> CompatibilityReport:
        with trace.timed(StepKind.COMPATIBILITY, "input_compatibility",
                         inputs=[p.filename for p in passports]) as timer:
            report = validate_inputs(passports)
            timer.update(
                status="ok" if report.compatible else "rejected",
                detail=(f"configuration: {report.configuration.value}" if report.compatible
                        else "; ".join(report.reasons) or "inputs are not compatible"),
                outputs=report.to_dict(),
            )
        for check in report.checks:
            if check.status is CheckStatus.WARN:
                trace.warn(f"{check.name}: {check.detail}")
        return report

    def _stage_interpret(
        self, trace: ExecutionTrace, query: str, configuration: InputConfiguration,
        force_task: Task | None,
    ) -> QueryInterpretation:
        with trace.timed(StepKind.INTERPRET, "query_interpretation") as timer:
            interpretation = self.interpreter.interpret(query, configuration)
            if force_task is not None and force_task is not interpretation.task:
                interpretation.note = (
                    f"task overridden by the caller: '{interpretation.task.value}' -> "
                    f"'{force_task.value}'"
                )
                interpretation.task = force_task
                interpretation.confidence = 1.0
            timer.update(
                status="ok",
                detail=f"task '{interpretation.task.value}' "
                       f"(confidence {interpretation.confidence:.2f})"
                       + (f"; {interpretation.note}" if interpretation.note else ""),
                outputs=interpretation.to_dict(),
                confidence=interpretation.confidence,
            )
        return interpretation

    def _plan_tools(
        self, task: Task, configuration: InputConfiguration, passports: list[ImagePassport],
    ) -> tuple[list[ToolSpec], list[Candidate], list[str], list[str]]:
        """Choose the tools to run, and explain the choice.

        Returns the plan, the full candidate list, the notes that explain the
        routing, and the subset of those notes that are genuine warnings. A note
        saying the verifier runs last is information; a note saying the right
        tool is missing is a warning, and mixing the two trains people to ignore
        the warnings list.
        """
        eligible, candidates = self.registry.select(
            task=task,
            configuration=configuration,
            passports=passports,
            gpu_available=self.gpu_available,
        )
        plan: list[ToolSpec] = []
        notes: list[str] = []
        warnings: list[str] = []

        if not eligible:
            # Nothing can run, but the trace must still show the intended
            # routing: which tool would have served this task, and what stopped
            # it. A reviewer needs to see that the controller routed correctly
            # even when the machine could not execute the result.
            blocked = [c for c in candidates
                       if task in c.spec.tasks and configuration in c.spec.configurations]
            if blocked:
                notes.append(
                    f"'{task.value}' would route to "
                    + ", ".join(f"{c.spec.name} ({c.reason})" for c in blocked)
                    + ", but none can run here"
                )
            else:
                notes.append(f"no registered tool serves '{task.value}' for these inputs")
            warnings.extend(notes)
            return plan, candidates, notes, warnings

        primary = eligible[0]
        plan.append(primary)
        if not primary.implemented:
            fallback = next((s for s in eligible[1:] if s.implemented), None)
            if fallback is not None:
                plan.append(fallback)
                message = (f"'{primary.name}' is not implemented yet; '{fallback.name}' runs as "
                           "the fallback so the query still returns a grounded answer")
                notes.append(message)
                warnings.append(message)
            else:
                message = f"'{primary.name}' is the correct tool but is not implemented yet"
                notes.append(message)
                warnings.append(message)

        verifier = self.registry.get(VERIFIER_TOOL)
        if (
            verifier is not None
            and verifier.implemented
            and verifier not in plan
            and task in verifier.tasks
            and configuration in verifier.configurations
        ):
            plan.append(verifier)
            notes.append("physics_verifier runs last to cross-check the answer against indices")

        return plan, candidates, notes, warnings

    def _stage_route(
        self, trace: ExecutionTrace, task: Task, configuration: InputConfiguration,
        passports: list[ImagePassport],
    ) -> list[ToolSpec]:
        with trace.timed(StepKind.ROUTE, "tool_selection") as timer:
            plan, candidates, notes, warnings = self._plan_tools(task, configuration, passports)
            timer.update(
                status="ok" if plan else "no_tool",
                detail="; ".join(notes) if notes
                else f"selected {', '.join(s.name for s in plan)}",
                outputs={
                    "selected": [s.name for s in plan],
                    "intended": [c.spec.name for c in candidates
                                 if task in c.spec.tasks
                                 and configuration in c.spec.configurations],
                    "candidates": [c.to_dict() for c in candidates],
                    "gpu_available": self.gpu_available,
                },
            )
        for warning in warnings:
            trace.warn(warning)
        trace.tools = [s.name for s in plan]
        return plan

    def _stage_execute(
        self, trace: ExecutionTrace, plan: list[ToolSpec], ctx_base: dict,
        params_by_tool: dict[str, dict] | None,
    ) -> list[tuple[str, ToolResult]]:
        executed: list[tuple[str, ToolResult]] = []
        for spec in plan:
            accepted, rejected = spec.validate_params((params_by_tool or {}).get(spec.name))
            ctx = ToolContext(params=accepted, prior=list(executed), **ctx_base)
            with trace.timed(
                StepKind.TOOL, spec.name,
                params=accepted,
                rejected_params=rejected,
                inputs=[p.filename for p in ctx.passports],
            ) as timer:
                result = spec.run(ctx) if spec.run else ToolResult.pending(spec.name, "a later phase")
                executed.append((spec.name, result))
                timer.update(
                    status=result.status.value,
                    detail=result.detail or (result.answer or "")[:200],
                    outputs={"metrics": result.metrics, "answer": result.answer},
                    artifacts=result.artifacts,
                    confidence=result.confidence,
                )
            for key, message in rejected.items():
                trace.warn(f"{spec.name}: parameter '{key}' rejected - {message}")
        return executed

    def _stage_integrate(
        self, trace: ExecutionTrace, executed: list[tuple[str, ToolResult]],
        interpretation: QueryInterpretation,
    ) -> None:
        answering = [(n, r) for n, r in executed
                     if r.status is ToolStatus.OK and r.answer]
        pending = [n for n, r in executed if r.status is ToolStatus.NOT_IMPLEMENTED]

        with trace.timed(StepKind.INTEGRATE, "output_integration") as timer:
            if answering:
                # One paragraph per tool. A verification note reads as a
                # separate remark about the answer, not as more of the answer.
                answer = "\n\n".join(r.answer for _, r in answering)
                tool_conf = min(r.confidence or 0.5 for _, r in answering)
                # Overall confidence discounts the tool's own confidence by how
                # sure the router was that this was the right task at all.
                confidence = round(tool_conf * (0.6 + 0.4 * interpretation.confidence), 3)
                detail = f"answer composed from {', '.join(n for n, _ in answering)}"
            else:
                answer = None
                confidence = None
                detail = (
                    "no tool produced an answer; "
                    + (f"pending implementation: {', '.join(pending)}" if pending
                       else "no eligible tool ran")
                )
            if pending:
                trace.warn(
                    f"selected but not yet implemented: {', '.join(pending)} - "
                    "the trace records the intended routing"
                )
            timer.update(
                status="ok" if answering else "no_answer",
                detail=detail,
                outputs={
                    "answering_tools": [n for n, _ in answering],
                    "pending_tools": pending,
                },
                confidence=confidence,
            )
        trace.finish(answer, confidence)

    # --- entry point -------------------------------------------------------
    def run(
        self,
        query: str,
        image_paths: list[str],
        *,
        params_by_tool: dict[str, dict] | None = None,
        force_task: Task | None = None,
    ) -> RunResult:
        """Execute the full pipeline for one query."""
        trace = ExecutionTrace(
            query=query,
            input_files=[os.path.basename(p) for p in image_paths],
        )
        result = RunResult(trace=trace)

        # 1. validate every image
        try:
            passports = self._stage_validate(trace, image_paths)
        except PassportError as exc:
            trace.step(StepKind.VALIDATE, "sensor_passport", status="failed", detail=str(exc))
            trace.reject([str(exc)])
            trace.finish(None, None)
            return result
        result.passports = passports

        # 2. compatibility - rejection happens here, before any model
        report = self._stage_compatibility(trace, passports)
        result.compatibility = report
        trace.configuration = report.configuration
        if not report.compatible:
            trace.reject(report.reasons or ["inputs are not compatible"])
            trace.finish(None, None)
            return result

        # 3. interpret the query
        interpretation = self._stage_interpret(trace, query, report.configuration, force_task)
        result.interpretation = interpretation
        trace.task = interpretation.task

        # 4. route to tools
        plan = self._stage_route(trace, interpretation.task, report.configuration, passports)

        # 5. execute with permitted parameters only
        run_dir = os.path.join(self.workdir, trace.run_id)
        os.makedirs(run_dir, exist_ok=True)
        ctx_base = {
            "passports": passports,
            "query": query,
            "task": interpretation.task,
            "configuration": report.configuration,
            "workdir": run_dir,
            "gpu_available": self.gpu_available,
            "entities": interpretation.entities,
        }
        result.executed = self._stage_execute(trace, plan, ctx_base, params_by_tool)

        # 6. integrate
        self._stage_integrate(trace, result.executed, interpretation)
        return result
