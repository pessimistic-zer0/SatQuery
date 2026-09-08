"""Query interpretation and task routing.

The controller has to turn a plain-English question into one of the seven task
types, and it has to do so without a language model available - the demo must
survive with no GPU. So the default interpreter is a scored keyword and pattern
matcher, and an optional language-model interpreter can be dropped in behind the
same interface later.

Routing is constrained before it is scored: a task that the uploaded images
cannot support is never a candidate, however strongly the words suggest it. That
is what stops "what changed here?" over a single image from selecting a change
model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .enums import DEFAULT_TASK, InputConfiguration, Task, TASK_CONFIGURATIONS

# --- Pattern tables ---------------------------------------------------------
# Each entry is (compiled pattern, weight, short label used as trace evidence).
_PATTERNS: dict[Task, list[tuple[str, float, str]]] = {
    Task.GROUNDING: [
        (r"\b(highlight|outline|delineate|mark|circle)\b", 3.0, "grounding verb"),
        (r"\b(show|point out|locate|find)\s+(me\s+)?(the|a|all)\b", 2.5, "locate phrasing"),
        (r"\b(bounding box|bbox|segment|mask|region of)\b", 3.0, "explicit region request"),
        (r"\bwhere\s+(is|are)\s+the\b", 2.0, "where-is question"),
        (r"\breferred to in the query\b", 3.0, "referring expression"),
    ],
    Task.CAPTION: [
        (r"\b(describe|caption|summari[sz]e|overview|tell me about)\b", 3.0, "description verb"),
        (r"\bwhat (does|do) (this|these) image", 2.0, "open-ended scene question"),
        (r"\b(land[- ]?cover|scene|contents?)\b", 1.0, "scene vocabulary"),
    ],
    Task.VQA: [
        (r"\bhow many\b", 3.0, "counting question"),
        (r"\b(is|are|does|do|can|has|have)\s+(there|the|this|it)\b", 2.0, "yes/no question"),
        (r"\bwhat (is|are|kind|type)\b", 2.0, "factual question"),
        (r"\bwhich\b", 1.5, "selection question"),
        (r"\?", 0.5, "question mark"),
    ],
    Task.CHANGE_DESCRIPTION: [
        (r"\bwhat (has\s+)?chang(ed|es)\b", 3.5, "what-changed question"),
        (r"\bdescribe .*\bchang", 3.5, "describe-change request"),
        (r"\b(before and after|between (these|the two) (dates|images|times))\b", 2.5,
         "before/after phrasing"),
        (r"\b(since|compared to|relative to)\b", 1.5, "comparison phrasing"),
    ],
    Task.CHANGE_VQA: [
        (r"\b(increas|decreas|grown|grew|shrunk|shrank|expand|reduc|remain)\w*\b", 3.0,
         "change-direction verb"),
        (r"\bhas the\b.*\b(chang|increas|decreas|grow|shrink)\w*", 3.0, "has-it-changed question"),
        (r"\bhow much\b.*\b(chang|increas|decreas)\w*", 3.0, "change-magnitude question"),
        (r"\b(more|less|larger|smaller)\s+than\b", 1.5, "comparative"),
    ],
    Task.CHANGE_MAP: [
        (r"\bwhere\b.*\b(chang|occur|happen)\w*", 3.5, "where-did-change-occur question"),
        (r"\bchange (map|mask|areas?)\b", 3.5, "explicit change map request"),
        (r"\b(map|highlight|show)\b.*\bchang\w*", 2.5, "map-the-change request"),
    ],
    Task.CROSS_MODAL_ANALYSIS: [
        (r"\b(optical|multispectral)\b.*\b(sar|radar)\b", 3.5, "both modalities named"),
        (r"\b(sar|radar)\b.*\b(optical|multispectral)\b", 3.5, "both modalities named"),
        (r"\b(both|together|combine|fuse|jointly|complementary)\b", 2.0, "fusion phrasing"),
        (r"\b(cloud|through cloud|day.?night)\b", 1.5, "cloud/day-night motivation"),
    ],
}

# Land-cover classes the query may be asking about. Phase B tools use these to
# pick which index to compute.
_CLASS_PATTERNS: dict[str, str] = {
    "water": r"\b(water|lake|river|pond|reservoir|flood|sea|coast|wetland)\w*\b",
    "built_up": r"\b(built[- ]?up|urban|building|settlement|city|town|construction|infrastructure)\w*\b",
    "vegetation": r"\b(vegetation|forest|tree|green|canopy|crop|agricultur|farm|field)\w*\b",
    "bare_soil": r"\b(bare|soil|barren|sand|desert|exposed ground)\w*\b",
    "road": r"\b(road|highway|street|runway|railway)\w*\b",
    "cloud": r"\b(cloud|haze|overcast)\w*\b",
}

# Phrase that a grounding request refers to, e.g. "highlight the water body".
_TARGET_PATTERN = re.compile(
    r"\b(?:highlight|outline|delineate|mark|circle|locate|find|show(?:\s+me)?|segment)\s+"
    r"(?:the\s+|a\s+|all\s+(?:the\s+)?)?([a-z][a-z\s\-]{2,40}?)"
    r"(?=\s*(?:\bin\b|\bon\b|\bthat\b|\bwhich\b|\breferred\b|[.,?!]|$))",
    re.IGNORECASE,
)


@dataclass
class TaskScore:
    task: Task
    score: float
    evidence: list[str] = field(default_factory=list)
    eligible: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task.value,
            "score": round(self.score, 2),
            "evidence": self.evidence,
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass
class QueryInterpretation:
    """What the controller decided the user asked for."""

    task: Task
    confidence: float
    method: str
    scores: list[TaskScore] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    note: str = ""

    @property
    def alternatives(self) -> list[Task]:
        return [s.task for s in self.scores if s.eligible and s.task is not self.task][:3]

    def to_dict(self) -> dict:
        return {
            "task": self.task.value,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "entities": self.entities,
            "note": self.note,
            "scores": [s.to_dict() for s in self.scores],
        }


class QueryInterpreter(Protocol):
    """Interface a language-model interpreter can implement later."""

    name: str

    def interpret(self, query: str, configuration: InputConfiguration) -> QueryInterpretation: ...


def extract_entities(query: str) -> dict:
    """Pull out the land-cover classes and the referring phrase, if any."""
    lowered = query.lower()
    classes = [name for name, pattern in _CLASS_PATTERNS.items() if re.search(pattern, lowered)]
    entities: dict = {"classes": classes}

    match = _TARGET_PATTERN.search(query)
    if match:
        target = re.sub(r"\s+", " ", match.group(1)).strip(" .,")
        # Drop a trailing filler word that the pattern may have swallowed.
        target = re.sub(r"\s+(referred|visible|shown)$", "", target, flags=re.IGNORECASE)
        if target and target.lower() not in {"me", "it", "them"}:
            entities["target_phrase"] = target
    return entities


class KeywordInterpreter:
    """Offline task classifier. Deterministic, and the demo's safe default."""

    name = "keyword_interpreter/1.0"

    @staticmethod
    def _resolve_compound_change_query(lowered: str, scores: list[TaskScore]) -> None:
        """Send "what changed, and where?" to the change map.

        The problem statement's own example query asks both at once. The
        "what changed" phrasing carries more matching words, so on raw score it
        beats the "where" phrasing and the spatial half of the question is
        silently dropped. The change map is the task that answers both - it
        carries the location, and the change tool emits a description alongside
        it - so an explicit spatial request wins the tie.
        """
        if not re.search(r"\bwhere\b", lowered):
            return
        by_task = {s.task: s for s in scores}
        change_map = by_task.get(Task.CHANGE_MAP)
        description = by_task.get(Task.CHANGE_DESCRIPTION)
        if change_map is None or description is None:
            return
        if change_map.score > 0 and description.score >= change_map.score:
            change_map.score = description.score + 0.5
            change_map.evidence.append("compound what-and-where change query")

    def interpret(self, query: str, configuration: InputConfiguration) -> QueryInterpretation:
        lowered = (query or "").lower().strip()
        legal = {
            task for task, configs in TASK_CONFIGURATIONS.items() if configuration in configs
        }

        scores: list[TaskScore] = []
        for task, patterns in _PATTERNS.items():
            total = 0.0
            evidence: list[str] = []
            for pattern, weight, label in patterns:
                if re.search(pattern, lowered):
                    total += weight
                    evidence.append(label)
            eligible = task in legal
            scores.append(TaskScore(
                task=task,
                score=total,
                evidence=evidence,
                eligible=eligible,
                reason="" if eligible
                else f"not available for a {configuration.value} input",
            ))

        self._resolve_compound_change_query(lowered, scores)
        scores.sort(key=lambda s: (-s.score, s.task.value))
        eligible_scores = [s for s in scores if s.eligible]

        fallback = DEFAULT_TASK.get(configuration, Task.VQA)
        if not eligible_scores or eligible_scores[0].score <= 0.0:
            return QueryInterpretation(
                task=fallback,
                confidence=0.35,
                method=self.name,
                scores=scores,
                entities=extract_entities(query),
                note=("the query matched no task pattern, so the default task for a "
                      f"{configuration.value} input was used"),
            )

        best = eligible_scores[0]
        runner_up = eligible_scores[1].score if len(eligible_scores) > 1 else 0.0
        # Confidence rises with the absolute score and with the margin over the
        # next candidate, so a clear, specific query scores high.
        margin = (best.score - runner_up) / max(best.score, 1e-6)
        confidence = min(0.97, 0.45 + 0.30 * min(1.0, best.score / 6.0) + 0.25 * margin)

        note = ""
        dropped = [s for s in scores if not s.eligible and s.score > best.score]
        if dropped:
            note = (f"'{dropped[0].task.value}' scored higher on wording but is not available "
                    f"for a {configuration.value} input")

        return QueryInterpretation(
            task=best.task,
            confidence=confidence,
            method=self.name,
            scores=scores,
            entities=extract_entities(query),
            note=note,
        )


DEFAULT_INTERPRETER = KeywordInterpreter()
