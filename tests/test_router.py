"""Routing must respect what the uploaded images can actually support."""

from __future__ import annotations

import pytest

from satquery.enums import InputConfiguration, Task
from satquery.router import KeywordInterpreter, extract_entities

SINGLE = InputConfiguration.SINGLE
BITEMP = InputConfiguration.BI_TEMPORAL_PAIR
CROSS = InputConfiguration.CROSS_MODAL_PAIR


@pytest.fixture()
def interpreter() -> KeywordInterpreter:
    return KeywordInterpreter()


@pytest.mark.parametrize(("query", "configuration", "expected"), [
    # The five representative queries from the problem statement.
    ("Describe the land-cover and major objects visible in this image.", SINGLE, Task.CAPTION),
    ("Highlight the water body referred to in the query.", SINGLE, Task.GROUNDING),
    ("What changed between these two dates, and where did the change occur?",
     BITEMP, Task.CHANGE_MAP),
    ("Use the optical and SAR images together to identify built-up and water-covered regions.",
     CROSS, Task.CROSS_MODAL_ANALYSIS),
    ("Has the built-up area increased, decreased, or remained unchanged?", BITEMP, Task.CHANGE_VQA),
    # Plain single-image questions.
    ("How many water bodies are visible?", SINGLE, Task.VQA),
    ("Is there a river in this image?", SINGLE, Task.VQA),
])
def test_representative_queries_route_correctly(interpreter, query, configuration, expected):
    assert interpreter.interpret(query, configuration).task is expected


def test_change_query_on_a_single_image_falls_back(interpreter):
    """A change question over one image must not select a change task."""
    result = interpreter.interpret("What changed between these two dates?", SINGLE)
    assert result.task in {Task.VQA, Task.CAPTION}
    assert "change" in result.note
    assert all(s.task.value.startswith("change") is False or not s.eligible
               for s in result.scores)


def test_grounding_is_unavailable_for_a_pair(interpreter):
    result = interpreter.interpret("Highlight the water body.", BITEMP)
    assert result.task is not Task.GROUNDING
    grounding = next(s for s in result.scores if s.task is Task.GROUNDING)
    assert not grounding.eligible


def test_empty_query_uses_the_configuration_default(interpreter):
    assert interpreter.interpret("", SINGLE).task is Task.VQA
    assert interpreter.interpret("", BITEMP).task is Task.CHANGE_DESCRIPTION
    assert interpreter.interpret("", CROSS).task is Task.CROSS_MODAL_ANALYSIS


def test_confidence_is_higher_for_an_unambiguous_query(interpreter):
    specific = interpreter.interpret("Highlight the water body in this image.", SINGLE)
    vague = interpreter.interpret("hmm", SINGLE)
    assert specific.confidence > vague.confidence


def test_entity_extraction_finds_classes_and_target():
    entities = extract_entities("Highlight the water body referred to in the query.")
    assert "water" in entities["classes"]
    assert "water body" in entities["target_phrase"].lower()

    entities = extract_entities("Has the built-up area increased since 2019?")
    assert "built_up" in entities["classes"]


def test_scores_record_why_a_task_was_dropped(interpreter):
    result = interpreter.interpret("Where did the change occur?", SINGLE)
    dropped = next(s for s in result.scores if s.task is Task.CHANGE_MAP)
    assert not dropped.eligible
    assert "single" in dropped.reason
