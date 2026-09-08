"""Prepared scenes for the demo.

A live demonstration should never begin with a file picker. Each scene here
loads a known set of imagery in one click and carries the questions that scene
can actually answer, so the operator moves straight to the part worth watching.

The four scenes cover, between them, every demonstration the problem statement
requires: single-image question answering, a second single-image task,
multitemporal change, an optical-SAR pair, and the routing that ties them
together - plus one scene that must be refused.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .enums import InputConfiguration


@dataclass(frozen=True)
class DemoScene:
    key: str
    title: str
    blurb: str
    fixtures: tuple[str, ...]
    queries: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "key": self.key, "title": self.title, "blurb": self.blurb,
            "queries": list(self.queries), "images": len(self.fixtures),
        }


SCENES: tuple[DemoScene, ...] = (
    DemoScene(
        key="single_optical",
        title="One optical scene",
        blurb="Sentinel-2, 12 bands, 10 m. A lake, a river, farmland and a town.",
        fixtures=("single_optical",),
        queries=(
            "How many water bodies are visible in this image?",
            "Describe the land cover and major objects visible in this image.",
            "Highlight the water body referred to in the query.",
        ),
    ),
    DemoScene(
        key="bitemporal",
        title="Two dates, same ground",
        blurb="Sentinel-2 from January 2022 and January 2024, 736 days apart.",
        fixtures=("bitemporal_t1", "bitemporal_t2"),
        queries=(
            "What changed between these two dates, and where did the change occur?",
            "Has the built-up area increased, decreased, or remained unchanged?",
        ),
    ),
    DemoScene(
        key="cross_modal",
        title="Optical and radar together",
        blurb="Sentinel-2 with cloud over the town, and the Sentinel-1 pass that sees through it.",
        fixtures=("cross_optical", "cross_sar"),
        queries=(
            "Use the optical and SAR images together to identify built-up and "
            "water-covered regions.",
            "Which parts of this scene can the optical sensor not see?",
        ),
    ),
    DemoScene(
        key="mismatched",
        title="A pair that does not match",
        blurb="Two scenes 200 km apart. The controller has to refuse this before it runs a model.",
        fixtures=("bitemporal_t1", "incompatible_far"),
        queries=("What changed between these two dates?",),
    ),
)

SCENES_BY_KEY = {scene.key: scene for scene in SCENES}


# Questions offered for imagery the user supplies themselves, chosen by what the
# uploaded configuration can actually support.
SUGGESTIONS: dict[InputConfiguration, tuple[str, ...]] = {
    InputConfiguration.SINGLE: (
        "Describe the land cover and major objects visible in this image.",
        "How many water bodies are visible in this image?",
        "Highlight the water body referred to in the query.",
    ),
    InputConfiguration.BI_TEMPORAL_PAIR: (
        "What changed between these two dates, and where did the change occur?",
        "Has the built-up area increased, decreased, or remained unchanged?",
    ),
    InputConfiguration.CROSS_MODAL_PAIR: (
        "Use the optical and SAR images together to identify built-up and "
        "water-covered regions.",
        "Which parts of this scene can the optical sensor not see?",
    ),
}


def load_manifest(fixtures_dir: str) -> dict | None:
    path = os.path.join(fixtures_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def scene_paths(scene: DemoScene, manifest: dict) -> list[str]:
    """Resolve a scene's fixture keys to file paths, skipping anything missing."""
    files = manifest.get("files", {})
    return [files[key]["path"] for key in scene.fixtures
            if key in files and os.path.isfile(files[key]["path"])]
