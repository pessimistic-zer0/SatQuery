"""Shared fixtures: one synthetic demo set, generated once per test session."""

from __future__ import annotations

import pytest

from satquery.fixtures.synth import make_demo_set


@pytest.fixture(scope="session")
def demo_set(tmp_path_factory) -> dict:
    """Generate the fixture set in a temporary directory."""
    outdir = tmp_path_factory.mktemp("fixtures")
    return make_demo_set(str(outdir), size=192, seed=3)


@pytest.fixture(scope="session")
def paths(demo_set) -> dict:
    return {key: entry["path"] for key, entry in demo_set["files"].items()}
