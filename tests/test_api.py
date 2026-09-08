"""The HTTP surface the interface depends on.

A live demonstration fails in the browser, not in the library, so the endpoints
the console calls are worth their own tests: loading a scene, reading a preview
of imagery a browser cannot display, and fetching the evidence a run produced.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """An app instance pointed at its own data and fixture directories."""
    import importlib
    import os

    from satquery.fixtures.synth import make_demo_set

    root = tmp_path_factory.mktemp("api")
    fixtures = root / "fixtures"
    make_demo_set(str(fixtures), size=160, seed=5)
    os.environ["SATQUERY_DATA"] = str(root / "data")
    os.environ["SATQUERY_FIXTURES"] = str(fixtures)
    os.environ["SATQUERY_FORCE_CPU"] = "1"

    import satquery.api as api

    importlib.reload(api)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_runtime_capabilities(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["tools_registered"] >= 12
    assert body["tools_implemented"] >= 7
    assert body["gpu"]["available"] is False


def test_registry_is_published_with_its_parameters(client):
    tools = client.get("/api/tools").json()["tools"]
    spectral = next(t for t in tools if t["name"] == "spectral_index")
    assert spectral["implemented"]
    assert {p["name"] for p in spectral["params"]} == {"index", "threshold", "min_region_px"}


def test_every_scene_is_available(client):
    scenes = client.get("/api/demos").json()["scenes"]
    assert len(scenes) == 4
    assert all(s["available"] for s in scenes), "a scene is missing its imagery"
    assert all(s["queries"] for s in scenes)


def test_loading_a_scene_validates_it(client):
    body = client.post("/api/demos/cross_modal").json()
    assert body["compatibility"]["configuration"] == "cross_modal_pair"
    assert body["compatibility"]["compatible"]
    assert len(body["passports"]) == 2
    assert body["suggestions"]


def test_unknown_scene_is_a_404(client):
    assert client.post("/api/demos/not_a_scene").status_code == 404


def test_preview_renders_imagery_a_browser_cannot_open(client):
    """A 12-band GeoTIFF and a dB radar image both have to become viewable."""
    session = client.post("/api/demos/cross_modal").json()["session_id"]
    for index in (0, 1):
        response = client.get(f"/api/sessions/{session}/preview/{index}.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_preview_of_a_missing_image_is_a_404(client):
    session = client.post("/api/demos/single_optical").json()["session_id"]
    assert client.get(f"/api/sessions/{session}/preview/5.png").status_code == 404


def test_a_query_returns_evidence_that_can_be_fetched(client):
    session = client.post("/api/demos/bitemporal").json()["session_id"]
    body = client.post("/api/query", json={
        "session_id": session,
        "query": "What changed between these two dates, and where did the change occur?",
    }).json()
    trace = body["trace"]
    assert trace["task"] == "change_map"
    assert trace["answer"]

    overlays = [a for a in trace["artifacts"] if a["format"] == "png"]
    assert overlays, "a change query must produce visual evidence"
    name = overlays[0]["path"].split("/")[-1]
    image = client.get(f"/api/runs/{trace['run_id']}/artifacts/{name}")
    assert image.status_code == 200
    assert image.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_artifact_paths_cannot_escape_the_run_directory(client):
    session = client.post("/api/demos/single_optical").json()["session_id"]
    run = client.post("/api/query", json={"session_id": session,
                                          "query": "Describe this image."}).json()
    run_id = run["trace"]["run_id"]
    for attempt in ("../../../etc/passwd", "..%2f..%2fsession.json"):
        assert client.get(f"/api/runs/{run_id}/artifacts/{attempt}").status_code in (404, 400)


def test_rejected_pair_returns_a_trace_and_a_report(client):
    session = client.post("/api/demos/mismatched").json()["session_id"]
    body = client.post("/api/query", json={"session_id": session,
                                           "query": "What changed?"}).json()
    trace = body["trace"]
    assert trace["rejected"]
    assert trace["tools"] == []
    assert client.get(f"/api/runs/{trace['run_id']}/report.pdf").status_code == 200


def test_unsupported_upload_is_refused_with_a_reason(client):
    response = client.post("/api/upload",
                           files={"files": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert "accepted formats" in response.json()["detail"]


def test_the_console_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "SatQuery" in page.text
    assert client.get("/static/fonts/Barlow-400.woff2").status_code == 200
