"""HTTP API for the SatQuery AI controller.

Endpoints follow the flow of the technical-approach slide: upload and validate
first, then ask a question against that validated session, then read the trace
or download the report.

Uploading is deliberately separated from querying so the compatibility verdict
is visible in the interface before any query is run - the user sees the pair
rejected, and why, rather than a failed query.
"""

from __future__ import annotations

import os
from typing import Annotated

import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .compat import validate_inputs
from .controller import Controller
from .demos import SCENES, SCENES_BY_KEY, SUGGESTIONS, load_manifest, scene_paths
from .enums import InputConfiguration, Task
from .passport import PassportError, RASTER_EXTENSIONS, build_passport
from .registry import REGISTRY
from .report import PDF_AVAILABLE
from .runtime import detect_gpu
from .store import Store

MAX_IMAGES = 2
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

store = Store(os.environ.get("SATQUERY_DATA", "data"))
controller = Controller(workdir=store.runs)
FIXTURES_DIR = os.environ.get("SATQUERY_FIXTURES", "data/fixtures")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(
    title="SatQuery AI",
    version="0.1.0",
    summary="Agentic vision-language assistant for multimodal remote-sensing imagery.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _session_payload(session) -> dict:
    """Session data plus the questions this configuration can answer."""
    payload = session.to_dict()
    configuration = (session.compatibility or {}).get("configuration")
    try:
        suggestions = SUGGESTIONS.get(InputConfiguration(configuration), ())
    except ValueError:
        suggestions = ()
    payload["suggestions"] = list(suggestions)
    return payload


class QueryRequest(BaseModel):
    session_id: str
    query: str = Field(min_length=1, max_length=2000)
    task: str | None = Field(default=None, description="Override the routed task.")
    params: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-tool parameter overrides, keyed by tool name. "
                    "Parameters outside a tool's schema are rejected and recorded.",
    )


@app.get("/api/health")
def health() -> dict:
    """Runtime capabilities, so the interface can show the no-GPU fallback."""
    gpu = detect_gpu()
    tools = REGISTRY.all()
    return {
        "status": "ok",
        "version": app.version,
        "gpu": gpu.to_dict(),
        "pdf_export": PDF_AVAILABLE,
        "tools_registered": len(tools),
        "tools_implemented": sum(1 for t in tools if t.implemented),
        "accepted_extensions": sorted(RASTER_EXTENSIONS),
    }


@app.get("/api/tools")
def list_tools() -> dict:
    """The predefined model and tool registry, with permitted parameters."""
    return {"tools": [spec.to_dict() for spec in REGISTRY.all()]}


@app.post("/api/upload")
async def upload(files: Annotated[list[UploadFile], File()]) -> dict:
    """Accept one or two images, build their passports and check compatibility."""
    if not 1 <= len(files) <= MAX_IMAGES:
        raise HTTPException(
            400,
            f"supply 1 or {MAX_IMAGES} images; the supported configurations are a single "
            f"image, a cross-modal pair, or a bi-temporal pair",
        )

    session = store.new_session()
    saved: list[str] = []
    for upload_file in files:
        name = os.path.basename(upload_file.filename or "upload")
        extension = os.path.splitext(name)[1].lower()
        if extension not in RASTER_EXTENSIONS:
            store.delete_session(session.session_id)
            raise HTTPException(
                415,
                f"'{name}' has extension '{extension}'; accepted formats are "
                f"{sorted(RASTER_EXTENSIONS)} (PNG and JPEG for benchmark datasets only)",
            )
        destination = os.path.join(session.directory, name)
        size = 0
        with open(destination, "wb") as handle:
            while chunk := await upload_file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    store.delete_session(session.session_id)
                    raise HTTPException(413, f"'{name}' exceeds the {MAX_UPLOAD_BYTES >> 20} MB limit")
                handle.write(chunk)
        saved.append(destination)

    try:
        passports = [build_passport(path) for path in saved]
    except PassportError as exc:
        store.delete_session(session.session_id)
        raise HTTPException(422, str(exc)) from exc

    report = validate_inputs(passports)
    session.files = saved
    session.passports = passports
    session.compatibility = report.to_dict()
    store.save_session(session)
    return _session_payload(session)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"no session '{session_id}'")
    return _session_payload(session)


@app.post("/api/query")
def run_query(request: QueryRequest) -> dict:
    """Route and execute one natural-language query over an uploaded session."""
    session = store.get_session(request.session_id)
    if session is None:
        raise HTTPException(404, f"no session '{request.session_id}'")

    forced: Task | None = None
    if request.task:
        try:
            forced = Task(request.task)
        except ValueError as exc:
            raise HTTPException(
                400, f"unknown task '{request.task}'; valid tasks are {[t.value for t in Task]}"
            ) from exc

    result = controller.run(
        request.query, session.files, params_by_tool=request.params, force_task=forced,
    )
    store.save_run(result)
    return result.to_dict()


@app.get("/api/runs")
def list_runs(limit: int = 50) -> dict:
    return {"runs": store.list_runs(limit)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    report = store.load_run(run_id)
    if report is None:
        raise HTTPException(404, f"no run '{run_id}'")
    return report


@app.get("/api/runs/{run_id}/report.json")
def download_json(run_id: str) -> FileResponse:
    path = os.path.join(store.runs, run_id, "report.json")
    if not os.path.isfile(path):
        raise HTTPException(404, f"no run '{run_id}'")
    return FileResponse(path, media_type="application/json",
                        filename=f"satquery-{run_id}.json")


@app.get("/api/runs/{run_id}/report.pdf")
def download_pdf(run_id: str) -> FileResponse:
    if not PDF_AVAILABLE:
        raise HTTPException(501, "reportlab is not installed; PDF export is unavailable")
    directory = os.path.join(store.runs, run_id)
    if not os.path.isdir(directory):
        raise HTTPException(404, f"no run '{run_id}'")
    path = os.path.join(directory, "report.pdf")
    if not os.path.isfile(path):
        raise HTTPException(500, "the PDF for this run could not be rendered")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"satquery-{run_id}.pdf")


@app.get("/api/demos")
def list_demos() -> dict:
    """Prepared scenes, and whether their imagery is on disk yet."""
    manifest = load_manifest(FIXTURES_DIR)
    scenes = []
    for scene in SCENES:
        entry = scene.to_dict()
        entry["available"] = bool(manifest) and len(scene_paths(scene, manifest)) == len(scene.fixtures)
        scenes.append(entry)
    return {
        "scenes": scenes,
        "fixtures_dir": FIXTURES_DIR,
        "hint": None if manifest else
        "no fixtures found; run 'python -m satquery.cli fixtures' to generate them",
    }


@app.post("/api/demos/{key}")
def load_demo(key: str) -> dict:
    """Copy a prepared scene into a new session and validate it."""
    scene = SCENES_BY_KEY.get(key)
    if scene is None:
        raise HTTPException(404, f"no scene '{key}'")
    manifest = load_manifest(FIXTURES_DIR)
    if manifest is None:
        raise HTTPException(
            409, "no fixtures on disk; run 'python -m satquery.cli fixtures' first")
    sources = scene_paths(scene, manifest)
    if len(sources) != len(scene.fixtures):
        raise HTTPException(409, f"scene '{key}' is missing some of its imagery")

    session = store.new_session()
    saved = []
    for source in sources:
        destination = os.path.join(session.directory, os.path.basename(source))
        shutil.copyfile(source, destination)
        saved.append(destination)

    passports = [build_passport(path) for path in saved]
    report = validate_inputs(passports)
    session.files = saved
    session.passports = passports
    session.compatibility = report.to_dict()
    store.save_session(session)

    payload = _session_payload(session)
    payload["scene"] = scene.to_dict()
    payload["suggestions"] = list(scene.queries)
    return payload


@app.get("/api/sessions/{session_id}/preview/{index}.png")
def session_preview(session_id: str, index: int) -> FileResponse:
    """A viewable RGB rendering of an uploaded image.

    Neither a 12-band GeoTIFF nor a dB-scaled radar image can be shown by a
    browser, so the interface asks for this instead.
    """
    session = store.get_session(session_id)
    if session is None or not 0 <= index < len(session.passports):
        raise HTTPException(404, "no such image in this session")

    cached = os.path.join(session.directory, f"preview_{index}.png")
    if not os.path.isfile(cached):
        from .imaging import rgb_preview, save_png
        from .raster import read_bands

        passport = session.passports[index]
        roles = tuple(r for r in ("blue", "green", "red", "vv")
                      if passport.band_index(r) is not None)
        bands, _grid, _note = read_bands(passport, roles or None, max_side=900)
        save_png(cached, rgb_preview(bands))
    return FileResponse(cached, media_type="image/png")


@app.get("/api/runs/{run_id}/artifacts/{name}")
def run_artifact(run_id: str, name: str) -> FileResponse:
    """Serve one piece of visual evidence produced by a run."""
    directory = os.path.abspath(os.path.join(store.runs, run_id))
    path = os.path.abspath(os.path.join(directory, name))
    if not path.startswith(directory + os.sep) or not os.path.isfile(path):
        raise HTTPException(404, f"no artifact '{name}' in run '{run_id}'")
    media = "image/png" if path.endswith(".png") else "image/tiff"
    return FileResponse(path, media_type=media)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """A minimal console. The designed interface arrives in Phase C."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "static", "index.html")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    return "<h1>SatQuery AI</h1><p>API is running. See <a href='/docs'>/docs</a>.</p>"
