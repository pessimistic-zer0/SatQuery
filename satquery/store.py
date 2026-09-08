"""Session and run storage.

Deliberately small: the deck's production stack is PostGIS plus a Celery queue,
but for a single-machine demo that is machinery without a purpose. Uploads live
on disk under a session directory and every finished run is written next to its
trace as JSON, so the API survives a restart without a database.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field

from .controller import RunResult
from .passport import ImagePassport


@dataclass
class Session:
    """One upload: the files, their passports and the compatibility verdict."""

    session_id: str
    directory: str
    files: list[str] = field(default_factory=list)
    passports: list[ImagePassport] = field(default_factory=list)
    compatibility: dict | None = None
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "files": [os.path.basename(f) for f in self.files],
            "passports": [p.to_dict() for p in self.passports],
            "compatibility": self.compatibility,
            "created_at": self.created_at,
        }


class Store:
    """Filesystem-backed store for sessions and runs."""

    def __init__(self, root: str = "data") -> None:
        self.root = os.path.abspath(root)
        self.uploads = os.path.join(self.root, "uploads")
        self.runs = os.path.join(self.root, "runs")
        os.makedirs(self.uploads, exist_ok=True)
        os.makedirs(self.runs, exist_ok=True)
        self._sessions: dict[str, Session] = {}

    # --- sessions ----------------------------------------------------------
    def new_session(self) -> Session:
        import datetime as dt

        session_id = uuid.uuid4().hex[:12]
        directory = os.path.join(self.uploads, session_id)
        os.makedirs(directory, exist_ok=True)
        session = Session(
            session_id=session_id,
            directory=directory,
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def save_session(self, session: Session) -> None:
        self._sessions[session.session_id] = session
        with open(os.path.join(session.directory, "session.json"), "w", encoding="utf-8") as fh:
            json.dump(session.to_dict(), fh, indent=2)

    def delete_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        shutil.rmtree(session.directory, ignore_errors=True)
        return True

    # --- runs --------------------------------------------------------------
    def run_dir(self, run_id: str) -> str:
        path = os.path.join(self.runs, run_id)
        os.makedirs(path, exist_ok=True)
        return path

    def save_run(self, result: RunResult) -> str:
        """Persist the run as JSON, and as a PDF when reportlab is available."""
        from .report import PDF_AVAILABLE, build_report_dict, write_pdf_report

        directory = self.run_dir(result.run_id)
        with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(build_report_dict(result), fh, indent=2)
        if PDF_AVAILABLE:
            try:
                write_pdf_report(result, os.path.join(directory, "report.pdf"))
            except Exception as exc:  # noqa: BLE001 - a PDF failure must not fail the run
                with open(os.path.join(directory, "report.pdf.error"), "w",
                          encoding="utf-8") as fh:
                    fh.write(f"{type(exc).__name__}: {exc}\n")
        return directory

    def load_run(self, run_id: str) -> dict | None:
        path = os.path.join(self.runs, run_id, "report.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def list_runs(self, limit: int = 50) -> list[dict]:
        entries = []
        for run_id in os.listdir(self.runs):
            report = self.load_run(run_id)
            if report is None:
                continue
            trace = report.get("trace", {})
            entries.append({
                "run_id": run_id,
                "created_at": trace.get("created_at"),
                "query": trace.get("query"),
                "task": trace.get("task"),
                "tools": trace.get("tools", []),
                "rejected": trace.get("rejected", False),
                "confidence": trace.get("confidence"),
            })
        entries.sort(key=lambda e: e.get("created_at") or "", reverse=True)
        return entries[:limit]
