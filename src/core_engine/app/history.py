"""Persistent history store — reports & chats that survive app restarts.

The frontend used to keep history only in the browser's localStorage, which is lost
if the webview storage is cleared and is invisible to the backend. This module gives
history a durable home on disk in the per-user app-data dir (next to settings.json and
the generated_pdfs/ store), so past reports and conversations are retained across
sessions and the server can read them on startup.

Storage: a single JSON file, user_data_dir()/history.json, holding a list of entries
(newest first). Each entry:

    {
      "id":          str,     # stable history id
      "kind":        str,     # "report" | "chat"
      "title":       str,     # topic or first user message
      "ts":          int,     # epoch millis (created)
      "status":      str,     # running | completed | out_of_scope | blocked | error
      "job_id":      str|None, # the report job whose PDF is on disk (for Download)
      "session_id":  str|None, # database session (brief + user docs), if any
      "pdf_ready":   bool,    # a downloadable PDF exists for job_id
    }

PDFs themselves are stored by runtime.store_pdf() keyed by job_id, so a completed
history entry can always be re-downloaded across restarts via /api/report/{id}/pdf.

Import-cheap: no pipeline imports, safe to touch at startup.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_engine.app import runtime

_MAX_ENTRIES = 500  # bound the file so it can't grow without limit


def _history_path() -> Path:
    return runtime.user_data_dir() / "history.json"


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _load() -> list[dict]:
    p = _history_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        # A corrupt file must not crash startup — treat as empty.
        return []


def _save(items: list[dict]) -> None:
    p = _history_path()
    try:
        p.write_text(
            json.dumps(items[:_MAX_ENTRIES], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        # Best-effort: never let a persistence failure break a request.
        pass


def list_entries() -> list[dict]:
    """Return all history entries, newest first."""
    return _load()


def add_entry(
    *,
    kind: str,
    title: str,
    status: str = "running",
    job_id: str | None = None,
    session_id: str | None = None,
    pdf_ready: bool = False,
) -> str:
    """Append a new history entry (newest first). Returns its id."""
    items = _load()
    entry_id = f"h{_now_ms()}_{uuid.uuid4().hex[:5]}"
    items.insert(0, {
        "id": entry_id,
        "kind": kind,
        "title": title,
        "ts": _now_ms(),
        "status": status,
        "job_id": job_id,
        "session_id": session_id,
        "pdf_ready": pdf_ready,
    })
    _save(items)
    return entry_id


def update_entry(entry_id: str, **patch: Any) -> None:
    """Merge `patch` into the entry with this id, if present."""
    if not entry_id:
        return
    items = _load()
    for i, it in enumerate(items):
        if it.get("id") == entry_id:
            items[i] = {**it, **patch}
            _save(items)
            return


def clear() -> None:
    """Remove all history entries."""
    _save([])
