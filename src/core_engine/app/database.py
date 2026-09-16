"""Database Mode store — per-topic sessions of brief report + user documents.

Database Mode is a two-step report flow:

  1. The agent generates a BRIEF report from its own web search. We snapshot that
     brief's verified claims and the sources it used into a session.
  2. The user uploads their own documents (articles, spreadsheets, PDFs). Each is
     parsed to text and stored as a USER_PROVIDED source in the same session.
  3. The COMPREHENSIVE report re-runs the pipeline over (brief sources + user
     documents), with no new web search — so the final report synthesizes the user's
     material together with the brief.

Storage: one JSON file per session under user_data_dir()/database/<id>/session.json.
This mirrors runtime.py's approach (plain JSON in the per-user app-data dir) — no
external database engine, single-user desktop app. Source/Claim objects are stored as
plain dicts and rehydrated into the dataclasses the pipeline expects.

Nothing here imports the pipeline at module load (only inside functions) so it stays
import-cheap and safe to touch at startup.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_engine.app import runtime


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_root() -> Path:
    d = runtime.user_data_dir() / "database"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_dir(session_id: str) -> Path:
    d = _db_root() / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session.json"


# --------------------------------------------------------------------------
# Serialization helpers: Source <-> dict.
# --------------------------------------------------------------------------
def _source_to_dict(src: Any) -> dict:
    return {
        "url": src.url,
        "domain": src.domain,
        "title": src.title,
        "kind": src.kind.value,
        "text": src.text,
        "fetched_at": src.fetched_at,
    }


def _dict_to_source(d: dict) -> Any:
    from core_engine.report.models import Source, SourceKind

    try:
        kind = SourceKind(d.get("kind", "general_web"))
    except ValueError:
        kind = SourceKind.GENERAL_WEB
    return Source(
        url=d["url"],
        domain=d.get("domain", ""),
        title=d.get("title", ""),
        kind=kind,
        text=d.get("text", ""),
        fetched_at=d.get("fetched_at", _now_iso()),
    )


# --------------------------------------------------------------------------
# Session CRUD.
# --------------------------------------------------------------------------
def create_session(topic: str) -> str:
    """Create a new database session for a topic. Returns the session id."""
    session_id = uuid.uuid4().hex
    data = {
        "id": session_id,
        "topic": topic,
        "created": _now_iso(),
        "brief_job_id": None,
        "brief_status": "pending",     # pending | completed | out_of_scope | error
        "brief_sources": [],           # list[dict] snapshot of the brief's sources
        "articles": [],                # list[dict] user-uploaded documents
        "comprehensive_job_id": None,
        "history_id": None,            # linked persistent-history entry (survives restart)
    }
    _write(session_id, data)
    return session_id


def _write(session_id: str, data: dict) -> None:
    _session_path(session_id).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_session(session_id: str) -> dict | None:
    p = _session_path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _require(session_id: str) -> dict:
    data = get_session(session_id)
    if data is None:
        raise KeyError(f"Unknown database session: {session_id}")
    return data


def save_brief(
    session_id: str, *, job_id: str, status: str, sources: list[Any]
) -> None:
    """Snapshot the brief report's outcome + the sources it used into the session, so
    the comprehensive step can reuse them without re-searching."""
    data = _require(session_id)
    data["brief_job_id"] = job_id
    data["brief_status"] = status
    data["brief_sources"] = [_source_to_dict(s) for s in sources]
    _write(session_id, data)


def set_comprehensive_job(session_id: str, job_id: str) -> None:
    data = _require(session_id)
    data["comprehensive_job_id"] = job_id
    _write(session_id, data)


def set_history_id(session_id: str, history_id: str) -> None:
    """Link this session to its persistent-history entry so the comprehensive run can
    update the same sidebar item as the brief."""
    data = _require(session_id)
    data["history_id"] = history_id
    _write(session_id, data)


# --------------------------------------------------------------------------
# User articles.
# --------------------------------------------------------------------------
def add_article(session_id: str, *, title: str, filename: str, text: str) -> dict:
    """Store a parsed user document as a USER_PROVIDED article. Returns the stored
    article record (without the full text, for a light API response)."""
    data = _require(session_id)
    article_id = uuid.uuid4().hex[:12]
    record = {
        "id": article_id,
        "title": title,
        "filename": filename,
        "text": text,
        "chars": len(text),
        "added": _now_iso(),
    }
    data["articles"].append(record)
    _write(session_id, data)
    return _article_summary(record)


def list_articles(session_id: str) -> list[dict]:
    """Return light summaries (no body text) of the session's user articles."""
    data = _require(session_id)
    return [_article_summary(a) for a in data.get("articles", [])]


def remove_article(session_id: str, article_id: str) -> bool:
    data = _require(session_id)
    before = len(data.get("articles", []))
    data["articles"] = [a for a in data.get("articles", []) if a.get("id") != article_id]
    changed = len(data["articles"]) != before
    if changed:
        _write(session_id, data)
    return changed


def _article_summary(record: dict) -> dict:
    return {
        "id": record["id"],
        "title": record["title"],
        "filename": record["filename"],
        "chars": record.get("chars", len(record.get("text", ""))),
        "added": record.get("added", ""),
    }


# --------------------------------------------------------------------------
# Building the comprehensive source pool.
# --------------------------------------------------------------------------
def sources_for_comprehensive(session_id: str) -> list[Any]:
    """Return the Source objects for the comprehensive run: the brief's sources plus
    one USER_PROVIDED Source per uploaded article. Deduped by URL upstream in the
    pipeline; here we just build the list."""
    from core_engine.report.models import Source, SourceKind

    data = _require(session_id)
    sources: list[Any] = [_dict_to_source(d) for d in data.get("brief_sources", [])]
    for art in data.get("articles", []):
        # Synthetic, stable URL so the source is addressable and dedupes cleanly. The
        # host segment becomes the 'domain' the scope gate counts, so each article is a
        # distinct authoritative domain.
        art_id = art["id"]
        url = f"userdoc://{session_id}/{art_id}"
        sources.append(Source(
            url=url,
            domain=f"user-document-{art_id}",
            title=art.get("title") or art.get("filename") or "User document",
            kind=SourceKind.USER_PROVIDED,
            text=art.get("text", ""),
            fetched_at=art.get("added", _now_iso()),
        ))
    return sources


def article_count(session_id: str) -> int:
    data = get_session(session_id)
    return len(data.get("articles", [])) if data else 0
