"""Wiring test: a REPORT_GENERATION message starts the unified brief-first flow —
it creates a database session AND a persistent history entry, and returns both ids.

Offline (fake providers), no network. Uses FastAPI's TestClient for the request/
response wiring only (it does not drain the SSE job stream).

    pytest -q tests/test_report_session_wiring.py
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Fake providers so intent classification routes deterministically and no network
    # is touched. Route all app-data to a temp dir.
    monkeypatch.setenv("CE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    from core_engine.config import get_settings
    get_settings.cache_clear()

    from core_engine.app import runtime
    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)

    from fastapi.testclient import TestClient
    from core_engine.app.server import create_app

    return TestClient(create_app())


def test_report_message_creates_session_and_history(client):
    # A clear report request -> FakeLLM classifier returns report_generation.
    r = client.post("/api/message", json={"message": "Generate a report on solar cells"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "report"
    assert body["job_id"]
    assert body["session_id"], "unified flow must create a database session"
    assert body["history_id"], "report must be recorded in persistent history"

    # The session exists and is linked to the history entry.
    from core_engine.app import database
    sess = database.get_session(body["session_id"])
    assert sess is not None
    assert sess["history_id"] == body["history_id"]


def test_report_appears_in_history_endpoint(client):
    r = client.post("/api/message", json={"message": "Analyze the EV battery market"})
    hid = r.json()["history_id"]

    h = client.get("/api/history")
    assert h.status_code == 200
    items = h.json()["items"]
    assert any(it["id"] == hid and it["kind"] == "report" for it in items)


def test_clear_history_endpoint(client):
    client.post("/api/message", json={"message": "Research renewable energy trends"})
    assert client.get("/api/history").json()["items"]     # non-empty
    d = client.delete("/api/history")
    assert d.status_code == 200
    assert client.get("/api/history").json()["items"] == []


def test_chat_message_is_recorded_in_history(client):
    # A greeting -> FakeLLM classifier returns general_chat.
    r = client.post("/api/message", json={"message": "hello there"})
    assert r.status_code == 200
    assert r.json()["type"] == "chat"
    items = client.get("/api/history").json()["items"]
    assert any(it["kind"] == "chat" for it in items)


def test_history_report_without_uploads_reports_zero_files(client):
    """A brief (no user uploads) must expose files=[] / file_count=0 so the UI can
    show the 'Brief' badge."""
    r = client.post("/api/message", json={"message": "Generate a report on hydrogen fuel"})
    hid = r.json()["history_id"]
    items = client.get("/api/history").json()["items"]
    entry = next(it for it in items if it["id"] == hid)
    assert entry["file_count"] == 0
    assert entry["files"] == []


def test_history_report_lists_uploaded_filenames(client):
    """After a document is uploaded to the report's session, its filename appears in
    that history entry's files list (drives the hover tooltip)."""
    r = client.post("/api/message", json={"message": "Analyze the lithium supply chain"})
    body = r.json()
    sid = body["session_id"]
    hid = body["history_id"]

    up = client.post(
        f"/api/database/{sid}/articles",
        files={"file": ("my_notes.txt", b"A verifiable fact about lithium.", "text/plain")},
    )
    assert up.status_code == 200, up.text

    items = client.get("/api/history").json()["items"]
    entry = next(it for it in items if it["id"] == hid)
    assert entry["file_count"] == 1
    assert entry["files"] == ["my_notes.txt"]
