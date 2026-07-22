"""Tests for the persistent history store (app/history.py).

Uses a temp app-data dir so nothing touches the real per-user directory. Offline.

    pytest -q tests/test_history_store.py
"""
from __future__ import annotations

import pytest


@pytest.fixture
def temp_history(monkeypatch, tmp_path):
    from core_engine.app import runtime

    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    from core_engine.app import history

    return history


def test_add_and_list(temp_history):
    h = temp_history
    assert h.list_entries() == []
    hid = h.add_entry(kind="report", title="solar cells", status="running",
                      job_id="job1", session_id="sess1")
    items = h.list_entries()
    assert len(items) == 1
    assert items[0]["id"] == hid
    assert items[0]["title"] == "solar cells"
    assert items[0]["status"] == "running"
    assert items[0]["job_id"] == "job1"
    assert items[0]["pdf_ready"] is False


def test_newest_first(temp_history):
    h = temp_history
    h.add_entry(kind="chat", title="first")
    h.add_entry(kind="chat", title="second")
    items = h.list_entries()
    assert [it["title"] for it in items] == ["second", "first"]


def test_update_entry(temp_history):
    h = temp_history
    hid = h.add_entry(kind="report", title="topic", status="running", job_id="j1")
    h.update_entry(hid, status="completed", pdf_ready=True, job_id="j2")
    it = h.list_entries()[0]
    assert it["status"] == "completed"
    assert it["pdf_ready"] is True
    assert it["job_id"] == "j2"


def test_update_unknown_id_is_noop(temp_history):
    h = temp_history
    h.add_entry(kind="chat", title="only")
    h.update_entry("does-not-exist", status="completed")
    assert len(h.list_entries()) == 1


def test_clear(temp_history):
    h = temp_history
    h.add_entry(kind="chat", title="a")
    h.add_entry(kind="chat", title="b")
    h.clear()
    assert h.list_entries() == []


def test_persists_to_disk_across_reload(temp_history, tmp_path):
    """A fresh read of the store (simulating a restart) sees prior entries."""
    h = temp_history
    h.add_entry(kind="report", title="persisted topic", status="completed",
                job_id="jX", pdf_ready=True)
    # The file exists in the temp app-data dir and reloads independently.
    assert (tmp_path / "history.json").exists()
    reloaded = h.list_entries()
    assert reloaded[0]["title"] == "persisted topic"
    assert reloaded[0]["pdf_ready"] is True


def test_corrupt_file_is_treated_as_empty(temp_history, tmp_path):
    (tmp_path / "history.json").write_text("{not json", encoding="utf-8")
    assert temp_history.list_entries() == []
