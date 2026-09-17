"""Tests for the Database Mode session store (app/database.py).

Uses a temp app-data dir (monkeypatched runtime.user_data_dir) so nothing touches
the real per-user directory. Offline, no network.

    pytest -q tests/test_database_store.py
"""
from __future__ import annotations

import pytest

from core_engine.report.models import Source, SourceKind


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Point runtime.user_data_dir at a temp dir for the duration of the test."""
    from core_engine.app import runtime

    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    from core_engine.app import database

    return database


def test_session_roundtrip(temp_db):
    db = temp_db
    sid = db.create_session("solid-state batteries")
    data = db.get_session(sid)
    assert data is not None
    assert data["topic"] == "solid-state batteries"
    assert data["brief_status"] == "pending"
    assert data["articles"] == []


def test_save_brief_snapshots_sources(temp_db):
    db = temp_db
    sid = db.create_session("topic")
    srcs = [
        Source("http://iea.org/a", "iea.org", "IEA", SourceKind.INDUSTRY_INSTITUTION,
               text="body one"),
        Source("http://bls.gov/b", "bls.gov", "BLS", SourceKind.GOVERNMENT,
               text="body two"),
    ]
    db.save_brief(sid, job_id="job1", status="completed", sources=srcs)
    data = db.get_session(sid)
    assert data["brief_status"] == "completed"
    assert len(data["brief_sources"]) == 2
    assert data["brief_sources"][0]["url"] == "http://iea.org/a"


def test_add_list_remove_articles(temp_db):
    db = temp_db
    sid = db.create_session("topic")
    rec = db.add_article(sid, title="My Article", filename="a.txt",
                         text="Important finding about the market.")
    assert rec["title"] == "My Article"
    assert "text" not in rec  # summary must not leak the body
    listing = db.list_articles(sid)
    assert len(listing) == 1
    assert db.article_count(sid) == 1

    assert db.remove_article(sid, rec["id"]) is True
    assert db.article_count(sid) == 0
    assert db.remove_article(sid, "nonexistent") is False


def test_sources_for_comprehensive_combines_brief_and_articles(temp_db):
    db = temp_db
    sid = db.create_session("topic")
    db.save_brief(sid, job_id="job1", status="completed", sources=[
        Source("http://iea.org/a", "iea.org", "IEA", SourceKind.INDUSTRY_INSTITUTION,
               text="brief body"),
    ])
    db.add_article(sid, title="User Doc", filename="u.txt", text="user body text")

    pool = db.sources_for_comprehensive(sid)
    assert len(pool) == 2
    kinds = {s.kind for s in pool}
    assert SourceKind.INDUSTRY_INSTITUTION in kinds
    assert SourceKind.USER_PROVIDED in kinds
    # The user doc carries its parsed text and a synthetic addressable URL.
    user = next(s for s in pool if s.kind is SourceKind.USER_PROVIDED)
    assert user.text == "user body text"
    assert user.url.startswith("userdoc://")


def test_get_unknown_session_returns_none(temp_db):
    assert temp_db.get_session("does-not-exist") is None
