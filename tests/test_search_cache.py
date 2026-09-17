"""Tests for the on-disk search cache (report/search_cache.py).

Uses a temp app-data dir (monkeypatched runtime.user_data_dir, same pattern as
test_history_store.py) so nothing touches the real per-user directory. Offline.

    pytest -q tests/test_search_cache.py
"""
from __future__ import annotations

import json
import time

import pytest

from core_engine.report.models import SearchHit


@pytest.fixture
def cache(monkeypatch, tmp_path):
    from core_engine.app import runtime

    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    from core_engine.report.search_cache import SearchCache

    return SearchCache()


def _hits() -> list[SearchHit]:
    return [
        SearchHit("https://cdc.gov/a", "A", "snip a", "cdc.gov"),
        SearchHit("https://iea.org/b", "B", "", "iea.org"),
    ]


def test_put_get_roundtrip(cache):
    assert cache.get("tavily", "solar") is None          # cold miss
    cache.put("tavily", "solar", _hits())
    got = cache.get("tavily", "solar")
    assert got is not None
    assert [h.url for h in got] == ["https://cdc.gov/a", "https://iea.org/b"]
    assert got[0].title == "A" and got[0].snippet == "snip a"
    assert got[1].domain == "iea.org"


def test_key_namespaces_by_backend(cache):
    cache.put("tavily", "solar", _hits())
    assert cache.get("duckduckgo", "solar") is None      # same query, other backend
    assert cache.get("tavily", "wind") is None           # same backend, other query


def test_expired_entry_is_a_miss(cache, tmp_path):
    from core_engine.report.search_cache import SearchCache

    cache.put("tavily", "solar", _hits())
    # Age the entry by rewriting created_at into the past (avoids flaky sleeps).
    p = cache._path("tavily", "solar")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["created_at"] = time.time() - 100.0
    p.write_text(json.dumps(data), encoding="utf-8")

    short_ttl = SearchCache(base_dir=tmp_path / "search_cache", ttl_s=10.0)
    assert short_ttl.get("tavily", "solar") is None     # expired
    long_ttl = SearchCache(base_dir=tmp_path / "search_cache", ttl_s=1000.0)
    assert long_ttl.get("tavily", "solar") is not None  # still fresh under a longer TTL


def test_corrupt_file_is_a_miss(cache):
    cache.put("tavily", "solar", _hits())
    p = cache._path("tavily", "solar")
    p.write_text("{not json", encoding="utf-8")
    assert cache.get("tavily", "solar") is None


def test_write_is_atomic_no_tmp_left(cache, tmp_path):
    cache.put("tavily", "solar", _hits())
    files = [f.name for f in (tmp_path / "search_cache").iterdir()]
    assert files == [f"{cache.key('tavily', 'solar')}.json"]
