"""On-disk cache for search results.

Search calls are the slowest and (for paid backends) the most expensive step of the
gather phase, and research re-runs often repeat the exact same queries. This cache
stores SearchHit lists as small JSON files under user_data_dir()/search_cache/,
keyed by sha256(backend + "\n" + query)[:16] so the SAME query on a DIFFERENT
backend never collides.

Design follows app/history.py: TTL checked on read, corrupt files read as a miss,
and writes are atomic (sibling tmp file + os.replace) so a crash mid-write can never
leave a truncated entry behind. Everything here is best-effort: a cache failure must
never break a search.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

from core_engine.config import get_settings
from core_engine.report.models import SearchHit

log = logging.getLogger(__name__)


def _default_dir() -> Path:
    # Lazy import, resolved per instance: the engine layer must not import the app
    # layer at module scope (see app/__init__), and resolving at construction time
    # lets tests redirect runtime.user_data_dir via monkeypatch.
    from core_engine.app import runtime

    return runtime.user_data_dir() / "search_cache"


class SearchCache:
    """TTL'd JSON-file cache of search hits. Not thread-safe by design — gather
    searches are awaited sequentially per query, and worst case a concurrent put is
    just a redundant write."""

    def __init__(self, base_dir: Path | None = None, ttl_s: float | None = None) -> None:
        self._dir = Path(base_dir) if base_dir is not None else _default_dir()
        self._ttl = get_settings().search_cache_ttl_s if ttl_s is None else ttl_s
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(backend: str, query: str) -> str:
        return hashlib.sha256(f"{backend}\n{query}".encode()).hexdigest()[:16]

    def _path(self, backend: str, query: str) -> Path:
        return self._dir / f"{self.key(backend, query)}.json"

    def get(self, backend: str, query: str) -> list[SearchHit] | None:
        """Return cached hits, or None on miss / expired / corrupt entry."""
        p = self._path(backend, query)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            created = float(data["created_at"])
            if 0 <= self._ttl < (time.time() - created):
                return None  # expired (ttl < 0 means 'never expires')
            return [
                SearchHit(
                    url=h["url"],
                    title=h.get("title", ""),
                    snippet=h.get("snippet", ""),
                    domain=h.get("domain", ""),
                )
                for h in data["hits"]
            ]
        except Exception:
            log.info("search cache: corrupt entry %s treated as miss", p.name)
            return None

    def put(self, backend: str, query: str, hits: list[SearchHit]) -> None:
        p = self._path(backend, query)
        payload = {
            "created_at": time.time(),
            "backend": backend,
            "query": query,
            "hits": [
                {"url": h.url, "title": h.title, "snippet": h.snippet,
                 "domain": h.domain}
                for h in hits
            ],
        }
        try:
            # Atomic write: tmp file + os.replace, same pattern as history.py.
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, p)
        except Exception:
            # Best-effort: never let a persistence failure break a search.
            log.debug("search cache write failed for %s", p, exc_info=True)
