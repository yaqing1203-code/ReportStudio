"""Tests for the search router (report/router.py) — fully offline.

Rule classification, LLM tiebreak + failure fallback, backend fallback chains,
no-key backend skipping, per-backend caching, and the gather-level router hook.

    pytest -q tests/test_search_router.py
"""
from __future__ import annotations

import pytest

from core_engine.report.models import SearchHit
from core_engine.report.router import (
    BackendRegistry,
    QueryType,
    classify_query,
    routed_search,
)
from core_engine.report.scrape import SearchUnavailableError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _hit(url: str) -> SearchHit:
    return SearchHit(url=url, title="T", snippet="s", domain=url.split("/")[2])


class _StubLLM:
    def __init__(self, reply: str | None = None, *, raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.raises:
            raise RuntimeError("llm down")
        return self.reply or ""


# --------------------------------------------------------------------------
# classify_query — rule short-circuits
# --------------------------------------------------------------------------
async def test_rules_classify_fact():
    assert await classify_query("solar subsidy statistics site:.gov") is QueryType.FACT
    assert await classify_query("enrollment data site:.edu archive") is QueryType.FACT


async def test_rules_classify_chinese():
    assert await classify_query("固态电池供应链 龙头企业") is QueryType.CHINESE
    # Mostly-English query must NOT be classified chinese (<=30% CJK).
    assert await classify_query("solid state battery supply chain 固态") is not QueryType.CHINESE


async def test_rules_classify_structured_academic_deep():
    assert await classify_query("EV battery revenue table by vendor") is QueryType.STRUCTURED
    # CJK structured terms fire when the query is not predominantly Chinese (<30%).
    assert await classify_query("global EV market 市场规模 figures") is QueryType.STRUCTURED
    assert await classify_query("perovskite stability study arxiv") is QueryType.ACADEMIC
    assert await classify_query("latest paper on fusion") is QueryType.ACADEMIC
    assert await classify_query("EV market in-depth analysis") is QueryType.DEEP_RESEARCH


async def test_no_rule_no_llm_is_general():
    assert await classify_query("solid state battery supply chain") is QueryType.GENERAL


async def test_llm_tiebreak_when_no_rule_fires():
    llm = _StubLLM("<think>thinking…</think>structured")
    qt = await classify_query("latest developments in fusion energy", llm)
    assert qt is QueryType.STRUCTURED
    assert llm.calls == 1


async def test_llm_failure_falls_back_to_general():
    llm = _StubLLM(raises=True)
    assert await classify_query("latest developments in fusion energy", llm) is QueryType.GENERAL
    llm2 = _StubLLM("I cannot classify this at all, sorry")
    assert await classify_query("latest developments in fusion energy", llm2) is QueryType.GENERAL


# --------------------------------------------------------------------------
# routed_search — fallback chains
# --------------------------------------------------------------------------
def _patch_factory(monkeypatch, backends: dict[str, object], unavailable: set[str] = frozenset()):
    """Route backend names to stub providers; names in `unavailable` raise on
    instantiation (simulating a missing API key)."""
    import core_engine.report.router as router_mod

    def _factory(name: str | None = None):
        if name in unavailable:
            raise RuntimeError(f"no key for {name}")
        if name in backends:
            return backends[name]
        raise RuntimeError(f"unknown backend {name}")

    monkeypatch.setattr(router_mod, "get_search_provider", _factory)


class _FailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, topic: str, *, max_results: int):
        self.calls += 1
        raise SearchUnavailableError("blocked")


class _OkProvider:
    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits
        self.calls = 0

    async def search(self, topic: str, *, max_results: int):
        self.calls += 1
        return self._hits[:max_results]


async def test_fallback_chain_switches_to_second_backend(monkeypatch):
    bad, good = _FailingProvider(), _OkProvider([_hit("https://cdc.gov/a")])
    _patch_factory(monkeypatch, {"bad": bad, "good": good})
    registry = BackendRegistry(chains={QueryType.GENERAL: ("bad", "good")})

    hits = await routed_search("a rule-free query string", None,
                               max_results=5, registry=registry)
    assert [h.url for h in hits] == ["https://cdc.gov/a"]
    assert bad.calls == 1 and good.calls == 1
    # via_backend diagnostics recorded registry-side.
    assert registry.routes["a rule-free query string"] == "good"


async def test_all_backends_failing_raises(monkeypatch):
    _patch_factory(monkeypatch, {"bad": _FailingProvider(), "bad2": _FailingProvider()})
    registry = BackendRegistry(chains={QueryType.GENERAL: ("bad", "bad2")})
    with pytest.raises(SearchUnavailableError, match="All search backends failed"):
        await routed_search("a rule-free query string", None,
                            max_results=5, registry=registry)


async def test_unconfigured_backend_is_skipped(monkeypatch):
    good = _OkProvider([_hit("https://iea.org/x")])
    _patch_factory(monkeypatch, {"good": good}, unavailable={"nokey"})
    registry = BackendRegistry(chains={QueryType.GENERAL: ("nokey", "good")})

    hits = await routed_search("a rule-free query string", None,
                               max_results=5, registry=registry)
    assert [h.url for h in hits] == ["https://iea.org/x"]
    assert good.calls == 1


async def test_routed_search_uses_cache_before_calling_backend(monkeypatch, tmp_path):
    from core_engine.report.search_cache import SearchCache

    cache = SearchCache(base_dir=tmp_path)
    q = "a rule-free query string"
    cache.put("good", q, [_hit("https://cdc.gov/cached")])
    good = _OkProvider([_hit("https://cdc.gov/live")])
    _patch_factory(monkeypatch, {"good": good})
    registry = BackendRegistry(chains={QueryType.GENERAL: ("good",)})

    hits = await routed_search(q, None, max_results=5, registry=registry, cache=cache)
    assert [h.url for h in hits] == ["https://cdc.gov/cached"]
    assert good.calls == 0  # served from cache, backend never called


async def test_chinese_query_prefers_jina_chain(monkeypatch):
    jina = _OkProvider([_hit("https://example.cn/a")])
    _patch_factory(monkeypatch, {"jina": jina})
    hits = await routed_search("固态电池供应链", None, max_results=5)
    assert jina.calls == 1
    assert [h.url for h in hits] == ["https://example.cn/a"]


# --------------------------------------------------------------------------
# Factory extensions (scrape.get_search_provider(name))
# --------------------------------------------------------------------------
def test_factory_named_backends(monkeypatch):
    from core_engine.config import get_settings
    from core_engine.report.scrape import (
        FirecrawlSearchProvider,
        JinaSearchProvider,
        get_search_provider,
    )

    get_settings.cache_clear()
    monkeypatch.delenv("CE_FIRECRAWL_API_KEY", raising=False)
    try:
        # Firecrawl without a key cannot be instantiated (router skips it).
        with pytest.raises(RuntimeError, match="FIRECRAWL"):
            get_search_provider("firecrawl")
        # Jina works keyless.
        assert isinstance(get_search_provider("jina"), JinaSearchProvider)
        monkeypatch.setenv("CE_FIRECRAWL_API_KEY", "fc-test")
        get_settings.cache_clear()
        assert isinstance(get_search_provider("firecrawl"), FirecrawlSearchProvider)
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------
# gather-level integration: provider=None + router enabled routes per query
# --------------------------------------------------------------------------
async def test_gather_routes_when_no_provider_pinned(monkeypatch, tmp_path):
    from core_engine.app import runtime
    from core_engine.config import get_settings
    from core_engine.report.scrape import FakeFetcher, gather_sources

    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    get_settings.cache_clear()
    monkeypatch.setenv("CE_MULTI_QUERY_RESEARCH", "false")
    monkeypatch.setenv("CE_RELEVANCE_FILTER", "false")

    body = "Official report on the topic with substantial body text. " * 20
    provider = _OkProvider([_hit("https://cdc.gov/routed")])
    _patch_factory(monkeypatch, {"duckduckgo": provider})
    fetcher = FakeFetcher({"https://cdc.gov/routed": body})
    try:
        sources, _rejected = await gather_sources(
            "a rule-free gather topic", None, fetcher)
    finally:
        get_settings.cache_clear()

    assert provider.calls >= 1
    assert [s.url for s in sources] == ["https://cdc.gov/routed"]
