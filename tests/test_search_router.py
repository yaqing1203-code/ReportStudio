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
    SearchPurpose,
    classify_query,
    routed_search,
)
from core_engine.report.scrape import SearchUnavailableError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def clean_settings_cache(monkeypatch):
    from core_engine.config import get_settings
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setenv("CE_SEARCH_API_KEY", "")
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


@pytest.mark.parametrize("query", ["固态电池供应链", "EV revenue table", "battery paper",
    "official statistics site:.gov", "EV in-depth analysis", "everyday question"])
async def test_configured_primary_wins_for_every_query_type(monkeypatch, query):
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "tavily")
    primary = _OkProvider([_hit("https://example.edu/primary")])
    fallback = _OkProvider([_hit("https://example.edu/fallback")])
    _patch_factory(monkeypatch, {"tavily": primary, "duckduckgo": fallback})
    llm = _StubLLM("academic")
    hits = await routed_search(query, llm, max_results=5)
    assert [h.url for h in hits] == ["https://example.edu/primary"]
    assert primary.calls == 1 and fallback.calls == 0 and llm.calls == 0


@pytest.mark.parametrize(("provider", "key", "expected"), [
    ("tavily", "", ("tavily", "duckduckgo")),
    ("duckduckgo", "", ("duckduckgo",)),
    ("duckduckgo", "configured-key", ("duckduckgo", "tavily")),
    ("duckduckgo", "   ", ("duckduckgo",)),
    ("jina", "configured-key", ("jina", "duckduckgo")),
    ("crossref", "configured-key", ("crossref", "duckduckgo")),
    ("fake", "configured-key", ("fake",)),
])
def test_default_chains_have_one_configured_primary_and_at_most_one_fallback(monkeypatch, provider, key, expected):
    monkeypatch.setenv("CE_SEARCH_PROVIDER", provider)
    monkeypatch.setenv("CE_SEARCH_API_KEY", key)
    registry = BackendRegistry()
    assert all(registry.chain_for(qtype) == expected for qtype in QueryType)


async def test_default_primary_uses_only_one_fallback(monkeypatch):
    import core_engine.report.router as router_mod
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "tavily")
    requested = []
    def factory(name):
        requested.append(name)
        return _FailingProvider()
    monkeypatch.setattr(router_mod, "get_search_provider", factory)
    with pytest.raises(SearchUnavailableError):
        await routed_search("中文查询不应该改换主渠道", max_results=3)
    assert requested == ["tavily", "duckduckgo"]


async def test_default_fallback_can_rescue_primary_but_explicit_chains_keep_all_entries(monkeypatch):
    primary, second = _FailingProvider(), _OkProvider([_hit("https://example.edu/ok")])
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "tavily")
    _patch_factory(monkeypatch, {"tavily": primary, "duckduckgo": second})
    assert len(await routed_search("中文话题", max_results=3)) == 1
    assert primary.calls == second.calls == 1
    first, middle, last = _FailingProvider(), _FailingProvider(), _OkProvider([_hit("https://example.edu/custom")])
    registry = BackendRegistry({QueryType.GENERAL: ("first", "middle", "last")})
    _patch_factory(monkeypatch, {"first": first, "middle": middle, "last": last})
    assert len(await routed_search("custom route", max_results=3, registry=registry, purpose="gap")) == 1
    assert first.calls == middle.calls == last.calls == 1


@pytest.mark.parametrize("query", ["solid-state battery supply chain", "固态电池最新消息", "paperwork automation"])
async def test_generic_success_does_not_start_academic_complement(monkeypatch, query):
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    primary, extra = _OkProvider([_hit("https://example.edu/main")]), _OkProvider([_hit("https://example.edu/paper")])
    _patch_factory(monkeypatch, {"duckduckgo": primary, "openalex": extra})
    hits = await routed_search(query, max_results=4)
    assert [h.url for h in hits] == ["https://example.edu/main"]
    assert primary.calls == 1 and extra.calls == 0


@pytest.mark.parametrize(("query", "purpose"), [
    ("battery paper", "general"), ("固态电池最新论文", "general"),
    ("固态电池文献综述", SearchPurpose.GENERAL), ("scholarly journal studies", "general"),
    ("battery performance", "academic"), ("企业最新消息", SearchPurpose.GAP),
])
async def test_academic_signals_or_explicit_purpose_trigger_complement(monkeypatch, query, purpose):
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    primary, extra = _OkProvider([_hit("https://example.edu/main")]), _OkProvider([_hit("https://example.edu/paper")])
    _patch_factory(monkeypatch, {"duckduckgo": primary, "openalex": extra})
    hits = await routed_search(query, max_results=4, purpose=purpose)
    assert {h.url for h in hits} == {"https://example.edu/main", "https://example.edu/paper"}
    assert primary.calls == extra.calls == 1


@pytest.mark.parametrize("primary_hits", [[], [_hit("https://reddit.com/no-evidence")],
    [SearchHit("javascript:alert(1)", "not a page", "", "")]])
async def test_no_admissible_primary_results_triggers_complement(monkeypatch, primary_hits):
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    primary, extra = _OkProvider(primary_hits), _OkProvider([_hit("https://example.edu/paper")])
    _patch_factory(monkeypatch, {"duckduckgo": primary, "openalex": extra})
    hits = await routed_search("a narrow unfamiliar topic", max_results=4)
    assert "https://example.edu/paper" in [h.url for h in hits]
    assert extra.calls == 1


async def test_generic_primary_outage_can_be_rescued_by_complement(monkeypatch):
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    primary, extra = _FailingProvider(), _OkProvider([_hit("https://example.edu/paper")])
    _patch_factory(monkeypatch, {"duckduckgo": primary, "openalex": extra})
    assert [h.url for h in await routed_search("unfamiliar topic", max_results=4)] == ["https://example.edu/paper"]


@pytest.mark.parametrize("failure", [RuntimeError, ValueError, TypeError])
async def test_unexpected_complement_failure_never_discards_primary(monkeypatch, failure):
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    class BrokenComplement:
        async def search(self, *args, **kwargs):
            raise failure("optional adapter failed")
    primary = _OkProvider([_hit("https://example.edu/main")])
    _patch_factory(monkeypatch, {"duckduckgo": primary, "openalex": BrokenComplement()})
    assert [h.url for h in await routed_search("battery paper", max_results=4)] == ["https://example.edu/main"]


async def test_gap_purpose_cannot_bypass_disabled_openalex_switch(monkeypatch):
    primary, extra = _OkProvider([_hit("https://example.edu/main")]), _FailingProvider()
    _patch_factory(monkeypatch, {"duckduckgo": primary, "openalex": extra})
    await routed_search("battery paper", max_results=4, purpose="gap")
    assert primary.calls == 1 and extra.calls == 0


@pytest.mark.parametrize("purpose", ["everything", "ACADEMIC", "", None, 1])
async def test_invalid_purpose_rejected_before_search(monkeypatch, purpose):
    primary = _FailingProvider()
    _patch_factory(monkeypatch, {"duckduckgo": primary})
    with pytest.raises(ValueError):
        await routed_search("topic", max_results=3, purpose=purpose)
    assert primary.calls == 0


async def test_cached_generic_primary_does_not_use_cached_or_live_complement(monkeypatch, tmp_path):
    from core_engine.report.search_cache import SearchCache
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    cache = SearchCache(tmp_path)
    cache.put("duckduckgo", "ordinary topic", [_hit("https://example.edu/cached")])
    cache.put("openalex", "ordinary topic", [_hit("https://example.edu/paper")])
    primary, extra = _FailingProvider(), _FailingProvider()
    _patch_factory(monkeypatch, {"duckduckgo": primary, "openalex": extra})
    assert [h.url for h in await routed_search("ordinary topic", max_results=3, cache=cache)] == ["https://example.edu/cached"]
    assert primary.calls == extra.calls == 0


async def test_nonpositive_limit_stays_offline_and_provider_overreturn_is_bounded(monkeypatch):
    class Unbounded:
        calls = 0
        async def search(self, *args, **kwargs):
            self.calls += 1
            return [_hit(f"https://example.edu/{i}") for i in range(10)]
    primary = Unbounded()
    _patch_factory(monkeypatch, {"duckduckgo": primary})
    assert await routed_search("topic", max_results=0) == []
    assert await routed_search("topic", max_results=-1) == []
    assert primary.calls == 0
    assert len(await routed_search("topic", max_results=2)) == 2


async def test_complement_maximum_is_enforced_even_when_adapter_overreturns(monkeypatch):
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    monkeypatch.setenv("CE_OPENALEX_MAX_RESULTS", "2")
    class Unbounded:
        def __init__(self):
            self.limits = []
        async def search(self, topic, *, max_results):
            self.limits.append(max_results)
            return [_hit(f"https://example.edu/{i}") for i in range(10)]
    extra = Unbounded()
    _patch_factory(monkeypatch, {"duckduckgo": _OkProvider([]), "openalex": extra})
    assert len(await routed_search("ordinary topic", max_results=5)) == 2
    assert extra.limits == [2]


async def test_openalex_primary_does_not_call_it_again_as_complement(monkeypatch):
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "openalex")
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    primary, fallback = _OkProvider([_hit("https://example.edu/paper")]), _FailingProvider()
    _patch_factory(monkeypatch, {"openalex": primary, "duckduckgo": fallback})
    assert len(await routed_search("battery paper", max_results=4, purpose="gap")) == 1
    assert primary.calls == 1 and fallback.calls == 0


async def test_optional_complement_does_not_hide_primary_programming_error(monkeypatch):
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    class BrokenPrimary:
        async def search(self, *args, **kwargs):
            raise TypeError("primary contract is broken")
    extra = _OkProvider([_hit("https://example.edu/paper")])
    _patch_factory(monkeypatch, {"duckduckgo": BrokenPrimary(), "openalex": extra})
    with pytest.raises(TypeError, match="primary contract is broken"):
        await routed_search("battery paper", max_results=4)


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


async def test_fake_provider_stays_offline_even_when_router_is_enabled(monkeypatch):
    from core_engine.config import get_settings
    from core_engine.report import router, scrape

    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    monkeypatch.setenv("CE_SEARCH_ROUTER_ENABLED", "true")
    get_settings.cache_clear()

    async def must_not_route(*args, **kwargs):
        raise AssertionError("offline gather must not use network routing")

    real_routed_search = router.routed_search
    monkeypatch.setattr(router, "routed_search", must_not_route)
    try:
        assert await scrape.gather_sources("offline topic", None, scrape.FakeFetcher()) == ([], [])
        # Active verification calls the router directly; it must also stay offline.
        assert await real_routed_search("离线检索", max_results=3) == []
    finally:
        get_settings.cache_clear()
