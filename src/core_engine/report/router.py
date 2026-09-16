"""Query-type-aware search routing (Search Router).

One search backend cannot serve every query shape well: Chinese-language queries are
often blocked on DuckDuckGo, table/market-size queries benefit from an extraction-
oriented backend (Firecrawl), and official-fact lookups want high-precision API
search. This module classifies each query and walks an ordered backend chain for
that type, falling through on SearchUnavailableError until one backend answers.

    classify_query(query) -> QueryType        (rules first, optional LLM tiebreak)
    BackendRegistry                           (QueryType -> ordered backend-name chain)
    routed_search(query, llm, ...)            (classify -> walk chain -> hits)

Diagnostics: the registry records `routes[query] = backend_name` (the via_backend
trail) instead of mutating the SearchHit model. Disk caching is per (backend, query)
via SearchCache — the chain checks the cache for each backend it tries.
"""
from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import ClassVar

from core_engine.config import get_settings
from core_engine.report.llm import strip_think_tags
from core_engine.report.models import SearchHit
from core_engine.report.scrape import SearchUnavailableError, get_search_provider
from core_engine.report.search_cache import SearchCache

log = logging.getLogger(__name__)


class QueryType(StrEnum):
    FACT = "fact"                    # official / primary-source lookups (gov, standards)
    DEEP_RESEARCH = "deep_research"  # long-form analysis / comparisons
    CHINESE = "chinese"              # predominantly Chinese-language query
    STRUCTURED = "structured"        # tables / revenue / market-size figures
    ACADEMIC = "academic"            # papers / studies
    GENERAL = "general"              # fallback


_CJK_RE = re.compile(r"[一-鿿]")

# Rule tables, checked in classify_query() order (first match wins).
_FACT_MARKERS = ("site:.gov", "site:.edu", "site:gov.cn", "site:.mil")
_STRUCTURED_TERMS = ("表格", "营收", "市场规模", "数据表", "table", "revenue", "figures")
_ACADEMIC_TERMS = ("paper", "study", "arxiv", "论文")
_DEEP_RESEARCH_TERMS = ("对比", "深度", "调研", "in-depth", "analysis")


def _rules_classify(query: str) -> QueryType | None:
    """Deterministic short-circuit rules. None = 'no rule fired'."""
    ql = query.lower()
    if any(m in ql for m in _FACT_MARKERS):
        return QueryType.FACT
    non_space = sum(1 for c in query if not c.isspace())
    cjk = len(_CJK_RE.findall(query))
    if non_space and cjk / non_space > 0.30:
        return QueryType.CHINESE
    if any(t in ql for t in _STRUCTURED_TERMS):
        return QueryType.STRUCTURED
    if any(t in ql for t in _ACADEMIC_TERMS):
        return QueryType.ACADEMIC
    if any(t in ql for t in _DEEP_RESEARCH_TERMS):
        return QueryType.DEEP_RESEARCH
    return None


async def _llm_classify(query: str, llm) -> QueryType:
    """LLM tiebreak for queries no rule matched. Any failure degrades to GENERAL —
    classification must never block a search."""
    system = (
        "You are a search-query classifier. Reply with EXACTLY ONE word, the query "
        "type: fact, deep_research, chinese, structured, academic, or general. "
        "structured = asks for tables/revenue/market-size figures; academic = asks for "
        "papers/studies; fact = asks for an official/primary source; deep_research = "
        "asks for in-depth comparison or analysis; general = anything else."
    )
    try:
        raw = strip_think_tags(await llm.complete(system, query)).lower()
        m = re.search(r"(deep_research|structured|academic|chinese|fact|general)", raw)
        if m:
            return QueryType(m.group(1))
        log.info("router: LLM classifier returned unparseable %r -> general", raw[:80])
    except Exception as e:
        log.warning("router: LLM classification failed (%s) -> general", type(e).__name__)
    return QueryType.GENERAL


async def classify_query(query: str, llm=None) -> QueryType:
    """Classify a query. Rules short-circuit first (cheap + deterministic); the LLM
    is consulted only when no rule fires, and any LLM failure falls back to GENERAL."""
    qt = _rules_classify(query)
    if qt is not None:
        return qt
    if llm is not None:
        return await _llm_classify(query, llm)
    return QueryType.GENERAL


class BackendRegistry:
    """QueryType -> ordered backend-name chain. Each name is instantiated lazily via
    scrape.get_search_provider(name); a backend that cannot be constructed (missing
    API key) is skipped by routed_search, which moves to the next chain entry."""

    # Chains for the special query types. All other types use _generic_chain().
    DEFAULT_CHAINS: ClassVar[dict[QueryType, tuple[str, ...]]] = {
        QueryType.CHINESE: ("jina", "tavily", "duckduckgo"),
        QueryType.STRUCTURED: ("firecrawl", "tavily", "duckduckgo"),
    }

    def __init__(self, chains: dict[QueryType, tuple[str, ...]] | None = None) -> None:
        self._chains = dict(self.DEFAULT_CHAINS)
        if chains:
            self._chains.update(chains)
        # via_backend diagnostics: query -> name of the backend that served it.
        self.routes: dict[str, str] = {}

    @staticmethod
    def _generic_chain() -> tuple[str, ...]:
        """Default chain for non-special types: the configured provider first
        (unless it is the test fake), then tavily, then the keyless duckduckgo."""
        s = get_settings()
        head = s.search_provider if s.search_provider != "fake" else "tavily"
        return tuple(dict.fromkeys((head, "tavily", "duckduckgo")))

    def chain_for(self, qtype: QueryType) -> tuple[str, ...]:
        return self._chains.get(qtype, self._generic_chain())


async def routed_search(
    query: str,
    llm=None,
    *,
    max_results: int,
    on_progress=None,
    registry: BackendRegistry | None = None,
    cache: SearchCache | None = None,
) -> list[SearchHit]:
    """Classify `query` and walk its backend chain until one backend returns hits.

    - A backend that cannot be instantiated (missing key) is skipped silently-ish.
    - A backend raising SearchUnavailableError (auth failure, blocked, timeout after
      retries) falls through to the next chain entry.
    - If a cache is given, each backend tried checks the cache first and a successful
      search is written back.
    - If EVERY chain entry fails, raises SearchUnavailableError naming all attempts.

    `on_progress(detail: str)` (optional) receives one-line routing decisions.
    """
    registry = registry or BackendRegistry()
    qtype = await classify_query(query, llm)
    chain = registry.chain_for(qtype)
    log.info("router: %r -> %s, chain=%s", query, qtype, chain)

    def _emit(detail: str) -> None:
        if on_progress:
            try:
                on_progress(detail)
            except Exception:
                pass  # progress is best-effort

    errors: list[str] = []
    for name in chain:
        try:
            provider = get_search_provider(name)
        except Exception as e:
            # Backend not configured (missing key) — not a failure, just skip it.
            log.info("router: backend %r unavailable (%s) — skipping", name, e)
            errors.append(f"{name}: not configured ({e})")
            continue
        if cache is not None:
            cached = cache.get(name, query)
            if cached is not None:
                registry.routes[query] = name
                log.info("router: cache hit for %r via %s", query, name)
                return cached[:max_results]
        try:
            hits = await provider.search(query, max_results=max_results)
        except SearchUnavailableError as e:
            errors.append(f"{name}: {e}")
            log.warning("router: backend %s failed for %r — trying next", name, query)
            _emit(f"Search backend '{name}' unavailable; falling back…")
            continue
        except (AttributeError, NameError, TypeError, KeyError, ImportError):
            raise  # code bug, not a backend failure — surface loudly (same as gather)
        except Exception as e:  # genuinely transient — try the next backend
            errors.append(f"{name}: {type(e).__name__}: {e}")
            log.warning("router: backend %s errored for %r (%s) — trying next",
                        name, query, type(e).__name__)
            continue
        registry.routes[query] = name
        if cache is not None and hits:
            cache.put(name, query, hits)
        return hits
    raise SearchUnavailableError(
        f"All search backends failed for {query!r} (type={qtype}, chain={chain}): "
        + "; ".join(errors))
