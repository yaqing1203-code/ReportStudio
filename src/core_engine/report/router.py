"""Configured primary search, one fallback, and on-demand academic discovery.

Default runs honor the configured provider for every query language and shape.
OpenAlex is a bounded complement for academic queries, explicit academic/gap work,
or a primary route without admissible results. Explicit registries retain their
caller-defined query-type chains without automatic complementary search.

    classify_query(query) -> QueryType        (rules first, optional LLM tiebreak)
    BackendRegistry                           (QueryType -> ordered backend-name chain)
    routed_search(query, llm, ...)            (classify -> walk chain -> hits)

Diagnostics: the registry records `routes[query] = backend_name` (the via_backend
trail) instead of mutating the SearchHit model. Disk caching is per (backend, query)
via SearchCache — the chain checks the cache for each backend it tries.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import replace
from enum import StrEnum
from urllib.parse import urlsplit

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


class SearchPurpose(StrEnum):
    GENERAL = "general"
    ACADEMIC = "academic"
    GAP = "gap"


_CJK_RE = re.compile(r"[一-鿿]")

# Rule tables, checked in classify_query() order (first match wins).
_FACT_MARKERS = ("site:.gov", "site:.edu", "site:gov.cn", "site:.mil")
_STRUCTURED_TERMS = ("表格", "营收", "市场规模", "数据表", "table", "revenue", "figures")
_ACADEMIC_TERMS = ("paper", "study", "arxiv", "论文")
_DEEP_RESEARCH_TERMS = ("对比", "深度", "调研", "in-depth", "analysis")
_ACADEMIC_SIGNALS_EN = re.compile(
    r"\b(papers?|stud(?:y|ies)|arxiv|academic|journals?|doi|pubmed|scholar|peer[- ]reviewed)\b"
    r"|\b(systematic review|meta-analysis|clinical trial|scientific research)\b", re.IGNORECASE)
_ACADEMIC_SIGNALS_ZH = ("论文", "学术", "期刊", "文献", "综述", "同行评审", "实验研究")


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

    def __init__(self, chains: dict[QueryType, tuple[str, ...]] | None = None) -> None:
        self._chains = dict(chains or {})
        # via_backend diagnostics: query -> name of the backend that served it.
        self.routes: dict[str, str] = {}

    @staticmethod
    def _generic_chain() -> tuple[str, ...]:
        """The configured provider plus at most one useful fallback."""
        s = get_settings()
        head = s.search_provider
        if head == "fake":
            return ("fake",)
        if head == "duckduckgo":
            return (head, "tavily") if s.search_api_key.strip() else (head,)
        return (head, "duckduckgo")

    def chain_for(self, qtype: QueryType) -> tuple[str, ...]:
        return self._chains.get(qtype, self._generic_chain())


async def _routed_primary_search(
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
    # The offline provider must never fall through to a real network backend.
    # Explicit registries remain available for router contract tests/custom callers.
    if registry is None and get_settings().search_provider == "fake":
        return await get_search_provider("fake").search(query, max_results=max_results)
    registry = registry or BackendRegistry()
    # Default routing does not depend on query type, so avoid spending a model
    # request on a classification that cannot alter the configured route.
    qtype = await classify_query(query, llm if registry._chains else None)
    chain = registry.chain_for(qtype)
    log.info("router: %r -> %s, chain=%s", query, qtype, chain)

    def _emit(detail: str) -> None:
        if on_progress:
            try:
                on_progress(detail)
            except Exception:
                pass  # progress is best-effort

    errors: list[str] = []
    successful_empty = False
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
            if cached:
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
        if not hits:
            successful_empty = True
            _emit(f"Search backend '{name}' returned no results; trying another source…")
            continue
        registry.routes[query] = name
        if cache is not None and hits:
            cache.put(name, query, hits)
        return hits[:max_results]
    if successful_empty:
        return []  # No result is different from every backend being unavailable.
    raise SearchUnavailableError(
        f"All search backends failed for {query!r} (type={qtype}, chain={chain}): "
        + "; ".join(errors))


def _merge_complement(
    primary: list[SearchHit], extra: list[SearchHit], limit: int, *, prefer_openalex_record: bool = False,
) -> list[SearchHit]:
    """Reserve bounded room for complementary hits even when the primary list is full."""
    from core_engine.report.openalex import api_work_id

    # Also applies to pre-fix cached hits: a resolver/landing page must not occupy
    # the only academic slot while a directly readable abstract sits below it.
    extra = sorted(extra, key=lambda hit: api_work_id(hit.url) is None)
    by_url = {hit.url: hit for hit in primary}
    additions = []
    for hit in extra:
        if hit.url in by_url:
            old = by_url[hit.url]
            by_url[hit.url] = replace(old, metadata={**hit.metadata, **old.metadata})
        else:
            by_url[hit.url] = hit
            additions.append(hit)
    # Academic gap rounds may fetch only the first candidate from each query.
    # Give the directly fetchable abstract a slot, without elevating unresolved
    # publisher landing pages or treating discovery metadata as evidence.
    first = extra[0].url if prefer_openalex_record and extra and api_work_id(extra[0].url) else None
    primary = list({hit.url: by_url[hit.url] for hit in primary if hit.url != first}.values())
    additions = [hit for hit in additions if hit.url != first]
    reserved = int(first is not None)
    room = max(0, limit - reserved)
    quota = min(len(additions), max(0, max(1, limit // 3) - reserved)) if limit > 1 else 0
    retained = primary[:max(0, room - quota)]
    supplements = additions[:max(0, room - len(retained))]
    merged = [by_url[first]] if first is not None else []
    for index in range(max(len(retained), len(supplements))):
        merged.extend(retained[index * 2:index * 2 + 2])
        merged.extend(supplements[index:index + 1])
    return [by_url[hit.url] for hit in merged][:limit]


async def routed_search(
    query: str,
    llm=None,
    *,
    max_results: int,
    on_progress=None,
    registry: BackendRegistry | None = None,
    cache: SearchCache | None = None,
    purpose: SearchPurpose | str = SearchPurpose.GENERAL,
) -> list[SearchHit]:
    """Search the configured route, supplementing with OpenAlex only when needed.

    Explicit registries and offline fake runs retain their caller-defined chains.
    OpenAlex failure cannot discard successful primary results; conversely it can
    rescue a failed primary search. The merged list obeys the original hit limit.
    `purpose` must be general, academic or gap; it never bypasses the master switch.
    """
    purpose = SearchPurpose(purpose)
    if max_results <= 0:
        return []
    settings = get_settings()
    if registry is None and settings.search_provider == "fake":
        return await get_search_provider("fake").search(query, max_results=max_results)
    complement = (registry is None and settings.openalex_search_enabled
                  and settings.search_provider != "openalex")
    active_registry = registry or BackendRegistry()
    primary_call = _routed_primary_search(query, llm, max_results=max_results,
        on_progress=on_progress, registry=active_registry, cache=cache)
    if not complement:
        return await primary_call
    academic = bool(_ACADEMIC_SIGNALS_EN.search(query)) or any(t in query for t in _ACADEMIC_SIGNALS_ZH)
    prefer_openalex_record = purpose is SearchPurpose.ACADEMIC or (purpose is SearchPurpose.GAP and academic)

    async def supplementary() -> list[SearchHit]:
        limit = min(max_results, settings.openalex_max_results)
        try:
            provider = get_search_provider("openalex")
            cached = cache.get("openalex", query) if cache is not None else None
            if cached:
                if prefer_openalex_record:
                    from core_engine.report.openalex import api_work_id

                    cached = sorted(cached, key=lambda hit: api_work_id(hit.url) is None)
                return cached[:limit]
            hits = await asyncio.wait_for(provider.search(query, max_results=limit),
                                          timeout=settings.openalex_timeout_s)
            if hits and cache is not None:
                cache.put("openalex", query, hits)
            return hits[:limit]
        except Exception as e:
            # Complementary discovery is optional, including an unexpected adapter
            # failure. Cancellation (BaseException) still propagates normally.
            log.warning("router: OpenAlex complement unavailable (%s); retaining primary results",
                        type(e).__name__)
            if on_progress:
                try:
                    on_progress("OpenAlex complement unavailable; continuing with other sources")
                except Exception:
                    pass
            return []

    primary: list[SearchHit] | BaseException
    extra: list[SearchHit] | BaseException = []
    if purpose in (SearchPurpose.ACADEMIC, SearchPurpose.GAP) or academic:
        primary, extra = await asyncio.gather(primary_call, supplementary(), return_exceptions=True)
    else:
        try:
            primary = await primary_call
        except SearchUnavailableError as e:
            primary = e
        if isinstance(primary, SearchUnavailableError) or not _has_useful_primary(primary):
            extra = await supplementary()
    if isinstance(extra, BaseException):
        raise extra
    if isinstance(primary, BaseException):
        if not isinstance(primary, SearchUnavailableError) or not extra:
            raise primary
        active_registry.routes[query] = "openalex"
        if prefer_openalex_record:
            return _merge_complement([], extra, max_results, prefer_openalex_record=True)
        return extra[:max_results]
    if extra:
        previous = active_registry.routes.get(query)
        active_registry.routes[query] = f"{previous}+openalex" if previous else "openalex"
    return _merge_complement(primary, extra, max_results, prefer_openalex_record=prefer_openalex_record)


def _has_useful_primary(hits: list[SearchHit]) -> bool:
    """An HTTP(S) hit admitted by source policy; actual page relevance is checked
    after fetch. A title/snippet alone is never treated as verified evidence."""
    from core_engine.report.models import SourceKind
    from core_engine.report.sources import classify

    for hit in hits:
        try:
            parsed = urlsplit(hit.url)
            if (parsed.scheme in {"http", "https"} and parsed.hostname
                    and classify(hit.url) is not SourceKind.REJECTED):
                return True
        except ValueError:
            continue
    return False
