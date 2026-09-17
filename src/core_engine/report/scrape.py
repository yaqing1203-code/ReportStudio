"""Search + scrape layer.

Two responsibilities, split behind interfaces so neither the network nor an API
key is required to test the pipeline logic:

  SearchProvider  : topic -> list[SearchHit]        (search engine)
  Fetcher         : url   -> extracted main text    (HTTP + readability)

The pipeline calls search(), passes the hits through the STRICT source filter
(sources.py) BEFORE any page is fetched, then fetches only the authoritative
survivors. Order matters: we never download a rejected domain.

Providers:
  - "fake"       : deterministic in-memory fixtures. Default. No network, no keys.
  - "duckduckgo" : keyless web search (zero-setup default).
  - "tavily"     : Tavily search API (needs CE_SEARCH_API_KEY).
  - "firecrawl"  : Firecrawl search API (needs CE_FIRECRAWL_API_KEY).
  - "jina"       : Jina Reader search (CE_JINA_API_KEY optional).

When no provider is explicitly injected and `search_router_enabled` is on, queries
are routed per query type through report/router.py (backend fallback chains) and
results are cached on disk via report/search_cache.py.

The real fetcher honours robots.txt and a timeout; extraction strips boilerplate
to main content so the verifier quotes real prose, not nav chrome.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from core_engine.config import get_settings
from core_engine.report.models import SearchHit, Source, SourceKind
from core_engine.report.search_cache import SearchCache
from core_engine.report.sources import (
    classify,
    credibility_for,
    extract_domain,
    filter_hits,
)

log = logging.getLogger(__name__)


class SearchUnavailableError(RuntimeError):
    """The search backend could not be reached / is blocked / is misconfigured.

    Raised (instead of silently returning []) so the pipeline can tell the user
    'search is unavailable — check your provider/key' rather than the misleading
    'this topic is out of scope'. Silent [] was exactly what masked the DuckDuckGo
    block that made the agent 'know nothing'.
    """


class SearchProvider(Protocol):
    async def search(self, topic: str, *, max_results: int) -> list[SearchHit]: ...


class Fetcher(Protocol):
    async def fetch(self, url: str) -> str | None: ...


# --------------------------------------------------------------------------
# Fake provider + fetcher: deterministic fixtures for tests / offline dev.
# --------------------------------------------------------------------------
class FakeSearchProvider:
    """Returns canned hits keyed loosely by topic. Lets us exercise the filter,
    verifier, scope gate, and renderer with zero network access.

    Register fixtures with add(); unknown topics return the default set so the
    'out-of-scope' path is easy to trigger (pass a topic with no fixtures)."""

    def __init__(self) -> None:
        self._fixtures: dict[str, list[SearchHit]] = {}

    def add(self, topic_key: str, hits: list[SearchHit]) -> None:
        self._fixtures[topic_key.lower()] = hits

    async def search(self, topic: str, *, max_results: int) -> list[SearchHit]:
        for key, hits in self._fixtures.items():
            if key in topic.lower():
                return hits[:max_results]
        return []


class FakeFetcher:
    """Serves canned page text by URL. Missing URLs return None (fetch failure)."""

    def __init__(self, pages: dict[str, str] | None = None) -> None:
        self._pages = pages or {}

    def add(self, url: str, text: str) -> None:
        self._pages[url] = text

    async def fetch(self, url: str) -> str | None:
        return self._pages.get(url)


# --------------------------------------------------------------------------
# Real providers (thin; imports deferred so the package works without the deps).
# --------------------------------------------------------------------------
class TavilySearchProvider:
    """Tavily search API — reliable, agent-oriented web search (recommended provider).

    Raises SearchUnavailableError with an actionable message on auth (401/403) or
    connectivity failure, so a bad/missing key surfaces clearly instead of silently
    producing an empty, 'out of scope' report.
    """

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    async def search(self, topic: str, *, max_results: int) -> list[SearchHit]:
        import httpx

        s = get_settings()
        # RETRY with exponential backoff for transient errors (network blips, 429 rate
        # limits, 5xx). Auth failures (401/403) and bad requests (4xx) are NOT retried —
        # they won't fix themselves. A strict per-attempt timeout means a silent hang
        # fails fast instead of blocking the pipeline.
        attempts = max(1, s.search_max_retries)
        last_err: str = ""
        for attempt in range(1, attempts + 1):
            log.info("tavily search attempt %d/%d: %r (timeout=%.0fs)",
                     attempt, attempts, topic, s.search_timeout_s)
            try:
                async with httpx.AsyncClient(timeout=s.search_timeout_s) as client:
                    resp = await client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": self._key,
                            "query": topic,
                            "max_results": max_results,
                            "search_depth": "advanced",
                        },
                    )
            except Exception as e:
                # Network/timeout — transient, so retry (with backoff) before giving up.
                last_err = f"{type(e).__name__}: {e}"
                log.warning("tavily attempt %d failed (network): %s", attempt, last_err)
                if attempt < attempts:
                    await asyncio.sleep(s.search_retry_base_s * (2 ** (attempt - 1)))
                    continue
                raise SearchUnavailableError(
                    f"Could not reach Tavily after {attempts} attempt(s): {last_err}"
                ) from e

            # Auth / client errors are permanent — surface immediately, do not retry.
            if resp.status_code in (401, 403):
                raise SearchUnavailableError(
                    "Tavily rejected the API key (HTTP "
                    f"{resp.status_code}). Check CE_SEARCH_API_KEY in Settings."
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                # Rate-limited or server error — transient, retry with backoff.
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                log.warning("tavily attempt %d got %s", attempt, last_err)
                if attempt < attempts:
                    await asyncio.sleep(s.search_retry_base_s * (2 ** (attempt - 1)))
                    continue
                raise SearchUnavailableError(
                    f"Tavily unavailable after {attempts} attempt(s): {last_err}")
            if resp.status_code >= 400:
                raise SearchUnavailableError(
                    f"Tavily returned HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            hits: list[SearchHit] = []
            for r in data.get("results", []):
                url = r.get("url", "")
                hits.append(SearchHit(
                    url=url,
                    title=r.get("title", ""),
                    snippet=r.get("content", ""),
                    domain=extract_domain(url),
                ))
            log.info("tavily search %r -> %d hit(s)", topic, len(hits))
            return hits
        # Unreachable (loop either returns or raises), but keeps type checkers happy.
        raise SearchUnavailableError(f"Tavily search failed: {last_err}")


class DuckDuckGoSearchProvider:
    """Keyless web search via DuckDuckGo's HTML endpoint — the zero-setup default.

    No API key, no account: we POST the query to the html.duckduckgo.com lite
    endpoint and parse result anchors out of the returned HTML. This is what makes
    'autonomous crawling out-of-the-box' true — the packaged .exe can search the web
    with nothing to configure. API providers (Tavily/SerpAPI) remain optional quality
    upgrades for higher-recall search.

    We keep this dependency-light: httpx (already bundled for the fetcher) + a small
    regex parse, so nothing extra needs installing. DuckDuckGo's markup is scraped
    defensively — if the layout shifts we return what we can rather than crashing.
    """

    _ENDPOINT = "https://html.duckduckgo.com/html/"

    async def search(self, topic: str, *, max_results: int) -> list[SearchHit]:
        # Primary: the maintained ddgs/duckduckgo_search library (handles the VQD
        # challenge token DuckDuckGo now requires). Fallback: raw HTML scrape.
        hits = await self._via_library(topic, max_results)
        if hits:
            return hits
        hits = await self._via_html(topic, max_results)
        if hits:
            return hits
        # Both paths returned nothing. DuckDuckGo is very likely challenge-blocking this
        # environment (it now serves HTTP 202 stubs to programmatic clients). Surface
        # that loudly instead of masquerading as 'no results / out of scope'.
        raise SearchUnavailableError(
            "DuckDuckGo returned no results — it is likely blocking automated queries "
            "from this network (HTTP 202 bot-challenge). Configure a Tavily API key in "
            "Settings (search provider = 'tavily') for reliable search."
        )

    async def _via_library(self, topic: str, max_results: int) -> list[SearchHit]:
        """Use ddgs (preferred) or the legacy duckduckgo_search package if present.
        The library is sync, so run it in a worker thread."""
        import asyncio

        def _run() -> list[dict]:
            try:
                try:
                    from ddgs import DDGS  # new package name
                except ImportError:
                    from duckduckgo_search import DDGS  # legacy name
            except Exception:
                return []
            try:
                with DDGS() as d:
                    return list(d.text(topic, max_results=max_results))
            except Exception as e:
                log.warning("ddgs library search failed for %r: %s", topic, e)
                return []

        try:
            rows = await asyncio.to_thread(_run)
        except Exception:
            rows = []
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for r in rows:
            url = r.get("href") or r.get("url") or ""
            if not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            hits.append(SearchHit(
                url=url, title=r.get("title", ""), snippet=r.get("body", ""),
                domain=extract_domain(url),
            ))
        return hits

    async def _via_html(self, topic: str, max_results: int) -> list[SearchHit]:
        """Raw-HTML fallback scrape of the DDG lite endpoint."""
        import html as _html
        import re
        from urllib.parse import parse_qs, unquote, urlsplit

        import httpx

        s = get_settings()
        headers = {"User-Agent": s.scrape_user_agent}
        try:
            async with httpx.AsyncClient(
                timeout=s.scrape_timeout_s, headers=headers, follow_redirects=True
            ) as client:
                resp = await client.post(self._ENDPOINT, data={"q": topic})
                # HTTP 202 = DDG bot challenge; there will be no results to parse.
                if resp.status_code == 202:
                    return []
                resp.raise_for_status()
                body = resp.text
        except Exception as e:
            log.warning("duckduckgo HTML search failed for %r: %s", topic, e)
            return []

        hits: list[SearchHit] = []
        seen: set[str] = set()
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
        )
        for m in pattern.finditer(body):
            raw_href, raw_title = m.group(1), m.group(2)
            url = _html.unescape(raw_href)
            if "uddg=" in url:
                qs = parse_qs(urlsplit(url).query)
                if qs.get("uddg"):
                    url = unquote(qs["uddg"][0])
            if url.startswith("//"):
                url = "https:" + url
            if not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            title = _html.unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
            hits.append(SearchHit(
                url=url, title=title, snippet="", domain=extract_domain(url),
            ))
            if len(hits) >= max_results:
                break
        return hits


async def _api_search_request(
    name: str,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict | None = None,
):
    """Shared retry/error-handling for the API search providers (Firecrawl, Jina).

    Same policy as TavilySearchProvider: auth failures (401/403) and other 4xx are
    permanent and raise immediately; network errors, 429 and 5xx retry with
    exponential backoff; exhausting attempts raises SearchUnavailableError so the
    router can fall through to the next backend.
    """
    import httpx

    s = get_settings()
    attempts = max(1, s.search_max_retries)
    last_err = ""
    for attempt in range(1, attempts + 1):
        log.info("%s search attempt %d/%d (timeout=%.0fs)",
                 name, attempt, attempts, s.search_timeout_s)
        try:
            async with httpx.AsyncClient(timeout=s.search_timeout_s) as client:
                resp = await client.request(method, url, headers=headers, json=json_body)
        except Exception as e:
            # Network/timeout — transient, retry with backoff before giving up.
            last_err = f"{type(e).__name__}: {e}"
            log.warning("%s attempt %d failed (network): %s", name, attempt, last_err)
            if attempt < attempts:
                await asyncio.sleep(s.search_retry_base_s * (2 ** (attempt - 1)))
                continue
            raise SearchUnavailableError(
                f"Could not reach {name} after {attempts} attempt(s): {last_err}"
            ) from e

        if resp.status_code in (401, 403):
            raise SearchUnavailableError(
                f"{name} rejected the API key (HTTP {resp.status_code}).")
        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            log.warning("%s attempt %d got %s", name, attempt, last_err)
            if attempt < attempts:
                await asyncio.sleep(s.search_retry_base_s * (2 ** (attempt - 1)))
                continue
            raise SearchUnavailableError(
                f"{name} unavailable after {attempts} attempt(s): {last_err}")
        if resp.status_code >= 400:
            raise SearchUnavailableError(
                f"{name} returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp
    # Unreachable (loop either returns or raises), but keeps type checkers happy.
    raise SearchUnavailableError(f"{name} search failed: {last_err}")


class FirecrawlSearchProvider:
    """Firecrawl /v1/search API — search tuned for structured data extraction.

    Preferred by the router for STRUCTURED queries (tables / revenue / market-size
    figures): Firecrawl returns page content geared at downstream extraction rather
    than bare links. Needs CE_FIRECRAWL_API_KEY; the factory raises without one, so
    the router skips it in the fallback chain.
    """

    _ENDPOINT = "https://api.firecrawl.dev/v1/search"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    async def search(self, topic: str, *, max_results: int) -> list[SearchHit]:
        resp = await _api_search_request(
            "firecrawl", "POST", self._ENDPOINT,
            headers={"Authorization": f"Bearer {self._key}"},
            json_body={"query": topic, "limit": max_results},
        )
        hits: list[SearchHit] = []
        for r in resp.json().get("data", []):
            url = r.get("url", "")
            if not url:
                continue
            hits.append(SearchHit(
                url=url,
                title=r.get("title", ""),
                snippet=r.get("description", ""),
                domain=extract_domain(url),
            ))
        log.info("firecrawl search %r -> %d hit(s)", topic, len(hits))
        return hits


class JinaSearchProvider:
    """Jina Reader search — GET https://s.jina.ai/{query} with Accept: application/json.

    Works WITHOUT an API key (CE_JINA_API_KEY optional; a key just loosens the rate
    limit), which makes it a useful router fallback for CHINESE queries where the
    keyless DuckDuckGo endpoint is often blocked.
    """

    _ENDPOINT = "https://s.jina.ai/"

    def __init__(self, api_key: str | None = None) -> None:
        self._key = api_key or ""

    async def search(self, topic: str, *, max_results: int) -> list[SearchHit]:
        from urllib.parse import quote

        headers = {"Accept": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        resp = await _api_search_request(
            "jina", "GET", self._ENDPOINT + quote(topic, safe=""), headers=headers,
        )
        data = resp.json()
        rows = data.get("data", data if isinstance(data, list) else [])
        hits: list[SearchHit] = []
        for r in rows[:max_results]:
            url = r.get("url", "")
            if not url:
                continue
            hits.append(SearchHit(
                url=url,
                title=r.get("title", ""),
                snippet=r.get("description", "") or r.get("content", "")[:300],
                domain=extract_domain(url),
            ))
        log.info("jina search %r -> %d hit(s)", topic, len(hits))
        return hits


class HttpFetcher:
    """robots-aware HTTP fetcher with readability extraction.

    Deliberately conservative: honours robots.txt (configurable), sets a
    descriptive UA, hard timeout, and only returns extracted main text."""

    def __init__(self) -> None:
        self._s = get_settings()
        self._robots_cache: dict[str, bool] = {}

    async def _allowed(self, client, url: str) -> bool:
        if not self._s.scrape_respect_robots:
            return True
        from urllib.parse import urlsplit
        from urllib.robotparser import RobotFileParser

        parts = urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root in self._robots_cache:
            allowed_root = self._robots_cache[root]
            if not allowed_root:
                return False
        rp = RobotFileParser()
        try:
            # Short, dedicated timeout: a slow/absent robots.txt must not double the
            # per-page latency. If we can't read it quickly, the fail-open/fail-closed
            # policy (scrape_robots_fail_open) decides; fetches are still rate-limited
            # via the page fetch's own timeout.
            resp = await client.get(f"{root}/robots.txt",
                                    timeout=self._s.robots_timeout_s)
            rp.parse(resp.text.splitlines())
            ok = rp.can_fetch(self._s.scrape_user_agent, url)
        except Exception:
            # No robots.txt reachable -> permit or skip per scrape_robots_fail_open
            # (either way the page fetch itself is still rate-limited).
            ok = self._s.scrape_robots_fail_open
        self._robots_cache[root] = ok
        return ok

    async def fetch(self, url: str) -> str | None:
        # HARD per-URL ceiling: robots + page combined can never exceed this, so one
        # hanging server cannot stall the whole gather step. Degrades to None (skip).
        try:
            return await asyncio.wait_for(
                self._fetch_inner(url), timeout=self._s.scrape_per_url_timeout_s)
        except TimeoutError:
            log.warning("fetch timed out (>%.0fs) for %s — skipping",
                        self._s.scrape_per_url_timeout_s, url)
            return None
        except Exception as e:
            log.warning("fetch failed for %s: %s", url, e)
            return None

    async def _fetch_inner(self, url: str) -> str | None:
        import httpx

        headers = {"User-Agent": self._s.scrape_user_agent}
        try:
            async with httpx.AsyncClient(
                timeout=self._s.scrape_timeout_s, headers=headers, follow_redirects=True
            ) as client:
                if not await self._allowed(client, url):
                    log.info("robots.txt disallows %s", url)
                    return None
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            log.warning("fetch failed for %s: %s", url, e)
            return None
        return _extract_main_text(html)


def _extract_main_text(html: str) -> str:
    """Strip boilerplate to main content. Uses trafilatura if available, else a
    minimal tag-stripping fallback so the module works without the dep."""
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if text:
            return text
    except Exception:
        pass
    # Fallback: crude tag strip. Good enough for the fake path / tests.
    import re

    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------
# Factory + orchestration
# --------------------------------------------------------------------------
def get_search_provider(name: str | None = None) -> SearchProvider:
    """Instantiate a search backend. With `name=None` (default) the configured
    `settings.search_provider` is used — the historical behavior. With an explicit
    name the router instantiates chain backends by name; a backend whose required
    key is missing raises, so the router skips it and tries the next chain entry."""
    s = get_settings()
    chosen = name or s.search_provider
    if chosen == "duckduckgo":
        return DuckDuckGoSearchProvider()          # keyless, zero-setup default
    if chosen == "tavily":
        if not s.search_api_key:
            raise RuntimeError("CE_SEARCH_API_KEY required for tavily provider")
        return TavilySearchProvider(s.search_api_key)
    if chosen == "firecrawl":
        if not s.firecrawl_api_key:
            raise RuntimeError("CE_FIRECRAWL_API_KEY required for firecrawl provider")
        return FirecrawlSearchProvider(s.firecrawl_api_key)
    if chosen == "jina":
        return JinaSearchProvider(s.jina_api_key or None)   # key optional
    if chosen == "fake":
        return FakeSearchProvider()
    raise RuntimeError(f"Unknown search provider: {chosen}")


def get_fetcher() -> Fetcher:
    return HttpFetcher()


# Section-targeted research angles. One generic search on the raw topic cannot feed a
# 5-section industry report — policy directives, chain structure, market sizing, and
# competitor data each live on different pages. We expand the topic into these angles
# so every mandated section gets its own supporting material. The plain topic is always
# included so a fixture/provider keyed on the bare topic still matches.
_RESEARCH_ANGLES: tuple[str, ...] = (
    "{topic}",
    "{topic} industry overview market report",
    "{topic} government policy regulation subsidy tariff",
    "{topic} supply chain upstream midstream downstream suppliers",
    "{topic} market size TAM forecast revenue growth CAGR",
    "{topic} competitive landscape market share leading companies",
    # Advanced-search angles: actively target high-authority corpora rather than
    # relying on generic web results. `site:` operators steer Tavily/DDG toward
    # academic, government, and international-organization domains.
    "{topic} research study evidence site:nature.com OR site:science.org OR site:ncbi.nlm.nih.gov",
    "{topic} statistics official data site:.gov OR site:.edu OR site:who.int OR site:un.org",
    "{topic} peer-reviewed analysis findings",
)


def build_research_queries(topic: str) -> list[str]:
    """Expand a topic into section-targeted search queries (deduped, order-stable)."""
    seen: set[str] = set()
    out: list[str] = []
    for angle in _RESEARCH_ANGLES:
        q = angle.format(topic=topic).strip()
        low = q.lower()
        if q and low not in seen:
            seen.add(low)
            out.append(q)
    return out


def _topic_terms(topic: str) -> list[str]:
    """Meaningful lowercased terms from the topic (drop short stopwords)."""
    import re

    _STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with",
             "vs", "via", "&"}
    words = re.findall(r"[a-z0-9]+", topic.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP]


def _stem(word: str) -> str:
    """Very light suffix stemmer so morphological variants of a topic term count as the
    same term (e.g. 'batteries'/'battery', 'manufacturing'/'manufacture', 'suppliers'/
    'supply' partially). Not linguistically perfect — deliberately cheap and dependency-
    free — but enough to broaden the lexical match toward 'approximately related' words
    without any model or network call."""
    w = word
    # Order matters: strip the longest plausible suffix first.
    for suf in ("ization", "isation", "ational", "iveness", "ements", "ability",
                "ingly", "ement", "ности", " tions", "ities", "ional", "ances",
                "ences", "ments", "ical", "ing", "ers", "est", "ies", "ied",
                " ition", "ation", "ness", " ment", "ance", "ence", "able", " ible",
                "ize", "ise", "ous", "ive", "er", "ed", "ly", "es", "s"):
        suf = suf.strip()
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    # 'ies' -> 'y' style repair so 'batteries' and 'battery' converge on 'batter…y'.
    if word.endswith("ies") and len(word) > 4:
        w = word[:-3] + "y"
    return w


# Small curated set of near-synonym / related-term expansions. Keeps the match "semantic"
# in spirit — a page about 'automobiles' should count for a topic about 'cars' — while
# staying a cheap static lookup rather than an embeddings call. Bidirectional.
_RELATED_TERMS: dict[str, set[str]] = {
    "car": {"automobile", "vehicle", "auto"},
    "ev": {"electric", "battery"},
    "ai": {"artificial", "intelligence", "ml", "machine"},
    "cost": {"price", "pricing", "expense"},
    "revenue": {"sales", "turnover", "income"},
    "market": {"industry", "sector"},
    "growth": {"expansion", "increase", "rise"},
    "supply": {"supplier", "sourcing", "procurement"},
    "policy": {"regulation", "regulatory", "legislation", "law"},
    "company": {"firm", "corporation", "business", "enterprise"},
}


def _expand_terms(stems: set[str]) -> set[str]:
    """Grow a set of (stemmed) topic terms with their curated related terms, so a page
    that uses a synonym instead of the exact topic word still registers as relevant."""
    out = set(stems)
    for s in list(stems):
        for k, syns in _RELATED_TERMS.items():
            ks = _stem(k)
            syn_stems = {_stem(x) for x in syns}
            # If the topic term matches the key OR any synonym, pull the whole cluster in.
            if s == ks or s in syn_stems:
                out.add(ks)
                out.update(syn_stems)
    return out


def relevance_score(topic: str, body: str) -> tuple[float, int]:
    """Score how substantially `body` covers `topic`, reading the FULL body text.

    Returns (density, distinct_hits):
      - density: fraction of body tokens that are topic terms (superficial mentions
        in a long off-topic page score low; deep-dive articles score high).
      - distinct_hits: how many distinct topic terms appear (coverage of the topic,
        not just one word repeated).

    This is the 'high recall then high-precision filter' step: search casts wide,
    this discards pages that only name-drop the topic.
    """
    import re

    terms = _topic_terms(topic)
    if not terms:
        return 1.0, 0  # no usable terms -> don't filter on relevance
    tokens = re.findall(r"[a-z0-9]+", body.lower())
    if not tokens:
        return 0.0, 0

    # SEMANTIC-LEANING LEXICAL MATCH (broadened per requirement #2): instead of exact
    # whole-word equality, we (a) stem both the topic terms and the body tokens so
    # morphological variants match ('batteries' ~ 'battery'), (b) expand the topic terms
    # with curated related/synonym terms ('car' also matches 'automobile'/'vehicle'), and
    # (c) count a token as a hit if its stem shares a prefix with a topic stem (so
    # 'manufacturing' matches 'manufacture'). This lets "approximately related" pages
    # through without any model or network call.
    term_stems = {_stem(t) for t in terms}
    term_stems = _expand_terms(term_stems)

    def _matches(tok_stem: str) -> bool:
        if tok_stem in term_stems:
            return True
        # Prefix/substring overlap for near-variants, guarded by a length floor so short
        # tokens don't match everything.
        for ts in term_stems:
            if len(ts) >= 4 and (tok_stem.startswith(ts) or ts.startswith(tok_stem)):
                return True
        return False

    matched_stems: set[str] = set()
    hits = 0
    for t in tokens:
        ts = _stem(t)
        if _matches(ts):
            hits += 1
            matched_stems.add(ts)
    distinct = len(matched_stems)
    density = hits / len(tokens)
    return density, distinct


def _backend_name(provider) -> str:
    """Stable cache-key name for a provider instance ('tavily', 'duckduckgo', ...)."""
    return type(provider).__name__.lower().removesuffix("searchprovider")


def _freshness_score(fetched_at: str) -> float:
    """exp(-age in years) of the fetch timestamp; 0.5 (neutral) if unparseable.

    fetched_at is the time WE fetched the page (publication dates are not reliably
    extractable), so freshly gathered sources all score ~1.0; the term matters for
    cached/older Source objects entering the pool."""
    import math
    from datetime import UTC, datetime

    try:
        ts = datetime.fromisoformat(fetched_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_years = max(0.0, (datetime.now(UTC) - ts).total_seconds() / 86400.0 / 365.25)
        return math.exp(-age_years)
    except Exception:
        return 0.5


async def gather_sources(
    topic: str,
    provider: SearchProvider | None,
    fetcher: Fetcher,
    *,
    on_progress=None,
    llm=None,
) -> tuple[list[Source], list[SearchHit]]:
    """Full gather step: search -> STRICT filter -> fetch survivors -> extract.

    With `multi_query_research` on (default) this issues several SECTION-TARGETED
    queries and merges the authoritative survivors, so each mandated section has its
    own research to draw on rather than starving on a single generic search. Hits are
    deduped by URL across sub-queries; the global fetch cap (`scrape_max_pages`) still
    bounds total downloads.

    SEARCH ROUTING: when `provider` is None and `search_router_enabled` is on, each
    query goes through report/router.py (query-type classification + per-type backend
    fallback chain). An explicitly-passed provider always takes the single-backend
    path, byte-for-byte the historical behavior (this is how tests pin the fake).
    Results are cached on disk (SearchCache, keyed by backend+query) for the real
    builtin backends; fake/custom providers bypass the cache.

    RANKING: after the relevance filter, kept sources are ordered by the
    credibility-weighted score (relevance * rank_w_relevance + credibility.weight *
    rank_w_source + freshness * rank_w_freshness) so a weak (L4) source cannot
    outrank L1/L2 anchors in the downstream context.

    PERFORMANCE / STABILITY: searches run CONCURRENTLY, and fetches run CONCURRENTLY
    under a bounded semaphore (`scrape_concurrency`). Every fetch has a hard per-URL
    ceiling (see HttpFetcher.fetch), so one slow/hanging server can never stall the
    whole step. `on_progress(done, total, detail)` (optional) is called as pages
    complete so the UI keeps receiving events and its inactivity watchdog never trips.

    Returns (authoritative_sources, rejected_hits). The rejected list is kept for the
    audit trail so we can show WHY sources were dropped.
    """
    s = get_settings()
    queries = build_research_queries(topic) if s.multi_query_research else [topic]

    # Router mode only when the caller did NOT pin a provider. With routing off and
    # no provider given, fall back to the configured single provider.
    use_router = provider is None and s.search_router_enabled
    if provider is None and not use_router:
        provider = get_search_provider()
    # The disk cache is an optimization for the real networked backends; fakes and
    # custom test providers bypass it so tests never touch the user data dir.
    cacheable = provider is not None and isinstance(
        provider, (TavilySearchProvider, DuckDuckGoSearchProvider,
                   FirecrawlSearchProvider, JinaSearchProvider))
    cache = SearchCache() if (s.search_cache_ttl_s > 0 and (use_router or cacheable)) else None
    backend_name = _backend_name(provider) if provider is not None else ""

    def _emit(done: int, total: int, detail: str) -> None:
        if on_progress:
            try:
                on_progress(done, total, detail)
            except Exception:
                pass  # progress is best-effort, never break the gather

    # 1. Run all search angles CONCURRENTLY (each provider call has its own timeout and
    #    retries). A single angle that errors is isolated — SearchUnavailableError from a
    #    hard failure (bad key / blocked backend) still propagates so the pipeline can
    #    surface the real reason, but a lone transient miss just yields no hits.
    log.info("gather: searching %d angle(s) for %r", len(queries), topic)
    _emit(0, 0, f"Searching {len(queries)} query angle(s)…")

    async def _one_search(q: str):
        try:
            if use_router:
                # Lazy import: router.py imports this module (factory), so importing
                # it here avoids a module-level cycle.
                from core_engine.report.router import routed_search

                hits = await routed_search(
                    q, llm, max_results=s.scrape_max_results, cache=cache,
                    on_progress=lambda msg: _emit(0, 0, msg))
                return q, hits, None
            if cache is not None:
                cached = cache.get(backend_name, q)
                if cached is not None:
                    log.info("gather: cache hit for %r via %s", q, backend_name)
                    return q, cached, None
            hits = await provider.search(q, max_results=s.scrape_max_results)
            if cache is not None and hits:
                cache.put(backend_name, q, hits)
            return q, hits, None
        except SearchUnavailableError:
            raise  # hard failure — let it propagate (bad key / blocked backend)
        except (AttributeError, NameError, TypeError, KeyError, ImportError):
            # These are CODE bugs, not transient network misses. Do NOT swallow them
            # as "skip this angle" — that once masked a config-name typo as "no results
            # / out of scope". Let them propagate so the real error surfaces loudly.
            raise
        except Exception as e:  # genuinely transient (network/parse) — isolate this angle
            log.warning("gather: search angle %r failed (%s) — skipping",
                        q, type(e).__name__)
            return q, [], e

    search_results = await asyncio.gather(*[_one_search(q) for q in queries])

    # Merge KEPT hits by URL (dedupe pages found by multiple angles); collect rejections.
    kept_by_url: dict[str, SearchHit] = {}
    rejected: list[SearchHit] = []
    seen_rejected: set[str] = set()
    for q, hits, _err in search_results:
        kept, rej = filter_hits(hits)
        # Cap how many NEW pages any single angle contributes, so one broad query can't
        # crowd out the section-specific ones (breadth across sections > depth on one).
        added = 0
        for hit in kept:
            if hit.url in kept_by_url:
                continue
            if added >= s.max_pages_per_query:
                break
            kept_by_url[hit.url] = hit
            added += 1
        for r in rej:
            if r.url not in seen_rejected:
                seen_rejected.add(r.url)
                rejected.append(r)

    to_fetch = list(kept_by_url.values())[: s.scrape_max_pages]
    log.info("gather: %d queries -> %d candidate page(s) to fetch",
             len(queries), len(to_fetch))
    _emit(0, len(to_fetch), f"Found {len(to_fetch)} page(s); fetching…")

    # 2. Fetch survivors CONCURRENTLY (bounded), reading the FULL body and applying the
    #    relevance filter. Each fetch is hard-timeout-bounded, so the whole step finishes
    #    in ~ceil(pages / concurrency) * per_url_timeout at worst, not the sum.
    sources: list[Source] = []
    dropped_thin = dropped_irrelevant = 0
    sem = asyncio.Semaphore(max(1, s.scrape_concurrency))
    done = 0
    total = len(to_fetch)
    min_chars = s.relevance_min_body_chars if s.relevance_filter else 200

    # Defensive ceiling at the GATHER level too: we do not trust the fetcher to bound
    # itself. Even a custom/misbehaving fetcher that ignores its own timeout cannot
    # stall the pipeline — we cap slightly above the fetcher's per-URL ceiling so its
    # own (cleaner) timeout normally fires first.
    fetch_budget = s.scrape_per_url_timeout_s + 5.0

    async def _fetch_one(hit: SearchHit):
        async with sem:
            try:
                text = await asyncio.wait_for(
                    fetcher.fetch(hit.url), timeout=fetch_budget)
            except TimeoutError:
                log.warning("gather: fetch exceeded %.0fs for %s — skipping",
                            fetch_budget, hit.url)
                text = None
            except Exception as e:  # belt-and-suspenders; fetcher already guards
                log.warning("gather: fetch crashed for %s (%s)", hit.url, type(e).__name__)
                text = None
        return hit, text

    tasks = [asyncio.create_task(_fetch_one(h)) for h in to_fetch]
    densities: dict[str, float] = {}   # url -> relevance density (for ranking below)
    for coro in asyncio.as_completed(tasks):
        hit, text = await coro
        done += 1
        _emit(done, total, f"Fetched {done}/{total} page(s), kept {len(sources)}")
        if not text or len(text) < min_chars:
            rejected.append(hit)
            dropped_thin += 1
            continue
        kind = classify(hit.url)
        if kind is SourceKind.REJECTED:  # denylisted — never used
            rejected.append(hit)
            continue
        if s.relevance_filter:
            density, distinct = relevance_score(topic, text)
            if density < s.relevance_min_score or distinct < s.relevance_min_topic_hits:
                # Mentions the topic only superficially — discard (high-precision step).
                log.info("relevance drop %s (density=%.3f distinct=%d)",
                         hit.url, density, distinct)
                rejected.append(hit)
                dropped_irrelevant += 1
                continue
            densities[hit.url] = density
        sources.append(Source(
            url=hit.url, domain=extract_domain(hit.url), title=hit.title,
            kind=kind, text=text,
            credibility=credibility_for(hit.url, kind),
        ))

    # CREDIBILITY-WEIGHTED RANKING: order the kept sources by
    #   relevance * rank_w_relevance + credibility.weight * rank_w_source
    #   + freshness * rank_w_freshness
    # (stable sort; density 0.0 when the relevance filter is off). A high-relevance
    # but weak-trust (L4) source can no longer outrank L1/L2 anchors, and the
    # scrape_max_pages cap is applied on the RANKED list.
    def _rank_score(src: Source) -> float:
        return (densities.get(src.url, 0.0) * s.rank_w_relevance
                + src.credibility.weight * s.rank_w_source
                + _freshness_score(src.fetched_at) * s.rank_w_freshness)

    sources.sort(key=_rank_score, reverse=True)
    sources = sources[: s.scrape_max_pages]
    log.info("gather: %d queries -> %d source(s) kept, %d rejected "
             "(%d thin, %d irrelevant)",
             len(queries), len(sources), len(rejected), dropped_thin, dropped_irrelevant)
    return sources, rejected
