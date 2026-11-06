"""Bounded OpenAlex discovery; reconstructed abstracts are labelled, not full text.

Bibliographic and citation records are discovery leads, never independent proof of
a scientific result. A work is fetched again before its abstract becomes evidence.
"""
from __future__ import annotations

import asyncio
import re
from datetime import date
from urllib.parse import urlencode, urlsplit

import httpx

from core_engine.config import get_settings
from core_engine.report.discovery import equivalent_terms
from core_engine.report.models import SearchHit
from core_engine.report.scholarly import FetchedDocument, normalize_doi
from core_engine.security.network import read_public_text

_WORK_ID = re.compile(r"W[1-9]\d{0,19}", re.IGNORECASE)
_FIELDS = ("id,doi,title,type,publication_date,authorships,abstract_inverted_index,"
           "primary_location,best_oa_location,locations,is_retracted")


class _InvalidOpenAlexResponse(ValueError):
    """The remote response does not conform to the expected record envelope."""


def work_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.startswith("https://openalex.org/"):
        value = value.removeprefix("https://openalex.org/")
    return value.upper() if _WORK_ID.fullmatch(value) else None


def api_work_id(url: str) -> str | None:
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc.lower() != "api.openalex.org"
            or parts.query or parts.fragment or not parts.path.startswith("/works/")):
        return None
    return work_id(parts.path.removeprefix("/works/"))


def reconstruct_abstract(index: object) -> str | None:
    """Reject holes, conflicting positions and oversized indexes instead of inventing text."""
    if not isinstance(index, dict) or not index or len(index) > 10000:
        return None
    words: dict[int, str] = {}
    characters = 0
    for token, positions in index.items():
        if (not isinstance(token, str) or not token.strip() or len(token) > 2000
                or not isinstance(positions, list) or not positions):
            return None
        for position in positions:
            if (type(position) is not int or not 0 <= position < 10000
                    or position in words):
                return None
            words[position] = token
            characters += len(token) + 1
            if characters > 100000:
                return None
    if set(words) != set(range(len(words))):
        return None
    return " ".join(words[i] for i in range(len(words)))


def _locations(record: dict) -> list[dict]:
    candidates = [record.get("best_oa_location"), record.get("primary_location")]
    others = record.get("locations")
    candidates += others[:10] if isinstance(others, list) else []
    result: list[dict] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        clean = {key: item.get(key) for key in
                 ("landing_page_url", "pdf_url", "version", "license", "is_oa")}
        if clean not in result:
            result.append(clean)
    return result[:10]


def _ordinary_work(record: dict) -> bool:
    title = record.get("title")
    kind = record.get("type")
    return bool(isinstance(title, str) and title.strip()
                and (record.get("is_retracted") is None or record.get("is_retracted") is False)
                and (kind is None or isinstance(kind, str))
                and kind not in {"paratext", "retraction", "erratum"}
                and not re.match(r"retraction[: ]|retracted[: ]|撤稿", title, re.IGNORECASE))


def openalex_document(record: object, expected_id: str) -> FetchedDocument | None:
    if not isinstance(record, dict) or work_id(record.get("id")) != work_id(expected_id):
        return None
    if not work_id(expected_id) or not _ordinary_work(record):
        return None
    text = reconstruct_abstract(record.get("abstract_inverted_index"))
    if not text:
        return None  # A title, citation count or OA link alone is not a result.
    primary = record.get("primary_location")
    primary = primary if isinstance(primary, dict) else {}
    authorships = record.get("authorships")
    authors = []
    for item in authorships[:50] if isinstance(authorships, list) else []:
        author = item.get("author") if isinstance(item, dict) else None
        if isinstance(author, dict) and isinstance(author.get("display_name"), str):
            authors.append(author["display_name"])
    published = record.get("publication_date")
    try:
        published = date.fromisoformat(published).isoformat() if isinstance(published, str) else None
    except ValueError:
        published = None
    return FetchedDocument(text, "abstract", published, {
        "openalex_id": work_id(expected_id), "doi": normalize_doi(str(record.get("doi", ""))),
        "title": record["title"], "authors": authors, "publication_type": record.get("type"),
        "publication_status": "preprint" if (record.get("type") == "preprint"
            or primary.get("version") == "submittedVersion") else "review_status_unknown",
        "source_role": "aggregated_abstract", "full_text_accessed": False,
        "abstract_reconstruction": "openalex_token_positions",
        "abstract_inverted_index": record["abstract_inverted_index"],
        "retraction_status": ("not_flagged_by_openalex_not_independently_checked"
                              if record.get("is_retracted") is False else "not_determined"),
        "landing_url": primary.get("landing_page_url"), "locations": _locations(record),
        # A location's licence describes that copy, not automatically this abstract.
        "licenses": [], "abstract_license": "not_determined",
    })


def search_query(topic: str) -> str:
    # Curated query-only translations; retain years and the rest of the question.
    aliases = equivalent_terms(topic)
    if aliases:
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(r"[一-鿿]", alias):
                topic = topic.replace(alias, aliases[0])
    return topic.strip()[:500]


def _public_candidate(url: object) -> str | None:
    if not isinstance(url, str) or len(url) > 4000:
        return None
    try:
        parts = urlsplit(url)
        if (parts.scheme != "https" or not parts.hostname or parts.username
                or parts.password or parts.port not in {None, 443}):
            return None
        if parts.hostname.lower() in {"doi.org", "dx.doi.org"}:
            return None  # Resolver-only links cross hosts and contain no paper text.
        if any(key in parts.query.lower() for key in ("api_key=", "access_token=", "token=")):
            return None
    except ValueError:
        return None
    # The fetch boundary still performs public DNS validation and pins the IP.
    return url


def search_hits(records: object, limit: int) -> list[SearchHit]:
    if not isinstance(records, list):
        raise _InvalidOpenAlexResponse("Invalid OpenAlex results envelope")
    abstracts: list[SearchHit] = []
    first: list[SearchHit] = []
    additional: list[SearchHit] = []
    seen: set[str] = set()
    for record in records[:8]:
        if not isinstance(record, dict) or not _ordinary_work(record):
            continue
        identifier = work_id(record.get("id"))
        if not identifier:
            continue
        metadata = {"discovered_via": "openalex", "discovery_openalex_id": identifier,
                    "discovery_doi": normalize_doi(str(record.get("doi", "")))}
        candidates = []
        if reconstruct_abstract(record.get("abstract_inverted_index")):
            candidates.append(SearchHit(f"https://api.openalex.org/works/{identifier}", record["title"],
                "OpenAlex aggregated abstract; per-work fetch required; not full text", "api.openalex.org", dict(metadata)))
        for location in _locations(record):
            for field in ("pdf_url", "landing_page_url"):
                url = _public_candidate(location.get(field))
                if url:
                    candidates.append(SearchHit(url, record["title"], "Full-text candidate; must be fetched and checked",
                        urlsplit(url).hostname or "", {**metadata, "discovery_location": location,
                            "discovery_publication_type": record.get("type")}))
        unique = []
        for hit in candidates:
            if hit.url not in seen:
                seen.add(hit.url)
                unique.append(hit)
        if unique:
            if api_work_id(unique[0].url):
                abstracts.append(unique[0])
            else:
                first.append(unique[0])
            additional.extend(unique[1:2])  # At most two leads per work; prefer work diversity.
    return (abstracts + first + additional)[:max(0, limit)]


class OpenAlexSearchProvider:
    async def search(self, topic: str, *, max_results: int) -> list[SearchHit]:
        from core_engine.report.scrape import SearchUnavailableError

        settings = get_settings()
        if not settings.openalex_search_enabled or max_results <= 0 or not topic.strip():
            return []
        limit = min(max_results, settings.openalex_max_results, 8)
        params = {"search": search_query(topic), "per_page": str(limit), "select": _FIELDS}
        headers = {"User-Agent": settings.scrape_user_agent}
        if settings.openalex_api_key:
            headers["Authorization"] = "Bearer " + settings.openalex_api_key

        async def request():
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False, headers=headers) as client:
                for attempt in range(2):
                    try:
                        response = await read_public_text(client, "https://api.openalex.org/works?" + urlencode(params),
                            max_bytes=3 * 1024 * 1024, timeout=settings.openalex_timeout_s)
                        data = response.json()
                        if not isinstance(data, dict) or "results" not in data:
                            raise _InvalidOpenAlexResponse("Invalid OpenAlex response envelope")
                        return search_hits(data["results"], limit)
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        if (status == 429 or status >= 500) and attempt == 0:
                            await asyncio.sleep(0.5)
                            continue
                        raise SearchUnavailableError(f"OpenAlex unavailable (HTTP {status}); check quota/key") from None
            return []

        try:
            return await asyncio.wait_for(request(), timeout=settings.openalex_timeout_s)
        except SearchUnavailableError:
            raise
        except (TimeoutError, httpx.HTTPError, ValueError, OSError) as exc:
            # Credentials are headers, never URLs, cache entries or audit messages.
            raise SearchUnavailableError(f"OpenAlex unavailable ({type(exc).__name__})") from None
