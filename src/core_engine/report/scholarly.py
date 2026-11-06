"""Bounded Crossref discovery and explicitly labelled publisher-deposited abstracts.

Registry metadata confirms a record exists, not that its scientific claims are true,
peer-reviewed, replicated, unretracted, or representative of commercial batteries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import quote, urlencode, urlsplit

from core_engine.report.models import SearchHit


@dataclass
class FetchedDocument:
    text: str
    document_format: str
    published_at: str | None = None
    metadata: dict = field(default_factory=dict)


class _AbstractText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"p", "jats:p", "title", "jats:title"}:
            self.parts.append("\n\n")


def abstract_text(value: str) -> str:
    parser = _AbstractText()
    parser.feed(value)
    return " ".join("".join(parser.parts).split())


def normalize_doi(value: str) -> str | None:
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/|^doi:\s*", "", value.strip(), flags=re.IGNORECASE).lower()
    return value if re.fullmatch(r"10\.\d{4,9}/[^\s<>]+", value) and len(value) <= 250 else None


def _list(value) -> list:
    return value if isinstance(value, list) else []


def crossref_document(record: dict, expected_doi: str) -> FetchedDocument | None:
    doi = normalize_doi(str(record.get("DOI", "")))
    if doi != normalize_doi(expected_doi):
        return None
    abstract = record.get("abstract")
    if not isinstance(abstract, str) or not abstract.strip():
        return None  # A bibliographic record/title is a lead, not evidence of a result.
    kind = record.get("type", "unknown")
    title = " ".join(t for t in _list(record.get("title")) if isinstance(t, str))
    if re.match(r"retraction[: ]|retracted[: ]|撤稿", title, re.IGNORECASE):
        return None  # Notices must not be presented as an ordinary study abstract.
    dates = record.get("published") or record.get("published-online") or record.get("published-print") or {}
    date_parts = _list(dates.get("date-parts")) if isinstance(dates, dict) else []
    parts = _list(date_parts[0]) if date_parts else []
    published = ("-".join(str(part) if i == 0 else f"{part:02d}" for i, part in enumerate(parts[:3]))
                 if parts and all(type(part) is int for part in parts[:3]) else None)
    return FetchedDocument(abstract_text(abstract), "abstract", published, {
        "doi": doi, "title": title, "publisher": record.get("publisher"),
        "authors": [" ".join(str(a.get(k, "")) for k in ("given", "family")).strip()
                    for a in _list(record.get("author")) if isinstance(a, dict)][:50],
        "publication_type": kind, "publication_status": "preprint" if kind == "posted-content" else "review_status_unknown",
        "source_role": "publisher_deposited_abstract", "full_text_accessed": False,
        "update_metadata_present": bool(record.get("update-to") or record.get("relation")),
        "retraction_status": "not_determined", "landing_url": record.get("URL"),
        "licenses": [item["URL"] for item in _list(record.get("license")) if isinstance(item, dict) and isinstance(item.get("URL"), str)],
    })


class CrossrefSearchProvider:
    async def search(self, topic: str, *, max_results: int) -> list[SearchHit]:
        import httpx

        from core_engine.config import get_settings
        from core_engine.security.network import read_public_text

        settings = get_settings()
        if not settings.scholarly_search_enabled or max_results <= 0:
            return []
        params = {"query.bibliographic": topic, "rows": str(min(max_results, 8)),
                  "filter": "type:posted-content" if re.search(r"preprint|预印本", topic, re.IGNORECASE) else "type:journal-article"}
        if settings.scholarly_mailto:
            params["mailto"] = settings.scholarly_mailto
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            response = await read_public_text(client, "https://api.crossref.org/works?" + urlencode(params),
                                              max_bytes=3 * 1024 * 1024, timeout=settings.scrape_timeout_s)
        hits = []
        for row in response.json().get("message", {}).get("items", []):
            if not isinstance(row, dict):
                continue
            doi = normalize_doi(str(row.get("DOI", "")))
            if not doi:
                continue
            title = " ".join(str(t) for t in _list(row.get("title"))) or doi
            if row.get("abstract"):
                hits.append(SearchHit("https://api.crossref.org/works/" + quote(doi, safe=""), title,
                                      "Publisher-deposited abstract; not accessed full text", "api.crossref.org"))
            resource = row.get("resource") or {}
            primary_record = resource.get("primary") if isinstance(resource, dict) else None
            primary = primary_record.get("URL", "") if isinstance(primary_record, dict) else ""
            if isinstance(primary, str) and urlsplit(primary).scheme == "https":
                hits.append(SearchHit(primary, title, "Publisher full-text candidate; fetch required", urlsplit(primary).hostname or ""))
            for link in _list(row.get("link")):
                if not isinstance(link, dict):
                    continue
                url = link.get("URL", "")
                if link.get("content-type") == "application/pdf" and urlsplit(url).scheme == "https":
                    hits.append(SearchHit(url, title, "Full-text candidate; must be fetched and verified", urlsplit(url).hostname or ""))
            if len(hits) >= max_results:
                break
        return hits[:max_results]
