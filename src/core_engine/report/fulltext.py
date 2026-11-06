"""Open-copy discovery and structured article parsing. Links are leads, never evidence.

Every copy is read through the existing robots/public-network boundary and keeps
its actual URL. No paywall, redirect or access-control bypass is attempted.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

from core_engine.report.models import SearchHit, Source
from core_engine.report.scholarly import FetchedDocument, normalize_doi

_EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/"


def europe_pmc_lookup(doi: str) -> str | None:
    identifier = normalize_doi(doi)
    return (_EUROPE_PMC + "search?" + urlencode({"query": "DOI:" + identifier, "format": "json", "pageSize": "1"})
            if identifier else None)


def europe_pmc_lookup_doi(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "www.ebi.ac.uk" or parts.path != "/europepmc/webservices/rest/search" or parts.fragment:
        return None
    query = parse_qs(parts.query)
    if set(query) != {"query", "format", "pageSize"} or query["format"] != ["json"] or query["pageSize"] != ["1"]:
        return None
    values = query["query"]
    return normalize_doi(values[0][4:]) if len(values) == 1 and values[0].startswith("DOI:") else None


def europe_pmc_fulltext(payload: object, doi: str) -> str | None:
    """A matching OA record is a download lead, never scientific evidence."""
    if not isinstance(payload, dict) or not isinstance(payload.get("resultList"), dict):
        return None
    records = payload["resultList"].get("result", [])
    for row in records[:5] if isinstance(records, list) else []:
        if (isinstance(row, dict) and row.get("isOpenAccess") == "Y"
                and normalize_doi(str(row.get("doi", ""))) == normalize_doi(doi)
                and isinstance(row.get("pmcid"), str) and re.fullmatch(r"PMC[1-9]\d{0,14}", row["pmcid"])):
            return _EUROPE_PMC + row["pmcid"] + "/fullTextXML"
    return None


def public_copy_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 4000:
        return None
    try:
        parts = urlsplit(value)
        if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
                or parts.port not in {None, 443} or parts.hostname.lower() in {"doi.org", "dx.doi.org"}
                or re.search(r"\.(?:png|jpe?g|gif|webp|svg|ico|css|js)$", parts.path, re.IGNORECASE)
                or any(key in parts.query.lower() for key in ("api_key=", "access_token=", "token="))):
            return None
    except ValueError:
        return None
    return value


def fulltext_candidates(sources: list[Source], per_work: int) -> list[SearchHit]:
    """Use locations from fetched records, including repositories beyond the best copy."""
    leads: list[SearchHit] = []
    seen = {s.url for s in sources}
    fulfilled = {s.metadata.get("fulltext_parent_url") for s in sources if s.metadata.get("full_text_accessed")}
    fulltext_dois = {normalize_doi(str(s.metadata.get("doi") or "")) for s in sources if s.metadata.get("full_text_accessed")}
    for source in sources:
        if source.document_format != "abstract" or source.url in fulfilled:
            continue
        doi = normalize_doi(str(source.metadata.get("doi") or ""))
        if doi and doi in fulltext_dois:
            continue
        locations = source.metadata.get("locations", [])
        if not isinstance(locations, list):
            continue
        locations = [loc for loc in locations if isinstance(loc, dict) and loc.get("is_oa") is True]
        # Try direct open PDFs and repositories before publisher landing pages.
        locations.sort(key=lambda loc: (not bool(loc.get("pdf_url")),
                                        loc.get("version") == "submittedVersion"))
        added = 0
        for location in locations:
            for field in ("pdf_url", "landing_page_url"):
                url = public_copy_url(location.get(field))
                if not url or url in seen or added >= per_work:
                    continue
                seen.add(url)
                added += 1
                leads.append(SearchHit(url, source.title, "Open full-text copy; download and identity check required",
                    urlsplit(url).hostname or "", {"discovered_via": "openalex",
                        "discovery_openalex_id": source.metadata.get("openalex_id"),
                        "discovery_publication_type": source.metadata.get("publication_type"),
                        "discovery_doi": source.metadata.get("doi"), "discovery_location": location,
                        "fulltext_parent_url": source.url, "require_fulltext": True}))
        # Reserve one fallback for an exact DOI-to-OA lookup, rather than using
        # every slot on publisher copies that may share the same access failure.
        lookup = europe_pmc_lookup(str(source.metadata.get("doi") or ""))
        if lookup and lookup not in seen and (per_work > 1 or added == 0) and per_work > 0:
            start = len(leads) - added
            if added >= per_work:
                leads.pop()
            seen.add(lookup)
            leads.insert(start + min(1, added), SearchHit(lookup, source.title,
                "Exact DOI open-copy lookup; metadata alone is not full text", "www.ebi.ac.uk", {
                    "discovered_via": "openalex" if source.metadata.get("openalex_id") else "crossref",
                    "discovery_openalex_id": source.metadata.get("openalex_id"), "discovery_doi": source.metadata["doi"],
                    "discovery_publication_type": source.metadata.get("publication_type"),
                    "fulltext_parent_url": source.url, "require_fulltext": True, "fulltext_resolver": "europe_pmc"}))
    return leads


class _ArticleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.metadata: dict[str, str] = {}
        self.pdf_links: list[str] = []

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        href = data.get("href", "")
        if tag in {"a", "link"} and isinstance(href, str) and (data.get("type") == "application/pdf"
                or re.search(r"/(?:bitstream|bitstreams)/", href)):
            self.pdf_links.append(href)
        if tag == "meta":
            name = str(data.get("name") or data.get("property") or "").lower()
            if name.startswith("citation_") and isinstance(data.get("content"), str):
                self.metadata[name] = data["content"]


def article_metadata(html: str, url: str) -> dict:
    parser = _ArticleHTML()
    parser.feed(html)
    meta = parser.metadata
    pdf = public_copy_url(urljoin(url, meta.get("citation_pdf_url", ""))) if meta.get("citation_pdf_url") else None
    if not pdf:
        pdf = next((candidate for href in parser.pdf_links
                    if (candidate := public_copy_url(urljoin(url, href)))), None)
    return {"doi": normalize_doi(meta.get("citation_doi", "")),
            "title": meta.get("citation_title"), "article_pdf_url": pdf,
            "article_abstract": meta.get("citation_abstract"),
            "published_date_declared": meta.get("citation_publication_date")}


def html_article_document(html: str, url: str, text: str | None) -> FetchedDocument | None:
    """A landing page's reference table is not the paper's body or abstract."""
    from core_engine.report.scholarly import abstract_text

    metadata = article_metadata(html, url)
    abstract = metadata.pop("article_abstract", None)
    if not (metadata.get("title") or metadata.get("doi")):
        return None
    metadata["actual_url"] = url
    sections = re.findall(r"<(?:h[1-6])\b[^>]*>.*?(?:introduction|methods|results|discussion|conclusion).*?</h[1-6]>",
                          html, re.IGNORECASE | re.DOTALL)
    metadata["full_text_accessed"] = bool(text and len(text) >= 4000 and len(sections) >= 2)
    # Prefer an explicitly declared abstract on a non-fulltext landing page.
    # Generic SEO descriptions are not an abstract, and index titles are not claims.
    if not metadata["full_text_accessed"] and isinstance(abstract, str) and 80 <= len(abstract) <= 30000:
        body = abstract_text(abstract)
        if len(body) >= 80:
            metadata["source_role"] = "article_declared_abstract"
            return FetchedDocument(body, "abstract", metadata=metadata)
    if not text:
        return None
    metadata["source_role"] = "article_fulltext" if metadata["full_text_accessed"] else "article_landing_page"
    return FetchedDocument(text, "html", metadata=metadata)


def jats_document(raw: str) -> FetchedDocument | None:
    # No entity expansion/external access. Reject declarations before parsing.
    if re.search(r"<!ENTITY", raw, re.IGNORECASE):
        return None
    # Common JATS PUBLIC/SYSTEM declarations are labels; do not load their DTDs.
    # Internal subsets and entity declarations are refused outright.
    raw = re.sub(r'<!DOCTYPE\s+article\s+(?:PUBLIC\s+"[^"]*"\s+"[^"]*"|SYSTEM\s+"[^"]*")\s*>',
                 "", raw, flags=re.IGNORECASE)
    if re.search(r"<!DOCTYPE", raw, re.IGNORECASE):
        return None
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None
    if root.tag.split("}")[-1] != "article":
        return None
    body = next((el for el in root.iter() if el.tag.split("}")[-1] == "body"), None)
    if body is None:
        return None  # Abstract-only XML must not masquerade as full text.
    from core_engine.report.methods import MAX_METHOD_CHARS, MAX_METHOD_SPANS, jats_methods_heading
    from core_engine.report.tables import append_tables, extract_tables

    try:
        tables = extract_tables(ET.tostring(body, encoding="unicode"), "jats")
    except Exception:
        tables = []  # Optional structure extraction must not discard readable prose.
    original_tables = {id(el): index for index, el in enumerate(
        (el for el in body.iter() if el.tag.split("}")[-1] == "table"), 1)}
    represented = {index for table in tables for index in
                   [table["locator"]["table_index"], *table.get("_duplicate_table_indices", [])]}

    def prose(element, section=None):
        tag = element.tag.split("}")[-1]
        if tag == "table":
            if original_tables.get(id(element)) not in represented:
                # Unsupported/oversized table layouts keep the previous plain
                # text fallback, without inventing a cell interpretation.
                yield element, " ".join(element.itertext()).strip(), None
            return
        if tag == "sec":
            title_node = next((child for child in element if child.tag.split("}")[-1] == "title"), None)
            heading = " ".join("".join(title_node.itertext()).split()) if title_node is not None else ""
            if section is not None or (title_node is not None and jats_methods_heading(heading, element.get("sec-type", ""))):
                root_heading = section["root_heading"] if section is not None else {"node": title_node}
                section = {"section_id": element.get("id"), "section_title": heading[:240],
                           "root_heading": root_heading}
        elif tag in {"table-wrap", "fig", "caption", "fn", "ref-list", "supplementary-material"}:
            section = None
        if tag in {"title", "p"}:
            yield element, " ".join(element.itertext()).strip(), section
        else:
            for child in element:
                yield from prose(child, section)

    parts: list[str] = []
    methods_spans: list[dict] = []
    offset, methods_complete = 0, True
    for index, (element, fragment, section) in enumerate(prose(body), 1):
        start = offset + (2 if parts else 0)
        end = start + len(fragment)
        parts.append(fragment[:max(0, 200000 - start)])
        offset = end
        if end > 200000:
            methods_complete = False
            break
        if section is None or not fragment:
            continue
        root_heading = section["root_heading"]
        root_heading["section_end"] = end
        if element.tag.split("}")[-1] == "title":
            if element is root_heading["node"]:
                root_heading.update(text=fragment, start=start, end=end)
            continue
        if len(methods_spans) >= MAX_METHOD_SPANS or len(fragment) > MAX_METHOD_CHARS:
            methods_complete = False
            continue
        refs = [{"id": ref.get("rid", ""), "text": " ".join(ref.itertext()).strip()}
                for ref in element.iter() if ref.tag.split("}")[-1] == "xref" and ref.get("ref-type") == "table"]
        if len(refs) > MAX_METHOD_SPANS:
            methods_complete = False
            continue
        methods_spans.append({"start": start, "end": end, "text": fragment,
            "locator": {"format": "jats", "section_id": section["section_id"], "section_title": section["section_title"],
                        "prose_index": index, "paragraph_id": element.get("id"), "methods_heading": root_heading},
            "table_refs": refs})
    for span in methods_spans:
        locator = span["locator"]
        locator["methods_heading"] = {key: value for key, value in locator["methods_heading"].items() if key != "node"}
    text = "\n\n".join(parts)[:200000]
    text, tables = append_tables(text, tables)
    if len(text) < 500:
        return None
    doi = next((normalize_doi("".join(el.itertext())) for el in root.iter()
                if el.tag.split("}")[-1] == "article-id" and el.get("pub-id-type") == "doi"), None)
    title = next((el for el in root.iter() if el.tag.split("}")[-1] == "article-title"), None)
    return FetchedDocument(text, "jats", metadata={"doi": doi,
        "title": "".join(title.itertext()) if title is not None else None,
        "article_type_declared": root.get("article-type"),
        "full_text_accessed": True, "source_role": "article_fulltext",
        **({"tables": tables} if tables else {}),
        **({"methods_spans": methods_spans, "methods_spans_complete": methods_complete} if methods_spans else {})})


def identity_matches(source: Source, hit: SearchHit) -> bool:
    """A conflicting declared DOI is a rejected copy, never merged as corroboration."""
    expected = normalize_doi(str(hit.metadata.get("discovery_doi") or ""))
    actual = normalize_doi(str(source.metadata.get("doi") or ""))
    return not (expected and actual and expected != actual)
