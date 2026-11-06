"""Evidence dimensions and conservative origin grouping, never a truth probability."""
from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from core_engine.report.models import Claim, SentenceCitation, Source
from core_engine.report.scholarly import normalize_doi
from core_engine.report.text import lexical_tokens, normalize_text


def work_identifier(source: Source) -> str | None:
    doi = None
    parts = urlsplit(source.url)
    if parts.hostname in {"doi.org", "dx.doi.org"}:
        doi = normalize_doi(unquote(parts.path.lstrip("/")))
    elif parts.hostname == "api.crossref.org" and parts.path.startswith("/works/"):
        doi = normalize_doi(unquote(parts.path[len("/works/"):]))
    if doi:
        return "doi:" + doi
    if parts.hostname in {"arxiv.org", "export.arxiv.org"}:
        return "arxiv:" + re.sub(r"v\d+$", "", parts.path.removeprefix("/abs/").removeprefix("/pdf/").removesuffix(".pdf"))
    if parts.hostname and (parts.hostname == "nature.com" or parts.hostname.endswith(".nature.com")) and "/articles/" in parts.path:
        slug = parts.path.removesuffix(".pdf").rstrip("/").split("/")[-1]
        return "doi:10.1038/" + slug if re.fullmatch(r"s\d{5}-\d{3}-[a-z0-9-]+", slug) else "nature:" + slug
    doi = normalize_doi(str(source.metadata.get("doi", "")))
    if doi:
        return "doi:" + doi
    from core_engine.report.openalex import api_work_id, work_id

    identifier = api_work_id(source.url) or work_id(source.metadata.get("openalex_id"))
    return "openalex:" + identifier if identifier else None


def _work_keys(source: Source) -> set[str]:
    from core_engine.report.openalex import work_id

    identity = work_identifier(source)
    result = {identity} if identity else set()
    # Registry-declared links are conservatively merged to avoid counting a PDF
    # and its abstract as independent. A conflicting document DOI takes precedence.
    declared = normalize_doi(str(source.metadata.get("discovery_doi", "")))
    if identity and identity.startswith("doi:") and declared and identity != "doi:" + declared:
        return result
    identifier = work_id(source.metadata.get("openalex_id"))
    discovered = (work_id(source.metadata.get("discovery_openalex_id"))
                  if source.metadata.get("discovered_via") == "openalex" else None)
    for value in (identifier, discovered):
        if value:
            result.add("openalex:" + value)
    if discovered and declared:
        result.add("doi:" + declared)
    return result


def origin_groups(sources: list[Source]) -> dict[str, str]:
    """Merge same-work and near-identical copies; a different URL is not corroboration.

    Groups estimate provenance, not independent experiments, ownership or datasets.
    """
    parents = list(range(len(sources)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    identities = [_work_keys(s) for s in sources]
    text = [normalize_text(s.text) for s in sources]
    shingles = []
    for source in sources:
        tokens = lexical_tokens(source.text[:100000])
        shingles.append({tuple(tokens[i:i + 5]) for i in range(len(tokens) - 4)})
    for i in range(len(sources)):
        for j in range(i):
            same_work = bool(identities[i] & identities[j])
            exact = bool(text[i]) and text[i] == text[j]
            a, b = shingles[i], shingles[j]
            copied = (len(a) >= 40 and len(b) >= 40 and len(a & b) / max(1, len(a | b)) >= .85)
            if same_work or exact or copied:
                parents[root(i)] = root(j)
    names: dict[int, str] = {}
    result = {}
    for i, source in enumerate(sources):
        names.setdefault(root(i), f"origin-{len(names) + 1}")
        result[source.url] = names[root(i)]
    return result


def assess_evidence(claims: list[Claim], sources: list[Source], citations: list[SentenceCitation]) -> dict:
    groups = origin_groups(sources)
    by_url = {s.url: s for s in sources}
    for claim in claims:
        urls = sorted({c.source_url for c in citations if c.claim_id == claim.id and c.source_url in by_url})
        count = len({groups[url] for url in urls})
        caveats = []
        if count == 1:
            caveats.append("single_origin_not_independently_confirmed")
        if len(urls) > count:
            caveats.append("same_work_or_republication_not_additional_corroboration")
        fulltext_supported = any(by_url[url].metadata.get("full_text_accessed") for url in urls)
        if any(by_url[url].document_format == "abstract" for url in urls) and not fulltext_supported:
            caveats.append("abstract_only_methods_and_full_results_not_checked")
        if any(by_url[url].metadata.get("source_role") == "aggregated_abstract" for url in urls) and not fulltext_supported:
            caveats.append("aggregated_abstract_original_publication_not_directly_checked")
        if any("arxiv.org" in (urlsplit(url).hostname or "") or by_url[url].metadata.get("publication_status") == "preprint" for url in urls):
            caveats.append("preprint_peer_review_not_established")
        if any(by_url[url].metadata.get("source_role") == "self_report" for url in urls):
            caveats.append("interested_party_self_report")
        claim.evidence_quality = {
            "source_support": "anchored" if urls else "unanchored", "supporting_urls": urls,
            "origin_group_count": count, "supporting_url_count": len(urls),
            "corroboration": "multiple_origins" if count > 1 else "single_origin" if count else "none",
            "truth_status": "not_established", "caveats": caveats,
            "model_review_flags": [flag for flag in claim.evidence_quality.get("model_review_flags", [])
                                   if flag in {"internal_consistency_attention", "outlier_attention", "source_disagreement_attention"}],
        }
    return {"source_count": len(sources), "origin_group_count": len(set(groups.values())),
            "groups": groups, "independence_is_not_proven": True,
            "claims": {c.id: c.evidence_quality for c in claims},
            "truth_probability": None}
