"""Retain candidates; publish qualified, source-bound statements rather than presumed truths."""
from __future__ import annotations

import re
from decimal import Decimal

from core_engine.report.evidence_quality import origin_groups
from core_engine.report.models import Claim, ConfidenceTier, SentenceCitation, Source
from core_engine.report.text import normalize_text, numbers_supported


def context_differences(a: Claim, b: Claim) -> dict[str, list[str]]:
    """Explicit scope differences preclude a same-condition contradiction.

    This deliberately detects differences, not completeness: missing conditions
    never prove comparability or independent replication.
    """
    result = {}
    for key in ("entity", "attribute", "qualifier", "time_scope"):
        x, y = getattr(a, key), getattr(b, key)
        if (x or y) and normalize_text(x or "") != normalize_text(y or ""):
            result[key] = [x or "not_stated", y or "not_stated"]
    for key in ("name", "entity", "period", "geography", "scope", "price_basis", "rate_basis", "value_kind"):
        x, y = (a.metric or {}).get(key), (b.metric or {}).get(key)
        if (x or y) and normalize_text(str(x or "")) != normalize_text(str(y or "")):
            result["metric_" + key] = [str(x or "not_stated"), str(y or "not_stated")]
    patterns = {
        "year": r"(?<!\d)(?:19|20|21)\d{2}(?!\d)",
        "release": r"\b(?:preliminary|final)\b|初步|最终核实|最终",
        "cell_scope": r"\b(?:full[- ]cell|half[- ]cell|material[- ]level)\b|全电芯|完整电芯|半电池|材料级",
        "battery_scope": r"\ball[- ]solid[- ]state\b|\bsemi[- ]solid(?:[- ]state)?\b|全固态|半固态|准固态",
    }
    for key, pattern in patterns.items():
        x, y = (set(re.findall(pattern, normalize_text(c.text), re.IGNORECASE)) for c in (a, b))
        if (x or y) and x != y:
            result[key] = [", ".join(sorted(x)), ", ".join(sorted(y))]
    for key, pattern, factors in (
        ("temperature", r"(-?\d+(?:\.\d+)?)\s*(?:°\s*c|℃|摄氏度)", {}),
        ("pressure", r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*(MPa|kPa|Pa)(?![A-Za-z])", {"MPa": 1000000, "kPa": 1000, "Pa": 1}),
    ):
        values = []
        for claim in (a, b):
            found = re.findall(pattern, claim.text, re.IGNORECASE if key == "temperature" else 0)
            values.append({Decimal(row[0]) * factors[row[1]] for row in found} if factors else
                          {Decimal(row) for row in found})
        if any(values) and values[0] != values[1]:
            result[key] = [", ".join(str(v) for v in sorted(group)) for group in values]
    # Equivalent amounts expressed in different units are not contradictory.
    # Restrict this to one structured metric per claim; pooled values are ambiguous.
    if a.value and b.value:
        from core_engine.report.metrics import MetricError, convert_value, quantities

        try:
            x, y = quantities(a.value), quantities(b.value)
            if len(x) == len(y) == 1 and x[0][1] != y[0][1] and convert_value(*x[0], y[0][1]) == y[0][0]:
                result["equivalent_unit_representation"] = [a.value, b.value]
        except MetricError:
            pass
    return result


def apply_policy(claims: list[Claim], sources: list[Source], citations: list[SentenceCitation]) -> dict:
    """Derive publication status only from validated exact source passages.

    Semantic entailment is performed by SentenceGrounder before this call; this
    layer independently checks locations/units and never assigns factual truth.
    """
    by_url = {source.url: source for source in sources}
    groups = origin_groups(sources)
    links: dict[str, list[str]] = {}
    by_claim = {claim.id: claim for claim in claims}
    for citation in citations:
        source = by_url.get(citation.source_url)
        claim = by_claim.get(citation.claim_id)
        if (source is not None and source.authoritative and claim is not None and type(citation.start_char) is int
                and type(citation.end_char) is int and 0 <= citation.start_char < citation.end_char <= len(source.text)
                and source.text[citation.start_char:citation.end_char] == citation.quote
                and source.content_sha256 == citation.source_sha256
                and citation.source_url in claim.candidate_source_urls
                and numbers_supported(claim.text, citation.quote)
                and (not claim.value or numbers_supported(claim.value, claim.text))):
            links.setdefault(claim.id, []).append(source.url)
    decisions = {}
    for claim in claims:
        urls = sorted(set(links.get(claim.id, [])))
        conflicts = sorted({peer for peer in claim.conflicting_claim_ids if peer in links and claim.id in links
                            and peer in by_claim and claim.id in by_claim[peer].conflicting_claim_ids
                            and not context_differences(claim, by_claim[peer])})
        count = len({groups[url] for url in urls})
        status = ("lead_only" if not urls else "disputed" if conflicts else
                  "corroborated" if count > 1 else "source_reported")
        claim.reporting_status = status
        # Legacy verified means source support for uncontested use, never truth.
        claim.verified = status in {"source_reported", "corroborated"}
        claim.confidence = (ConfidenceTier.CORROBORATED if status == "corroborated" else
                            ConfidenceTier.LIKELY if status == "source_reported" else None)
        decisions[claim.id] = {"status": status, "supporting_urls": urls, "origin_group_count": count,
                               "conflicting_claim_ids": conflicts, "truth_status": "not_established"}
    return {"version": 1, "default": "retain_as_candidate_not_true", "claims": decisions,
            "source_reported": sum(v["status"] == "source_reported" for v in decisions.values()),
            "corroborated": sum(v["status"] == "corroborated" for v in decisions.values()),
            "disputed": sum(v["status"] == "disputed" for v in decisions.values()),
            "lead_only": sum(v["status"] == "lead_only" for v in decisions.values()),
            "independence_is_not_proven": True}


def lead_records(claims: list[Claim]) -> list[dict]:
    return [{"claim_id": claim.id, "text": claim.text, "candidate_source_urls": claim.candidate_source_urls,
             "status": "lead_only", "reason": "no_anchored_original_passage"}
            for claim in claims if claim.reporting_status == "lead_only"]


def attribution(claims: list[Claim], sources: list[Source], citations: list[SentenceCitation], locale: str) -> tuple[str, str]:
    state = ("disputed" if any(c.reporting_status == "disputed" for c in claims) else
             "corroborated" if all(c.reporting_status == "corroborated" for c in claims) else "source_reported")
    zh = locale.lower().startswith("zh")
    if state == "disputed":
        return state, "来源存在分歧，以下仅为该来源的陈述：" if zh else "Disputed source statement: "
    if state == "corroborated":
        return state, "多来源原文支持：" if zh else "Multiple source groups report: "
    urls = {c.source_url for c in citations}
    titles = list(dict.fromkeys(s.title.strip() or s.domain for s in sources if s.url in urls))
    if len(titles) == 1:
        title = titles[0][:120]
        return state, f"据《{title}》报告：" if zh else f"According to {title}: "
    return state, "据所引来源分别报告：" if zh else "According to the cited sources: "
