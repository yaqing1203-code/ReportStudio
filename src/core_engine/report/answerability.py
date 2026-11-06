"""Question coverage checks: source volume cannot substitute for the requested fact.

Deterministic guards cover named metrics, entities and periods. Unrecognized broad
research questions remain explicitly undetermined instead of being declared covered.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from core_engine.report.metrics import (
    METRIC_ALIASES,
    mentions_metric,
    observed_quantities,
    parse_unit,
    quantities,
)
from core_engine.report.models import Claim, Source
from core_engine.report.text import content_terms, normalize_text

_EXTRA = {
    "growth_rate": ("growth", "grow", "grew", "增长", "增速", "increase", "increased", "decrease", "变化率"),
    "cpi": ("cpi", "consumer price index", "inflation", "通胀", "物价"),
    "capacity": ("capacity", "装机", "产能"),
    "core_inflation": ("core inflation", "core cpi", "核心通胀", "excluding food and energy", "less food and energy"),
}
_MISSING = re.compile(r"未披露|未提供|没有.{0,8}数据|无法确定|not (?:disclosed|available|reported)|unknown", re.IGNORECASE)
_BOILERPLATE = re.compile(r"(?:根据|所给|资料|说明|报告|分析|确定|查询|统计|关于|please|report|state|describe)", re.IGNORECASE)


@dataclass
class Coverage:
    status: str  # answerable | insufficient | undetermined
    requested_metrics: list[str] = field(default_factory=list)
    requested_entities: list[str] = field(default_factory=list)
    requested_periods: list[str] = field(default_factory=list)
    witnesses: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    irrelevant_claim_ids: list[str] = field(default_factory=list)
    requested_qualifiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _mentions(text: str, metric: str) -> bool:
    text = normalize_text(text).replace("net-profit", "net profit")
    if metric == "growth_rate":
        return any(alias in text for alias in _EXTRA[metric])
    if metric == "cpi":
        return any(alias in text for alias in ("cpi", "price index", "inflation", "all items", "通胀", "物价"))
    if metric in _EXTRA:
        return any(alias in text for alias in _EXTRA[metric])
    return mentions_metric(text, metric)


def _entities(question: str) -> list[str]:
    entities = re.findall(r"\b([A-Z][A-Za-z0-9&.-]*)['’]s\b", question)
    entities += re.findall(r"\b(?:for|of)\s+([A-Z][A-Za-z0-9&.-]*)\b", question)
    for match in re.finditer(r"([\u3400-\u9fffA-Za-z]{1,24})(公司|集团|股份)", question):
        stem = re.split(r"根据|资料|说明|报告|分析|确定|查询|统计|关于|的", match[1])[-1]
        entities.append(stem + match[2])
    for match in re.finditer(r"([\u3400-\u9fff]{2,20})(?=(?:20\d{2}年?)(?:的)?(?:净利润|营收|营业收入))", question):
        entities.append(re.split(r"根据|资料|说明|报告|分析|确定|查询|统计|关于|的", match[1])[-1])
    excluded = {"US", "U.S.", "China", "GDP", "CPI", "BEA", "BLS", "IRENA"}
    return list(dict.fromkeys(entity for entity in entities if entity and entity not in excluded
                             and not any(word in entity for word in ("某", "一家", "企业", "这家", "该公司"))))


def _periods(text: str) -> set[str]:
    """Keep explicit quarter/fiscal periods separate from annual observations."""
    specific = set()
    for year, quarter in re.findall(r"((?:19|20|21)\d{2})\s*(?:年\s*)?[Qq]([1-4])", text):
        specific.add(f"{year}Q{quarter}")
    for quarter, year in re.findall(r"[Qq]([1-4])\s+((?:19|20|21)\d{2})", text):
        specific.add(f"{year}Q{quarter}")
    for year, quarter in re.findall(r"((?:19|20|21)\d{2})年?第?([一二三四1-4])季度", text):
        specific.add(f"{year}Q" + {"一": "1", "二": "2", "三": "3", "四": "4"}.get(quarter, quarter))
    for quarter, year in re.findall(r"(first|second|third|fourth) quarter (?:of )?((?:19|20|21)\d{2})", text, re.IGNORECASE):
        specific.add(f"{year}Q" + {"first": "1", "second": "2", "third": "3", "fourth": "4"}[quarter.lower()])
    specific.update("FY" + year for year in re.findall(r"\bFY\s*((?:19|20|21)\d{2})", text, re.IGNORECASE))
    years = set(re.findall(r"(?<!\d)(?:19|20|21)\d{2}(?!\d)", text))
    specific_years = {match[0] for period in specific if (match := re.search(r"\d{4}", period))}
    return specific | (years - specific_years)


def _contains_entity(text: str, entity: str) -> bool:
    pattern = re.escape(normalize_text(entity))
    if re.search(r"[a-z0-9]", entity, re.IGNORECASE):
        pattern = r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])"
    return bool(re.search(pattern, normalize_text(text)))


def assess_coverage(question: str, claims: list[Claim], sources: list[Source], *, topic_aliases=()) -> Coverage:
    normalized = normalize_text(question).replace("net-profit", "net profit")
    metrics = [name for name in (*METRIC_ALIASES, "capacity", "core_inflation") if _mentions(normalized, name)]
    # Generic growth is an attribute of specific headline metrics, not a demand for
    # an additional independent series. Capacity's explicit subtypes cover it.
    if "capacity" in metrics and any(name in metrics for name in ("installed_capacity", "capacity_addition")):
        metrics.remove("capacity")
    entities = _entities(question)
    periods = sorted(_periods(question))
    coverage = Coverage("undetermined", metrics, entities, periods)
    if not metrics:
        stop = {"the", "and", "for", "of", "on", "in", "is", "a", "an", "report", "research",
                "generate", "please", "write", "analysis", "topic", "test", "sector", "资料", "报告", "研究", "分析"}
        terms = {term for term in content_terms(question, stop) if not term.isdigit()}
        available = set().union(*(content_terms(claim.text, stop) for claim in claims))
        if (terms and not terms.intersection(available)
                and not any(normalize_text(alias) in normalize_text(claim.text) for alias in topic_aliases for claim in claims)):
            coverage.status = "insufficient"
            coverage.missing = ["question_subject"]
        scale_question = re.search(r"量产|商业化|中试|mass[- ]production|commercial (?:deployment|production)|pilot[- ](?:line|production)", question, re.IGNORECASE)
        barriers = re.search(r"挑战|障碍|瓶颈|难点|机理|风险|challenge|barrier|risk|limitation|mechanism", question, re.IGNORECASE)
        direct_status = re.search(r"是否|何时|情况|现状|进展|已经|证明|确定|status|when|has .*begun", question, re.IGNORECASE)
        if scale_question and (not barriers or direct_status):
            production = re.compile(r"量产|中试|mass[- ]production|commercial (?:deployment|production)|pilot[- ](?:line|production)", re.IGNORECASE)
            hypothetical = re.compile(r"potential|could|might|\bmay\b|计划|预计|有望|拟|可能", re.IGNORECASE)
            witnesses = [claim.id for claim in claims if production.search(claim.text) and not hypothetical.search(claim.text)
                         and all(_contains_entity(claim.text, entity) for entity in entities)
                         and (not periods or set(periods).intersection(_periods(claim.text)))]
            coverage.requested_qualifiers.append("production_scale_evidence")
            coverage.witnesses["production_scale_evidence"] = witnesses
            if not witnesses:
                coverage.status = "insufficient"
                coverage.missing.append("production_scale_evidence")
        return coverage
    by_url = {source.url: source for source in sources}
    claim_periods = {}
    for claim in claims:
        title_periods = _periods("\n".join(by_url[url].title for url in claim.candidate_source_urls if url in by_url))
        claim_periods[claim.id] = _periods(claim.text) or (title_periods if len(title_periods) == 1 else set())
    relevant = set()
    company_request = bool(re.search(r"某一家|一家.{0,12}企业|company|企业|公司", normalized))
    for metric in metrics:
        ids = []
        for claim in claims:
            try:
                values = quantities(claim.text)
            except ValueError:
                continue
            if not _mentions(claim.text, metric) or _MISSING.search(claim.text) or not values:
                continue
            if (metric in {"net_profit", "revenue", "market_size", "gdp", "tam", "sam", "som"}
                    and "growth_rate" not in metrics
                    and not any(parse_unit(unit).dimension == "money"
                                for _, unit in observed_quantities(claim.text, metric))):
                continue
            if entities and not any(_contains_entity(claim.text, entity) for entity in entities):
                continue
            if (company_request and metric in {"net_profit", "revenue"} and not entities
                    and (not claim.entity or any(term in claim.entity for term in ("行业", "全国", "aggregate", "sector")))):
                continue
            if periods and not claim_periods[claim.id].intersection(periods):
                continue
            ids.append(claim.id)
            relevant.add(claim.id)
        coverage.witnesses[metric] = ids
        if not ids:
            coverage.missing.append(metric)
        for entity in entities or [""]:
            for period in periods or [""]:
                witnesses = [claim.id for claim in claims if claim.id in ids
                             and (not entity or _contains_entity(claim.text, entity))
                             and (not period or period in claim_periods[claim.id])]
                key = metric + (f"/{entity}" if entity else "") + (f"@{period}" if period else "")
                coverage.witnesses[key] = witnesses
                if not witnesses and key not in coverage.missing:
                    coverage.missing.append(key)
    coverage.irrelevant_claim_ids = [claim.id for claim in claims if claim.id not in relevant]
    # A preliminary release is not an answer to an explicitly final-release task.
    # Publication/version labels may be stated in the document title.
    for qualifier, pattern in (("final_release", r"最终|\bfinal\b"),
                               ("preliminary_release", r"初步|\bpreliminary\b")):
        positive_mentions = [match for match in re.finditer(pattern, question, re.IGNORECASE) if not re.search(
            r"(?:不要|不用|而非|不是|不采用|不使用)(?:用|采用|使用)?\s*$|(?:\bnot|rather than|instead of)\s+(?:the\s+)?$",
            question[:match.start()], re.IGNORECASE)]
        if positive_mentions:
            coverage.requested_qualifiers.append(qualifier)
            witnesses = [claim.id for claim in claims if claim.id in relevant and re.search(pattern,
                claim.text + "\n" + "\n".join(by_url[url].title for url in claim.candidate_source_urls if url in by_url),
                re.IGNORECASE)]
            coverage.witnesses[qualifier] = witnesses
            if not witnesses:
                coverage.missing.append(qualifier)
    coverage.status = "insufficient" if coverage.missing else "answerable"
    return coverage
