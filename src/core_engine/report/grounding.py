"""Sentence-level evidence audit. Unsupported drafts are retained for review, not published.

The audit establishes support in a captured source, not the truth of that source.
Offsets and hashes always refer to the exact extracted text saved in the evidence packet.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Coroutine
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core_engine.report.models import (
    Claim,
    DeepDive,
    Evidence,
    ReportData,
    ReportSection,
    ReportSentence,
    SentenceCitation,
    Source,
)
from core_engine.report.publication import apply_policy, attribution
from core_engine.report.text import normalize_text, numbers_supported, split_sentences
from core_engine.security.files import atomic_write_text


def _citation_page(source: Source, start: int) -> int | None:
    if source.document_format != "pdf" and "\f" not in source.text:
        return None
    # Appended table linearizations retain their physical PDF page separately;
    # counting form feeds would otherwise attribute every table to the last page.
    for table in source.metadata.get("tables", []):
        if not isinstance(table, dict):
            continue
        left, right = table.get("start"), table.get("end")
        locator = table.get("locator", {})
        if not isinstance(locator, dict):
            continue
        page = locator.get("page")
        limit = locator.get("page_count") or source.metadata.get("physical_page_count")
        if (isinstance(left, int) and isinstance(right, int) and left <= start < right
                and isinstance(page, int) and page > 0 and (not isinstance(limit, int) or page <= limit)):
            return page
    return source.text[:start].count("\f") + 1

if TYPE_CHECKING:
    from core_engine.config import Settings
    from core_engine.report.llm import LLM

_MARKER = re.compile(r"\[\[([A-Za-z0-9_, -]+)\]\]")
_HEDGE = "[UNVERIFIED - single source]"


def _identity(text: str) -> str:
    return normalize_text(text.replace(_HEDGE, "")).rstrip(".。!?！？ ")


def parse_draft(raw: str) -> list[ReportSentence]:
    """Accept structured drafts; legacy plain prose gets no inferred paraphrase citations."""
    raw = raw.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        data = json.loads(fenced)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("sentences"), list):
        sentences = []
        for item in data["sentences"]:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            ids = item.get("claim_ids", [])
            valid_ids = isinstance(ids, list) and all(isinstance(cid, str) for cid in ids)
            paragraph = item.get("paragraph_index", 0)
            for text in split_sentences(item["text"]):
                sentences.append(ReportSentence(
                    text=text, claim_ids=list(dict.fromkeys(ids)) if valid_ids else ["INVALID_CITATIONS"],
                    paragraph_index=paragraph if isinstance(paragraph, int) and paragraph >= 0 else 0))
        return sentences
    if not raw:
        return []
    # Put suffix citation markers before sentence punctuation so they stay attached.
    raw = re.sub(r"([。.!?！？])\s*(\[\[[A-Za-z0-9_, -]+\]\])", r"\2\1", raw)
    sentences = []
    for paragraph_index, paragraph in enumerate(re.split(r"\n\s*\n", raw)):
        for part in split_sentences(paragraph):
            ids = [cid.strip() for marker in _MARKER.findall(part) for cid in marker.split(",")]
            text = _MARKER.sub("", part).strip()
            if text:
                sentences.append(ReportSentence(text=text, claim_ids=list(dict.fromkeys(ids)),
                                                paragraph_index=paragraph_index))
    return sentences


@dataclass
class AnchorReuse:
    """Successful quote checks shared by stages of one report run."""

    quotes: dict[tuple[str, str, str, str], str] = field(default_factory=dict)
    locks: dict[tuple[str, str, str, str], asyncio.Lock] = field(default_factory=dict)
    requests: int = 0
    hits: int = 0


class SentenceGrounder:
    def __init__(self, llm: LLM, settings: Settings, *, anchor_reuse: AnchorReuse | None = None) -> None:
        self.llm = llm
        self.settings = settings
        self._semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))
        self._claim_cache: dict[str, list[SentenceCitation]] = {}
        self._check_cache: dict[tuple[str, str], bool] = {}
        self._check_tasks: dict[tuple[str, str], asyncio.Task[bool]] = {}
        self._claim_tasks: dict[str, asyncio.Task[list[SentenceCitation]]] = {}
        self.citations: list[SentenceCitation] = []
        self._anchor_timed_out = False
        self._anchor_reuse = anchor_reuse if anchor_reuse is not None else AnchorReuse()

    async def _entailed(self, text: str, context: str) -> bool:
        if not context or not numbers_supported(text, context):
            return False
        # An identical complete source sentence is an auditable extractive citation.
        if any(_identity(text) == _identity(part) for part in split_sentences(context)):
            return True
        from core_engine.report.llm import FakeLLM
        if type(self.llm) is FakeLLM:
            return False  # the offline baseline is extractive, not a semantic judge
        key = (text, context)
        if key in self._check_cache:
            return self._check_cache[key]
        if key not in self._check_tasks:
            async def check():
                async with self._semaphore:
                    try:
                        verdict = await asyncio.wait_for(
                            self.llm.check_entailment(text, context),
                            timeout=self.settings.llm_timeout_s)
                        return bool(verdict.supports and verdict.quote.strip()
                                    and verdict.quote.strip() in context
                                    and numbers_supported(text, verdict.quote))
                    except Exception:
                        return False
            self._check_tasks[key] = asyncio.create_task(check())
        self._check_cache[key] = await self._check_tasks[key]
        return self._check_cache[key]

    async def _claim_evidence(self, claim: Claim, sources: dict[str, Source]) -> list[SentenceCitation]:
        if claim.id in self._claim_cache:
            return self._claim_cache[claim.id]
        if claim.id not in self._claim_tasks:
            self._claim_tasks[claim.id] = asyncio.create_task(self._build_claim_evidence(claim, sources))
        self._claim_cache[claim.id] = await self._claim_tasks[claim.id]
        return self._claim_cache[claim.id]

    async def _build_claim_evidence(self, claim: Claim, sources: dict[str, Source]) -> list[SentenceCitation]:
        from core_engine.report.llm import FakeLLM
        from core_engine.report.verify import claim_context

        records: list[SentenceCitation] = []
        if claim.value and not numbers_supported(claim.value, claim.text):
            return records
        # Check only actual sources associated with this claim, never arbitrary model URLs.
        urls = list(dict.fromkeys(claim.candidate_source_urls + claim.supporting_sources()))
        # When a work's open copy was read, anchor that actual document before
        # spending the citation budget on duplicate registry abstracts.
        urls.sort(key=lambda url: not bool(sources.get(url) and sources[url].metadata.get("full_text_accessed")))
        for url in urls[:self.settings.max_cross_ref_sources]:
            source = sources.get(url)
            if source is None or not source.authoritative:
                continue
            quote = ""
            method = "verbatim"
            for part in split_sentences(source.text):
                if _identity(claim.text) == _identity(part):
                    quote = part
                    break
            if not quote and type(self.llm) is not FakeLLM:
                excerpt = claim_context(claim, source, self.settings.llm_context_chars)
                key = (source.url, source.content_sha256, claim.text, excerpt)
                async with self._anchor_reuse.locks.setdefault(key, asyncio.Lock()):
                    quote = self._anchor_reuse.quotes.get(key, "")
                    if quote:
                        self._anchor_reuse.hits += 1
                    else:
                        self._anchor_reuse.requests += 1
                        try:
                            async with self._semaphore:
                                verdict = await asyncio.wait_for(
                                    self.llm.check_entailment(claim.text, excerpt),
                                    timeout=self.settings.llm_timeout_s)
                            candidate = verdict.quote.strip()
                            if (verdict.supports and candidate and candidate in source.text
                                    and numbers_supported(claim.text, candidate)):
                                quote = candidate
                                self._anchor_reuse.quotes[key] = candidate
                        except Exception:
                            pass
                method = "model_entailment"
            if not quote or not numbers_supported(claim.text, quote):
                continue
            start = source.text.find(quote)
            citation = SentenceCitation(
                id=f"e{len(self.citations) + 1}", claim_id=claim.id, source_url=url,
                quote=quote, start_char=start, end_char=start + len(quote),
                paragraph=source.text[:start].count("\n\n") + 1,
                page=_citation_page(source, start),
                source_sha256=source.content_sha256, method=method)
            self.citations.append(citation)
            records.append(citation)
            # The verifier can discover an additional supporting source. Once it
            # has an exact anchored passage, persist that provenance on the claim
            # as well as its citation; export and independent replay must agree.
            if url not in claim.candidate_source_urls:
                claim.candidate_source_urls.append(url)
            claim.evidence.append(Evidence(url, quote, True, f"sentence audit: {method}"))
        return records

    async def audit_sentence(self, sentence: ReportSentence, claims: dict[str, Claim],
                             sources: dict[str, Source], *, allow_disputed: bool = False) -> ReportSentence:
        if sentence.status == "editorial":
            return sentence
        # Re-auditing an existing report must not preserve stale publication state
        # when a newly added numeric or provenance check rejects the sentence.
        sentence.status = "withheld"
        sentence.citation_ids = []
        sentence.reason = ""
        sentence.attribution = ""
        if not sentence.claim_ids:
            # Legacy extractive drafts can be recovered exactly, never by fuzzy overlap.
            sentence.claim_ids = [c.id for c in claims.values() if _identity(c.text) == _identity(sentence.text)]
        if not sentence.claim_ids:
            sentence.reason = "missing_claim_citation"
            return sentence
        if any(cid not in claims for cid in sentence.claim_ids):
            sentence.reason = "unknown_or_out_of_section_claim"
            return sentence
        if (any(claims[cid].reporting_status == "disputed" for cid in sentence.claim_ids)
                and (not allow_disputed or any(claims[cid].reporting_status != "disputed" for cid in sentence.claim_ids))):
            sentence.status = "withheld"
            sentence.reason = "publication_policy_excludes_unsettled_premise"
            sentence.citation_ids = []
            return sentence
        exact = [cid for cid in sentence.claim_ids if _identity(claims[cid].text) == _identity(sentence.text)]
        if exact:
            sentence.claim_ids = exact  # remove decorative citations on extractive sentences
        records = []
        for cid in sentence.claim_ids:
            evidence = await self._claim_evidence(claims[cid], sources)
            if not evidence:
                sentence.reason = "claim_has_no_anchored_support"
                return sentence
            records.extend(evidence)
        context = "\n\n".join(record.quote for record in records)
        clean = sentence.text.replace(_HEDGE, "").strip()
        if not numbers_supported(clean, context):
            sentence.reason = "numeric_value_or_unit_not_supported"
            return sentence
        if not await self._entailed(clean, context):
            sentence.reason = "draft_sentence_not_entailed_by_sources"
            return sentence
        sentence.status = "supported"
        sentence.reason = "source_support_audited"
        sentence.citation_ids = list(dict.fromkeys(record.id for record in records))
        sentence.evidence_status, sentence.attribution = attribution(
            [claims[cid] for cid in sentence.claim_ids], list(sources.values()), records, self.settings.report_locale)
        return sentence

    async def audit_report(self, report: ReportData) -> ReportData:
        sources = {s.url: s for s in report.sources}
        all_claims = report.claims + report.rumors + report.disputed_claims
        claims = {c.id: c for c in all_claims}
        await self.anchor_claims(all_claims, report.sources)
        # Recompute after anchoring; never trust serialized states from callers.
        leads = [Claim(row["claim_id"], row["text"], candidate_source_urls=row.get("candidate_source_urls", []))
                 for row in report.leads if row["claim_id"] not in claims]
        report.evidence_policy = apply_policy(all_claims + leads, report.sources, self.citations)
        report.evidence_policy["leads"] = report.leads
        work: list[Coroutine[Any, Any, ReportSentence]] = []
        containers: list[ReportSection | DeepDive] = []
        for section in report.sections:
            containers.append(section)
            containers.extend(section.deep_dives)
        if report.rumors:
            report.rumor_sentences = [ReportSentence(text=f"{_HEDGE} {c.text}", claim_ids=[c.id])
                                      for c in report.rumors]
            containers.append(ReportSection(
                heading="Unverified signals", claim_ids=[c.id for c in report.rumors],
                sentences=report.rumor_sentences))
        if report.disputed_claims:
            report.dispute_sentences = [ReportSentence(text=c.text, claim_ids=[c.id]) for c in report.disputed_claims]
            containers.append(ReportSection(heading="Source disagreements", section_id="disputes",
                                           claim_ids=[c.id for c in report.disputed_claims],
                                           sentences=report.dispute_sentences))
        for container in containers:
            if not container.sentences:
                container.sentences = parse_draft("\n\n".join(container.paragraphs))
            allowed = {cid: claims[cid] for cid in container.claim_ids if cid in claims}
            for sentence in container.sentences:
                if self._anchor_timed_out and any(not self._claim_cache.get(cid) for cid in sentence.claim_ids):
                    sentence.reason = "sentence_audit_deadline"
                    continue
                if len(work) >= self.settings.grounding_max_sentences and sentence.status != "editorial":
                    sentence.reason = "sentence_count_limit"
                    continue
                work.append(self.audit_sentence(sentence, allowed, sources, allow_disputed=isinstance(container, ReportSection)
                                                 and container.section_id == "disputes"))
        try:
            await asyncio.wait_for(asyncio.gather(*work), timeout=self.settings.grounding_deadline_s)
        except TimeoutError:
            # Partial results remain reviewable; no in-progress sentence is published.
            for task in (*self._claim_tasks.values(), *self._check_tasks.values()):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._claim_tasks.values(), *self._check_tasks.values(),
                                 return_exceptions=True)
            for sentence in report.all_sentences():
                if sentence.status == "withheld" and not sentence.reason:
                    sentence.reason = "sentence_audit_deadline"
        for container in containers:
            container.paragraphs = [sentence.attribution + sentence.text for sentence in container.sentences
                                    if sentence.status in {"supported", "editorial"}]
        report.citations = list(self.citations)
        sentences = [s for s in report.all_sentences() if s.status != "editorial"]
        supported = sum(s.status == "supported" for s in sentences)
        chart_count = sum(len(section.latex_blocks) for section in report.sections)
        report.grounding_audit = {
            "scope": "drafted_prose", "attempted_sentences": len(sentences),
            "supported_sentences": supported, "withheld_sentences": len(sentences) - supported,
            "supported_ratio": supported / len(sentences) if sentences else 0.0,
            "citation_count": len(report.citations), "chart_blocks_requiring_review": chart_count,
            "anchor_model_requests": self._anchor_reuse.requests,
            "anchor_checks_reused": self._anchor_reuse.hits,
            "source_support_is_not_truth": True,
        }
        report.quality_status = ("source_supported" if sentences and supported == len(sentences)
                                 and not chart_count else "needs_review")
        report.abstract = (
            f"This report contains {supported} source-supported sentence(s); "
            f"{len(sentences) - supported} draft sentence(s) were withheld for review. "
            "Citations identify exact passages in captured sources. Source support does not establish truth."
            if self.settings.report_locale != "zh" else
            f"本报告收录 {supported} 句有原文支持的正文，另有 {len(sentences) - supported} 句草稿需审阅、未进入正文。"
            "引用可追溯至留存原文；原文支持不等于事实已被独立证实。")
        return report

    async def anchor_claims(self, claims: list[Claim], sources: list[Source]) -> tuple[list[Claim], list[SentenceCitation]]:
        """Source-bind research premises before coverage checks or follow-up searches."""
        by_url = {source.url: source for source in sources}
        selected = claims[:self.settings.grounding_max_sentences]
        try:
            await asyncio.wait_for(asyncio.gather(*(
                self._claim_evidence(claim, by_url) for claim in selected)),
                timeout=self.settings.grounding_deadline_s)
        except TimeoutError:
            self._anchor_timed_out = True
            for task in self._claim_tasks.values():
                task.cancel()
            await asyncio.gather(*self._claim_tasks.values(), return_exceptions=True)
        ids = {record.claim_id for record in self.citations}
        return [claim for claim in selected if claim.id in ids], self.citations


def evidence_packet(report: ReportData) -> dict:
    """Portable audit artifact, including exact source snapshots and withheld drafts."""
    return {
        "schema_version": 1, "topic": report.topic, "generated_at": report.generated_at,
        "quality_status": report.quality_status, "audit": report.grounding_audit,
        "sources": [{**asdict(source), "content_sha256": source.content_sha256} for source in report.sources],
        "claims": [asdict(c) for c in report.claims + report.rumors + report.disputed_claims],
        "sections": [asdict(section) for section in report.sections],
        "rumor_sentences": [asdict(sentence) for sentence in report.rumor_sentences],
        "dispute_sentences": [asdict(sentence) for sentence in report.dispute_sentences],
        "leads": report.leads, "evidence_policy": report.evidence_policy,
        "citations": [asdict(record) for record in report.citations],
        "metrics": report.metrics, "calculations": report.calculations,
        "metric_issues": report.metric_issues, "chart_audit": report.chart_audit,
        "answerability": report.answerability,
        "research_plan": report.research_plan, "research_audit": report.research_audit,
        "research_evidence": report.research_evidence,
        "analysis": report.analysis, "analysis_audit": report.analysis_audit,
        "source_quality": report.source_quality,
    }


def write_evidence(report: ReportData, tex_path: Path) -> Path:
    path = tex_path.with_suffix(".evidence.json")
    atomic_write_text(path, json.dumps(evidence_packet(report), ensure_ascii=False, indent=2))
    return path
