"""Data models for the report-generation pipeline.

The pipeline is a strict linear flow with a hard verification gate:

    topic -> search -> SOURCE FILTER -> scrape -> extract claims
          -> VERIFY (triple-check) -> [scope gate] -> LaTeX -> compile PDF

Every stage passes one of these immutable-ish dataclasses to the next. The
`PipelineResult` is the single object the caller inspects: it says whether we
produced a PDF, halted out-of-scope, or were blocked by the harness, and it
carries the full audit trail either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, IntEnum
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SourceKind(str, Enum):
    # --- High-trust tier (curated allowlists) ---
    ACADEMIC = "academic"                  # peer-reviewed / scientific databases
    GOVERNMENT = "government"              # .gov / ministries / IGOs
    AUTHORITATIVE_MEDIA = "media"          # allowlisted mainstream outlets
    INVESTMENT_BANK = "investment_bank"    # IB / research-house / market-data reports
    INDUSTRY_INSTITUTION = "institution"   # trade associations / standards bodies / IGOs
    NONPROFIT = "nonprofit"               # NGO / non-profit research orgs
    # --- General tier (any non-denylisted site with a real article body) ---
    GENERAL_WEB = "general_web"            # admitted for coverage; lower trust weight
    # --- User-supplied tier (Database Mode: the user vouches for these) ---
    USER_PROVIDED = "user_provided"        # uploaded by the user; high trust, still cross-checked
    # --- Blocked ---
    REJECTED = "rejected"                 # denylisted (tabloid/UGC/social) — never used

    @property
    def high_trust(self) -> bool:
        """True for curated-allowlist tiers (used to weight corroboration).

        USER_PROVIDED counts as high-trust: the user explicitly supplied the document,
        so it bypasses the domain allowlist. Its claims are still run through the
        extraction + verification/conflict harness like any other source.
        """
        return self in (
            SourceKind.ACADEMIC, SourceKind.GOVERNMENT, SourceKind.AUTHORITATIVE_MEDIA,
            SourceKind.INVESTMENT_BANK, SourceKind.INDUSTRY_INSTITUTION,
            SourceKind.NONPROFIT, SourceKind.USER_PROVIDED,
        )


class CredibilityLevel(IntEnum):
    """Source credibility grading (L1-L4), orthogonal to SourceKind.

    SourceKind answers "which curated bucket admitted this URL"; CredibilityLevel
    answers "how much weight should this source carry when ranking and when
    judging isolated (single-source) claims".

      L1 — authoritative anchor: official releases, audited filings, legal
           instruments, national standards, top-tier academic venues.
      L2 — professional endorsement: leading media, think tanks, listed-company
           disclosures, research houses.
      L3 — industry consensus: trade associations, mainstream in-depth self-media,
           expert interviews; the default for unrecognized-but-allowed sites.
      L4 — weak-signal / isolated evidence: personal blogs, anonymous forums,
           unattributed material. Never ranked ahead of L1/L2 anchors.
    """

    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4

    @property
    def weight(self) -> float:
        """Ranking weight: higher credibility -> larger weight."""
        return {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.2}[int(self)]

    @property
    def label(self) -> str:
        return {
            1: "L1 Authoritative Anchor",
            2: "L2 Professional Endorsement",
            3: "L3 Industry Consensus",
            4: "L4 Weak Signal / Isolated",
        }[int(self)]


@dataclass(slots=True)
class SearchHit:
    """A raw search-engine result, before filtering. Cheap: URL + snippet only."""

    url: str
    title: str
    snippet: str
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Source:
    """An authoritative page we actually fetched and extracted text from."""

    url: str
    domain: str
    title: str
    kind: SourceKind
    text: str                         # extracted main content (not raw HTML)
    fetched_at: str = field(default_factory=_now)
    # Credibility grade (L1-L4) assigned at gather time via sources.credibility_for().
    # Defaults to L3 (industry consensus) so pre-existing constructions keep a
    # neutral weight in the credibility-weighted ranking.
    credibility: CredibilityLevel = CredibilityLevel.L3
    published_at: str | None = None
    document_format: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_sha256(self) -> str:
        import hashlib
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def authoritative(self) -> bool:
        # Every kind except REJECTED is authoritative. Kept as an explicit negative
        # so adding a new authoritative SourceKind doesn't silently exclude it.
        return self.kind is not SourceKind.REJECTED


@dataclass(slots=True)
class Evidence:
    """A single (claim, source) support check produced by the verifier.

    `quote` is the verbatim span from the source the verifier says supports the
    claim. If we cannot point to a real span, the claim is NOT verified — this is
    the anti-hallucination anchor: no quote, no support.
    """

    source_url: str
    quote: str
    supports: bool
    note: str = ""


class ConfidenceTier(str, Enum):
    """Legacy ranking values, retained for stored-packet compatibility.

    Explicit reporting_status controls publication. No tier establishes truth,
    independent replication or source reliability.
    """

    HIGH = "high"
    CORROBORATED = "corroborated"
    LIKELY = "likely"
    RUMOR = "rumor"

    @property
    def label(self) -> str:
        return {"high": "Legacy Source Support", "corroborated": "Multiple Source Groups",
                "likely": "Source Reported", "rumor": "Needs Source Review"}[self.value]


@dataclass(slots=True)
class Claim:
    """An atomic, checkable factual statement extracted from the sources.

    Candidates are retained without a truth presumption. Publication requires
    an anchored source passage and explicit source attribution. Unresolved
    same-scope conflicts are displayed separately from conclusions and calculations.

    Structured-slot semantics (filled by the alignment/fusion layer, all optional):
      entity      — the subject the claim is about (normalized name, e.g. a company).
      attribute   — the measured/claimed property (e.g. "market_share", "revenue").
      value       — the claimed value as stated (e.g. "32%").
      qualifier   — scope/calibre qualifiers (口径: e.g. "GAAP", "domestic only").
      time_scope  — the period the claim refers to (e.g. "FY2023").
    When extraction cannot fill the slots they stay None and the claim degrades to a
    plain-text statement handled by the lexical paths.
      credibility — credibility of the claim's (strongest) supporting source; None
                    until the verification layer assigns it.
      isolated    — True for 孤证: supported by a single source whose credibility is
                    L3 or weaker. Isolated claims get hedged wording in the report.
    """

    id: str
    text: str
    # URLs proposed as supporting this claim (populated at extraction, pruned at verify).
    candidate_source_urls: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    verified: bool = False                       # legacy: source support, never factual truth
    verification_rounds: int = 0
    confidence: ConfidenceTier | None = None    # set by the harness
    # Credibility / isolation grading (see class docstring). Filled by the
    # verification/alignment layer; None / False until then.
    credibility: CredibilityLevel | None = None
    isolated: bool = False
    # Structured slots (entity-attribute-value), all optional — see class docstring.
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    qualifier: str | None = None
    time_scope: str | None = None
    metric: dict[str, Any] | None = None
    evidence_quality: dict[str, Any] = field(default_factory=dict)
    # Retrieval hints from actual extraction windows; never proof of support.
    source_windows: dict[str, list[list[int]]] = field(default_factory=dict)
    # Publication decisions are separate from legacy ranking/confidence fields.
    reporting_status: str = "pending"
    conflicting_claim_ids: list[str] = field(default_factory=list)

    def supporting_sources(self) -> list[str]:
        """Distinct source URLs with confirmed supporting evidence."""
        return sorted({e.source_url for e in self.evidence if e.supports})

    @property
    def is_rumor(self) -> bool:
        return self.confidence is ConfidenceTier.RUMOR


@dataclass(slots=True)
class DeepDive:
    """A multi-paragraph analytical insight on a high-value data point.

    Distinct from ordinary section prose: a DeepDive is triggered when the harness
    surfaces a critical finding (market bottleneck, core policy impact, key competitor
    advantage) and the model is asked for comprehensive analysis rather than a summary.

    Grounding is preserved: `claim_ids` are the verified claims the analysis is built
    on, and the drafting prompt forbids introducing any fact not in those claims. The
    analysis interprets and connects verified facts; it does not add new ones.
    """

    title: str
    kind: str                          # "bottleneck" | "policy_impact" | "competitor_edge" | ...
    paragraphs: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    sentences: list[ReportSentence] = field(default_factory=list)


@dataclass(slots=True)
class SentenceCitation:
    id: str
    claim_id: str
    source_url: str
    quote: str
    start_char: int
    end_char: int
    paragraph: int
    page: int | None
    source_sha256: str
    method: str  # verbatim | model_entailment (source support, not factual truth)


@dataclass(slots=True)
class ReportSentence:
    text: str
    claim_ids: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    status: str = "withheld"  # supported | withheld | editorial
    reason: str = ""
    paragraph_index: int = 0
    evidence_status: str = ""
    attribution: str = ""


@dataclass(slots=True)
class ReportSection:
    heading: str
    # Stable id, one of the 5 mandated section keys (see Settings.report_sections).
    section_id: str = ""
    # Body is composed only from verified claims; each paragraph cites claim ids.
    paragraphs: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    # Pre-rendered LaTeX chart/table blocks (TikZ / pgfplots / booktabs) for the
    # structured sections. Already escaped + compile-safe; the template emits them
    # verbatim. Empty for prose-only sections.
    latex_blocks: list[str] = field(default_factory=list)
    # Deep-dive analytical insights attached to this section (may be empty). Rendered
    # as titled subsections after the main prose + charts.
    deep_dives: list[DeepDive] = field(default_factory=list)
    sentences: list[ReportSentence] = field(default_factory=list)


@dataclass(slots=True)
class ReportData:
    """Report payload, with separate claim tiers and sentence-level source support.

    The pipeline audits drafted sentences before rendering. Withheld drafts remain
    in the evidence packet; source support is not a guarantee of factual truth.
    """

    topic: str
    title: str
    abstract: str
    sections: list[ReportSection] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    generated_at: str = field(default_factory=_now)
    # The synthesized KG behind the structured sections. Typed as Any to avoid a
    # circular import (kg.py imports this module); it is a ReportKnowledgeGraph.
    kg: Any = None
    # Uncorroborated single-source claims, kept and rendered in a clearly-labeled
    # "Unverified Signals & Rumors" section rather than mixed with verified facts.
    rumors: list[Claim] = field(default_factory=list)
    citations: list[SentenceCitation] = field(default_factory=list)
    quality_status: str = "not_audited"
    grounding_audit: dict[str, Any] = field(default_factory=dict)
    rumor_sentences: list[ReportSentence] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    calculations: list[dict[str, Any]] = field(default_factory=list)
    metric_issues: list[dict[str, Any]] = field(default_factory=list)
    chart_audit: list[dict[str, Any]] = field(default_factory=list)
    answerability: dict[str, Any] = field(default_factory=dict)
    research_plan: dict[str, Any] = field(default_factory=dict)
    research_audit: dict[str, Any] = field(default_factory=dict)
    research_evidence: dict[str, Any] = field(default_factory=dict)
    analysis: list[dict[str, Any]] = field(default_factory=list)
    analysis_audit: dict[str, Any] = field(default_factory=dict)
    source_quality: dict[str, Any] = field(default_factory=dict)
    disputed_claims: list[Claim] = field(default_factory=list)
    dispute_sentences: list[ReportSentence] = field(default_factory=list)
    leads: list[dict[str, Any]] = field(default_factory=list)
    evidence_policy: dict[str, Any] = field(default_factory=dict)

    def all_sentences(self) -> list[ReportSentence]:
        return ([sentence for section in self.sections
                 for sentence in (section.sentences + [s for dive in section.deep_dives for s in dive.sentences])]
                + self.rumor_sentences + self.dispute_sentences)

    def bibliography(self) -> list[Source]:
        """Sources cited by any kept claim (verified OR rumor), deduped by URL."""
        cited = {u for c in (self.claims + self.rumors + self.disputed_claims)
                 for u in c.supporting_sources()}
        seen: dict[str, Source] = {}
        for s in self.sources:
            if s.url in cited and s.url not in seen:
                seen[s.url] = s
        return list(seen.values())


class PipelineStatus(str, Enum):
    COMPLETED = "completed"            # PDF produced
    OUT_OF_SCOPE = "out_of_scope"      # insufficient authoritative info -> polite halt
    BLOCKED = "blocked"               # harness blocked; could not verify after retries
    ERROR = "error"                   # unexpected failure (e.g. LaTeX toolchain missing)


# The exact user-facing message mandated by the spec for the out-of-scope case.
OUT_OF_SCOPE_MESSAGE = "This topic is outside my current business scope."
OUT_OF_SCOPE_MESSAGE_ZH = "该主题超出我目前的业务范围。"


def out_of_scope_message(locale: str = "en") -> str:
    """Locale-aware out-of-scope message; defaults to the mandated English text."""
    if (locale or "en").lower().startswith("zh"):
        return OUT_OF_SCOPE_MESSAGE_ZH
    return OUT_OF_SCOPE_MESSAGE


@dataclass(slots=True)
class PipelineResult:
    status: PipelineStatus
    topic: str
    message: str = ""
    pdf_path: str | None = None
    tex_path: str | None = None
    evidence_path: str | None = None
    report: ReportData | None = None
    answerability: dict[str, Any] = field(default_factory=dict)
    research_plan: dict[str, Any] = field(default_factory=dict)
    research_audit: dict[str, Any] = field(default_factory=dict)
    research_evidence: dict[str, Any] = field(default_factory=dict)
    evidence_policy: dict[str, Any] = field(default_factory=dict)
    # Ordered, human-readable trace of every gate decision — for audit + debugging.
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, stage: str, **fields: Any) -> None:
        self.trace.append({"stage": stage, "at": _now(), **fields})

    @property
    def ok(self) -> bool:
        return self.status is PipelineStatus.COMPLETED
