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
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


@dataclass(slots=True)
class SearchHit:
    """A raw search-engine result, before filtering. Cheap: URL + snippet only."""

    url: str
    title: str
    snippet: str
    domain: str


@dataclass(slots=True)
class Source:
    """An authoritative page we actually fetched and extracted text from."""

    url: str
    domain: str
    title: str
    kind: SourceKind
    text: str                         # extracted main content (not raw HTML)
    fetched_at: str = field(default_factory=_now)

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
    """How well-supported a claim is, assigned by the verification harness.

    Semantic + circumstantial standard (not verbatim matching): a source SUPPORTS a
    claim if it states, implies, or is consistent with the claim's underlying fact,
    even in different wording. The pipeline keeps all non-contradicted claims and
    labels them:

      HIGH        — corroborated across multiple independent domains, OR endorsed by
                    a high-trust (academic / gov / IGO / allowlisted) source. Fact.
      CORROBORATED— consistently supported by >1 source, below the HIGH bar. Fact.
      LIKELY      — supported by one credible source or strong circumstantial /
                    consistent evidence. Reported as "Likely True".
      RUMOR       — traceable to a source but weak/single low-trust signal. Kept,
                    flagged as unverified.
    """

    HIGH = "high"
    CORROBORATED = "corroborated"
    LIKELY = "likely"
    RUMOR = "rumor"

    @property
    def label(self) -> str:
        return {"high": "High Confidence", "corroborated": "Corroborated",
                "likely": "Likely True", "rumor": "Unverified / Rumor"}[self.value]


@dataclass(slots=True)
class Claim:
    """An atomic, checkable factual statement extracted from the sources.

    Broad-collection model: a claim is kept as long as it is TRACEABLE to at least one
    real fetched source and NOT flagged as hallucinated (a source contradicting it or
    asserting facts no source supports). Its `confidence` tier records how well it is
    corroborated, so the report can separate verified facts from rumors.
    """

    id: str
    text: str
    # URLs proposed as supporting this claim (populated at extraction, pruned at verify).
    candidate_source_urls: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    verified: bool = False                       # True for HIGH/CORROBORATED tiers
    verification_rounds: int = 0
    confidence: "ConfidenceTier | None" = None    # set by the harness

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
    deep_dives: list["DeepDive"] = field(default_factory=list)


@dataclass(slots=True)
class ReportData:
    """The structured, fully-verified payload handed to the LaTeX renderer.

    By construction everything in here has passed the harness — the renderer does
    NOT get to see unverified claims, so it cannot typeset a hallucination.
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

    def bibliography(self) -> list[Source]:
        """Sources cited by any kept claim (verified OR rumor), deduped by URL."""
        cited = {u for c in (self.claims + self.rumors)
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


@dataclass(slots=True)
class PipelineResult:
    status: PipelineStatus
    topic: str
    message: str = ""
    pdf_path: str | None = None
    tex_path: str | None = None
    report: ReportData | None = None
    # Ordered, human-readable trace of every gate decision — for audit + debugging.
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, stage: str, **fields: Any) -> None:
        self.trace.append({"stage": stage, "at": _now(), **fields})

    @property
    def ok(self) -> bool:
        return self.status is PipelineStatus.COMPLETED
