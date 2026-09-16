"""Report-generation pipeline — the linear orchestrator.

This is the top-level flow the pivot asked for. It is deliberately a strict,
deterministic state machine, NOT an autonomous agent loop: each stage either
advances or halts with a typed PipelineResult. The verification harness is the
gate every path must clear before any LaTeX is produced.

    topic
      │
      ▼
   [1] gather_sources ── search → STRICT source filter → fetch survivors
      │                  (only .gov / IGO / .edu / allowlisted media reach here)
      ▼
   [2] SCOPE GATE (pre-verify) ── enough distinct authoritative domains?
      │   no → OUT_OF_SCOPE (polite halt, exact mandated message)
      ▼
   [3] extract claims (LLM proposes; never trusted as truth)
      │
      ▼
   [4] VERIFY (triple-check + cross-reference) ──┐ blocked?
      │                                          │  re-scrape / re-verify
      │◄─────────────────────────────────────────┘  up to max_reverify_attempts
      ▼
   [5] SCOPE GATE (post-verify) ── enough VERIFIED claims + domains?
      │   no → OUT_OF_SCOPE
      ▼
   [6] assemble ReportData (verified claims only)
      │
      ▼
   [7] render LaTeX → [8] compile PDF (degrades to .tex if no TeX toolchain)
      │
      ▼
   COMPLETED

Every decision is appended to result.trace for a full audit.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from typing import ClassVar

from core_engine.config import get_settings
from core_engine.report import charts as charts_mod
from core_engine.report.compile import CompileError, compile_pdf, engine_available
from core_engine.report.kg import ReportKnowledgeGraph, build_report_kg
from core_engine.report.latex import write_tex
from core_engine.report.llm import LLM, get_llm
from core_engine.report.models import (
    Claim,
    DeepDive,
    PipelineResult,
    PipelineStatus,
    ReportData,
    ReportSection,
    Source,
    out_of_scope_message,
)
from core_engine.report.scrape import (
    Fetcher,
    SearchProvider,
    SearchUnavailableError,
    gather_sources,
    get_fetcher,
    get_search_provider,
)
from core_engine.report.verify import VerificationHarness, VerificationReport

log = logging.getLogger(__name__)

# on_progress(stage, detail): stage is a stable machine key ("scraping", "verifying",
# "analyzing", "generating_latex", "compiling", "completed", "out_of_scope"); detail is
# a human-readable line for the GUI. Kept intentionally simple (no async) so any caller
# — sync GUI thread or async server — can pass a plain function.
ProgressFn = Callable[[str, str], None]

# Wall-clock cap on a human-in-the-loop review pause. Chosen below the server's
# overall run budget (verify_deadline_s + assemble_deadline_s + 180s, >= 600s) so the
# gate's own timeout fires first and the run degrades to "continue" instead of being
# killed by the run-level watchdog.
HITL_CONFIRM_TIMEOUT_S = 600.0


class ConfirmationGate:
    """Lightweight HITL rendezvous between a running pipeline and a human reviewer.

    The pipeline calls `await gate.wait()` at a checkpoint; the server/UI calls
    `gate.decide("continue" | "abort")` when the user responds. Internally just an
    asyncio.Event plus the decision string. A wait that exceeds `timeout` resolves to
    "continue" (fail-open: a forgotten review must never hang a run) and logs it.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.decision: str | None = None

    async def wait(self, timeout: float = HITL_CONFIRM_TIMEOUT_S) -> str:
        """Block until decide() is called or `timeout` elapses. Returns "continue" or
        "abort"; a timeout or an unexpected decision value resolves to "continue"."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except TimeoutError:
            log.warning("confirmation gate timed out after %.0fs — defaulting to "
                        "'continue'", timeout)
            return "continue"
        if self.decision not in ("continue", "abort"):
            log.warning("confirmation gate woke with unexpected decision %r — "
                        "defaulting to 'continue'", self.decision)
            return "continue"
        return self.decision

    def decide(self, decision: str) -> None:
        """Record the human decision and wake the waiting pipeline. Idempotent:
        a second call after the gate already fired just updates the record."""
        if decision not in ("continue", "abort"):
            raise ValueError(f"decision must be 'continue' or 'abort', got {decision!r}")
        self.decision = decision
        self._event.set()


def _make_emitter(on_progress: ProgressFn | None):
    """Wrap the caller's callback so a raising/misbehaving callback can never break a
    run. Progress is strictly best-effort telemetry."""
    def emit(stage: str, detail: str = "") -> None:
        if on_progress is None:
            return
        try:
            on_progress(stage, detail)
        except Exception:  # telemetry must never break the pipeline
            log.debug("progress callback raised (ignored)", exc_info=True)
    return emit


class ReportPipeline:
    def __init__(
        self,
        *,
        llm: LLM | None = None,
        search: SearchProvider | None = None,
        fetcher: Fetcher | None = None,
    ) -> None:
        self._s = get_settings()
        self._llm = llm or get_llm()
        # An explicitly injected search provider pins the single-backend path; when
        # none is given (the app default) gather_sources may route per query type.
        self._search_is_default = search is None
        self._search = search or get_search_provider()
        self._fetcher = fetcher or get_fetcher()
        self._harness = VerificationHarness(self._llm, fetcher=self._fetcher)

    async def run(
        self,
        topic: str,
        *,
        compile_to_pdf: bool = True,
        on_progress: ProgressFn | None = None,
        extra_sources: list[Source] | None = None,
        skip_search: bool = False,
        confirmation_gate: ConfirmationGate | None = None,
    ) -> PipelineResult:
        """Run the pipeline. `on_progress(stage, detail)` is called at each phase so a
        GUI can show live status ('Scraping…', 'Verifying…', 'Generating LaTeX…'). It is
        best-effort: a callback that raises is swallowed so it can never break a run.

        Database Mode hooks:
          - `extra_sources`: pre-built Source objects (e.g. user-uploaded documents, or
            the sources a prior brief already gathered) merged into the pool, deduped by
            URL. They are trusted as-is — no re-fetch, no allowlist re-check — but their
            claims still pass through extraction + the verification/conflict harness.
          - `skip_search`: do NOT hit the web search/fetch layer; build the report purely
            from `extra_sources`. Used by the 'comprehensive' step so it enriches the
            brief with the user's documents instead of re-searching.

        HITL hook (workflow F): when `confirmation_gate` is given AND
        settings.hitl_on_isolated_core_claim is enabled, the run pauses after
        verification if any CORE isolated claim (single source, credibility L3 or
        stronger) survived, emits an 'awaiting_confirmation' event with a JSON summary,
        and waits for gate.decide(). "abort" halts the run (PipelineStatus.BLOCKED);
        "continue" (or a gate timeout) proceeds unchanged. With the switch off or no
        gate, behavior is exactly as before.
        """
        emit = _make_emitter(on_progress)
        result = PipelineResult(status=PipelineStatus.ERROR, topic=topic)
        topic = topic.strip()
        extra_sources = extra_sources or []
        if not topic:
            result.status = PipelineStatus.OUT_OF_SCOPE
            result.message = out_of_scope_message(self._s.report_locale)
            result.log("intake", ok=False, reason="empty topic")
            emit("out_of_scope", "Empty topic.")
            return result

        try:
            rejected: list = []
            if skip_search:
                # Comprehensive step: no web search — the pool IS the injected sources.
                emit("scraping_start",
                     f"Using {len(extra_sources)} stored/uploaded source(s); "
                     "skipping web search…")
                sources = _merge_sources([], extra_sources)
                emit("scraping_done",
                     f"Prepared {len(sources)} source(s) from the database.")
            else:
                emit("scraping_start", "Searching sources and scraping pages…")

                # Feed progress DURING gather so the UI's inactivity watchdog never trips
                # while pages are being fetched (this stall was the 'no progress' hang).
                def _gather_progress(done: int, total: int, detail: str) -> None:
                    emit("scraping_progress", detail if not total
                         else f"Fetching sources: {detail}")

                try:
                    fetched, rejected = await gather_sources(
                        # None (only when the provider was not explicitly injected)
                        # lets gather route per query type via the search router.
                        topic, None if self._search_is_default else self._search,
                        self._fetcher,
                        on_progress=_gather_progress, llm=self._llm)
                except SearchUnavailableError as e:
                    # Search backend blocked/unconfigured — this is NOT 'out of scope'.
                    # Surface the real, actionable reason (the DDG-block masking bug).
                    result.status = PipelineStatus.ERROR
                    result.message = f"Search unavailable: {e}"
                    result.log("gather", ok=False, error=str(e))
                    emit("error", f"Search unavailable: {e}")
                    return result
                # Merge any injected sources (brief sources + user docs) with the freshly
                # fetched pool, deduped by URL.
                sources = _merge_sources(fetched, extra_sources)
                result.log("gather", authoritative=len(sources), rejected=len(rejected),
                           injected=len(extra_sources),
                           kept_domains=sorted({s.domain for s in sources}))
                emit("scraping_done",
                     f"Collected {len(sources)} source(s) ({len(rejected)} discarded"
                     + (f", {len(extra_sources)} user-provided" if extra_sources else "")
                     + ").")

            # --- Scope gate #1: do we even have enough footing to proceed? ---
            # Count distinct authoritative domains directly off the Source objects'
            # assigned kind, NOT by re-classifying the URL — user-provided documents have
            # synthetic URLs that the allowlist would reject, but they ARE authoritative.
            domains = _distinct_authoritative_domains(sources)
            if len(domains) < self._s.min_authoritative_sources:
                # Distinguish the failure modes so the reason is actionable, not a
                # generic 'out of scope' for what is really a data-gathering shortfall.
                if not sources and not rejected:
                    reason = ("no search results returned — the search provider may be "
                              "unreachable or not configured")
                elif not sources:
                    reason = (f"found {len(rejected)} page(s) but none passed the source "
                              f"filter or relevance check")
                else:
                    reason = (f"only {len(domains)} distinct source domain(s); "
                              f"need {self._s.min_authoritative_sources}")
                emit("scraping", f"Insufficient sources: {reason}.")
                return self._out_of_scope(result, reason)

            # --- Extract + verify, with bounded re-verify loop ---
            accept_first = self._s.verify_strategy in ("conflict_only", "aligned")
            emit("verifying_start",
                 "Accepting claims and resolving conflicts…" if accept_first
                 else "Cross-verifying claims across sources…")
            verification = await self._extract_and_verify(topic, sources, result, emit)
            if accept_first:
                dropped = len(verification.rejected_claims)
                emit("verifying_done",
                     f"Accepted {verification.kept_count} claim(s); "
                     f"dropped {dropped} conflicting claim(s).")
            else:
                n_high = sum(1 for c in verification.verified_claims)
                n_rumor = len(verification.rumor_claims)
                emit("verifying_done",
                     f"Classified {verification.kept_count} claim(s): {n_high} verified, "
                     f"{n_rumor} unverified/rumor.")

            # --- Scope gate #2: broad-collection model. A report is out-of-scope only
            #     if too FEW claims survive at ANY tier (rumors included). Rumors are
            #     kept and labeled, so they count toward having something to analyze. ---
            if verification.kept_count < self._s.min_total_claims:
                return self._out_of_scope(
                    result,
                    f"post-verification: only {verification.kept_count} claim(s) "
                    f"survived (need {self._s.min_total_claims}); "
                    f"{verification.verified_count} verified, "
                    f"{len(verification.rumor_claims)} rumor",
                )

            # --- HITL checkpoint (workflow F, default OFF): if core isolated claims
            #     survived verification, pause for a human decision before assembly. ---
            if confirmation_gate is not None and self._s.hitl_on_isolated_core_claim:
                decision = await self._isolated_claim_checkpoint(
                    verification, confirmation_gate, emit)
                result.log("hitl", decision=decision)
                if decision == "abort":
                    result.status = PipelineStatus.BLOCKED
                    result.message = ("Report generation aborted by the reviewer at the "
                                      "isolated-claim review checkpoint.")
                    result.log("scope_gate", ok=False,
                               reason="aborted by user at isolated-claim review")
                    emit("blocked", result.message)
                    return result

            # --- Assemble the report payload (verified facts + labeled rumors) ---
            emit("analyzing_start", "Synthesizing knowledge graph and analysis…")
            report = await self._assemble(topic, verification, sources)
            result.report = report
            result.log("assemble", sections=len(report.sections),
                       verified_claims=len(report.claims),
                       cited_sources=len(report.bibliography()))
            emit("analyzing_done",
                 f"Assembled {len(report.sections)} section(s) from "
                 f"{len(report.claims)} claim(s).")

            # --- Render LaTeX ---
            emit("generating_latex_start", "Generating LaTeX document…")
            tex_path = write_tex(report)
            result.tex_path = str(tex_path)
            result.log("render", tex=str(tex_path))
            emit("generating_latex_done", "LaTeX document generated.")

            # --- Compile PDF (degrade gracefully if no toolchain) ---
            if compile_to_pdf:
                emit("compiling_start", "Compiling PDF…")
                if not engine_available():
                    result.status = PipelineStatus.COMPLETED
                    result.message = (
                        f"Report generated. LaTeX written to {tex_path.name}; PDF "
                        f"compilation skipped (no '{self._s.latex_engine}' on PATH)."
                    )
                    result.log("compile", ok=False, reason="engine unavailable")
                    return result
                try:
                    pdf_path = compile_pdf(tex_path)
                    result.pdf_path = str(pdf_path)
                    result.log("compile", ok=True, pdf=str(pdf_path))
                    emit("compiling_done", "PDF compiled.")
                except CompileError as e:
                    result.status = PipelineStatus.ERROR
                    result.message = f"LaTeX compilation failed: {e}"
                    result.log("compile", ok=False, error=str(e))
                    return result

            result.status = PipelineStatus.COMPLETED
            result.message = "Report generated and verified."
            emit("completed", "Report generated and verified.")
            return result

        except Exception as e:  # never leak a stack trace as the product
            log.exception("pipeline crashed for topic %r", topic)
            result.status = PipelineStatus.ERROR
            result.message = f"Pipeline error: {e}"
            result.log("error", error=str(e))
            return result

    # ------------------------------------------------------------------
    async def _extract_and_verify(
        self, topic: str, sources: list[Source], result: PipelineResult,
        emit=None,
    ) -> VerificationReport:
        """Extract candidate claims and run the harness, retrying up to
        max_reverify_attempts. Each retry widens the candidate source set for a
        claim (cross-source re-check) rather than re-drafting — we re-verify, we
        do not re-imagine."""
        def _emit(stage: str, msg: str) -> None:
            if emit:
                emit(stage, msg)

        _emit("extracting_start",
              f"Extracting candidate claims from {len(sources)} source(s)…")
        claims = await self._extract_claims(sources, on_progress=_emit)
        result.log("extract", candidate_claims=len(claims))
        _emit("extracting_done", f"Extracted {len(claims)} candidate claim(s).")

        # Per-claim progress so the UI shows movement instead of a frozen spinner.
        def _claim_progress(i: int, total: int, text: str) -> None:
            _emit("verifying_progress",
                  f"Verifying claim {i}/{total}: {text[:60]}")

        attempt = 0
        _emit("verifying_progress",
              f"Classifying {len(claims)} claim(s) across {self._s.verify_rounds} rounds…")
        verification = await self._harness.verify(
            claims, sources, on_progress=_claim_progress)
        result.log("verify", attempt=attempt, verified=verification.verified_count,
                   kept=verification.kept_count,
                   rejected=len(verification.rejected_claims),
                   rounds=verification.rounds_run, timed_out=verification.timed_out)
        if verification.timed_out:
            _emit("verifying_progress",
                  f"Verification time limit reached — proceeding with "
                  f"{verification.kept_count} claim(s) gathered so far.")

        # Re-verify only if too few claims survived AND we didn't already time out
        # (re-running after a timeout would just hit the wall again). The accept-first
        # strategies (conflict_only / aligned) are deterministic over a fixed claim set —
        # re-running cannot surface more claims and would re-accept (duplicating
        # evidence), so we never loop them.
        while (self._s.verify_strategy not in ("conflict_only", "aligned")
               and verification.kept_count < self._s.min_total_claims
               and not verification.timed_out
               and attempt < self._s.max_reverify_attempts):
            attempt += 1
            _emit("verifying_progress", f"Below threshold — re-verifying (attempt {attempt + 1})…")
            verification = await self._harness.verify(
                claims, sources, on_progress=_claim_progress)
            result.log("verify", attempt=attempt, verified=verification.verified_count,
                       kept=verification.kept_count,
                       rejected=len(verification.rejected_claims),
                       rounds=verification.rounds_run, timed_out=verification.timed_out)

        return verification

    async def _isolated_claim_checkpoint(
        self, verification: VerificationReport, gate: ConfirmationGate, emit
    ) -> str:
        """Human review gate over CORE isolated claims (孤证): kept claims backed by a
        single source whose credibility is L3 or stronger (numeric level <= 3). Emits
        an 'awaiting_confirmation' event whose detail is a JSON summary (one entry per
        claim: truncated text, source domain, credibility label), then blocks on the
        gate. Returns "continue" or "abort"; with no core isolated claims it returns
        "continue" immediately without emitting anything."""
        core = [c for c in verification.kept_claims
                if c.isolated and c.credibility is not None and int(c.credibility) <= 3]
        if not core:
            return "continue"
        summary = {
            "claims": [
                {
                    "text": c.text if len(c.text) <= 160 else c.text[:157].rstrip() + "…",
                    "domain": (charts_mod._domain_of(c.candidate_source_urls[0])
                               if c.candidate_source_urls else "unknown"),
                    "credibility": c.credibility.label,
                }
                for c in core
            ],
            "timeout_s": HITL_CONFIRM_TIMEOUT_S,
        }
        emit("awaiting_confirmation", json.dumps(summary, ensure_ascii=False))
        log.info("hitl: awaiting human confirmation for %d core isolated claim(s)",
                 len(core))
        decision = await gate.wait(timeout=HITL_CONFIRM_TIMEOUT_S)
        emit("confirmation_resolved",
             f"Isolated-claim review finished (decision: {decision}).")
        return decision

    async def _extract_claims(self, sources: list[Source], on_progress=None) -> list[Claim]:
        claims: list[Claim] = []
        counter = 0
        # Index claims by normalized text so the same fact from multiple sources
        # merges into ONE claim with multiple candidate sources — that is what
        # makes cross-referencing possible.
        by_text: dict[str, Claim] = {}
        total = len(sources)

        # CONCURRENCY: extract from all sources in parallel under a bounded semaphore
        # (llm_max_concurrency, matching the verify stage's pattern). Per-source
        # resilience is preserved: a rate-limited/failed extraction on ONE source logs
        # and yields [] so it cannot abort the run. Results stay index-aligned with
        # `sources` so the merge below is byte-for-byte deterministic in source order.
        sem = asyncio.Semaphore(max(1, self._s.llm_max_concurrency))

        async def _one(src: Source):
            async with sem:
                try:
                    return await self._llm.extract_claims(src.url, src.text)
                except Exception as e:
                    log.warning("extract: source %s failed (%s) — skipping",
                                src.url, type(e).__name__)
                    return []

        extracted_by_source = await asyncio.gather(*(_one(s) for s in sources))

        for i, (src, extracted) in enumerate(
                zip(sources, extracted_by_source, strict=True), start=1):
            for ex in extracted:
                key = _normalize(ex.text)
                if key in by_text:
                    claim = by_text[key]
                    for u in ex.candidate_source_urls:
                        if u not in claim.candidate_source_urls:
                            claim.candidate_source_urls.append(u)
                    # Multi-source claim: credibility is the STRONGEST source's level
                    # (highest grade = smallest number). Fill any slot the first
                    # occurrence left empty from the duplicate's extraction.
                    claim.credibility = _min_credibility(claim.credibility,
                                                         src.credibility)
                    for slot in ("entity", "attribute", "value", "qualifier",
                                 "time_scope"):
                        if getattr(claim, slot) is None:
                            setattr(claim, slot, getattr(ex, slot))
                    continue
                counter += 1
                claim = Claim(
                    id=f"c{counter}",
                    text=ex.text,
                    candidate_source_urls=list(ex.candidate_source_urls),
                    credibility=src.credibility,
                    entity=ex.entity,
                    attribute=ex.attribute,
                    value=ex.value,
                    qualifier=ex.qualifier,
                    time_scope=ex.time_scope,
                )
                by_text[key] = claim
                claims.append(claim)
            # Per-source progress so the extraction phase shows real movement instead of
            # relying only on the connection heartbeat.
            if on_progress:
                on_progress("extracting_progress",
                            f"Read {i}/{total} source(s); {len(claims)} claim(s) so far…")
        return claims

    async def _assemble(
        self, topic: str, verification: VerificationReport, sources: list[Source]
    ) -> ReportData:
        """Build the report over the 5 MANDATED sections, KG-driven where required.

        Broad-collection model: VERIFIED facts (HIGH/CORROBORATED) drive the KG, charts,
        and section prose — the grounded core of the report is unchanged. RUMOR-tier
        claims are collected separately and rendered in a clearly-labeled 'Unverified
        Signals & Rumors' section so no early signal is lost, but they are never mixed
        into the verified analysis or the charts.
        """
        verified = verification.verified_claims
        rumors = verification.rumor_claims if self._s.include_rumors_in_report else []

        # PERFORMANCE: KG synthesis, the 5 section drafts, and the deep-dive analyses are
        # all independent LLM generations. Running them SEQUENTIALLY was the assembly
        # bottleneck (10-15 slow generations back-to-back with NO deadline — exactly what
        # hung the "Synthesizing…" stage until the run-level timeout killed it). We now
        # issue them CONCURRENTLY (bounded by the global LLM semaphore that already caps
        # provider load) and wrap every sub-task in an assembly deadline, so a slow batch
        # degrades to partial output instead of hanging.
        deadline = time.monotonic() + self._s.assemble_deadline_s

        async def _guard(coro, fallback):
            """Run an assembly sub-task under the remaining assembly budget; on timeout
            or error return `fallback` so one slow generation can't sink the report."""
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("assemble: deadline exhausted — using fallback")
                return fallback
            try:
                return await asyncio.wait_for(coro, timeout=remaining)
            except (TimeoutError, Exception) as e:
                log.warning("assemble: sub-task failed/timed out (%s) — using fallback",
                            type(e).__name__)
                return fallback

        # Route each verified claim to a mandated section by keyword (cheap, local).
        buckets = self._bucket_claims(verified)

        # Locale-aware section plan: the 5 mandated sections plus the synthesis-layer
        # sections (source_comparison / limitations) when enabled.
        plan = self._s.resolved_sections()

        # 1. KG synthesis + 2. deep-dives + 3. section prose all run in ONE concurrent
        #    batch. KG and deep-dives read verified claims independently; section drafts
        #    are independent per section. All are guarded by the assembly deadline.
        kg_task = asyncio.create_task(
            _guard(build_report_kg(topic, verified, self._llm), None))
        deep_dives_task = asyncio.create_task(
            _guard(self._build_deep_dives(topic, verified, buckets), {}))

        async def _draft(section_id: str, heading: str) -> str:
            bucket = buckets.get(section_id, [])
            if not bucket:
                return ""
            # Isolated (single-source, low-credibility) claims carry an explicit
            # marker so the drafting prompt hedges them (workflow E: hedging rules).
            return await self._llm.draft_section(
                heading, [_claim_line(c) for c in bucket])

        draft_tasks = {
            sid: asyncio.create_task(_guard(_draft(sid, heading), ""))
            for sid, heading in plan if sid not in _SYNTHESIS_SECTION_IDS
        }

        # The source-comparison section drafts from the alignment clusters (not from
        # the claim buckets); it joins the same concurrent batch under the deadline.
        plan_headings = dict(plan)
        sc_task = None
        if "source_comparison" in plan_headings and verification.clusters:
            sc_task = asyncio.create_task(_guard(
                self._draft_source_comparison(verification,
                                              plan_headings["source_comparison"]),
                None))

        kg = await kg_task
        if kg is None:  # KG timed out/failed — degrade to an empty graph (no LLM call)
            kg = ReportKnowledgeGraph(topic=topic)
        # Render the standardized chart/table fragments once, from the (possibly empty)
        # KG plus the alignment clusters (comparison matrix).
        chart_blocks = (charts_mod.build_all(kg, verification.clusters)
                        if self._s.enable_charts else {})
        deep_dives_by_section = await deep_dives_task
        drafted = {sid: await t for sid, t in draft_tasks.items()}
        sc_section = await sc_task if sc_task is not None else None

        sections: list[ReportSection] = []
        for section_id, heading in plan:
            if section_id == "source_comparison":
                # Inserted right after competitive_landscape (the plan order); skipped
                # entirely when there are no alignment clusters to compare.
                if sc_section is not None:
                    sections.append(sc_section)
                continue
            if section_id == "limitations":
                # Deterministic closing section — rendered from verification results,
                # never drafted by the LLM.
                sections.append(self._limitations_section(verification, heading))
                continue
            bucket = buckets.get(section_id, [])
            prose = drafted.get(section_id, "")
            latex_blocks = self._section_charts(section_id, chart_blocks)
            dives = deep_dives_by_section.get(section_id, [])
            # Only emit a section if it has prose OR a chart OR a deep-dive; an empty
            # mandated section gets a transparent "insufficient verified data" note.
            paragraphs = [prose] if prose else []
            if not paragraphs and not latex_blocks and not dives:
                paragraphs = [
                    ("Insufficient independently verified data was available for this "
                     "section from authoritative sources.")
                ]
            sections.append(ReportSection(
                heading=heading, section_id=section_id, paragraphs=paragraphs,
                claim_ids=[c.id for c in bucket], latex_blocks=latex_blocks,
                deep_dives=dives,
            ))

        title = _report_title(topic, locale=self._s.report_locale)
        subject = _clean_topic(topic)
        abstract = (
            f"This report synthesizes {len(verified)} verified finding(s) on "
            f"{subject} drawn from a mix of sources — high-trust outlets "
            f"(government, investment-bank, industry-institution, non-profit, and "
            f"authoritative media) alongside other public web sources — plus "
            f"{len(rumors)} unverified signal(s) reported separately. Verified "
            f"findings are corroborated across independent sources or endorsed by a "
            f"high-trust outlet; unverified signals come from a single or "
            f"lower-trust source and are clearly labeled. Structured sections "
            f"(industry chain, market size, competitive landscape) are generated "
            f"from a knowledge graph built ONLY from verified claims."
        )
        return ReportData(
            topic=topic, title=title, abstract=abstract,
            sections=sections, claims=verified, sources=sources, kg=kg,
            rumors=rumors,
        )

    # keyword routing for the 5 mandated sections
    _SECTION_KEYWORDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "industry_overview": ("overview", "background", "history", "sector", "industry",
                              "technology", "adoption"),
        "policy_analysis": ("policy", "regulat", "government", "directive", "law",
                            "subsidy", "tariff", "compliance", "ministry", "standard"),
        "industry_chain": ("upstream", "midstream", "downstream", "supplies", "supply",
                          "component", "material", "manufactur", "distribut"),
        "market_size": ("market", "tam", "sam", "som", "cagr", "growth", "billion",
                       "revenue", "forecast", "size"),
        "competitive_landscape": ("competitor", "share", "player", "leader", "rival",
                                 "advantage", "vendor", "incumbent"),
    }

    def _bucket_claims(self, verified: list[Claim]) -> dict[str, list[Claim]]:
        """Assign each verified claim to the best-matching mandated section by keyword
        score, falling back to round-robin so every claim is placed somewhere."""
        buckets: dict[str, list[Claim]] = {sid: [] for sid, _ in self._s.report_sections}
        section_ids = [sid for sid, _ in self._s.report_sections]
        unassigned: list[Claim] = []
        for claim in verified:
            low = claim.text.lower()
            best_id, best_score = "", 0
            for sid, kws in self._SECTION_KEYWORDS.items():
                score = sum(1 for kw in kws if kw in low)
                if score > best_score:
                    best_id, best_score = sid, score
            if best_score > 0:
                buckets[best_id].append(claim)
            else:
                unassigned.append(claim)
        # Round-robin the leftovers so overview/analysis sections aren't starved.
        for i, claim in enumerate(unassigned):
            buckets[section_ids[i % len(section_ids)]].append(claim)
        return buckets

    async def _build_deep_dives(
        self, topic: str, verified: list[Claim], buckets: dict[str, list[Claim]]
    ) -> dict[str, list[DeepDive]]:
        """Identify high-value verified findings and expand each into a multi-paragraph
        analytical insight. Returns deep-dives grouped by mandated section id.

        Grounding guarantee: every insight is tied to verified claim ids, and any
        candidate referencing an unknown claim id is dropped — a deep-dive can only be
        built on claims that already cleared the harness. The drafting prompt forbids
        introducing new facts, so the analysis interprets verified data, never invents.
        """
        out: dict[str, list[DeepDive]] = {}
        if not self._s.enable_deep_dive or not verified:
            return out

        verified_ids = {c.id for c in verified}
        text_by_id = {c.id: c.text for c in verified}
        pairs = [(c.id, c.text) for c in verified]

        candidates = await self._llm.identify_insights(topic, pairs)
        section_ids = {sid for sid, _ in self._s.report_sections}

        # Build the list of deep-dives to DRAFT (cheap selection first), then draft them
        # all CONCURRENTLY. Drafting is the expensive part — up to max_deep_dives LLM
        # generations — so running it in parallel (bounded by the global LLM semaphore)
        # is what stops assembly from serializing into a timeout. Order is preserved so
        # the report is deterministic.
        Spec = tuple  # (section_id, title, kind, claim_ids, claim_texts)
        specs: list[Spec] = []
        seen_sections_from_insights: set[str] = set()
        for cand in candidates:
            if len(specs) >= self._s.max_deep_dives:
                break
            claim_ids = [cid for cid in cand.claim_ids if cid in verified_ids]
            if not claim_ids:
                continue
            section_id = cand.section_id if cand.section_id in section_ids else "industry_overview"
            claim_texts = [text_by_id[cid] for cid in claim_ids]
            specs.append((section_id, cand.title, cand.kind, claim_ids, claim_texts))
            seen_sections_from_insights.add(section_id)

        # Fallback specs: guarantee analytical depth for every section with enough claims
        # but no insight-driven deep-dive yet (same policy as before, just queued not drafted).
        if buckets:
            for section_id in section_ids:
                if len(specs) >= self._s.max_deep_dives:
                    break
                if section_id in seen_sections_from_insights:
                    continue
                bucket_claims = buckets.get(section_id, [])
                if len(bucket_claims) < self._s.deep_dive_section_min_claims:
                    continue
                section_title = dict(self._s.report_sections).get(section_id, section_id)
                specs.append((
                    section_id, f"{section_title}: Analytical Perspective", "analysis",
                    [c.id for c in bucket_claims], [c.text for c in bucket_claims],
                ))

        if not specs:
            return out

        async def _draft(spec):
            section_id, title, kind, claim_ids, claim_texts = spec
            try:
                paragraphs = await self._llm.draft_deep_dive(title, kind, claim_texts)
            except Exception as e:  # one failed dive must not sink assembly
                log.warning("deep-dive draft failed for %r (%s) — skipping",
                            title, type(e).__name__)
                return None
            if len(paragraphs) < self._s.deep_dive_min_paragraphs:
                return None
            return section_id, DeepDive(title=title, kind=kind,
                                        paragraphs=paragraphs, claim_ids=claim_ids)

        drafted = await asyncio.gather(*[_draft(s) for s in specs])
        for res in drafted:
            if res is None:
                continue
            section_id, dive = res
            out.setdefault(section_id, []).append(dive)
        return out

    async def _draft_source_comparison(
        self, verification: VerificationReport, heading: str
    ) -> ReportSection | None:
        """Draft the 'Source Comparison & Divergence' section from the alignment
        clusters: each cluster becomes one material line naming the entity/attribute,
        every source's value + qualifier (口径), and the intra-cluster relation.
        Lines whose cells rest on an isolated claim carry the UNVERIFIED marker so
        the hedging rules apply. Returns None when there is nothing to compare."""
        clusters = verification.clusters
        if not clusters:
            return None
        id_by_text = {c.text: c.id for c in verification.kept_claims}
        isolated_texts = {c.text for c in verification.kept_claims if c.isolated}
        lines: list[str] = []
        claim_ids: list[str] = []
        for cluster in clusters:
            cells: list[str] = []
            isolated = False
            for cell in cluster.get("cells", []):
                domain = charts_mod._domain_of(str(cell.get("source_url") or ""))
                value = str(cell.get("value") or "n/a")
                qualifier = str(cell.get("qualifier") or "no stated qualifier")
                cells.append(f"{domain} reports {value} ({qualifier})")
                text = str(cell.get("claim_text") or "")
                if text in isolated_texts:
                    isolated = True
                cid = id_by_text.get(text)
                if cid and cid not in claim_ids:
                    claim_ids.append(cid)
            line = (f"{cluster.get('entity') or 'Unknown entity'} — "
                    f"{cluster.get('attribute') or 'metric'}: "
                    + "; ".join(cells)
                    + f". Relation among sources: "
                      f"{cluster.get('relation') or 'corroborate'}.")
            lines.append(f"[UNVERIFIED - single source] {line}" if isolated else line)
        prose = await self._llm.draft_section(
            heading, lines,
            instructions=(
                "This section compares how different sources report the SAME metrics. "
                "Attach the reporting source (domain) to EVERY sentence that states a "
                "figure. For clusters whose relation is 'complement', explain the "
                "difference in calibre/qualifier (口径) or time scope that reconciles "
                "the figures. For clusters whose relation is 'conflict', state which "
                "source prevailed on authority and that the lower-authority figure "
                "was discarded. Hedge every figure that rests on a single source."
            ))
        return ReportSection(
            heading=heading, section_id="source_comparison",
            paragraphs=[prose] if prose else [], claim_ids=claim_ids)

    def _limitations_section(
        self, verification: VerificationReport, heading: str
    ) -> ReportSection:
        """The deterministic closing 'Information Limitations' section — rendered
        straight from verification results, never drafted by the LLM. Lists every
        isolated (孤证) claim with its source domain, credibility label, and the
        review/downgrade note; also surfaces verification limits such as a timed-out
        harness. With no isolated claims it states full corroboration."""
        zh = (self._s.report_locale or "en").lower().startswith("zh")
        isolated = [c for c in verification.kept_claims if c.isolated]
        paragraphs: list[str] = []
        if isolated:
            paragraphs.append(
                "以下声明仅由单一来源支持，尚未获得独立来源的交叉验证，请谨慎采信："
                if zh else
                "The following claim(s) rest on a single source and have not been "
                "independently corroborated; read them with caution:")
            for c in isolated:
                domain = (charts_mod._domain_of(c.candidate_source_urls[0])
                          if c.candidate_source_urls else "unknown")
                label = (c.credibility.label if c.credibility is not None
                         else ("未评级" if zh else "ungraded"))
                reason = verification.reasons.get(c.id, "")
                if zh:
                    note = f" 审查注记：{reason}" if reason else ""
                    paragraphs.append(
                        f"「{c.text}」（来源：{domain}；可信度：{label}）。{note}".rstrip())
                else:
                    note = f" Review note: {reason}" if reason else ""
                    paragraphs.append(
                        f'"{c.text}" (source: {domain}; credibility: {label}).'
                        f"{note}".rstrip())
        else:
            paragraphs.append(
                "所有关键声明均已由多个独立来源交叉验证。" if zh else
                "All key claims were corroborated by multiple independent sources.")
        if verification.timed_out:
            paragraphs.append(
                "验证阶段达到时间预算上限，部分声明可能未完成全部交叉核查。" if zh else
                "Verification reached its time budget; some claims may not have been "
                "fully cross-checked.")
        return ReportSection(
            heading=heading, section_id="limitations", paragraphs=paragraphs,
            claim_ids=[c.id for c in isolated])

    @staticmethod
    def _section_charts(section_id: str, blocks: dict[str, str]) -> list[str]:
        """Pick the pre-rendered chart fragments that belong to a section, dropping
        empties (a fragment is "" when the KG lacked enough verified data)."""
        mapping = {
            "industry_chain": ["chain_tikz"],
            "market_size": ["market_callout", "market_plot"],
            "competitive_landscape": ["competitive_table", "competitive_bar",
                                      "comparison_matrix"],
        }
        keys = mapping.get(section_id, [])
        return [blocks[k] for k in keys if blocks.get(k)]

    def _out_of_scope(self, result: PipelineResult, reason: str) -> PipelineResult:
        result.status = PipelineStatus.OUT_OF_SCOPE
        result.message = out_of_scope_message(self._s.report_locale)
        result.log("scope_gate", ok=False, reason=reason)
        log.info("out-of-scope for %r: %s", result.topic, reason)
        return result


# Synthesis-layer section ids (workflow E): built from verification results, not from
# the claim buckets, so they are excluded from the generic draft loop in _assemble.
_SYNTHESIS_SECTION_IDS = frozenset({"source_comparison", "limitations"})


def _claim_line(claim: Claim) -> str:
    """Claim text as fed to section drafting; isolated (single-source, low-credibility)
    claims carry an explicit marker so the drafting prompt hedges them and tests can
    assert the marker survives (workflow E: hedging rules)."""
    if claim.isolated:
        return f"[UNVERIFIED - single source] {claim.text}"
    return claim.text


def _merge_sources(
    base: list[Source], extra: list[Source]
) -> list[Source]:
    """Combine two source lists, deduped by URL. `base` (freshly fetched) wins on a
    URL collision so a re-fetch is never clobbered by a stored copy; otherwise the
    injected source is appended. Order is preserved (base first, then new extras)."""
    seen: set[str] = set()
    merged: list[Source] = []
    for s in list(base) + list(extra):
        if s.url in seen:
            continue
        seen.add(s.url)
        merged.append(s)
    return merged


def _distinct_authoritative_domains(sources: list[Source]) -> set[str]:
    """Distinct authoritative domains counted from the Source objects' assigned kind.

    Unlike sources.distinct_authoritative_domains (which re-classifies a URL against the
    allowlist), this trusts the kind already on the Source — so USER_PROVIDED documents
    with synthetic URLs still count toward the scope gate."""
    from core_engine.report.models import SourceKind

    return {s.domain for s in sources if s.kind is not SourceKind.REJECTED}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _min_credibility(a, b):
    """Stronger (smaller-numbered) of two CredibilityLevels; None means unset."""
    if a is None:
        return b
    if b is None:
        return a
    return a if int(a) <= int(b) else b


# Leading command phrases users type ("Generate a report about …", "Write a
# report on …") that must NOT leak into the report's own title. We strip them so
# the title names only the actual subject.
_COMMAND_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:generate|create|make|write|produce|build|draft|prepare|give\s+me|"
    r"do|compile|research|analyze|analyse|investigate|study|explore|"
    r"summarize|summarise|document|explain|tell\s+me\s+about|"
    r"i\s+(?:want|need|would\s+like))\b"
    r"(?:\s+(?:me|us))?"
    r"(?:\s+(?:a|an|the|some))?"
    r"(?:\s+(?:brief|short|detailed|comprehensive|full|quick|formal|"
    r"in-?depth|thorough))?"
    r"(?:\s+(?:industry\s+)?(?:research\s+)?"
    r"(?:report|paper|analysis|overview|study|summary|document|breakdown|"
    r"deep\s*dive|write-?up))?"
    r"(?:\s+(?:about|on|of|for|regarding|concerning|covering|into|re))?"
    r"\b[\s:,-]*",
    re.IGNORECASE,
)

# Small words kept lowercase in title case unless they lead the title.
_TITLE_MINOR_WORDS = frozenset({
    "a", "an", "the", "and", "or", "nor", "but", "for", "of", "on", "in",
    "to", "at", "by", "vs", "via", "with", "from",
})


def _clean_topic(topic: str) -> str:
    """Strip any leading command phrase so we keep only the real subject.

    'Generate a report about solar cells' -> 'solar cells'
    'Solid-state battery supply chain'    -> 'Solid-state battery supply chain'
    Iterated so stacked prefixes ('please write a report on …') fully unwind,
    but never returns empty — if stripping would erase everything, keep the
    original topic.
    """
    cleaned = topic.strip()
    for _ in range(3):
        stripped = _COMMAND_PREFIX_RE.sub("", cleaned, count=1).strip()
        if not stripped or stripped == cleaned:
            break
        cleaned = stripped
    return cleaned or topic.strip()


def _title_case(text: str) -> str:
    """Title-case a subject while preserving existing acronyms (AI, EV, PV) and
    keeping minor words lowercase mid-title."""
    words = text.split()
    out: list[str] = []
    for i, w in enumerate(words):
        low = w.lower()
        if w.isupper() and len(w) > 1:
            out.append(w)                       # keep acronyms: AI, EV, GDP
        elif i != 0 and low in _TITLE_MINOR_WORDS:
            out.append(low)
        else:
            out.append(w[:1].upper() + w[1:])   # capitalise, keep inner caps
    return " ".join(out)


def _report_title(topic: str, locale: str = "en") -> str:
    """Build the report's own title from the topic, without echoing the user's
    command. E.g. 'Generate a report about solar cell' -> 'A Brief Industry
    Research Report of Solar Cell'. Under the zh locale: '<subject>行业简报'."""
    subject = _clean_topic(topic).rstrip(".").strip()
    if (locale or "en").lower().startswith("zh"):
        return f"{subject}行业简报" if subject else "行业简报"
    subject = _title_case(subject) if subject else subject
    if not subject:
        return "A Brief Industry Research Report"
    return f"A Brief Industry Research Report of {subject}"
