"""Question-led reporting with bounded recovery and auditable analytical judgments.

Plan -> gather -> extract/verify -> source-bind premises -> assess evidence gaps
     -> bounded supplemental search and re-verification -> question coverage gate
     -> draft/audit factual prose -> metric calculations and chart validation
     -> separate analytical judgments + critical review -> render/compile.

Document-only runs never search the web. Supplemental research has hard query,
round, page and time limits. Unsupported requests produce an evidence-gap artifact.
Model-proposed interpretations are labeled inferences, never source assertions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import ClassVar

from core_engine.config import get_settings
from core_engine.report import charts as charts_mod
from core_engine.report.answerability import assess_coverage
from core_engine.report.chart_audit import validate_charts
from core_engine.report.compile import CompileError, compile_pdf, engine_available
from core_engine.report.discovery import discovery_query, equivalent_terms
from core_engine.report.evidence_quality import assess_evidence, origin_groups
from core_engine.report.generation import profile_audit
from core_engine.report.grounding import AnchorReuse, SentenceGrounder, parse_draft, write_evidence
from core_engine.report.kg import ReportKnowledgeGraph, build_report_kg
from core_engine.report.latex import write_tex
from core_engine.report.llm import LLM, get_llm
from core_engine.report.metrics import catalog_from_claims
from core_engine.report.models import (
    Claim,
    DeepDive,
    PipelineResult,
    PipelineStatus,
    ReportData,
    ReportSection,
    ReportSentence,
    SentenceCitation,
    Source,
    out_of_scope_message,
)
from core_engine.report.publication import apply_policy, lead_records
from core_engine.report.reasoning import analyze
from core_engine.report.research import (
    ResearchPlan,
    ResearchSession,
    build_plan,
    evidence_requirement,
    related,
)
from core_engine.report.scientific import comparison_audit, table_candidates
from core_engine.report.scrape import (
    Fetcher,
    SearchProvider,
    SearchUnavailableError,
    gather_sources,
    get_fetcher,
    get_search_provider,
    retryable_fetch_failure,
)
from core_engine.report.text import normalize_text, numeric_tokens
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
        skip_initial_search: bool = False,
        confirmation_gate: ConfirmationGate | None = None,
        research_plan: ResearchPlan | None = None,
        seed_claims: list[Claim] | None = None,
        retained_leads: list[dict] | None = None,
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
          - `skip_initial_search`: resume from supplied sources and target evidence
            gaps online, without repeating the broad initial collection.

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
        generation_profile = profile_audit(self._s) if self._s.generation_depth else None
        if generation_profile:
            result.log("generation_profile", **generation_profile)
            result.research_audit["generation_profile"] = generation_profile
        topic = topic.strip()
        result.topic = topic
        extra_sources = extra_sources or []
        anchor_reuse = AnchorReuse()
        self._extraction_cache: dict = {}
        self._candidate_archive: dict[str, Claim] = {}
        for index, lead in enumerate(retained_leads or [], 1):
            candidate = Claim(f"retained_{index}", lead["text"], candidate_source_urls=lead.get("candidate_source_urls", []))
            self._candidate_archive[normalize_text(candidate.text)] = candidate
        self._research_questions: list[str] = []
        self._local_claims: list[Claim] = []
        self._next_local_id = max([self._s.extraction_max_claims] + [int(c.id[1:]) for c in (seed_claims or [])
                                                                 if re.fullmatch(r"c\d+", c.id)])
        if not topic:
            result.status = PipelineStatus.OUT_OF_SCOPE
            result.message = out_of_scope_message(self._s.report_locale)
            result.log("intake", ok=False, reason="empty topic")
            emit("out_of_scope", "Empty topic.")
            return result

        try:
            if skip_initial_search and not extra_sources:
                raise ValueError("Resuming evidence-gap research requires supplied source snapshots")
            if seed_claims and len({c.id for c in seed_claims}) != len(seed_claims):
                raise ValueError("Seed claim identifiers must be unique")
            session = None
            if self._s.research_enabled:
                emit("planning_start", "Planning research questions and evidence requirements…")
                if research_plan is None:
                    research_plan = await build_plan(topic, self._llm, self._s)
                if (research_plan.objective != topic or not research_plan.questions
                        or research_plan.questions[0].question != topic or not research_plan.questions[0].required):
                    raise ValueError("A replayed research plan must preserve the exact task and questions")
                self._research_questions = [q.question + " " + evidence_requirement(research_plan, q, self._s.report_locale)
                                            for q in research_plan.questions]
                session = ResearchSession(research_plan, self._llm, self._s, anchor_reuse=anchor_reuse)
                if generation_profile:
                    session.audit["generation_profile"] = generation_profile
                result.research_plan = research_plan.to_dict()
                result.log("research_plan", **result.research_plan)
                emit("planning_done", f"Planned {len(research_plan.questions)} question(s) with explicit evidence requirements.")
            rejected: list = []
            initial_search_diagnostics: list[dict] = []
            if skip_search or skip_initial_search:
                # Comprehensive step: no web search — the pool IS the injected sources.
                emit("scraping_start",
                     f"Using {len(extra_sources)} stored/uploaded source(s); "
                     + ("skipping broad collection; evidence-gap searches remain enabled…"
                        if skip_initial_search and not skip_search else "skipping web search…"))
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
                        max_pages=(max(0 if extra_sources else 1, self._s.scrape_max_pages - len(extra_sources)
                            - min(len(session.plan.questions), self._s.research_max_queries, self._s.research_max_rounds * 2))
                            if session else None),
                        on_progress=_gather_progress, llm=self._llm,
                        queries=session.plan.initial_queries() if session else None, settings=self._s,
                        diagnostics=initial_search_diagnostics)
                except SearchUnavailableError as e:
                    # Search backend blocked/unconfigured — this is NOT 'out of scope'.
                    # Surface the real, actionable reason (the DDG-block masking bug).
                    result.status = PipelineStatus.ERROR
                    result.message = f"Search unavailable: {e}"
                    result.log("gather", ok=False, error=str(e), query_diagnostics=initial_search_diagnostics)
                    if session is not None:
                        session.audit["initial_queries"] = initial_search_diagnostics
                        session.audit["stop_reason"] = "initial_retrieval_error"
                        result.research_audit = session.audit
                    emit("error", f"Search unavailable: {e}")
                    return result
                # Merge any injected sources (brief sources + user docs) with the freshly
                # fetched pool, deduped by URL.
                sources = _merge_sources(fetched, extra_sources)
                result.log("gather", authoritative=len(sources), rejected=len(rejected),
                           injected=len(extra_sources),
                           query_diagnostics=initial_search_diagnostics,
                           kept_domains=sorted({s.domain for s in sources}))
                emit("scraping_done",
                     f"Collected {len(sources)} source(s) ({len(rejected)} discarded"
                     + (f", {len(extra_sources)} user-provided" if extra_sources else "")
                     + ").")

            verification = None
            if session is not None:
                session.audit["initial_rejections"] = [{"url": hit.url, "reason": hit.metadata.get("rejection_reason", "source_filter")}
                                                        for hit in rejected]
                session.audit["initial_queries"] = [] if skip_search or skip_initial_search else [
                    {"query": query, "question_id": next(q.id for q in session.plan.questions if q.question == query),
                     "kind": next(q.kind for q in session.plan.questions if q.question == query),
                     **next((record for record in initial_search_diagnostics if record["query"] == query),
                            {"status": "completed"})}
                    for query in session.plan.initial_queries()]
                verification = (await self._extract_and_verify(topic, sources, result, emit,
                                    bounded_research=True, candidate_claims=seed_claims,
                                    isolation_reviews=session.isolation_reviews) if sources else VerificationReport())
                sources, verification = await self._fill_evidence_gaps(
                    session, sources, verification, result, emit, skip_search=skip_search)
                result.research_audit, result.research_evidence = session.audit, session.evidence
                result.log("research_summary", **session.audit)

            # --- Scope gate #1: do we even have enough footing to proceed? ---
            # Count distinct authoritative domains directly off the Source objects'
            # assigned kind, NOT by re-classifying the URL — user-provided documents have
            # synthetic URLs that the allowlist would reject, but they ARE authoritative.
            domains = _distinct_authoritative_domains(sources)
            grounded_coverage = assess_coverage(topic, session.claims, sources, topic_aliases=equivalent_terms(topic)) if session else None
            sparse_eligible = bool(self._s.sparse_evidence_enabled and session and session.claims and grounded_coverage
                                   and grounded_coverage.status != "insufficient")
            origins = len(set(origin_groups(sources).values()))
            limited_evidence = bool(sparse_eligible and (origins < self._s.min_authoritative_sources
                                   or (verification is not None and verification.kept_count < self._s.min_total_claims)))
            if limited_evidence:
                session.audit["evidence_mode"] = "limited_evidence"
                result.log("sparse_evidence", mode="limited_evidence", origin_groups=origins,
                           source_urls=len(sources), truth_established=False)
            if origins < self._s.min_authoritative_sources and not sparse_eligible:
                # Distinguish the failure modes so the reason is actionable, not a
                # generic 'out of scope' for what is really a data-gathering shortfall.
                if not sources and not rejected:
                    reason = ("no search results returned — the search provider may be "
                              "unreachable or not configured")
                elif not sources:
                    reason = (f"found {len(rejected)} page(s) but none passed the source "
                              f"filter or relevance check")
                else:
                    reason = (f"only {origins} distinguishable source origin(s) across {len(domains)} domain(s); "
                              f"need {self._s.min_authoritative_sources}")
                emit("scraping", f"Insufficient sources: {reason}.")
                result.answerability = {"status": "insufficient", "missing": ["authoritative_sources"],
                                        "reason": reason}
                result.log("scope_gate", ok=False, reason=reason)
                return self._unanswerable(result, sources, emit)

            # --- Extract + verify, with bounded re-verify loop ---
            accept_first = self._s.verify_strategy in ("conflict_only", "aligned")
            emit("verifying_start",
                 "Retaining candidates and reviewing disagreements…" if accept_first
                 else "Cross-verifying claims across sources…")
            if verification is None:
                verification = await self._extract_and_verify(topic, sources, result, emit,
                    bounded_research=skip_search, candidate_claims=seed_claims,
                    isolation_reviews=session.isolation_reviews if session is not None else None)
            # Every strategy passes this gate, including runs without research.
            # Retention/authority alone never makes an observation publishable.
            candidates = list({c.id: c for c in verification.kept_claims + verification.rejected_claims}.values())
            present = {normalize_text(c.text) for c in candidates}
            used_ids = {c.id for c in candidates}
            for index, claim in enumerate(self._candidate_archive.values(), 1):
                if normalize_text(claim.text) in present:
                    continue
                identifier = f"lead_{index}"
                while identifier in used_ids:
                    identifier += "_retained"
                used_ids.add(identifier)
                # Earlier retrieval rounds must not silently lose candidates.
                # New IDs prevent a later extraction's IDs from borrowing old proof.
                candidates.append(replace(claim, id=identifier, evidence=[], verified=False,
                                          confidence=None, conflicting_claim_ids=[]))
            if session is not None:
                source_citations = [SentenceCitation(**c) for c in session.evidence.get("citations", [])]
            else:
                _, source_citations = await SentenceGrounder(
                    self._llm, self._s, anchor_reuse=anchor_reuse).anchor_claims(candidates, sources)
            result.evidence_policy = apply_policy(candidates, sources, source_citations)
            verification.kept_claims = [c for c in candidates if c.reporting_status != "lead_only"]
            verification.rejected_claims = [c for c in candidates if c.reporting_status == "lead_only"]
            result.evidence_policy["leads"] = lead_records(candidates)
            if session is None:
                result.research_evidence = {"claims": [asdict(c) for c in candidates],
                                           "citations": [asdict(c) for c in source_citations]}
            result.log("publication_policy", **result.evidence_policy)
            emit("verifying_done", f"Source-bound: {len(verification.verified_claims)}; "
                 f"disputed: {result.evidence_policy['disputed']}; leads: {result.evidence_policy['lead_only']}.")

            # --- Scope gate #2: broad-collection model. A report is out-of-scope only
            #     if too FEW claims survive at ANY tier (rumors included). Rumors are
            #     kept and labeled, so they count toward having something to analyze. ---
            if verification.kept_count < self._s.min_total_claims and not sparse_eligible:
                result.answerability = {"status": "insufficient", "missing": ["supported_claims"],
                                        "kept_claims": verification.kept_count,
                                        "required_claims": self._s.min_total_claims}
                result.log("scope_gate", ok=False, reason="too few supported claims")
                return self._unanswerable(result, sources, emit)

            coverage = assess_coverage(topic, session.claims if session and session.audit["questions"]
                                       else verification.kept_claims, sources, topic_aliases=equivalent_terms(topic))
            result.answerability = coverage.to_dict()
            result.log("answerability", **result.answerability)
            if coverage.status == "insufficient":
                return self._unanswerable(result, sources, emit)

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

            # --- Assemble attributed statements, disagreements and retained leads ---
            emit("analyzing_start", "Synthesizing knowledge graph and analysis…")
            focus_ids = None
            if session and session.plan.requested_metrics:
                focus_ids = {cid for row in session.audit["questions"] for cid in row["claim_ids"]}
                focus_ids.update(c.id for c in verification.kept_claims if related(topic, c.text))
            report = await self._assemble(topic, verification, sources, research_plan=result.research_plan,
                                          focus_claim_ids=focus_ids)
            report.research_plan, report.research_audit = result.research_plan, result.research_audit
            report.research_evidence = result.research_evidence
            report.evidence_policy = result.evidence_policy
            emit("grounding_start", "Auditing each draft sentence against source passages…")
            report = await SentenceGrounder(self._llm, self._s, anchor_reuse=anchor_reuse).audit_report(report)
            result.evidence_policy = report.evidence_policy
            supported_ids = {cid for sentence in report.all_sentences() if sentence.status == "supported"
                             for cid in sentence.claim_ids}
            final_coverage = assess_coverage(topic, [c for c in report.claims + report.rumors + report.disputed_claims
                                             if c.id in supported_ids], sources,
                                             topic_aliases=equivalent_terms(topic))
            result.answerability = final_coverage.to_dict()
            result.log("grounded_answerability", **result.answerability)
            if final_coverage.status == "insufficient":
                return self._unanswerable(result, sources, emit)
            report.answerability = result.answerability
            if report.disputed_claims:
                report.answerability["unresolved_source_conflicts"] = [c.id for c in report.disputed_claims]
                report.answerability["status"] = "undetermined" if report.claims else "insufficient"
                report.answerability["missing"] = list(dict.fromkeys([
                    *report.answerability.get("missing", []), "unresolved_source_conflict"]))
            report.source_quality = assess_evidence(report.claims + report.rumors + report.disputed_claims, sources, report.citations)
            report.source_quality["evidence_mode"] = "limited_evidence" if limited_evidence else "standard"
            metrics, metric_issues = catalog_from_claims(report.claims, report.citations, sources=report.sources)
            if report.kg is not None:
                graph, clusters, derived, calculations, chart_audit, calculation_issues = validate_charts(
                    report.kg, metrics, report.claims, report.citations, verification.clusters)
                report.kg, report.calculations, report.chart_audit = graph, calculations, chart_audit
                metrics += derived
                metric_issues += calculation_issues
                blocks = charts_mod.build_all(graph, clusters) if self._s.enable_charts else {}
                for section in report.sections:
                    section.latex_blocks = self._section_charts(section.section_id, blocks)
                    if section.latex_blocks:
                        section.sentences = [s for s in section.sentences if not (
                            s.status == "editorial" and s.text.startswith("Insufficient independently"))]
                        section.paragraphs = [s.text for s in section.sentences if s.status != "withheld"]
                report.grounding_audit["chart_blocks_requiring_review"] = (sum(
                    entry["status"] == "withheld" for entry in chart_audit) if self._s.enable_charts else 0)
            report.metrics = [metric.to_dict() for metric in metrics]
            report.research_audit["scientific_comparisons"] = comparison_audit(metrics)
            report.metric_issues = metric_issues
            if session is not None:
                emit("reasoning_start", "Analyzing implications, assumptions and rival explanations…")
                report.analysis, report.analysis_audit = await analyze(
                    result.research_plan, result.research_audit, result.research_evidence,
                    report.calculations, report.metrics, self._llm, self._s)
                result.log("analysis_audit", **report.analysis_audit)
                emit("reasoning_done", f"{len(report.analysis)} analytical conclusion(s) retained; "
                     f"{len(report.analysis_audit['withheld'])} withheld.")
            if (metric_issues or report.grounding_audit["chart_blocks_requiring_review"] or limited_evidence
                    or report.disputed_claims or report.leads):
                report.quality_status = "needs_review"
            if (session is not None and (session.audit.get("coverage_status") != "complete"
                    or report.analysis or report.analysis_audit.get("withheld")
                    or report.analysis_audit.get("status") in {"unavailable", "deadline"})):
                report.quality_status = "needs_review"
            result.log("sentence_grounding", **report.grounding_audit,
                       quality_status=report.quality_status)
            emit("grounding_done", f"{report.grounding_audit['supported_sentences']} sentence(s) supported; "
                 f"{report.grounding_audit['withheld_sentences']} withheld for review.")
            result.report = report
            result.log("assemble", sections=len(report.sections),
                       verified_claims=len(report.claims),
                       cited_sources=len(report.bibliography()))
            emit("analyzing_done",
                 f"Assembled {len(report.sections)} section(s) from "
                 f"{len(report.claims)} claim(s).")

            # --- Render LaTeX ---
            emit("generating_latex_start", "Generating LaTeX document…")
            tex_path = await asyncio.to_thread(write_tex, report, self._s.output_dir, settings=self._s)
            result.tex_path = str(tex_path)
            result.evidence_path = str(await asyncio.to_thread(write_evidence, report, tex_path))
            result.log("render", tex=str(tex_path))
            emit("generating_latex_done", "LaTeX document generated.")

            # --- Compile PDF (degrade gracefully if no toolchain) ---
            if compile_to_pdf:
                emit("compiling_start", "Compiling PDF…")
                if not engine_available(self._s.latex_engine):
                    result.status = PipelineStatus.COMPLETED
                    result.message = (
                        f"Report generated. LaTeX written to {tex_path.name}; PDF "
                        f"compilation skipped (no '{self._s.latex_engine}' on PATH)."
                    )
                    result.log("compile", ok=False, reason="engine unavailable")
                    return result
                try:
                    pdf_path = await asyncio.to_thread(compile_pdf, tex_path, settings=self._s)
                    result.pdf_path = str(pdf_path)
                    result.log("compile", ok=True, pdf=str(pdf_path))
                    emit("compiling_done", "PDF compiled.")
                except CompileError as e:
                    result.status = PipelineStatus.ERROR
                    result.message = f"LaTeX compilation failed: {e}"
                    result.log("compile", ok=False, error=str(e))
                    return result

            result.status = PipelineStatus.COMPLETED
            result.message = (
                "Report generated with sentence-level source citations."
                if report.quality_status == "source_supported" else
                "Review draft generated; inspect research gaps, analytical judgments and the evidence audit."
            )
            emit("completed", result.message)
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
        emit=None, *, bounded_research=False, candidate_claims=None, strict_context=False,
        isolation_reviews: dict[tuple[str, str], str] | None = None,
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
        if candidate_claims is None:
            extraction_audit: list[dict] = []
            try:
                claims = await self._extract_claims(sources, on_progress=_emit, audit=extraction_audit)
            finally:
                result.log("extraction_sources", sources=extraction_audit)
            local = getattr(self, "_local_claims", [])
            local_text = {normalize_text(c.text) for c in local}
            claims = [c for c in claims if normalize_text(c.text) not in local_text]
            claims = claims[:max(0, self._s.extraction_max_claims - len(local))] + local
        else:
            claims = candidate_claims[:self._s.extraction_max_claims]
        claims = [replace(c, evidence=[], verified=False, confidence=None, verification_rounds=0,
                          candidate_source_urls=list(c.candidate_source_urls), conflicting_claim_ids=[])
                  for c in claims]
        archive = getattr(self, "_candidate_archive", None)
        if archive is not None:
            for claim in claims:
                archive[normalize_text(claim.text)] = claim
        result.log("extract", candidate_claims=len(claims))
        _emit("extracting_done", f"Extracted {len(claims)} candidate claim(s).")

        # Per-claim progress so the UI shows movement instead of a frozen spinner.
        def _claim_progress(i: int, total: int, text: str) -> None:
            _emit("verifying_progress",
                  f"Verifying claim {i}/{total}: {text[:60]}")

        # The research loop owns all supplemental retrieval. The old isolated-claim
        # proxy search must not escape its budget, or document-only mode.
        harness = self._harness
        if bounded_research or strict_context:
            harness = VerificationHarness(self._llm, fetcher=self._fetcher)
            harness._s = self._s.model_copy(update={"active_verify_enabled": False,
                **({"verify_strategy": "cross_reference", "verify_mode": "strict"} if strict_context else {})})
        attempt = 0
        _emit("verifying_progress",
              f"Classifying {len(claims)} claim(s) across {self._s.verify_rounds} rounds…")
        verification = await harness.verify(claims, sources, on_progress=_claim_progress,
                                            isolation_reviews=isolation_reviews)
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
        while (not bounded_research and self._s.verify_strategy not in ("conflict_only", "aligned")
               and verification.kept_count < self._s.min_total_claims
               and not verification.timed_out
               and attempt < self._s.max_reverify_attempts):
            attempt += 1
            _emit("verifying_progress", f"Below threshold — re-verifying (attempt {attempt + 1})…")
            verification = await harness.verify(
                claims, sources, on_progress=_claim_progress)
            result.log("verify", attempt=attempt, verified=verification.verified_count,
                       kept=verification.kept_count,
                       rejected=len(verification.rejected_claims),
                       rounds=verification.rounds_run, timed_out=verification.timed_out)

        return verification

    async def _fill_evidence_gaps(self, session, sources, verification, result, emit, *, skip_search):
        deadline = time.monotonic() + self._s.research_deadline_s
        queries_used = 0
        rejected_urls = {row["url"] for row in session.audit.get("initial_rejections", [])
                         if not retryable_fetch_failure(row.get("reason"))}
        session.audit["search_policy"] = "document_only" if skip_search else "web_allowed"
        session.audit["disagreements"] = [c for c in verification.clusters if c.get("relation") == "conflict"]

        async def bounded(awaitable):
            return await asyncio.wait_for(awaitable, timeout=max(0.001, deadline - time.monotonic()))

        try:
            emit("research_audit", "Mapping source-supported evidence to the research questions…")
            await bounded(session.evaluate(verification.kept_claims, sources))
            local_possible = self._s.research_local_max_chunks > 0 and any(s.metadata.get("full_text_accessed") for s in sources)
            if skip_search and not local_possible:
                session.audit["stop_reason"] = "document_only"
                return sources, verification
            if self._s.research_max_rounds == 0 or (self._s.research_max_queries == 0 and not local_possible):
                session.audit["stop_reason"] = "document_only" if skip_search else "retrieval_disabled"
                return sources, verification
            for round_number in range(1, self._s.research_max_rounds + 1):
                searched_questions = {row["question_id"] for row in session.audit["searches"]}
                unsearched = {row["id"] for row in session.audit["questions"]
                              if row["status"] != "supported" and row["id"] not in searched_questions}
                if (session.audit["coverage_status"] == "complete"
                        and len(set(origin_groups(sources).values())) >= self._s.min_authoritative_sources):
                    session.audit["stop_reason"] = "document_only" if skip_search else "questions_covered"
                    break
                if self._s.research_fulltext_enabled and not skip_search:
                    from core_engine.report.fulltext import fulltext_candidates

                    past_attempts = session.audit.get("fulltext_attempts", [])
                    counts = Counter(row["url"] for row in past_attempts)
                    blocked = {row["url"] for row in past_attempts if row.get("status") == "retrieved"
                               or not retryable_fetch_failure(row.get("reason"))}
                    fulltexts = sum(bool(s.metadata.get("fulltext_parent_url")) for s in sources)
                    leads = [h for h in fulltext_candidates(sources, self._s.research_fulltext_locations_per_work)
                             if h.url not in blocked and counts[h.url] < 2 and h.url not in rejected_urls]
                    allowance = min(self._s.research_fulltext_max_documents - fulltexts,
                                    self._s.scrape_max_pages - len(sources) - len(unsearched),
                                    self._s.research_fulltext_max_documents * self._s.research_fulltext_locations_per_work - len(past_attempts))
                    if leads and allowance > 0:
                        # Follow locations from fetched records, with no extra web search.
                        leads = leads[:allowance]

                        class OpenCopyLeads:
                            def __init__(self, hits):
                                self.hits = hits

                            async def search(self, _topic, *, max_results):
                                return self.hits[:max_results]

                        emit("fulltext_start", f"Reading {len(leads)} open full-text candidate(s)…")
                        attempts = [{"url": h.url, "parent_url": h.metadata["fulltext_parent_url"],
                                     "round": round_number, "status": "started"} for h in leads]
                        session.audit.setdefault("fulltext_attempts", []).extend(attempts)
                        try:
                            copies, rejected = await bounded(gather_sources(
                                session.plan.objective, OpenCopyLeads(leads), self._fetcher,
                                queries=[session.plan.objective], max_pages=len(leads),
                                settings=self._s.model_copy(update={"max_pages_per_query": len(leads)})))
                        except BaseException as exc:
                            for attempt in attempts:
                                attempt.update(status="deadline" if isinstance(exc, TimeoutError) else "error",
                                               error_type=type(exc).__name__)
                            raise
                        by_url = {s.metadata.get("retrieval_entry_url", s.url): s for s in copies}
                        failures = {h.url: h.metadata.get("rejection_reason", "source_filter") for h in rejected}
                        for attempt in attempts:
                            url = attempt["url"]
                            attempt.update(status="retrieved" if url in by_url else "rejected", reason=failures.get(url),
                                retrieved_url=by_url[url].url if url in by_url else None,
                                format=by_url[url].document_format if url in by_url else None)
                        if copies:
                            candidate_sources = _merge_sources(sources, copies)
                            sources = candidate_sources  # retain fetched originals even if later checks time out
                            candidate = await bounded(self._extract_and_verify(session.plan.objective, candidate_sources,
                                result, emit, bounded_research=True, isolation_reviews=session.isolation_reviews))
                            await bounded(session.evaluate(candidate.kept_claims, candidate_sources))
                            sources, verification = candidate_sources, candidate
                        emit("fulltext_done", f"Read {len(copies)} full-text document(s); retained abstract limitations for other works.")
                local_remaining = self._s.research_local_max_chunks - len(session.audit.get("local_reads", []))
                previous_reads = len(session.audit.get("local_reads", []))
                if local_remaining > 0 and (skip_search or not unsearched
                        or queries_used >= self._s.research_max_queries or len(sources) >= self._s.scrape_max_pages):
                    verification = await bounded(self._read_local_gaps(session, sources, verification, result, emit,
                                                                       min(2, local_remaining)))
                if skip_search:
                    session.audit["stop_reason"] = "document_only"
                    if (len(session.audit.get("local_reads", [])) > previous_reads
                            and len(session.audit["local_reads"]) < self._s.research_local_max_chunks):
                        continue
                    break
                if (session.audit["coverage_status"] == "complete"
                        and len(set(origin_groups(sources).values())) >= self._s.min_authoritative_sources):
                    session.audit["stop_reason"] = "questions_covered"
                    break
                remaining_queries = self._s.research_max_queries - queries_used
                remaining_pages = self._s.scrape_max_pages - len(sources)
                if remaining_queries <= 0 or remaining_pages <= 0:
                    session.audit["stop_reason"] = "query_budget" if remaining_queries <= 0 else "source_budget"
                    if (len(session.audit.get("local_reads", [])) > previous_reads
                            and len(session.audit["local_reads"]) < self._s.research_local_max_chunks):
                        continue
                    break
                # Give every uncovered question a first search before repeated
                # deep reads consume the shared wall-clock budget.
                queries = session.next_queries(min(remaining_queries, max(2, len(unsearched))))
                if not queries and len(set(origin_groups(sources).values())) < self._s.min_authoritative_sources:
                    for _ in range(min(remaining_queries, 2)):
                        key = ("q1", "source_diversity")
                        attempt = session.query_attempts.get(key, 0)
                        if attempt >= 4:
                            break
                        strategy, query = discovery_query(session.plan.objective, "source_diversity", attempt, session.plan.search_terms)
                        session.query_attempts[key] = attempt + 1
                        if normalize_text(query) not in session.used_queries:
                            session.used_queries.add(normalize_text(query))
                            queries.append({"question_id": "q1", "gap": "source_diversity", "kind": "fact", "strategy": strategy, "query": query})
                if not queries:
                    session.audit["stop_reason"] = "no_new_queries"
                    break
                emit("gap_search_start", f"Research round {round_number}: searching {len(queries)} evidence gap(s)…")
                # Reserve a document slot for each pending question that has not
                # received its first search. Query fairness alone is insufficient
                # if early questions exhaust the entire retained-source budget.
                future_questions = unsearched - {q["question_id"] for q in queries}
                round_pages = min(remaining_pages, max(len(queries), remaining_pages - len(future_questions)))
                searches = [{**query, "round": round_number, "status": "started"} for query in queries]
                session.audit["searches"].extend(searches)
                queries_used += len(searches)
                session.audit["queries_used"] = queries_used
                query_diagnostics: list[dict] = []
                try:
                    fetched, rejected = await bounded(gather_sources(
                        session.plan.objective, None if self._search_is_default else self._search, self._fetcher,
                        queries=[query["query"] for query in queries], exclude_urls={s.url for s in sources} | rejected_urls,
                        max_pages=round_pages, settings=self._s, llm=self._llm, diagnostics=query_diagnostics,
                        search_purpose="gap",
                        on_progress=lambda _done, _total, detail: emit("gap_search_progress", detail)))
                finally:
                    for search in searches:
                        detail = next((row for row in query_diagnostics if row["query"] == search["query"]), None)
                        if detail is not None:
                            search.update(detail)
                            search["rejected_urls"] = [row["url"] for row in detail.get("rejections", [])]
                rejected_urls.update(hit.url for hit in rejected if not retryable_fetch_failure(hit.metadata.get("rejection_reason")))
                if not fetched:
                    session.audit["stop_reason"] = "no_new_sources"
                    if queries_used < self._s.research_max_queries and round_number < self._s.research_max_rounds:
                        continue
                    break
                candidate_sources = _merge_sources(sources, fetched)
                sources = candidate_sources  # unverified data stays a source/lead, never a coverage witness
                candidate = await bounded(self._extract_and_verify(session.plan.objective, candidate_sources,
                    result, emit, bounded_research=True, isolation_reviews=session.isolation_reviews))
                previous = {normalize_text(c.text) for c in session.claims}
                await bounded(session.evaluate(candidate.kept_claims, candidate_sources))
                sources, verification = candidate_sources, candidate
                gained = len({normalize_text(c.text) for c in session.claims} - previous)
                session.audit["rounds"].append({"round": round_number, "new_sources": len(fetched),
                                                "new_grounded_claims": gained,
                                                "question_statuses": {q["id"]: q["status"] for q in session.audit["questions"]}})
                emit("gap_search_done", f"Added {gained} source-supported observation(s); reassessed research gaps.")
                if (session.audit["coverage_status"] == "complete"
                        and len(set(origin_groups(sources).values())) >= self._s.min_authoritative_sources):
                    session.audit["stop_reason"] = "questions_covered"
                    break
                if gained == 0:
                    session.audit["stop_reason"] = "no_new_evidence"
                    if queries_used < self._s.research_max_queries and round_number < self._s.research_max_rounds:
                        continue
                    break
            else:
                session.audit["stop_reason"] = "round_budget"
        except TimeoutError:
            session.audit["stop_reason"] = "deadline"
            for search in session.audit["searches"]:
                if search["status"] == "started":
                    search["status"] = "deadline"
            for read in session.audit.get("local_reads", []):
                if not read.get("round_completed"):
                    read["status"] = "deadline"
        except Exception as exc:
            session.audit["stop_reason"] = "retrieval_error"
            session.audit["error_type"] = type(exc).__name__
            for search in session.audit["searches"]:
                if search["status"] == "started":
                    search["status"] = "error"
        session.audit["queries_used"] = queries_used
        session.audit["disagreements"] = [c for c in verification.clusters if c.get("relation") == "conflict"]
        return sources, verification

    async def _read_local_gaps(self, session, sources, verification, result, emit, available):
        remaining = self._s.extraction_max_claims - len(verification.kept_claims)
        if remaining <= 0:
            return verification
        jobs = session.next_local_reads(sources, min(available, remaining))
        if not jobs:
            return verification
        emit("local_read_start", f"Reading {len(jobs)} original passage(s) for specific evidence gaps…")
        by_url = {source.url: source for source in sources}
        new_claims = []
        known = {normalize_text(c.text) for c in verification.kept_claims}
        records = []
        for job in jobs:
            record = {**job, "status": "started"}
            session.audit.setdefault("local_reads", []).append(record)
            records.append(record)
            source = by_url[job["source_url"]]
            start, end = job["start"], job["end"]
            excerpt = source.text[start:end]
            try:
                extractor = getattr(self._llm, "extract_research_claims", None)
                goal = next(q.question for q in session.plan.questions if q.id == job["question_id"])
                extracted = (await extractor(source.url, excerpt, [goal, "Missing evidence ONLY: " + job["gap"]], max_claims=3)
                             if extractor else await self._llm.extract_claims(source.url, excerpt))
                added = []
                for item in extracted[:min(3, remaining - len(new_claims))]:
                    if not item.text.strip() or normalize_text(item.text) in known:
                        continue
                    known.add(normalize_text(item.text))
                    self._next_local_id += 1
                    position = excerpt.find(item.source_quote) if item.source_quote else -1
                    span = [start, end] if position < 0 else [start + max(0, position - 300),
                            min(end, start + position + len(item.source_quote) + 300)]
                    claim = Claim(f"c{self._next_local_id}", item.text, candidate_source_urls=[source.url],
                        credibility=source.credibility, entity=item.entity, attribute=item.attribute, value=item.value,
                        qualifier=item.qualifier, time_scope=item.time_scope, metric=item.metric,
                        source_windows={source.url: [span]})
                    new_claims.append(claim)
                    added.append(claim.id)
                record.update(status="extracted" if added else "no_new_claims", candidate_claim_ids=added)
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
        if new_claims:
            candidate = await self._extract_and_verify(session.plan.objective, sources, result, emit,
                bounded_research=True, candidate_claims=verification.kept_claims + new_claims, strict_context=True)
            await session.evaluate(candidate.kept_claims, sources)
            anchored_ids = {c.id for c in session.claims}
            for record in records:
                record["anchored_claim_ids"] = [cid for cid in record.get("candidate_claim_ids", []) if cid in anchored_ids]
            self._local_claims.extend(c for c in candidate.kept_claims if c.id in {n.id for n in new_claims} & anchored_ids)
            verification = candidate
        for record in records:
            record["round_completed"] = True
        emit("local_read_done", f"Located {sum(len(r.get('anchored_claim_ids', [])) for r in records)} additional source-supported observation(s).")
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

    async def _extract_claims(self, sources: list[Source], on_progress=None, audit=None) -> list[Claim]:
        claims: list[Claim] = []
        counter = 0
        # Index claims by normalized text so the same fact from multiple sources
        # merges into ONE claim with multiple candidate sources — that is what
        # makes cross-referencing possible.
        by_text: dict[tuple[str, ...], Claim] = {}
        total = len(sources)

        # CONCURRENCY: extract from all sources in parallel under a bounded semaphore
        # (llm_max_concurrency, matching the verify stage's pattern). Per-source
        # resilience is preserved: a rate-limited/failed extraction on ONE source logs
        # and yields [] so it cannot abort the run. Results stay index-aligned with
        # `sources` so the merge below is byte-for-byte deterministic in source order.
        sem = asyncio.Semaphore(max(1, self._s.llm_max_concurrency))
        failed = []
        total_omitted = 0
        from core_engine.report.passages import extraction_windows, select_research_claims

        cache = getattr(self, "_extraction_cache", None)
        if cache is None:
            self._extraction_cache = cache = {}

        async def _one(src: Source):
            async with sem:
                try:
                    windows = extraction_windows(src.text, getattr(self, "_research_questions", []),
                        self._s.llm_context_chars, self._s.extraction_max_chunks_per_source)
                    structured, table_audit = table_candidates(src, min(4, self._s.extraction_max_claims_per_source))
                    parts, chunks = ([structured] if structured else []), []
                    last_error = None
                    for start, end in windows:
                        excerpt = src.text[start:end]
                        focus = tuple(getattr(self, "_research_questions", [])) if src.metadata.get("full_text_accessed") else ()
                        key = (src.url, excerpt, focus)
                        try:
                            reused = key in cache
                            if not reused:
                                focused_extractor = getattr(self._llm, "extract_research_claims", None)
                                cache[key] = (await focused_extractor(src.url, excerpt, list(focus)) if focus and focused_extractor
                                              else await self._llm.extract_claims(src.url, excerpt))
                            batch = []
                            for ex in cache[key]:
                                position = excerpt.find(ex.source_quote) if ex.source_quote else -1
                                span = [start, end] if position < 0 else [start + max(0, position - 300),
                                    min(end, start + position + len(ex.source_quote) + 300)]
                                batch.append(replace(ex, source_windows={src.url: [span]}))
                            parts.append(batch)
                            chunks.append({"start": start, "end": end, "status": "completed", "reused": reused})
                        except Exception as exc:
                            last_error = exc
                            chunks.append({"start": start, "end": end, "status": "error", "error_type": type(exc).__name__,
                                           "reason": getattr(exc, "reason", "extraction_unavailable")})
                    if not parts:
                        raise last_error or ValueError("all_source_chunks_failed")
                    # Give methods/results/limitations windows a share of the
                    # claim budget instead of letting the first window fill it.
                    extracted, seen = [], set()
                    for index in range(max((len(part) for part in parts), default=0)):
                        for part in parts:
                            if index < len(part) and part[index].text not in seen:
                                seen.add(part[index].text)
                                extracted.append(part[index])
                    retained = select_research_claims(extracted,
                        getattr(self, "_research_questions", []) if src.metadata.get("full_text_accessed") else [],
                        self._s.extraction_max_claims_per_source)
                    if audit is not None:
                        audit.append({"source_url": src.url, "status": "completed", "claim_count": len(retained),
                                      "budget_omitted": len(extracted) - len(retained), "chunks": chunks,
                                      "table_extraction": table_audit,
                                      "source_chars": len(src.text),
                                      "context_chars_read": sum(c["end"] - c["start"] for c in chunks if c["status"] == "completed")})
                    return retained
                except Exception as e:
                    log.warning("extract: source %s failed (%s) — skipping",
                                src.url, type(e).__name__)
                    failed.append(src.url)
                    if audit is not None:
                        audit.append({"source_url": src.url, "status": "error", "error_type": type(e).__name__,
                                      "reason": getattr(e, "reason", "extraction_unavailable")})
                    return []

        extracted_by_source = await asyncio.gather(*(_one(s) for s in sources))
        if sources and len(failed) == len(sources):
            from core_engine.report.llm import LLMResponseError

            raise LLMResponseError("all_source_extractions_failed")

        balanced = [(src, [part[index]]) for index in range(max((len(p) for p in extracted_by_source), default=0))
                    for src, part in zip(sources, extracted_by_source, strict=True) if index < len(part)]
        for i, (src, extracted) in enumerate(balanced, start=1):
            for ex in extracted:
                if not ex.text.strip():
                    continue
                key = (_normalize(ex.text), *(normalize_text(getattr(ex, slot) or "")
                       for slot in ("entity", "attribute", "value", "qualifier", "time_scope")),
                       json.dumps(ex.metric, sort_keys=True, ensure_ascii=False))
                if key in by_text:
                    claim = by_text[key]
                    for url, windows in ex.source_windows.items():
                        for window in windows:
                            if window not in claim.source_windows.setdefault(url, []):
                                claim.source_windows[url].append(window)
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
                if len(claims) >= self._s.extraction_max_claims:
                    total_omitted += 1
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
                    metric=ex.metric,
                    source_windows=ex.source_windows,
                )
                by_text[key] = claim
                claims.append(claim)
            # Per-source progress so the extraction phase shows real movement instead of
            # relying only on the connection heartbeat.
            if on_progress and (i == len(balanced) or i % max(1, total) == 0):
                on_progress("extracting_progress",
                            f"Merged {i}/{len(balanced)} observations across {total} source(s); {len(claims)} claim(s)…")
        if total_omitted and audit is not None:
            audit.append({"scope": "total_claim_budget", "budget_omitted": total_omitted,
                          "limit": self._s.extraction_max_claims})
        return claims

    async def _assemble(
        self, topic: str, verification: VerificationReport, sources: list[Source], *, research_plan=None, focus_claim_ids=None,
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
        if focus_claim_ids is not None:
            verified = [c for c in verified if c.id in focus_claim_ids]
            rumors = [c for c in rumors if c.id in focus_claim_ids]
        focused = bool(research_plan and research_plan.get("requested_metrics") and not re.search(
            r"industry|sector|supply chain|landscape|行业|产业", topic, re.IGNORECASE))
        if focused:
            # A table header alone is context, not a standalone factual finding.
            verified = [c for c in verified if "|" not in c.text or numeric_tokens(c.text)]

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
        if focused:
            buckets = {"findings": verified}
            plan = [("findings", "主要发现" if self._s.report_locale == "zh" else "Findings")] + [
                (sid, heading) for sid, heading in plan if sid in _SYNTHESIS_SECTION_IDS]

        # 1. KG synthesis + 2. deep-dives + 3. section prose all run in ONE concurrent
        #    batch. KG and deep-dives read verified claims independently; section drafts
        #    are independent per section. All are guarded by the assembly deadline.
        kg_task = asyncio.create_task(
            _guard(build_report_kg(topic, verified, self._llm), None))
        async def _dives():
            return {} if focused else await self._build_deep_dives(topic, verified, buckets)

        deep_dives_task = asyncio.create_task(_guard(_dives(), {}))

        async def _draft(section_id: str, heading: str) -> str:
            bucket = buckets.get(section_id, [])
            if not bucket:
                return ""
            # Isolated (single-source, low-credibility) claims carry an explicit
            # marker so the drafting prompt hedges them (workflow E: hedging rules).
            return await self._llm.draft_section(
                heading, [f"[[{c.id}]] {_claim_line(c)}" for c in bucket],
                instructions=f"Write in {'Chinese' if self._s.report_locale == 'zh' else 'English'}. "
                f"Address the user's task: {topic}. "
                "Use source assertions only here; interpretations and hypotheses belong in the separate analysis record. "
                "Describe what the cited sources report, preserving qualifications and test conditions. "
                "Source rank and absent contradictions do not establish truth or independent replication. "
                + (f"Research questions: {json.dumps(research_plan.get('questions', []), ensure_ascii=False)}" if research_plan else ""))

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
        # Rendering waits for sentence grounding and metric/scope validation.
        chart_blocks = {}
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
                    ("本节暂无足够的原文支持内容，不能据现有资料补造结论。" if self._s.report_locale == "zh" else
                     "Insufficient source-supported information was available for this section.")
                ]
            sections.append(ReportSection(
                heading=heading, section_id=section_id, paragraphs=paragraphs,
                claim_ids=[c.id for c in bucket], latex_blocks=latex_blocks,
                deep_dives=dives,
                sentences=([ReportSentence(text=paragraphs[0], status="editorial")]
                           if not prose and not latex_blocks and not dives else []),
            ))

        title = _report_title(topic, locale=self._s.report_locale)
        if focused:
            title = title.replace("行业简报", "研究简报").replace("Industry Research Report", "Research Report")
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
            disputed_claims=[c for c in verification.kept_claims if c.reporting_status == "disputed"],
            leads=lead_records(verification.rejected_claims),
        )

    # keyword routing for the 5 mandated sections
    _SECTION_KEYWORDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "industry_overview": ("overview", "background", "history", "sector", "industry",
                              "technology", "adoption"),
        "policy_analysis": ("policy", "regulat", "government", "directive", "law",
                            "subsidy", "tariff", "compliance", "ministry", "standard"),
        "industry_chain": ("upstream", "midstream", "downstream", "supplies", "supply",
                          "component", "material", "manufactur", "distribut"),
        "market_size": ("market", "tam", "sam", "som", "cagr", "billion", "revenue"),
        "competitive_landscape": ("competitor", "share", "player", "leader", "rival",
                                 "advantage", "vendor", "incumbent"),
    }

    def _bucket_claims(self, verified: list[Claim]) -> dict[str, list[Claim]]:
        """Route explicit subject keywords; unmatched technical facts stay in overview."""
        buckets: dict[str, list[Claim]] = {sid: [] for sid, _ in self._s.report_sections}
        section_ids = [sid for sid, _ in self._s.report_sections]
        unassigned: list[Claim] = []
        for claim in verified:
            low = normalize_text(claim.text)
            best_id, best_score = "", 0
            chinese_keywords = {
                "industry_overview": ("行业", "产业", "技术", "应用", "需求"),
                "policy_analysis": ("政策", "法规", "补贴", "关税", "监管", "标准"),
                "industry_chain": ("上游", "中游", "下游", "供应链", "原材料", "制造"),
                "market_size": ("市场规模", "营收", "亿元", "亿美元", "产值"),
                "competitive_landscape": ("竞争", "市场份额", "龙头", "竞争优势", "厂商"),
            }
            for sid, kws in self._SECTION_KEYWORDS.items():
                if sid not in buckets:
                    continue
                kws = (*kws, *chinese_keywords.get(sid, ()))
                def matches(keyword, text):
                    if keyword.isascii():
                        suffix = r"[a-z]*" if keyword in {"regulat", "manufactur", "distribut"} else ""
                        return bool(re.search(r"(?<![a-z])" + re.escape(keyword) + suffix + r"(?![a-z])", text))
                    return keyword in text

                score = sum(1 for kw in kws if matches(kw, low))
                if score > best_score:
                    best_id, best_score = sid, score
            if best_score > 0:
                buckets[best_id].append(claim)
            else:
                unassigned.append(claim)
        # Unclassified material stays in overview; never manufacture section relevance.
        fallback = "industry_overview" if "industry_overview" in buckets else section_ids[0]
        buckets[fallback].extend(unassigned)
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
            claim_texts = [f"[[{cid}]] {text_by_id[cid]}" for cid in claim_ids]
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
                    [c.id for c in bucket_claims], [f"[[{c.id}]] {_claim_line(c)}" for c in bucket_claims],
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
            if not paragraphs or not parse_draft("\n\n".join(paragraphs)):
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
            if cluster.get("relation") in {"conflict", "unresolved"}:
                continue  # deterministic, separately cited disagreement display
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
            marked = f"[UNVERIFIED - single source] {line}" if isolated else line
            lines.append(f"[[{','.join(claim_ids)}]] {marked}")
        prose = await self._llm.draft_section(
            heading, lines,
            instructions=(
                "This section compares how different sources report the SAME metrics. "
                "Attach the reporting source (domain) to EVERY sentence that states a "
                "figure. For clusters whose relation is 'complement', explain the "
                "difference in calibre/qualifier (口径) or time scope that reconciles "
                "the figures. Never resolve a disagreement by source rank or assert factual truth. "
                "Attribute every source-reported figure and preserve conditions."
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
                "来源数量与机构权威性不等于独立交叉验证；请查看逐句证据记录。" if zh else
                "Source count and authority do not establish independent corroboration; inspect sentence citations.")
        if verification.timed_out:
            paragraphs.append(
                "验证阶段达到时间预算上限，部分声明可能未完成全部交叉核查。" if zh else
                "Verification reached its time budget; some claims may not have been "
                "fully cross-checked.")
        # Internal run statistics are editorial metadata. Isolated factual claims
        # still go through the same source audit as the main body.
        metadata = [paragraphs[0], *([paragraphs[-1]] if verification.timed_out else [])]
        sentences = [ReportSentence(text=p, status="editorial") for p in metadata]
        for claim in isolated:
            domain = charts_mod._domain_of(claim.candidate_source_urls[0]) if claim.candidate_source_urls else "unknown"
            label = claim.credibility.label if claim.credibility is not None else "ungraded"
            sentences.append(ReportSentence(text=f"{claim.id}: {domain}; {label}.", status="editorial"))
            sentences.append(ReportSentence(text=f"[UNVERIFIED - single source] {claim.text}",
                                            claim_ids=[claim.id]))
        return ReportSection(
            heading=heading, section_id="limitations", paragraphs=metadata,
            claim_ids=[c.id for c in isolated], sentences=sentences)

    @staticmethod
    def _section_charts(section_id: str, blocks: dict[str, str]) -> list[str]:
        """Pick the pre-rendered chart fragments that belong to a section, dropping
        empties (a fragment is "" when the KG lacked enough verified data)."""
        mapping = {
            "findings": ["market_callout", "market_plot", "competitive_table", "competitive_bar", "comparison_matrix"],
            "industry_chain": ["chain_tikz"],
            "market_size": ["market_callout", "market_plot"],
            "competitive_landscape": ["competitive_table", "competitive_bar",
                                      "comparison_matrix"],
        }
        keys = mapping.get(section_id, [])
        return [blocks[k] for k in keys if blocks.get(k)]

    def _unanswerable(self, result: PipelineResult, sources: list[Source], emit) -> PipelineResult:
        from uuid import uuid4

        from core_engine.security.files import atomic_write_text

        # A failed draft may have a narrower policy than the research snapshot.
        # Rebind the refusal artifact to exactly the candidates/proof it exports.
        candidates = [Claim(**row) for row in result.research_evidence.get("claims", [])]
        ids = {c.id for c in candidates}
        candidates += [Claim(row["claim_id"], row["text"], candidate_source_urls=row.get("candidate_source_urls", []))
                       for row in result.evidence_policy.get("leads", []) if row["claim_id"] not in ids]
        citations = [SentenceCitation(**row) for row in result.research_evidence.get("citations", [])]
        if candidates or result.evidence_policy:
            result.evidence_policy = apply_policy(candidates, sources, citations)
            result.evidence_policy["leads"] = lead_records(candidates)
            result.research_evidence["claims"] = [asdict(c) for c in candidates]
        result.status = PipelineStatus.OUT_OF_SCOPE
        missing = ", ".join(result.answerability.get("missing", []))
        result.message = (
            f"现有资料无法回答所问问题，缺少有原文支持的指标或时期：{missing}。请补充对应资料。"
            if self._s.report_locale == "zh" else
            f"Available sources do not answer the requested question. Missing supported metric/period: {missing}. "
            "Please provide the corresponding evidence.")
        self._s.output_dir.mkdir(parents=True, exist_ok=True)
        path = self._s.output_dir / f"refusal-{uuid4().hex}.evidence.json"
        atomic_write_text(path, json.dumps({
            "schema_version": 1, "kind": "insufficient_evidence", "topic": result.topic,
            "answerability": result.answerability, "sources": [
                {**asdict(source), "content_sha256": source.content_sha256} for source in sources],
            "claims": result.research_evidence.get("claims", []), "sections": [],
            "citations": result.research_evidence.get("citations", []), "metrics": [], "calculations": [],
            "research_plan": result.research_plan, "research_audit": result.research_audit,
            "research_evidence": result.research_evidence,
            "evidence_policy": result.evidence_policy,
            "leads": result.evidence_policy.get("leads", []),
        }, ensure_ascii=False, indent=2))
        result.evidence_path = str(path)
        emit("out_of_scope", result.message)
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
    return normalize_text(text)


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
    explicit_title = re.match(r"(.{2,80}?(?:报告|简报))[:：]", subject)
    if explicit_title:
        return explicit_title[1]
    if (locale or "en").lower().startswith("zh"):
        return f"{subject}行业简报" if subject else "行业简报"
    subject = _title_case(subject) if subject else subject
    if not subject:
        return "A Brief Industry Research Report"
    return f"A Brief Industry Research Report of {subject}"
