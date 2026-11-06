"""Opt-in live A/B report run using the locally configured search and LLM credentials.

Real responses are cached by exact request for reproducible fixes and less duplicate
API spending. No credentials are serialized. Snapshot reuse is reported explicitly.
Run from the repository root: python scripts/compare_openalex.py --help
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from unittest.mock import patch

from core_engine.app.runtime import apply_settings_to_env
from core_engine.config import get_settings
from core_engine.report.evidence_quality import origin_groups
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import get_llm
from core_engine.report.models import (
    Claim,
    ConfidenceTier,
    CredibilityLevel,
    DeepDive,
    Evidence,
    ReportData,
    ReportSection,
    ReportSentence,
    SentenceCitation,
    Source,
    SourceKind,
)
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet
from core_engine.report.research import KINDS, ResearchPlan, ResearchQuestion, ResearchSession
from core_engine.report.scholarly import FetchedDocument
from core_engine.report.scrape import HttpFetcher
from core_engine.report.search_cache import SearchCache


def json_default(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def restore_claims(rows):
    return [Claim(**{**claim, "evidence": [Evidence(**e) for e in claim["evidence"]],
        "confidence": ConfidenceTier(claim["confidence"]) if claim["confidence"] else None,
        "credibility": CredibilityLevel(claim["credibility"]) if claim["credibility"] else None}) for claim in rows]


def restore_report(row):
    """Restore a trusted local result for layout-only replay, preserving evidence."""
    data = dict(row)
    data["sources"] = [Source(**{**source, "kind": SourceKind(source["kind"]),
        "credibility": CredibilityLevel(source["credibility"])}) for source in row["sources"]]
    for key in ("claims", "rumors", "disputed_claims"):
        data[key] = restore_claims(row.get(key, []))
    data["sections"] = [ReportSection(**{**section,
        "sentences": [ReportSentence(**s) for s in section["sentences"]],
        "deep_dives": [DeepDive(**{**dive, "sentences": [ReportSentence(**s) for s in dive["sentences"]]})
                       for dive in section["deep_dives"]]}) for section in row["sections"]]
    data["rumor_sentences"] = [ReportSentence(**s) for s in row["rumor_sentences"]]
    data["dispute_sentences"] = [ReportSentence(**s) for s in row.get("dispute_sentences", [])]
    data["citations"] = [SentenceCitation(**c) for c in row["citations"]]
    return ReportData(**data)


def render_saved(root, conditions, prefix="solid-state-battery"):
    from core_engine.report.compile import compile_pdf
    from core_engine.report.latex import write_tex

    for condition in conditions.split(","):
        saved = json.loads((root / condition / "result.json").read_text(encoding="utf-8"))
        report = restore_report(saved["report"])
        packet = evidence_packet(report)
        original = json.loads(Path(saved["evidence_path"]).read_text(encoding="utf-8"))
        if json.dumps(packet, sort_keys=True, default=json_default) != json.dumps(original, sort_keys=True):
            raise ValueError("Layout replay changed the evidence packet")
        if check_packet(packet)["integrity_errors"]:
            raise ValueError("Evidence integrity failed; refusing deliverable rendering")
        tex = write_tex(report, root / condition)
        pdf = compile_pdf(tex)
        evidence = tex.with_suffix(".evidence.json")
        if Path(saved["evidence_path"]) != evidence:
            shutil.copyfile(saved["evidence_path"], evidence)
        target = Path("output/pdf") / f"{prefix}-{condition}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, target)
        shutil.copyfile(evidence, target.with_suffix(".evidence.json"))
        saved.update(tex_path=str(tex), pdf_path=str(pdf), evidence_path=str(evidence))
        saved["trace"].append({"stage": "layout_replay", "at": datetime.now(UTC).isoformat(), "pdf": str(pdf)})
        write(root / condition / "result.json", saved)
        summary_path = root / condition / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(pdf=str(pdf), evidence=str(evidence), deliverable_pdf=str(target.resolve()),
                       layout_replayed_at=datetime.now(UTC).isoformat())
        write(summary_path, summary)
        comparison_path = root / "comparison.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison["conditions"][condition] = summary
        write(comparison_path, comparison)
        print("Layout-only replay:", target.resolve(), flush=True)


class SnapshotFetcher:
    def __init__(self, directory):
        self.directory = directory
        self.real = HttpFetcher()
        self.failures = {}
        self.network_reads = self.snapshot_reads = 0

    async def fetch(self, url):
        return await self._fetch(url, fulltext=False)

    async def fetch_fulltext(self, url):
        return await self._fetch(url, fulltext=True)

    async def _fetch(self, url, *, fulltext):
        path = self.directory / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            value = row["document"]
            from core_engine.report.fulltext import public_copy_url

            pdf = value.get("metadata", {}).get("article_pdf_url") if isinstance(value, dict) else None
            usable_fulltext = not fulltext or (isinstance(value, dict) and value.get("metadata", {}).get("full_text_accessed"))
            if usable_fulltext and (not pdf or public_copy_url(pdf)):
                self.snapshot_reads += 1
                return FetchedDocument(**value) if isinstance(value, dict) else value
        self.network_reads += 1
        value = await (self.real.fetch_fulltext(url) if fulltext else self.real.fetch(url))
        if value is None:
            self.failures[url] = self.real.failures.get(url, "no_readable_content")
            return None  # Failed downloads can be retried on a later run.
        write(path, {"url": url, "captured_at": datetime.now(UTC).isoformat(),
                     "document": asdict(value) if isinstance(value, FetchedDocument) else value})
        return value


def record_llm(llm, directory, stats):
    original = llm._complete
    locks = {}

    async def recorded(system, user):
        settings = get_settings()
        identity = json.dumps([settings.llm_provider, settings.llm_base_url, settings.llm_model,
            settings.llm_max_tokens, settings.llm_thinking_mode, system, user], ensure_ascii=False)
        key = hashlib.sha256(identity.encode()).hexdigest()
        path = directory / (key + ".json")
        async with locks.setdefault(key, asyncio.Lock()):
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))["response"]
                if isinstance(cached, str) and cached.strip():
                    stats["replayed_calls"] += 1
                    return cached
            started = time.monotonic()
            stats["live_calls"] += 1
            try:
                raw = await original(system, user)
            except Exception as exc:
                stats["failed_calls"] += 1
                print("LLM request failed:", type(exc).__name__, flush=True)
                raise
            write(path, {"model": get_settings().llm_model, "request_sha256": key,
                "system": system, "response": raw, "elapsed_s": round(time.monotonic() - started, 3),
                "recorded_at": datetime.now(UTC).isoformat()})
            return raw

    llm._complete = recorded


def summarize(result, stats, fetcher, elapsed):
    summary = {"status": result.status.value, "message": result.message, "elapsed_s": round(elapsed, 2),
        "live_model_evaluated": True, "llm_calls": dict(stats),
        "new_source_fetches": fetcher.network_reads, "replayed_source_snapshots": fetcher.snapshot_reads,
        "fetch_failures": dict(fetcher.failures)}
    if result.report is None:
        return summary
    report = result.report
    packet = evidence_packet(report)
    groups = origin_groups(report.sources)
    used = {c.source_url for c in report.citations}
    summary.update({"checks": check_packet(packet), "source_count": len(report.sources),
        "origin_groups": len(set(groups.values())), "body_used_sources": len(used),
        "body_used_origin_groups": len({groups[url] for url in used if url in groups}),
        "source_formats": dict(Counter(s.document_format for s in report.sources)),
        "source_publication_dates_known": sum(bool(s.published_at) for s in report.sources),
        "openalex_sources": sum(s.domain == "api.openalex.org" for s in report.sources),
        "claim_count": len(report.claims), "quality_status": report.quality_status,
        "disputed_claim_count": len(report.disputed_claims), "lead_count": len(report.leads),
        "publication_policy": {key: report.evidence_policy.get(key) for key in
                               ("version", "source_reported", "corroborated", "disputed", "lead_only")},
        "evidence_mode": report.source_quality.get("evidence_mode"),
        "research_questions": report.research_audit.get("questions", []),
        "fully_covered_questions": sum(q["status"] == "supported" for q in report.research_audit.get("questions", [])),
        "full_text_documents": sum(bool(s.metadata.get("full_text_accessed")) for s in report.sources),
        "fulltext_attempts": report.research_audit.get("fulltext_attempts", []),
        "local_reads": report.research_audit.get("local_reads", []),
        "research_stop_reason": report.research_audit.get("stop_reason"),
        "sources": [{"url": s.url, "title": s.title, "format": s.document_format,
                     "published_at": s.published_at, "origin_group": groups[s.url],
                     "body_used": s.url in used} for s in report.sources],
        "pdf": result.pdf_path, "evidence": result.evidence_path})
    return summary


async def run(args):
    apply_settings_to_env()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Same bounded settings in both conditions; no change to saved GUI settings.
    overrides = {"REPORT_LOCALE": "zh", "LATEX_ENGINE": "xelatex", "SCRAPE_MAX_PAGES": "12",
        "MAX_PAGES_PER_QUERY": "3", "LLM_CONTEXT_CHARS": "5000", "LLM_MAX_RETRIES": "2",
        "LLM_MAX_CONCURRENCY": "2", "LLM_MIN_INTERVAL_S": "2", "RESEARCH_PLAN_TIMEOUT_S": "40", "RESEARCH_DEADLINE_S": "180",
        "ENABLE_DEEP_DIVE": "false", "MAX_CROSS_REF_SOURCES": "2",
        "VERIFY_STRATEGY": "cross_reference", "VERIFY_MODE": "strict", "VERIFY_DEADLINE_S": "180"}
    overrides["EXTRACTION_MAX_CLAIMS_PER_SOURCE"] = "8"
    overrides["EXTRACTION_MAX_CLAIMS"] = "48"
    if args.profile == "coverage":
        overrides.update({"SCRAPE_MAX_PAGES": "20", "MAX_PAGES_PER_QUERY": "4", "LLM_CONTEXT_CHARS": "10000",
            "RESEARCH_MAX_ROUNDS": "4", "RESEARCH_MAX_QUERIES": "8", "RESEARCH_DEADLINE_S": "600",
            "RESEARCH_PLAN_TIMEOUT_S": "60", "EXTRACTION_MAX_CLAIMS": "96",
            "RESEARCH_COVERAGE_TIMEOUT_S": "60",
            "RESEARCH_FULLTEXT_MAX_DOCUMENTS": "6", "EXTRACTION_MAX_CHUNKS_PER_SOURCE": "3"})
    if get_settings().llm_model == "MiniMax-M3":
        overrides["LLM_THINKING_MODE"] = "disabled"
    for name, value in overrides.items():
        os.environ["CE_" + name] = value
    get_settings.cache_clear()
    if args.render_only:
        render_saved(root, args.conditions, args.deliverable_prefix)
        return
    if get_settings().llm_provider == "fake":
        raise RuntimeError("Live comparison requires a configured real LLM")
    cache = SearchCache(root / "search-cache")
    fixed_plan = None
    seed_report = None
    if args.plan_file:
        saved = json.loads(Path(args.plan_file).read_text(encoding="utf-8"))
        row = saved.get("research_plan", saved)
        questions = [ResearchQuestion(**q) for q in row["questions"]]
        if (row["objective"] != args.topic or not 1 <= len(questions) <= 12
                or len({q.id for q in questions}) != len(questions) or questions[0].question != args.topic or not questions[0].required
                or any(q.kind not in KINDS for q in questions)):
            raise ValueError("Frozen plan must preserve the same task and valid distinct questions")
        fixed_plan = ResearchPlan(**{key: value for key, value in row.items()
                                    if key in ResearchPlan.__dataclass_fields__ and key != "questions"}, questions=questions)
    if args.seed_result:
        saved_seed = json.loads(Path(args.seed_result).read_text(encoding="utf-8"))
        seed_report = restore_report(saved_seed["report"])
        if seed_report.topic != args.topic or check_packet(evidence_packet(seed_report))["integrity_errors"]:
            raise ValueError("Imported sources/claims must preserve the task and pass independent evidence replay")
    if args.local_only and seed_report is None:
        raise ValueError("Local refinement requires --seed-result")
    if args.resume_gaps and (seed_report is None or fixed_plan is None):
        raise ValueError("Resuming gap research requires --seed-result and --plan-file")
    reports = {}
    for existing in ("baseline", "openalex"):
        path = root / existing / "summary.json"
        if path.exists():
            reports[existing] = json.loads(path.read_text(encoding="utf-8"))
    for condition in ([] if args.reassess_only else args.conditions.split(",")):
        enabled = condition == "openalex"
        if condition not in {"baseline", "openalex"}:
            raise ValueError("Use baseline,openalex")
        os.environ["CE_OPENALEX_SEARCH_ENABLED"] = str(enabled).lower()
        os.environ["CE_OUTPUT_DIR"] = str(root / condition)
        get_settings.cache_clear()
        stats = Counter()
        llm = get_llm()
        record_llm(llm, root / "llm-cache", stats)
        fetcher = SnapshotFetcher(root / "source-snapshots")
        events = []
        def progress(stage, detail, *, events=events, condition=condition):
            event = {"at": datetime.now(UTC).isoformat(), "stage": stage, "detail": detail}
            events.append(event)
            write(root / condition / "progress.json", events)
            if stage not in {"verifying_progress", "grounding_progress", "gap_search_progress"}:
                print(f"[{condition}] {stage}: {detail[:180]}", flush=True)
        print(f"Starting {condition}; model={get_settings().llm_model}", flush=True)
        started = time.monotonic()
        with patch("core_engine.report.scrape.SearchCache", return_value=cache):
            result = await ReportPipeline(llm=llm, fetcher=fetcher).run(
                args.topic, compile_to_pdf=args.pdf, on_progress=progress, research_plan=fixed_plan,
                extra_sources=seed_report.sources if seed_report else None,
                seed_claims=restore_claims(seed_report.research_evidence["claims"]) if seed_report else None,
                skip_search=args.local_only, skip_initial_search=args.resume_gaps,
                retained_leads=seed_report.leads if seed_report else None)
        write(root / condition / "result.json", asdict(result))
        summary = summarize(result, stats, fetcher, time.monotonic() - started)
        if seed_report:
            summary.update(imported_source_snapshots=len(seed_report.sources), seed_result=str(Path(args.seed_result).resolve()),
                refinement_mode=("local_gap_reads_no_web" if args.local_only else "resumed_gap_search_no_broad_search"
                                 if args.resume_gaps else "imported_sources_plus_retrieval"),
                seed_packet_sha256=hashlib.sha256(json.dumps(evidence_packet(seed_report), sort_keys=True,
                    default=json_default).encode()).hexdigest())
        write(root / condition / "summary.json", summary)
        if result.pdf_path and not summary.get("checks", {}).get("integrity_errors"):
            target = Path("output/pdf") / f"{args.deliverable_prefix}-{condition}.pdf"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result.pdf_path, target)
            shutil.copyfile(result.evidence_path, target.with_suffix(".evidence.json"))
            summary["deliverable_pdf"] = str(target.resolve())
            write(root / condition / "summary.json", summary)
        reports[condition] = summary
        print(f"[{condition}] finished: {summary['status']}; claims={summary.get('claim_count', 0)}", flush=True)
    write(root / "comparison.json", {"topic": args.topic, "recorded_at": datetime.now(UTC).isoformat(),
        "method": ("Frozen real source snapshots and anchored claims from the preceding acquisition run; claims reverified and original full texts reread for specific gaps. No web calls in local-only mode."
                   if seed_report else "Same topic/model/settings, primary search cache and exact-prompt model replay. Adaptive evidence-gap queries may differ."),
        "scope": "One real case; automatic integrity checks are not an independent semantic truth or professional quality score.",
        "settings": overrides, "conditions": reports})
    reassessed = None
    if args.reassess_previous:
        saved = json.loads(Path(args.reassess_previous).read_text(encoding="utf-8"))
        old_report = restore_report(saved["report"])
        if not fixed_plan or old_report.research_plan["questions"] != fixed_plan.to_dict()["questions"]:
            raise ValueError("Reassessment must use identical original questions")
        if check_packet(evidence_packet(old_report))["integrity_errors"]:
            raise ValueError("Old evidence failed independent replay; cannot reassess coverage")
        llm = get_llm()
        stats = Counter()
        record_llm(llm, root / "llm-cache", stats)
        session = ResearchSession(fixed_plan, llm, get_settings())
        evidence = old_report.research_evidence
        # The independent replay above checked every old exact source span. Do
        # not retrieve new documents or change the previous evidence snapshot.
        await session._assess_grounded(restore_claims(evidence["claims"]), old_report.sources,
                                      [SentenceCitation(**c) for c in evidence["citations"]])
        reassessed = {"fully_covered_questions": sum(q["status"] == "supported" for q in session.audit["questions"]),
            "questions": session.audit["questions"], "llm_calls": dict(stats),
            "source_count": len(old_report.sources), "method": "Old unchanged source-bound evidence, new per-question assessor; no retrieval"}
        write(root / "previous-reassessed.json", reassessed)
    if args.previous:
        previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))
        previous = previous.get("conditions", {}).get("openalex", previous)
        current = reports.get("openalex", {})
        write(root / "coverage-comparison.json", {"previous": previous, "previous_reassessed": reassessed, "current": current,
            "same_questions": bool(fixed_plan and previous.get("research_questions") and
                [(q.id, q.question) for q in fixed_plan.questions] ==
                [(q["id"], q["question"]) for q in previous["research_questions"]]),
            "limitations": ["Coverage profile increases bounded retrieval and context budgets; not an isolated API A/B.",
                "Coverage judgments use the same model; traceability is independently replayed, semantic accuracy requires review."]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="固态电池")
    parser.add_argument("--output", default="output/solid-state-openalex-ab")
    parser.add_argument("--conditions", default="baseline,openalex")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Recompile saved results without search/model calls")
    parser.add_argument("--profile", choices=("paired", "coverage"), default="paired")
    parser.add_argument("--plan-file", help="Replay the exact saved six-question plan for a before/after comparison")
    parser.add_argument("--previous", help="Previous comparison or summary JSON, retained in the coverage comparison")
    parser.add_argument("--reassess-previous", help="Reassess the old unchanged evidence with the new assessor to separate evaluation changes")
    parser.add_argument("--reassess-only", action="store_true", help="Use saved report summaries; only reassess old evidence")
    parser.add_argument("--deliverable-prefix", default="solid-state-battery")
    parser.add_argument("--seed-result", help="Import real source snapshots and reverify anchored claims from a saved result")
    parser.add_argument("--local-only", action="store_true", help="Refine imported full-text evidence without web searches or downloads")
    parser.add_argument("--resume-gaps", action="store_true", help="Reverify imported evidence and search gaps online without repeating broad initial searches")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
