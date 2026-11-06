"""Offline research-loop regression using pinned real releases and scripted search.

Neither retrieval effectiveness nor live-model reasoning quality is measured.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from core_engine.config import get_settings
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import FakeLLM
from core_engine.report.models import CredibilityLevel, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet, load_dataset, packet_sentences
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider, SearchHit
from core_engine.security.files import atomic_write_text

DATASET = Path(__file__).resolve().parents[3] / "evals" / "research_quality" / "cases.json"


async def run_research_cases(output: Path, *, dataset_path: Path = DATASET) -> dict:
    source_data = load_dataset()
    rows = {s["id"]: s for s in source_data["sources"]}
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported research evaluation schema")
    results = []
    artifacts = output.parent / (output.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    for case in data["cases"]:
        initial, retrieved = ([rows[key] for key in case[field]] for field in ("initial", "retrieved"))

        class ScriptedSearch(FakeSearchProvider):
            async def search(self, topic, *, max_results, _case=case, _retrieved=retrieved):
                if topic == _case["question"]:
                    return []
                return [SearchHit(s["url"], s["title"], "", s["url"].split("/")[2]) for s in _retrieved][:max_results]

        class ScriptedAnalyst(FakeLLM):
            async def complete(self, system, user, _case=case):
                if _case.get("scripted_analysis") and "Evidence-based analyst" in system:
                    payload = json.loads(user)
                    premise = next(c for c in payload["claims"] if "最终核实" in c["text"])
                    return json.dumps({"findings": [{"question_id": "q1", "kind": "interpretation",
                        "conclusion": "GDP最终核实相对初步核算下调，提示采用不同发布版本会影响现价总量比较。",
                        "premise_claim_ids": [premise["id"]], "counter_claim_ids": [], "calculation_ids": [],
                        "assumptions": ["比较时保留来源声明的现价口径与发布版本。"],
                        "alternatives": ["统计修订也可能反映核算资料更新。"],
                        "uncertainty": "单次统计修订不能直接解释经济活动变化的原因。"}]}, ensure_ascii=False)
                if _case.get("scripted_analysis") and "Analysis critic" in system:
                    return '{"acceptable":true,"unsupported_assertions":[]}'
                return await super().complete(system, user)

        pipeline = ReportPipeline(llm=ScriptedAnalyst(), search=ScriptedSearch(), fetcher=FakeFetcher(
            {s["url"]: s["text"] for s in retrieved}))
        pipeline._s = get_settings().model_copy(update={
            "min_authoritative_sources": 1, "min_total_claims": 1, "verify_strategy": "cross_reference",
            "verify_mode": "strict", "report_locale": "zh", "enable_charts": False,
            "enable_deep_dive": False, "enable_synthesis_sections": False, "active_verify_enabled": False,
            # Short pinned excerpts are deliberate fixtures, not live-page admission criteria.
            "relevance_filter": True, "relevance_min_body_chars": 1, "relevance_min_score": 0,
            "relevance_min_topic_hits": 0, "output_dir": artifacts,
            "research_max_rounds": 2, "research_max_queries": 4, "research_deadline_s": 30,
            "research_plan_timeout_s": 2, "reasoning_deadline_s": 5,
        })
        pipeline._harness._s = pipeline._s
        sources = [Source(s["url"], s["url"].split("/")[2], s["title"], SourceKind.GOVERNMENT,
                          s["text"], published_at=s["published_at"], credibility=CredibilityLevel.L1) for s in initial]
        result = await pipeline.run(case["question"], compile_to_pdf=False, extra_sources=sources,
                                    skip_search=case.get("skip_search", False))
        packet = (evidence_packet(result.report) if result.report else
                  json.loads(Path(result.evidence_path).read_text(encoding="utf-8")) if result.evidence_path else {})
        selected = list({s["id"]: s for s in initial + retrieved}.values())
        checks = check_packet(packet, selected)
        audit = packet.get("research_audit", {})
        searches = audit.get("searches", [])
        text = "\n".join(s["text"] for s in packet_sentences(packet) if s.get("status") == "supported")
        errors = list(checks["integrity_errors"])
        if result.status.value != case["expected_status"]:
            errors.append("unexpected_pipeline_status")
        if len(searches) < case.get("minimum_searches", 0) or len(searches) > case.get("maximum_searches", 4):
            errors.append("unexpected_gap_search_count")
        if case.get("required_gap") and not any(s["gap"] == case["required_gap"] for s in searches):
            errors.append("did_not_target_annotated_gap")
        if case.get("required_text") and case["required_text"] not in text:
            errors.append("required_answer_not_published")
        if case.get("expected_partial") and (audit.get("coverage_status") != "partial"
                                             or result.report is None or result.report.quality_status != "needs_review"):
            errors.append("measurements_falsely_declared_complete_causal_research")
        if case.get("expected_analyses") != len(packet.get("analysis", [])) and "expected_analyses" in case:
            errors.append("unexpected_analytical_conclusion_count")
        if case["expected_status"] == "out_of_scope" and (result.pdf_path or result.tex_path or text
                or packet.get("kind") != "insufficient_evidence" or not packet.get("answerability", {}).get("missing")):
            errors.append("invalid_abstention")
        results.append({"case_id": case["id"], "pipeline_status": result.status.value,
                        "automatic_checks_passed": not errors, "errors": errors,
                        "evidence_path": result.evidence_path, "research_audit": audit,
                        "checked_analyses": checks["checked_analyses"]})
    summary = {"schema_version": 1, "dataset_version": data["version"], "cases_run": len(results),
               "automatic_checks_passed": sum(r["automatic_checks_passed"] for r in results),
               "live_model_evaluated": False, "retrieval": "scripted_pinned_source_snapshots",
               "analysis": "scripted_proposals_and_critic_when_requested", "human_review_required": True,
               "limitations": ["Tests orchestration, evidence binding, abstention and inference boundaries.",
                               "Does not measure live retrieval quality or independent semantic reasoning accuracy."],
               "results": results}
    review_path = output.with_name(f"{output.stem}-{uuid4().hex[:12]}.review.json")
    atomic_write_text(review_path, json.dumps({"dataset_version": data["version"],
        "reviews": [{"case_id": r["case_id"], "evidence_path": r["evidence_path"],
                     "scores": {key: None for key in ("question_coverage", "premise_relevance", "causal_restraint",
                                "counterevidence", "assumptions", "usefulness")}, "reviewer": "", "notes": ""} for r in results]},
        ensure_ascii=False, indent=2))
    summary["human_review_form"] = str(review_path.resolve())
    atomic_write_text(output, json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/research-quality.json"))
    args = parser.parse_args(argv)
    summary = asyncio.run(run_research_cases(args.output))
    print(f"Research regression: {summary['automatic_checks_passed']}/{summary['cases_run']} automatic checks passed; "
          "scripted retrieval/model responses, human review required.")
    return 0 if summary["automatic_checks_passed"] == summary["cases_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
