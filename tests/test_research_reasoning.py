from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import pytest

from core_engine.config import get_settings
from core_engine.report.calculations import calculate
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import FakeLLM
from core_engine.report.metrics import Metric
from core_engine.report.models import Claim, PipelineStatus, SentenceCitation, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet
from core_engine.report.reasoning import analyze, counterevidence_status, validate_finding
from core_engine.report.research import (
    ResearchPlan,
    ResearchQuestion,
    ResearchSession,
    bounded_rows,
    build_plan,
    related,
)
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider, SearchHit, gather_sources

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def settings(tmp_path, **updates):
    fields = {"min_authoritative_sources": 1, "min_total_claims": 1, "verify_strategy": "cross_reference",
              "verify_mode": "strict", "enable_deep_dive": False, "enable_synthesis_sections": False,
              "enable_charts": False, "active_verify_enabled": True, "max_reverify_attempts": 2,
              "relevance_filter": True, "relevance_min_body_chars": 1, "relevance_min_score": 0,
              "relevance_min_topic_hits": 0, "output_dir": tmp_path, "research_plan_timeout_s": 1,
              "research_deadline_s": 3, "reasoning_deadline_s": 1, "scrape_max_pages": 8}
    return get_settings().model_copy(update={**fields, **updates})


def pipe(tmp_path, search=None, fetcher=None, llm=None, **updates):
    pipeline = ReportPipeline(llm=llm or FakeLLM(), search=search or FakeSearchProvider(), fetcher=fetcher or FakeFetcher())
    pipeline._s = settings(tmp_path, **updates)
    pipeline._harness._s = pipeline._s
    return pipeline


class Searches(FakeSearchProvider):
    def __init__(self, task, supplements=()):
        super().__init__()
        self.task, self.supplements = task, list(supplements)
        self.calls = []

    async def search(self, topic, *, max_results):
        self.calls.append(topic)
        if topic == self.task:
            return []  # Initial evidence supplied separately, so no first-stage network needed.
        return list(self.supplements)[:max_results]


def source(text, key="seed", title="Report 2024"):
    return Source(f"https://example.gov/{key}", "example.gov", title, SourceKind.GOVERNMENT, text)


async def test_planner_keeps_original_task_rejects_scope_drift_and_caps_questions(tmp_path):
    llm = FakeLLM()
    task = "Report Acme's net profit in 2024."
    llm.set_complete("Research planner", json.dumps({"questions": [
        {"question": "Other's net profit in 2024", "kind": "fact"},
        {"question": "Acme's net profit in 2025", "kind": "fact"},
        {"question": "Quantum mechanics tutorial", "kind": "fact"},
        {"question": "Acme's net profit in 2024: source and scope?", "kind": "comparison", "required": True},
        {"question": "Acme's net profit in 2024: source and scope?", "kind": "comparison"},
    ]}))
    plan = await build_plan(task, llm, settings(tmp_path, research_max_questions=3))
    assert plan.objective == plan.questions[0].question == task
    assert plan.questions[0].required and not plan.questions[1].required
    assert len(plan.questions) == 2 and len(plan.notes) == 4
    assert plan.initial_queries()[0] == task
    assert not related("Acme net profit 2024", "Unrelated GDP in 2024")


@pytest.mark.parametrize("response", ["not json", "[]", '{"questions":"wrong"}'])
async def test_malformed_planning_falls_back_to_explicit_constraints(tmp_path, response):
    llm = FakeLLM()
    llm.set_complete("Research planner", response)
    plan = await build_plan("Acme's 2024 revenue", llm, settings(tmp_path))
    assert plan.method == "deterministic" and plan.requested_periods == ["2024"]
    assert plan.requested_metrics == ["revenue"]


async def test_planner_failure_and_timeout_do_not_lose_the_task(tmp_path):
    class Slow(FakeLLM):
        async def complete(self, system, user):
            await asyncio.sleep(1)

    plan = await build_plan("Why did revenue grow?", Slow(), settings(tmp_path, research_plan_timeout_s=.01))
    assert plan.questions[0].required and plan.notes
    assert any(q.kind == "counterevidence" for q in plan.questions)


async def test_measurements_alone_cannot_close_a_causal_research_question(tmp_path):
    task = "Why did Acme's 2024 net profit grow?"
    llm = FakeLLM()
    llm.set_complete("Research coverage assessor", '{"assessments":[{"id":"q1","status":"supported","claim_ids":["c1"]}]}')
    plan = ResearchPlan(task, [ResearchQuestion("q1", task, "explanation", True)])
    session = ResearchSession(plan, llm, settings(tmp_path))
    src = source("Acme's 2024 net profit grew by 5%.")
    await session.evaluate([Claim("c1", src.text, candidate_source_urls=[src.url], verified=True)], [src])
    assert session.audit["questions"][0]["status"] == "partial"
    assert session.audit["coverage_status"] == "partial"


async def test_gap_search_recovers_requested_fact_and_keeps_unrelated_gdp_out_of_body(tmp_path):
    task = "Report Acme's net profit in 2024."
    seed = source("GDP was 1200亿元 in 2024.")
    recovered = source("Acme's net profit in 2024 was 12亿元.", "profit")
    search = Searches(task, [SearchHit(recovered.url, recovered.title, "", recovered.domain),
                             SearchHit(seed.url, seed.title, "", seed.domain)])
    fetcher = FakeFetcher({recovered.url: recovered.text, seed.url: seed.text})
    result = await pipe(tmp_path, search, fetcher).run(task, compile_to_pdf=False, extra_sources=[seed])
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert len(search.calls) > 1 and result.research_audit["stop_reason"] == "questions_covered"
    assert result.research_audit["rounds"][0]["new_grounded_claims"] > 0
    assert any("net_profit" in query for query in search.calls[1:])
    assert all("GDP" not in claim.text for claim in result.report.claims)
    assert [section.section_id for section in result.report.sections] == ["findings"]
    assert not check_packet(evidence_packet(result.report))["integrity_errors"]


async def test_second_research_round_uses_remaining_period_gap(tmp_path):
    task = "Compare Acme's net profit in 2022, 2023 and 2024."
    seed = source("Acme's net profit in 2022 was 10亿元.")
    third = source("Acme's net profit in 2023 was 11亿元.", "2023")
    fourth = source("Acme's net profit in 2024 was 12亿元.", "2024")

    class Rounds(Searches):
        async def search(self, topic, *, max_results):
            self.calls.append(topic)
            if topic == task:
                return []
            selected = third if len(self.calls) <= 3 else fourth
            return [SearchHit(selected.url, selected.title, "", selected.domain)]

    search = Rounds(task)
    result = await pipe(tmp_path, search, FakeFetcher({third.url: third.text, fourth.url: fourth.text})).run(
        task, compile_to_pdf=False, extra_sources=[seed])
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert len(result.research_audit["rounds"]) == 2
    assert result.research_audit["queries_used"] <= 4
    assert result.research_audit["questions"][0]["status"] == "supported"


async def test_final_release_is_a_real_gap_even_when_preliminary_gdp_exists(tmp_path):
    task = "报告2024年GDP最终核实的现价总量。"
    seed = source("2024年GDP初步核算为1349084亿元。", title="2024年初步核算")
    final = source("最终核实，2024年GDP现价总量为1348066亿元。", "final", "2024年最终核实")
    search = Searches(task, [SearchHit(final.url, final.title, "", final.domain)])
    result = await pipe(tmp_path, search, FakeFetcher({final.url: final.text})).run(task, compile_to_pdf=False, extra_sources=[seed])
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert any(s["gap"] == "final_release" for s in result.research_audit["searches"])


async def test_document_only_mode_never_uses_new_or_legacy_active_search(tmp_path, monkeypatch):
    from core_engine.report import router

    async def forbidden(*args, **kwargs):
        pytest.fail("Document-only mode attempted a web search")

    monkeypatch.setattr(router, "routed_search", forbidden)
    task = "Acme's net profit in 2024"
    search = Searches(task)
    result = await pipe(tmp_path, search, verify_strategy="aligned").run(task, compile_to_pdf=False,
        skip_search=True, extra_sources=[source("GDP was 1200亿元 in 2024.")])
    assert result.status is PipelineStatus.OUT_OF_SCOPE and not search.calls
    assert result.research_audit["stop_reason"] == "document_only"
    assert not result.research_audit["searches"]


async def test_negated_preliminary_request_does_not_create_a_fake_preliminary_gap(tmp_path):
    task = "报告2024年GDP最终核实总量，不要用初步数。"
    final = source("最终核实，2024年GDP现价总量为100亿元。", title="2024年最终核实")
    result = await pipe(tmp_path).run(task, compile_to_pdf=False, skip_search=True, extra_sources=[final])
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert result.answerability["requested_qualifiers"] == ["final_release"]


@pytest.mark.parametrize(("rounds", "queries", "expected"), [(0, 4, "retrieval_disabled"), (2, 0, "retrieval_disabled"),
                                                            (2, 1, "no_new_sources")])
async def test_research_respects_budgets_and_still_refuses_missing_facts(tmp_path, rounds, queries, expected):
    task = "Acme's 2024 revenue"
    search = Searches(task)
    result = await pipe(tmp_path, search, research_max_rounds=rounds, research_max_queries=queries).run(
        task, compile_to_pdf=False, extra_sources=[source("GDP was 1200亿元 in 2024.")])
    assert result.status is PipelineStatus.OUT_OF_SCOPE and not result.tex_path
    assert result.research_audit["stop_reason"] == expected
    assert len(result.research_audit["searches"]) <= queries


async def test_repair_deadline_cancels_search_and_preserves_refusal_audit(tmp_path):
    task = "Acme's 2024 revenue"

    class Slow(Searches):
        cancelled = False

        async def search(self, topic, *, max_results):
            if topic == task:
                return []
            try:
                await asyncio.sleep(1)
            finally:
                self.cancelled = True

    search = Slow(task)
    result = await pipe(tmp_path, search, research_deadline_s=.03).run(task, compile_to_pdf=False,
        extra_sources=[source("GDP was 1200亿元 in 2024.")])
    assert search.cancelled and result.research_audit["stop_reason"] == "deadline"
    assert result.status is PipelineStatus.OUT_OF_SCOPE
    packet = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
    assert packet["research_plan"]["objective"] == task


async def test_explicit_gather_queries_do_not_expand_and_known_urls_are_not_fetched(tmp_path):
    class Fetch(FakeFetcher):
        def __init__(self, rows):
            super().__init__(rows)
            self.calls = []

        async def fetch(self, url):
            self.calls.append(url)
            return await super().fetch(url)

    task = "bounded gap"
    old, new = source("Old text"), source("New text", "new")
    search = Searches("other", [SearchHit(s.url, s.title, "", s.domain) for s in (old, new)])
    fetch = Fetch({new.url: new.text})
    sources, _ = await gather_sources(task, search, fetch, queries=[task], exclude_urls={old.url},
                                       max_pages=1, settings=settings(tmp_path))
    assert search.calls == [task] and fetch.calls == [new.url] and len(sources) == 1


async def test_cancelled_gather_cleans_up_fetches(tmp_path):
    class SlowFetch:
        active = 0

        async def fetch(self, url):
            self.active += 1
            try:
                await asyncio.sleep(1)
            finally:
                self.active -= 1

    search = Searches("other", [SearchHit(s.url, s.title, "", s.domain) for s in (source("a"), source("b", "b"))])
    fetcher = SlowFetch()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(gather_sources("gap", search, fetcher, queries=["gap"], settings=settings(tmp_path)), .02)
    assert fetcher.active == 0


def test_model_context_budget_preserves_whole_observations():
    small, large = {"id": "small", "text": "Source fact."}, {"id": "large", "text": "x" * 1000}
    assert bounded_rows([large, small], 100) == [small]
    assert bounded_rows([small], 0) == []


def test_cancelled_counterevidence_search_cannot_be_reported_as_completed():
    finding = {"counter_claim_ids": []}
    audit = {"searches": [{"kind": "counterevidence", "status": "deadline"}]}
    assert counterevidence_status(finding, audit) == "not_searched"
    audit["initial_queries"] = [{"kind": "counterevidence", "status": "completed"}]
    assert counterevidence_status(finding, audit) == "searched_but_no_counterevidence_selected"


def argument_inputs():
    text = "Acme's revenue was 100 CNY in 2023.\n\nAcme's revenue was 120 CNY in 2024."
    src = source(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    claims, citations, metrics = [], [], []
    for i, (year, value, quote) in enumerate(((2023, "100", text.split("\n\n")[0]), (2024, "120", text.split("\n\n")[1])), 1):
        claim = Claim(f"c{i}", quote, candidate_source_urls=[src.url], verified=True)
        claims.append(claim)
        start = text.index(quote)
        citations.append(SentenceCitation(f"r{i}", claim.id, src.url, quote, start, start + len(quote), 1,
                                          None, digest, "verbatim"))
        metrics.append(Metric(f"m{i}", "revenue", "Acme", Decimal(value), "CNY", str(year), "China",
                              "consolidated", "current", value_kind="actual", claim_ids=[claim.id]))
    derived, record = calculate("growth_rate", metrics, output_id="d1")
    task = "Compare Acme's revenue in 2023 and 2024"
    plan = ResearchPlan(task, [ResearchQuestion("q1", task, "comparison", True)]).to_dict()
    evidence = {"claims": [asdict(c) for c in claims], "citations": [asdict(c) for c in citations]}
    row = {"question_id": "q1", "kind": "interpretation", "conclusion": "Acme's revenue increased by 20% from 2023 to 2024; stronger sales may be one explanation.",
           "premise_claim_ids": ["c1", "c2"], "counter_claim_ids": [], "calculation_ids": [record["id"]],
           "assumptions": ["The stated revenue scope remains comparable."],
           "alternatives": ["Price changes may also explain the difference."],
           "uncertainty": "Revenue alone does not identify volume or demand changes."}
    return plan, evidence, [record], [m.to_dict() for m in [*metrics, derived]], row


def test_analysis_can_use_registered_calculations_and_keeps_an_explicit_inference_label():
    plan, evidence, records, metrics, row = argument_inputs()
    finding = validate_finding(row, plan, evidence, records, metrics)
    assert finding["status"] == "inference" and finding["confidence"] == "limited"
    assert finding["citation_ids"] == ["r1", "r2"] and finding["human_review_required"]


@pytest.mark.parametrize("mutation", ["unknown_claim", "unverified", "new_number", "wrong_calculation", "missing_basis",
                                     "causal_certainty", "unconditional_forecast", "missing_alternatives", "missing_uncertainty", "unknown_question"])
def test_analysis_rejects_broken_provenance_fabricated_numbers_and_causal_overreach(mutation):
    plan, evidence, records, metrics, row = argument_inputs()
    if mutation == "unknown_claim":
        row["premise_claim_ids"] = ["invented"]
    elif mutation == "unverified":
        evidence["claims"][0]["verified"] = False
    elif mutation == "new_number":
        row["conclusion"] = row["conclusion"].replace("20%", "200%")
    elif mutation == "wrong_calculation":
        row["calculation_ids"] = ["invented"]
    elif mutation == "missing_basis":
        row["premise_claim_ids"] = ["c1"]
    elif mutation == "causal_certainty":
        row["conclusion"] = "Revenue proves Acme's demand caused the increase."
    elif mutation == "unconditional_forecast":
        row["conclusion"] = "Acme's revenue will increase next year."
    elif mutation == "missing_alternatives":
        row.update(kind="causal_hypothesis", alternatives=[])
    elif mutation == "missing_uncertainty":
        row["uncertainty"] = ""
    else:
        row["question_id"] = "unplanned"
    with pytest.raises(ValueError):
        validate_finding(row, plan, evidence, records, metrics)


async def test_critic_veto_and_unavailable_critic_withhold_analysis(tmp_path):
    plan, evidence, records, metrics, row = argument_inputs()
    llm = FakeLLM()
    llm.set_complete("Evidence-based analyst", json.dumps({"findings": [row]}))
    findings, audit = await analyze(plan, {}, evidence, records, metrics, llm, settings(tmp_path))
    assert not findings and audit["withheld"]
    llm.set_complete("Analysis critic", '{"acceptable":true,"unsupported_assertions":[]}')
    findings, audit = await analyze(plan, {}, evidence, records, metrics, llm, settings(tmp_path))
    assert len(findings) == 1 and audit["accepted"] == 1
    assert findings[0]["counterevidence_status"] == "not_searched"


async def test_analysis_deadline_withholds_pending_interpretation(tmp_path):
    plan, evidence, records, metrics, row = argument_inputs()

    class SlowCritic(FakeLLM):
        async def complete(self, system, user):
            if "Analysis critic" in system:
                await asyncio.sleep(1)
            return await super().complete(system, user)

    llm = SlowCritic()
    llm.set_complete("Evidence-based analyst", json.dumps({"findings": [row]}))
    findings, audit = await analyze(plan, {}, evidence, records, metrics, llm, settings(tmp_path, reasoning_deadline_s=.01))
    assert not findings and audit["status"] == "deadline"


async def test_real_release_research_benchmark_recovers_abstains_and_limits_causal_claims(tmp_path):
    from core_engine.report.research_eval import run_research_cases

    summary = await run_research_cases(tmp_path / "research.json")
    assert summary["automatic_checks_passed"] == summary["cases_run"] == 6
    assert not summary["live_model_evaluated"] and summary["human_review_required"]
    review = json.loads(Path(summary["human_review_form"]).read_text(encoding="utf-8"))
    assert all(value is None for row in review["reviews"] for value in row["scores"].values())


async def test_independent_grader_rechecks_analysis_and_research_instead_of_trusting_scores(tmp_path):
    plan, _evidence, _records, _metrics, row = argument_inputs()
    llm = FakeLLM()
    llm.set_complete("Evidence-based analyst", json.dumps({"findings": [row]}))
    llm.set_complete("Analysis critic", '{"acceptable":true,"unsupported_assertions":[]}')
    # Source-only observations remain the real report's body; tested inference is
    # attached through the normal pipeline in the integration test below.
    seed = source("Acme's revenue was 100 CNY in 2023.\n\nAcme's revenue was 120 CNY in 2024.")
    row = {**row, "calculation_ids": [], "conclusion": "Acme's reported revenue differs across the cited periods; price changes may explain some difference."}
    llm.set_complete("Evidence-based analyst", json.dumps({"findings": [row]}))
    result = await pipe(tmp_path, llm=llm).run(plan["objective"], compile_to_pdf=False, skip_search=True, extra_sources=[seed])
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert len(result.report.analysis) == 1
    packet = evidence_packet(result.report)
    assert not check_packet(packet)["integrity_errors"]
    assert check_packet(packet)["checked_analyses"] == 1
    forged = copy.deepcopy(packet)
    forged["analysis"][0]["conclusion"] = "Acme revenue will grow 999% next year."
    assert check_packet(forged)["integrity_errors"]


async def test_audit_timeout_preserves_each_planned_question_in_the_refusal_packet(tmp_path, monkeypatch):
    async def delayed_evaluation(self, claims, sources):
        await asyncio.sleep(1)

    monkeypatch.setattr(ResearchSession, "evaluate", delayed_evaluation)
    result = await pipe(tmp_path, research_deadline_s=.01).run("Acme's 2024 revenue", compile_to_pdf=False,
        extra_sources=[source("Acme's 2024 revenue was 100亿元.")])
    assert result.status is PipelineStatus.OUT_OF_SCOPE and result.research_audit["stop_reason"] == "deadline"
    packet = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
    assert packet["research_audit"]["questions"][0]["status"] == "undetermined"
    assert not check_packet(packet)["integrity_errors"]
    forged = copy.deepcopy(packet)
    forged["research_audit"]["questions"][0]["claim_ids"] = ["invented"]
    assert check_packet(forged)["integrity_errors"]
