"""Publication policy regressions; all observations here are synthetic fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core_engine.config import get_settings
from core_engine.report.grounding import SentenceGrounder, evidence_packet
from core_engine.report.latex import render_tex
from core_engine.report.llm import FakeLLM
from core_engine.report.metrics import catalog_from_claims
from core_engine.report.models import Claim, ReportData, ReportSection, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.publication import apply_policy, context_differences
from core_engine.report.quality_eval import check_packet
from core_engine.report.research import ResearchPlan, ResearchQuestion, ResearchSession
from core_engine.report.verify import VerificationHarness

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def settings(**overrides):
    return get_settings().model_copy(update={
        "report_locale": "zh", "active_verify_enabled": False,
        "research_enabled": False, "reasoning_enabled": False,
        "enable_charts": False, "enable_deep_dive": False,
        "enable_synthesis_sections": False, "min_authoritative_sources": 1,
        "min_total_claims": 1, **overrides})


def pair():
    sources = [Source("https://lab.example/paper", "lab.example", "实验报告", SourceKind.ACADEMIC,
                      "测试电芯的市场份额为35%。"),
               Source("https://other.example/study", "other.example", "另一报告", SourceKind.GENERAL_WEB,
                      "测试电芯的市场份额为60%。")]
    claims = [Claim(f"c{i}", s.text, candidate_source_urls=[s.url], entity="测试电芯",
                    attribute="market_share", value=value)
              for i, (s, value) in enumerate(zip(sources, ("35%", "60%")), 1)]
    return claims, sources


async def grounded(claims, sources):
    return await SentenceGrounder(FakeLLM(), settings()).anchor_claims(claims, sources)


async def test_single_general_source_is_publishable_with_explicit_attribution():
    claims, sources = pair()
    source, claim = sources[1], claims[1]
    report = ReportData("测试电芯", "报告", "", claims=[claim], sources=[source], sections=[
        ReportSection("发现", claim_ids=[claim.id], paragraphs=[f"[[{claim.id}]] {claim.text}"])])
    report = await SentenceGrounder(FakeLLM(), settings()).audit_report(report)
    sentence = report.sections[0].sentences[0]
    assert sentence.status == "supported" and sentence.evidence_status == "source_reported"
    assert sentence.attribution == "据《另一报告》报告："
    assert report.evidence_policy["claims"][claim.id]["truth_status"] == "not_established"
    assert sentence.attribution in render_tex(report, settings=settings())
    assert check_packet(evidence_packet(report))["integrity_errors"] == []


async def test_same_work_and_republication_never_upgrade_to_multiple_origins():
    text = "实验材料的电导率达到3.5%。"
    sources = [Source(f"https://{domain}/paper", domain, "论文", SourceKind.ACADEMIC,
                      text + suffix, metadata={"doi": "10.1234/solid-state.1"})
               for domain, suffix in (("publisher.example", "原始全文。"), ("registry.example", "收录摘要。"))]
    claims = [Claim("c1", text, candidate_source_urls=[s.url for s in sources])]
    _, citations = await grounded(claims, sources)
    policy = apply_policy(claims, sources, citations)
    assert policy["source_reported"] == 1 and policy["corroborated"] == 0
    assert policy["claims"]["c1"]["origin_group_count"] == 1
    for s in sources:
        s.metadata = {}
        s.text = text
    _, citations = await grounded(claims, sources)
    assert apply_policy(claims, sources, citations)["corroborated"] == 0


async def test_different_source_groups_raise_support_without_claiming_replication():
    text = "测试电芯的市场份额为35%。"
    sources = [Source(f"https://{name}.example/a", name, name, SourceKind.GENERAL_WEB, text + suffix)
               for name, suffix in (("a", "本调查采用抽样。"), ("b", "本统计覆盖所有样本。"))]
    claims = [Claim("c1", text, candidate_source_urls=[s.url for s in sources])]
    _, citations = await grounded(claims, sources)
    policy = apply_policy(claims, sources, citations)
    assert policy["corroborated"] == 1
    assert policy["independence_is_not_proven"] is True
    assert policy["claims"]["c1"]["truth_status"] == "not_established"


@pytest.mark.parametrize(("left", "right", "key"), [
    ("全固态电池达到100圈。", "半固态电池达到200圈。", "battery_scope"),
    ("电芯在25°C完成100圈。", "电芯在60°C完成200圈。", "temperature"),
    ("电芯在5 MPa完成100圈。", "电芯在10 MPa完成200圈。", "pressure"),
    ("2023年电芯达到100圈。", "2024年电芯达到200圈。", "year"),
    ("全电芯达到100圈。", "材料级测试达到200圈。", "cell_scope"),
    ("电芯在25°C完成100圈。", "电芯完成200圈。", "temperature"),
])
async def test_conditions_precede_contradiction_model(left, right, key):
    class AggressiveModel(FakeLLM):
        async def check_contradiction(self, a, b):
            raise AssertionError("Different or missing conditions must be checked first")
    claims = [Claim("c1", left, entity="电芯", attribute="cycles"), Claim("c2", right, entity="电芯", attribute="cycles")]
    assert key in context_differences(*claims)
    harness = VerificationHarness(AggressiveModel())
    harness._s = settings(verify_strategy="aligned")
    result = await harness.verify(claims, [])
    assert result.clusters[0]["relation"] == "complement"
    assert not any(c.conflicting_claim_ids for c in claims)


def test_equivalent_pressure_and_amount_units_do_not_create_false_conflicts():
    assert not context_differences(Claim("a", "电芯在5 MPa运行。"), Claim("b", "电芯在5000 kPa运行。"))
    a = Claim("a", "营收100亿元。", value="100亿元")
    b = Claim("b", "营收1000000万元。", value="1000000万元")
    assert "equivalent_unit_representation" in context_differences(a, b)


async def test_anchored_conflict_retains_both_sides_but_blocks_prose_and_metrics():
    claims, sources = pair()
    harness = VerificationHarness(FakeLLM())
    harness._s = settings(verify_strategy="aligned")
    result = await harness.verify(claims, sources)
    assert len(result.kept_claims) == 2 and not result.rejected_claims
    _, citations = await grounded(claims, sources)
    assert apply_policy(claims, sources, citations)["disputed"] == 2
    assert catalog_from_claims(claims, citations)[0] == []
    report = ReportData("测试电芯", "报告", "", claims=[], disputed_claims=claims, sources=sources,
                        sections=[ReportSection("发现", claim_ids=["c1"], paragraphs=["[[c1]] " + claims[0].text])])
    report = await SentenceGrounder(FakeLLM(), settings()).audit_report(report)
    assert report.sections[0].sentences[0].status == "withheld"
    assert all(s.status == "supported" and s.evidence_status == "disputed" for s in report.dispute_sentences)
    assert len(report.citations) == 2
    assert "来源分歧与暂缓结论" in render_tex(report, settings=settings())
    assert check_packet(evidence_packet(report))["integrity_errors"] == []
    return report


async def test_unanchored_conflicting_candidate_is_only_a_lead():
    claims, sources = pair()
    claims[0].conflicting_claim_ids = ["c2"]
    claims[1].conflicting_claim_ids = ["c1"]
    sources[1].text = "未披露任何市场份额。"
    _, citations = await grounded(claims, sources)
    policy = apply_policy(claims, sources, citations)
    assert claims[0].reporting_status == "source_reported"
    assert claims[1].reporting_status == "lead_only" and not claims[1].verified
    assert policy["disputed"] == 0


async def test_independent_grader_rejects_changed_status_and_disputes_in_main_body():
    report = await test_anchored_conflict_retains_both_sides_but_blocks_prose_and_metrics()
    packet = evidence_packet(report)
    changed = copy.deepcopy(packet)
    changed["claims"][0].update(reporting_status="source_reported", verified=True)
    assert any("publication_status" in e for e in check_packet(changed)["integrity_errors"])
    changed = copy.deepcopy(packet)
    changed["sections"][0]["sentences"] = changed.pop("dispute_sentences")
    assert "invalid_sentence_evidence_attribution" in check_packet(changed)["integrity_errors"]


async def test_conflicts_cannot_complete_research_even_if_model_says_supported():
    class Optimist(FakeLLM):
        async def complete(self, system, user):
            if "coverage assessor" in system.lower():
                row = json.loads(user)["questions"][0]
                return json.dumps({"assessments": [{"id": row["id"], "status": "supported",
                                                   "claim_ids": ["c1", "c2"], "missing": []}]})
            return await super().complete(system, user)
    claims, sources = pair()
    for a, b in ((claims[0], claims[1]), (claims[1], claims[0])):
        a.conflicting_claim_ids = [b.id]
    plan = ResearchPlan("测试电芯的市场份额", [ResearchQuestion("q1", "测试电芯的市场份额", "fact", True, "原文数值")])
    session = ResearchSession(plan, Optimist(), settings())
    await session.evaluate(claims, sources)
    assert session.audit["coverage_status"] == "partial"
    assert session.audit["questions"][0]["status"] != "supported"
    assert "unresolved_source_conflict" in session.audit["questions"][0]["missing"]


@pytest.mark.parametrize("strategy", ["aligned", "conflict_only", "cross_reference"])
@pytest.mark.parametrize("research_enabled", [True, False])
async def test_full_pipeline_displays_disagreements_and_excludes_calculation_inputs(tmp_path, strategy, research_enabled):
    claims, sources = pair()
    pipeline = ReportPipeline(llm=FakeLLM())
    pipeline._s = settings(verify_strategy=strategy, verify_mode="strict", research_enabled=research_enabled,
                           research_max_rounds=0, output_dir=tmp_path)
    pipeline._harness._s = pipeline._s
    result = await pipeline.run("测试电芯的市场份额", extra_sources=sources, skip_search=True,
                                seed_claims=claims, compile_to_pdf=False)
    assert result.report is not None, result.message
    report = result.report
    assert len(report.disputed_claims) == 2 and report.claims == []
    assert report.metrics == [] and report.calculations == []
    assert result.evidence_policy["disputed"] == 2
    assert result.answerability["status"] == "insufficient"
    if research_enabled:
        assert report.research_audit["coverage_status"] == "partial"
    assert check_packet(evidence_packet(report))["integrity_errors"] == []


async def test_unlocated_candidates_are_retained_in_refusal_artifact(tmp_path):
    source = Source("https://lab.example/a", "lab.example", "固态电池", SourceKind.ACADEMIC, "资料只讨论化学组成。")
    llm = FakeLLM()
    llm.set_structured_claims(source.url, [{"text": "秘密电芯已实现量产。"}])
    pipeline = ReportPipeline(llm=llm)
    pipeline._s = settings(verify_strategy="aligned", output_dir=tmp_path)
    pipeline._harness._s = pipeline._s
    result = await pipeline.run("固态电池是否量产", extra_sources=[source], skip_search=True, compile_to_pdf=False)
    assert result.report is None and result.evidence_path
    packet = json.loads(Path(result.evidence_path).read_text(encoding='utf-8'))
    assert packet["leads"] and packet["sections"] == []
    assert packet["evidence_policy"]["lead_only"] == 1
    assert check_packet(packet)["integrity_errors"] == []


@pytest.mark.parametrize("timeout", [True, False])
async def test_same_scope_differing_values_never_become_settled_when_comparison_fails(timeout):
    from core_engine.report.llm import ContradictionJudgement

    class Inconclusive(FakeLLM):
        async def check_contradiction(self, a, b):
            if timeout:
                raise TimeoutError("simulated provider failure")
            return ContradictionJudgement(contradict=False)
    claims, sources = pair()
    harness = VerificationHarness(Inconclusive())
    harness._s = settings(verify_strategy="aligned")
    result = await harness.verify(claims, sources)
    assert len(result.kept_claims) == 2 and not result.rejected_claims
    _, citations = await grounded(claims, sources)
    assert apply_policy(claims, sources, citations)["disputed"] == 2
    assert catalog_from_claims(claims, citations)[0] == []


async def test_candidates_from_earlier_research_rounds_survive_as_reviewable_leads(tmp_path):
    from core_engine.report.verify import VerificationReport

    class LaterRound(ReportPipeline):
        async def _fill_evidence_gaps(self, session, sources, verification, result, emit, *, skip_search):
            kept = [c for c in verification.kept_claims if c.id == "good"]
            await session.evaluate(kept, sources)
            return sources, VerificationReport(kept_claims=kept)

    source = Source("https://lab.example/a", "lab.example", "原文", SourceKind.ACADEMIC, "固态电池使用固体电解质。")
    claims = [Claim("good", source.text, candidate_source_urls=[source.url]),
              Claim("lost", "固态电池已达到8000 Wh/kg。", candidate_source_urls=[source.url])]
    pipeline = LaterRound(llm=FakeLLM())
    pipeline._s = settings(research_enabled=True, output_dir=tmp_path)
    pipeline._harness._s = pipeline._s
    result = await pipeline.run("固态电池", extra_sources=[source], skip_search=True,
                                seed_claims=claims, compile_to_pdf=False)
    assert result.report is not None, result.message
    assert result.report.leads[0]["text"] == claims[1].text
    assert result.report.leads[0]["claim_id"] != "good"
    assert result.evidence_policy["lead_only"] == 1
    assert check_packet(evidence_packet(result.report))["integrity_errors"] == []


async def test_withheld_draft_exports_consistent_research_proof_in_refusal(tmp_path):
    class UnsupportedDraft(FakeLLM):
        async def draft_section(self, heading, claims, **kwargs):
            return '{"sentences":[{"text":"火星已有城市。","claim_ids":["good"]}]}'

    source = Source("https://lab.example/a", "lab.example", "原文", SourceKind.ACADEMIC, "固态电池使用固体电解质。")
    pipeline = ReportPipeline(llm=UnsupportedDraft())
    pipeline._s = settings(research_enabled=True, output_dir=tmp_path)
    pipeline._harness._s = pipeline._s
    result = await pipeline.run("固态电池", extra_sources=[source], skip_search=True,
                                seed_claims=[Claim("good", source.text, candidate_source_urls=[source.url])],
                                compile_to_pdf=False)
    assert result.report is None and result.status.value == "out_of_scope"
    packet = json.loads(Path(result.evidence_path).read_text(encoding='utf-8'))
    assert packet["sections"] == [] and packet["claims"][0]["id"] == "good"
    assert packet["evidence_policy"]["source_reported"] == 1
    assert check_packet(packet)["integrity_errors"] == []
