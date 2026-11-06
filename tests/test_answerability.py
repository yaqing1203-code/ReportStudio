from __future__ import annotations

import json
from decimal import Decimal

import pytest

from core_engine.config import get_settings
from core_engine.report.answerability import assess_coverage
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import FakeLLM
from core_engine.report.models import Claim, PipelineStatus, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet, load_dataset, score_case
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def coverage(question, texts, title="Report 2024"):
    source = Source("https://example.gov/report", "example.gov", title, SourceKind.GOVERNMENT, " ".join(texts))
    claims = [Claim(f"c{i}", text, candidate_source_urls=[source.url]) for i, text in enumerate(texts)]
    return assess_coverage(question, claims, [source])


@pytest.mark.parametrize("texts", [
    ["In 2024 GDP was 1200亿元."],
    ["Acme's 2023 net profit was 12亿元."],
    ["Other's 2024 net profit was 12亿元."],
    ["Acme2's 2024 net profit was 12亿元."],
    ["Acme's 2024 net profit margin was 12%."],
    ["Acme's 2024 net profit margin was 12%, and revenue was 200亿元."],
    ["Acme's 2024 net profit was not disclosed; revenue was 200亿元."],
])
def test_unrelated_missing_wrong_period_or_margin_does_not_answer_profit_question(texts):
    result = coverage("What was Acme's net profit in 2024?", texts)
    assert result.status == "insufficient" and result.missing


def test_each_requested_entity_and_year_needs_its_own_witness():
    question = "Compare Acme's and Other's net profit in 2023 and 2024."
    texts = [f"{entity}'s {year} net profit was 12亿元." for entity in ("Acme", "Other") for year in (2023, 2024)]
    result = coverage(question, texts)
    assert result.status == "answerable"
    assert len(result.witnesses["net_profit/Acme@2023"]) == 1
    result = coverage(question, texts[:-1])
    assert result.status == "insufficient" and "net_profit/Other@2024" in result.missing


@pytest.mark.parametrize(("question", "text"), [
    ("Acme's 2024Q1 revenue", "Acme's 2024 revenue was 20亿元."),
    ("Acme's 2024年第一季度营收", "Acme's 2024年第二季度营收为20亿元。"),
    ("Acme's FY2024 revenue", "Acme's 2024 revenue was 20亿元."),
])
def test_quarter_and_fiscal_periods_cannot_be_substituted_by_year(question, text):
    assert coverage(question, [text]).status == "insufficient"


def test_source_title_cannot_disguise_wrong_claim_year_and_broad_topic_needs_overlap():
    assert coverage("Acme's 2024 revenue", ["Acme's 2023 revenue was 20亿元."], "Report 2024").status == "insufficient"
    assert coverage("quantum computing", ["GDP increased by 5% in 2024."]).status == "insufficient"
    assert coverage("Acme's revenue", ["Acme's revenue was $12.5 billion."]).status == "answerable"
    assert coverage("Acme's 2025 revenue", ["Acme's revenue was 20亿元."], "2024 report (2025 release)").status == "insufficient"


def pipeline(tmp_path, llm=None):
    pipe = ReportPipeline(llm=llm or FakeLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    pipe._s = get_settings().model_copy(update={
        "min_authoritative_sources": 1, "min_total_claims": 1, "verify_strategy": "cross_reference",
        "verify_mode": "strict", "report_locale": "zh", "enable_charts": True,
        "enable_deep_dive": False, "enable_synthesis_sections": False, "active_verify_enabled": False,
        "max_reverify_attempts": 0, "output_dir": tmp_path,
    })
    pipe._harness._s = pipe._s
    return pipe


async def test_known_gdp_profit_failure_now_abstains_without_irrelevant_body_or_pdf(tmp_path):
    data = load_dataset()
    row = data["sources"][0]
    source = Source(row["url"], "stats.gov.cn", row["title"], SourceKind.GOVERNMENT, row["text"])
    case = next(c for c in data["cases"] if c["id"] == "insufficient-profit-evidence")
    result = await pipeline(tmp_path).run(case["question"], extra_sources=[source], skip_search=True)
    assert result.status is PipelineStatus.OUT_OF_SCOPE
    assert result.report is None and result.tex_path is None and result.pdf_path is None
    assert "net_profit" in result.message
    packet = json.loads(next(tmp_path.glob("refusal-*.evidence.json")).read_text(encoding="utf-8"))
    assert packet["answerability"]["missing"] and not packet["sections"]
    assert score_case(packet, case, data)["automatic_checks_passed"]
    assert not score_case({}, case, data)["automatic_checks_passed"]


async def test_full_pipeline_validates_trend_and_exports_replayable_calculation_ledger(tmp_path):
    text = " ".join(
        f"In {year} the market was {amount} USD bn "
        "[entity=Solar; geography=China; scope=passenger-cars; price_basis=current; value_kind=actual]."
        for year, amount in ((2023, "90"), (2024, "120")))
    source = Source("https://example.gov/market", "example.gov", "Synthetic market example",
                    SourceKind.GOVERNMENT, text)
    result = await pipeline(tmp_path).run("market size in 2023 and 2024", compile_to_pdf=False,
                                         extra_sources=[source], skip_search=True)
    assert result.status is PipelineStatus.COMPLETED, result.message
    report = result.report
    assert len(report.metrics) == 4 and len(report.calculations) == 2
    assert report.kg.market.cagr_pct == Decimal("33.333333")
    assert any(section.latex_blocks for section in report.sections)
    packet = evidence_packet(report)
    check = check_packet(packet)
    assert not check["integrity_errors"]
    assert check["calculation_replay_rate"] == 1 and check["checked_charts"] == 1
    tex = next(tmp_path.glob("*.tex")).read_text(encoding="utf-8")
    assert "Deterministic Calculations" in tex and "Chart Validation" in tex


async def test_grounded_but_unused_requested_fact_does_not_make_irrelevant_report_answerable(tmp_path):
    class DropsAnswer(FakeLLM):
        async def draft_section(self, heading, claims, instructions=""):
            return "GDP was 1200亿元 in 2024."

    text = "Acme's 2024 net profit was 12亿元. GDP was 1200亿元 in 2024."
    source = Source("https://example.gov/profit", "example.gov", "Acme 2024", SourceKind.GOVERNMENT, text)
    result = await pipeline(tmp_path, DropsAnswer()).run("Acme's 2024 net profit", compile_to_pdf=False,
                                                       extra_sources=[source], skip_search=True)
    assert result.status is PipelineStatus.OUT_OF_SCOPE and result.report is None
