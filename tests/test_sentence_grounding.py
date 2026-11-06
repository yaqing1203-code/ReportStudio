from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from core_engine.config import get_settings
from core_engine.report.grounding import (
    AnchorReuse,
    SentenceGrounder,
    evidence_packet,
    parse_draft,
    write_evidence,
)
from core_engine.report.latex import render_tex
from core_engine.report.llm import EntailmentJudgement, FakeLLM
from core_engine.report.models import (
    Claim,
    ReportData,
    ReportSection,
    ReportSentence,
    Source,
    SourceKind,
)
from core_engine.report.quality_eval import check_packet

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def report_with(text="利润增长3.5%。", draft=None):
    source = Source("https://example.gov/report", "example.gov", "原始报告", SourceKind.GOVERNMENT,
                    "说明段落。\n\n" + text)
    claim = Claim("c1", text, candidate_source_urls=[source.url], verified=True)
    section = ReportSection("概况", "industry_overview", claim_ids=["c1"],
                            paragraphs=[draft or json.dumps({"sentences": [{"text": text, "claim_ids": ["c1"]}]})])
    return ReportData("topic", "报告", "", sections=[section], claims=[claim], sources=[source])


async def audit(report, llm=None, **updates):
    settings = get_settings().model_copy(update={"grounding_deadline_s": 2, **updates})
    return await SentenceGrounder(llm or FakeLLM(), settings).audit_report(report)


async def test_sentence_to_claim_to_exact_snapshot_roundtrip(tmp_path):
    report = await audit(report_with())
    sentence = report.sections[0].sentences[0]
    assert sentence.status == "supported"
    record = report.citations[0]
    source = report.sources[0]
    assert source.text[record.start_char:record.end_char] == record.quote
    assert record.paragraph == 2
    assert record.source_sha256 == source.content_sha256
    path = write_evidence(report, tmp_path / "report.tex")
    packet = json.loads(path.read_text(encoding="utf-8"))
    assert check_packet(packet)["integrity_errors"] == []
    assert packet["sections"][0]["sentences"][0]["citation_ids"] == [record.id]
    tex = render_tex(report)
    assert f"hyperref[ev:{record.id}]" in tex
    assert record.quote.replace("%", r"\%") in tex
    assert "Snapshot SHA-256" in tex


async def test_anchor_reuse_deduplicates_concurrent_and_later_stages_but_checks_new_snapshot():
    quote = "The prototype was tested at 25 °C."
    source = Source("https://example.edu/paper", "example.edu", "Paper", SourceKind.ACADEMIC, quote)
    claim = Claim("c1", "Testing used a temperature of 25 °C.", candidate_source_urls=[source.url],
                  source_windows={source.url: [[0, len(quote)]]})

    class Reviewer(FakeLLM):
        calls = 0

        async def check_entailment(self, text, context):
            self.calls += 1
            await asyncio.sleep(0)
            assert context == quote
            return EntailmentJudgement(True, quote)

    llm = Reviewer()
    settings = get_settings().model_copy(update={"llm_max_concurrency": 2})
    reuse = AnchorReuse()
    _, citations = await SentenceGrounder(llm, settings, anchor_reuse=reuse).anchor_claims(
        [claim, replace(claim, id="c2", evidence=[])], [source])
    assert {c.claim_id for c in citations} == {"c1", "c2"}
    _, citations = await SentenceGrounder(llm, settings, anchor_reuse=reuse).anchor_claims([claim], [source])
    assert len(citations) == 1 and llm.calls == 1 and reuse.hits == 2
    revised = replace(source, text=quote + "\n\nA correction was appended.")
    _, citations = await SentenceGrounder(llm, settings, anchor_reuse=reuse).anchor_claims([claim], [revised])
    assert len(citations) == 1 and llm.calls == 2
    assert citations[0].source_sha256 == revised.content_sha256


async def test_anchor_failure_can_be_retried_in_a_later_stage():
    quote = "Tests were run at 25 °C."
    source = Source("https://example.edu/paper", "example.edu", "Paper", SourceKind.ACADEMIC, quote)
    claim = Claim("c1", "Testing used a temperature of 25 °C.", candidate_source_urls=[source.url])

    class Reviewer(FakeLLM):
        calls = 0

        async def check_entailment(self, text, context):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("first attempt expired")
            return EntailmentJudgement(True, quote)

    llm, settings, reuse = Reviewer(), get_settings(), AnchorReuse()
    first, _ = await SentenceGrounder(llm, settings, anchor_reuse=reuse).anchor_claims([claim], [source])
    second, _ = await SentenceGrounder(llm, settings, anchor_reuse=reuse).anchor_claims([claim], [source])
    assert first == [] and second == [claim] and llm.calls == 2 and reuse.hits == 0


async def test_new_cross_reference_is_persisted_in_claim_provenance_for_independent_replay():
    from core_engine.report.models import Evidence
    report = report_with()
    other = Source("https://example.edu/reprint", "example.edu", "Copy", SourceKind.ACADEMIC, report.sources[0].text)
    report.sources.append(other)
    report.claims[0].evidence.append(Evidence(other.url, "利润增长3.5%。", True, "Cross-reference"))
    report = await audit(report)
    assert other.url in report.claims[0].candidate_source_urls
    packet = evidence_packet(report)
    assert len(packet["citations"]) == 2
    assert not check_packet(packet)["integrity_errors"]


async def test_case_folded_verbatim_match_cannot_change_scientific_si_prefix():
    report = report_with("Conductivity was 4 mS/cm.")
    report.claims[0].text = "Conductivity was 4 MS/cm."
    report.sections[0].paragraphs = [json.dumps({"sentences": [{"text": report.claims[0].text, "claim_ids": ["c1"]}]})]
    report = await audit(report)
    assert report.sections[0].sentences[0].status == "withheld"
    assert not report.citations


async def test_review_marker_cannot_be_parsed_as_latex_optional_item_label():
    report = report_with()
    report.rumors = report.claims
    report.claims = []
    report = await audit(report)
    tex = render_tex(report, settings=get_settings().model_copy(update={"report_locale": "zh"}))
    assert "\\item{}" in tex and "\\item [UNVERIFIED" not in tex
    assert "待复核的来源陈述" in tex
    assert "\\addcontentsline{toc}{section}{Unverified" not in tex


@pytest.mark.parametrize("draft,reason", [
    ({"text": "利润增长35%。", "claim_ids": ["c1"]}, "numeric_value_or_unit_not_supported"),
    ({"text": "利润增长-3.5%。", "claim_ids": ["c1"]}, "numeric_value_or_unit_not_supported"),
    ({"text": "利润增长3.5个百分点。", "claim_ids": ["c1"]}, "numeric_value_or_unit_not_supported"),
    ({"text": "利润增长3.5%。", "claim_ids": ["does-not-exist"]}, "unknown_or_out_of_section_claim"),
    ({"text": "该企业一定会成为行业第一。", "claim_ids": []}, "missing_claim_citation"),
])
async def test_unsupported_draft_is_withheld_and_reviewable(draft, reason):
    report = await audit(report_with(draft=json.dumps({"sentences": [draft]})))
    sentence = report.sections[0].sentences[0]
    assert sentence.status == "withheld"
    assert sentence.reason == reason
    assert report.sections[0].paragraphs == []
    assert report.quality_status == "needs_review"
    assert draft["text"] not in render_tex(report)
    assert evidence_packet(report)["sections"][0]["sentences"][0]["text"] == draft["text"]


async def test_fabricated_quote_is_not_accepted_even_when_model_says_supported():
    class FabricatingLLM(FakeLLM):
        async def check_entailment(self, claim, source_text):
            return EntailmentJudgement(True, "fabricated source span")
    report = report_with()
    report.sources[0].text = "没有任何关于利润的数据。"
    report = await audit(report, FabricatingLLM())
    assert not report.citations
    assert report.sections[0].sentences[0].reason == "claim_has_no_anchored_support"


async def test_plain_paraphrase_without_citation_is_not_fuzzily_attributed():
    report = await audit(report_with(draft="这家公司的盈利表现有所提升。"))
    assert report.sections[0].sentences[0].status == "withheld"


async def test_exact_legacy_sentence_is_recovered_with_real_evidence():
    report = await audit(report_with(draft="利润增长3.5%。"))
    assert report.sections[0].sentences[0].status == "supported"


async def test_pdf_page_and_hash_survive_snapshot_storage():
    report = report_with()
    report.sources[0].text = "第一页正文。\f利润增长3.5%。"
    report.sources[0].document_format = "pdf"
    report = await audit(report)
    assert report.citations[0].page == 2


async def test_decorative_claim_citation_is_removed_from_exact_sentence():
    report = report_with(draft=json.dumps({"sentences": [{"text": "利润增长3.5%。", "claim_ids": ["c1", "c2"]}]}))
    report.claims.append(Claim("c2", "市场下降2%。", candidate_source_urls=[report.sources[0].url]))
    report.sources[0].text += "市场下降2%。"
    report.sections[0].claim_ids.append("c2")
    report = await audit(report)
    assert report.sections[0].sentences[0].claim_ids == ["c1"]


async def test_structured_value_conflicting_with_claim_text_is_not_published():
    report = report_with()
    report.claims[0].value = "35%"
    report = await audit(report)
    assert report.sections[0].sentences[0].status == "withheld"


async def test_audit_deadline_withholds_partial_work_without_hanging():
    import asyncio
    class HangingLLM(FakeLLM):
        async def check_entailment(self, claim, source_text):
            await asyncio.sleep(10)
    report = report_with()
    report.sources[0].text = "利润变化另见附录，未披露具体数字。"
    report = await audit(report, HangingLLM(), grounding_deadline_s=0.01)
    assert report.sections[0].sentences[0].status == "withheld"
    assert report.sections[0].sentences[0].reason == "sentence_audit_deadline"


async def test_rumor_without_source_support_never_bypasses_sentence_audit():
    report = report_with()
    report.rumors.append(Claim("r1", "秘密工厂将在2028年开业。", candidate_source_urls=[report.sources[0].url]))
    report = await audit(report)
    assert report.rumor_sentences[0].status == "withheld"
    assert "秘密工厂" not in render_tex(report)


def test_parser_splits_multi_sentence_json_and_preserves_ids():
    sentences = parse_draft('```json\n{"sentences":[{"text":"增长3.5%。下降2%。","claim_ids":["c1"]}]}\n```')
    assert len(sentences) == 2
    assert all(s.claim_ids == ["c1"] for s in sentences)
    assert parse_draft("增长3.5%。[[c1]]")[0].claim_ids == ["c1"]


async def test_sentence_count_budget_withholds_excess_instead_of_silently_skipping():
    report = report_with(draft=json.dumps({"sentences": [
        {"text": "利润增长3.5%。", "claim_ids": ["c1"]},
        {"text": "利润增长3.5%。", "claim_ids": ["c1"]}]}))
    report = await audit(report, grounding_max_sentences=1)
    assert [s.status for s in report.sections[0].sentences] == ["supported", "withheld"]
    assert report.sections[0].sentences[1].reason == "sentence_count_limit"
    assert report.grounding_audit["withheld_sentences"] == 1


def test_render_uses_per_run_locale_snapshot_instead_of_mutable_global_settings():
    settings = get_settings().model_copy(update={"report_locale": "zh"})
    assert r"\usepackage{ctex}" in render_tex(report_with(), settings=settings)


def test_all_withheld_section_shows_gap_without_leaking_original_draft():
    report = report_with()
    section = report.sections[0]
    section.paragraphs = ["DO NOT PUBLISH DRAFT"]
    section.sentences = [ReportSentence(text="DO NOT PUBLISH DRAFT", status="withheld")]
    tex = render_tex(report, settings=get_settings().model_copy(update={"report_locale": "zh"}))
    assert "本节暂无足够的原文支持内容。" in tex
    assert "DO NOT PUBLISH DRAFT" not in tex


async def test_job_persists_evidence_and_review_status_for_history(monkeypatch, tmp_path):
    import httpx

    from core_engine.app import history, runtime, server
    from core_engine.report import pipeline as pipeline_module
    from core_engine.report.models import PipelineResult, PipelineStatus

    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "_JOBS", {})
    monkeypatch.delenv("CE_APP_SESSION_TOKEN", raising=False)
    report = await audit(report_with())
    path = write_evidence(report, tmp_path / "draft.tex")

    class StubPipeline:
        async def run(self, *args, **kwargs):
            return PipelineResult(PipelineStatus.COMPLETED, "topic", report=report, evidence_path=str(path))

    monkeypatch.setattr(pipeline_module, "ReportPipeline", StubPipeline)
    job = server.Job(id="evidence-history", topic="topic")
    job.history_id = history.add_entry(kind="report", title="topic", job_id=job.id)
    server._JOBS[job.id] = job
    await server._run_job(job)
    assert job.terminal_event["has_evidence"] is True
    assert job.terminal_event["quality_status"] == "source_supported"
    server._JOBS.clear()  # simulate loss of process-local jobs
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.create_app()),
                                 base_url="http://127.0.0.1") as client:
        response = await client.get(f"/api/report/{job.id}/evidence")
        assert response.status_code == 200
        assert response.json()["citations"]
        items = (await client.get("/api/history")).json()["items"]
        assert items[0]["evidence_ready"] is True
        assert items[0]["quality_status"] == "source_supported"


async def test_pdf_parser_keeps_page_boundaries_for_real_uploaded_pdf():
    import io

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    from core_engine.report.documents import parse_document

    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for text in (b"First page introduction.", b"Revenue grew 3.5 percent."):
        page = writer.add_blank_page(300, 300)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 100 Td (" + text + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    raw = io.BytesIO()
    writer.write(raw)
    _, text = parse_document("two-pages.pdf", raw.getvalue())
    assert "\f" in text
    report = report_with(text="Revenue grew 3.5 percent.")
    report.sources[0].text = text
    report.sources[0].document_format = "pdf"
    report = await audit(report)
    assert report.citations[0].page == 2
