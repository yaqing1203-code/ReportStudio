from __future__ import annotations

import io
import zipfile

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from fastapi.testclient import TestClient

from core_engine.app import output_templates as templates
from core_engine.app import runtime


def make_template(mode="slots"):
    doc = Document()
    doc.sections[0].left_margin = Cm(3.1)
    doc.styles["Normal"].font.name = "宋体"
    doc.styles["Normal"].font.size = Pt(12)
    doc.sections[0].header.paragraphs[0].text = "固定学校页眉"
    doc.sections[0].footer.paragraphs[0].text = "固定页脚"
    p = doc.add_paragraph(style="Title")
    p.add_run("{{ti").bold = True
    p.add_run("tle}}")
    doc.add_paragraph("作者：{{作者}}；日期：{{date}}")
    if mode == "slots":
        doc.add_paragraph("{{content}}")
    elif mode == "outline":
        for heading in ("背景说明", "主要讨论", "结论与建议"):
            doc.add_heading(heading, 1)
            doc.add_paragraph("请替换这段示例正文。")
    elif mode == "named":
        for heading in ("背景说明", "主要讨论", "结论与建议"):
            doc.add_heading(heading, 1)
            doc.add_paragraph("{{section:" + heading + "}}")
    else:
        doc.add_paragraph("原有样例内容应被替换。")
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


def package(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def repack(parts):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return output.getvalue()


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)


def payload():
    return {"title": "固态电池课程文章", "date": "2026-11-04", "sections": {
        "背景说明": ["固态电池需要明确材料与结构边界。"],
        "主要讨论": ["第一段讨论界面。", "第二段讨论制造。"],
        "结论与建议": ["评价其前景应结合条件与应用场景。"]}, "references": []}


@pytest.mark.parametrize("mode", ["slots", "outline", "named", "styles"])
def test_template_fills_content_and_keeps_unmodified_package_parts(mode):
    raw = make_template(mode)
    info = templates.inspect_template(raw)
    output = templates.render_template(raw, info, payload(), {"作者": "张同学"})
    original, result = package(raw), package(output)
    assert set(original) == set(result)
    assert all(result[name] == data for name, data in original.items() if name != "word/document.xml")
    doc = Document(io.BytesIO(output))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "{{" not in text and "固态电池课程文章" in text
    assert "请替换这段示例正文" not in text and "原有样例内容应被替换" not in text
    assert "第一段讨论界面" in text and "第二段讨论制造" in text
    assert doc.sections[0].left_margin == Document(io.BytesIO(raw)).sections[0].left_margin
    assert doc.sections[0].header.paragraphs[0].text == "固定学校页眉"
    if mode != "styles":
        assert "作者：张同学" in text


def test_inline_placeholder_split_across_runs_keeps_surrounding_format():
    raw = make_template()
    doc = Document(io.BytesIO(templates.render_template(raw, templates.inspect_template(raw), payload(), {"作者": "A<&>B"})))
    assert doc.paragraphs[0].text == "固态电池课程文章"
    assert doc.paragraphs[0].runs[0].bold
    assert "A<&>B" in doc.paragraphs[1].text


def test_manual_numbered_headings_are_recognized_and_empty_body_does_not_inherit_heading_style():
    doc = Document()
    for name in ("一、背景说明", "二、主要讨论", "三、结论与建议"):
        p = doc.add_paragraph()
        p.add_run(name).bold = True
    stream = io.BytesIO()
    doc.save(stream)
    raw = stream.getvalue()
    info = templates.inspect_template(raw)
    assert info["mode"] == "outline" and len(info["sections"]) == 3
    data = payload()
    data["sections"] = {h: ["正文内容。"] for h in info["sections"]}
    output = templates.render_template(raw, info, data, {})
    doc = Document(io.BytesIO(output))
    assert sum(p.text == "正文内容。" for p in doc.paragraphs) == 3
    assert all(p.style.name == "Normal" for p in doc.paragraphs if p.text == "正文内容。")


def test_multiline_slot_in_table_and_header_inline_slots():
    doc = Document()
    doc.sections[0].header.paragraphs[0].text = "题目：{{title}}"
    table = doc.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "{{content}}"
    stream = io.BytesIO()
    doc.save(stream)
    raw = stream.getvalue()
    output = templates.render_template(raw, templates.inspect_template(raw), payload(), {})
    result = Document(io.BytesIO(output))
    assert "第二段讨论制造" in result.tables[0].cell(0, 0).text
    assert result.sections[0].header.paragraphs[0].text == "题目：固态电池课程文章"


@pytest.mark.parametrize("marker", ["前缀 {{content}}", "{{content}} {{section:正文}}"])
def test_block_slots_cannot_silently_overwrite_static_content(marker):
    doc = Document()
    doc.add_paragraph(marker)
    output = io.BytesIO()
    doc.save(output)
    with pytest.raises(templates.TemplateError):
        templates.inspect_template(output.getvalue())


def test_block_bookmarks_and_section_geometry_survive_expansion():
    doc = Document()
    p = doc.add_paragraph("{{content}}")
    start, end = OxmlElement("w:bookmarkStart"), OxmlElement("w:bookmarkEnd")
    start.set(qn("w:id"), "42")
    start.set(qn("w:name"), "report_body")
    end.set(qn("w:id"), "42")
    p._p.insert(0, start)
    p._p.append(end)
    buf = io.BytesIO()
    doc.save(buf)
    raw = buf.getvalue()
    output = templates.render_template(raw, templates.inspect_template(raw), payload(), {})
    root = templates._xml(package(output)["word/document.xml"])
    assert len(root.xpath(".//w:bookmarkStart[@w:id='42']", namespaces=templates._NS)) == 1
    assert len(root.xpath(".//w:bookmarkEnd[@w:id='42']", namespaces=templates._NS)) == 1
    assert len(Document(io.BytesIO(output)).sections) == 1


def test_page_fields_mark_for_refresh_without_rewriting_footer():
    doc = Document(io.BytesIO(make_template()))
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    doc.sections[0].footer.paragraphs[0]._p.append(field)
    stream = io.BytesIO()
    doc.save(stream)
    raw = stream.getvalue()
    result = package(templates.render_template(raw, templates.inspect_template(raw), payload(), {}))
    assert result["word/footer1.xml"] == package(raw)["word/footer1.xml"]
    root = templates._xml(result["word/settings.xml"])
    assert root.find(qn("w:updateFields")).get(qn("w:val")) == "true"


def test_harmless_printer_settings_binary_is_preserved():
    parts = package(make_template())
    parts["word/printerSettings/printerSettings1.bin"] = b"printer preferences"
    raw = repack(parts)
    result = templates.render_template(raw, templates.inspect_template(raw), payload(), {})
    assert package(result)["word/printerSettings/printerSettings1.bin"] == b"printer preferences"


@pytest.mark.parametrize("extra", ["word/vbaProject.bin", "word/embeddings/object.bin", "../escape.xml"])
def test_active_or_traversing_package_is_rejected(extra):
    parts = package(make_template())
    parts[extra] = b"untrusted"
    with pytest.raises(templates.TemplateError):
        templates.inspect_template(repack(parts))


def test_external_template_relationship_is_rejected():
    parts = package(make_template())
    parts["word/_rels/document.xml.rels"] = b'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="evil" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/attachedTemplate" Target="https://external.example/template.dotm" TargetMode="External"/></Relationships>'''
    with pytest.raises(templates.TemplateError, match="外部"):
        templates.inspect_template(repack(parts))


@pytest.mark.parametrize("instruction", ["DDEAUTO cmd /c whoami", 'INCLUDETEXT "C:/private.txt"', 'INCLUDEPICTURE "https://external.example/a"'])
def test_active_word_fields_cannot_run_during_pdf_export(instruction):
    doc = Document()
    p = doc.add_paragraph("{{content}}")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), instruction)
    p._p.append(field)
    stream = io.BytesIO()
    doc.save(stream)
    with pytest.raises(templates.TemplateError, match="活动域"):
        templates.inspect_template(stream.getvalue())


def test_macro_content_type_cannot_be_disguised_with_docx_extension():
    parts = package(make_template())
    parts["[Content_Types].xml"] = parts["[Content_Types].xml"].replace(b"wordprocessingml.document.main+xml", b"ms-word.document.macroEnabled.main+xml")
    with pytest.raises(templates.TemplateError, match="不含宏"):
        templates.inspect_template(repack(parts))


def test_store_validates_type_identity_and_independent_export(tmp_path):
    with pytest.raises(templates.TemplateError):
        templates.add_template("old.doc", make_template())
    with pytest.raises(templates.TemplateError):
        templates.add_template("broken.docx", b"Not a zip")
    info = templates.add_template("自己的模板.docx", make_template())
    snapshot = templates.get_template(info["id"])
    assert templates.list_templates()[0]["id"] == info["id"]
    with pytest.raises(templates.TemplateError):
        templates.get_template("../outside")
    with pytest.raises(templates.TemplateError):
        templates.selection(info["id"], {"wrong": "value"})
    templates.delete_template(info["id"])
    assert templates.list_templates() == []
    assert templates.render_template(snapshot[1], snapshot[0], payload(), {"作者": ""})


def test_pdf_failure_keeps_template_docx_and_clears_partial_pdf(monkeypatch):
    from core_engine.app import word_pdf
    def failed(source, destination):
        destination.write_bytes(b"partial")
        raise RuntimeError("No converter")
    monkeypatch.setattr(word_pdf, "convert_docx_pdf", failed)
    info = templates.add_template("t.docx", make_template())
    result = templates.export_document("job-id", templates.get_template(info["id"]), payload(), {})
    assert result["has_docx"] and not result["has_pdf"] and result["warnings"]
    assert templates.artifact_path("job-id", "docx").is_file()
    assert not templates.artifact_path("job-id", "pdf").exists()


def test_report_template_never_publishes_withheld_sentences_or_unchecked_paragraphs():
    from core_engine.report.models import ReportData, ReportSection, ReportSentence
    report = ReportData("主题", "报告", "摘要", sections=[ReportSection("行业概述", section_id="industry_overview",
        paragraphs=["未经审阅的草稿"], sentences=[ReportSentence("已审核表述。", status="supported", citation_ids=["e1"]),
                                                   ReportSentence("应当剔除的表述。", status="withheld")])])
    result = templates.report_payload(report, {"sections": ["研究方法", "行业概况"]})
    text = str(result)
    assert "未经审阅的草稿" not in text and "应当剔除的表述" not in text
    assert "[e1]" in text and "未覆盖本章节" in text
    assert result["sections"]["行业概况"][0].startswith("已审核表述")


@pytest.mark.parametrize("mode", ["slots", "outline", "named", "styles"])
def test_report_citation_links_and_references_survive_every_template_mode(mode):
    from core_engine.report.models import (
        ReportData,
        ReportSection,
        ReportSentence,
        SentenceCitation,
    )
    report = ReportData("格式测试", "报告", "摘要。", sections=[ReportSection("行业概述",
        sentences=[ReportSentence("已审核的格式测试段落。", status="supported", citation_ids=["e1"])])],
        citations=[SentenceCitation("e1", "c1", "https://example.org/source", "格式测试引文", 0, 6, 0, None, "fixture", "verbatim")])
    raw = make_template(mode)
    info = templates.inspect_template(raw)
    result = templates.render_template(raw, info, templates.report_payload(report, info), {})
    root = templates._xml(package(result)["word/document.xml"])
    anchors = root.xpath("//w:hyperlink/@w:anchor", namespaces=templates._NS)
    targets = set(root.xpath("//w:bookmarkStart/@w:name", namespaces=templates._NS))
    assert anchors and set(anchors) <= targets
    assert "格式测试引文" in "".join(root.xpath("//w:t/text()", namespaces=templates._NS))


def test_unused_evidence_is_not_inserted_as_references_and_metrics_keep_their_citations():
    from core_engine.report.models import ReportData, SentenceCitation
    citations = [SentenceCitation(key, "c1", "https://example.org/source", quote, 0, 3, 0, None, "fixture", "verbatim")
                 for key, quote in (("used", "指标原文"), ("unused", "不要输出这段无关证据"))]
    report = ReportData("测试", "报告", "摘要", metrics=[{"name": "market_size", "value": "100", "unit": "万元", "evidence_ids": ["used"]}], citations=citations)
    output = templates.report_payload(report, {"sections": []})
    assert "[used]" in "".join(output["sections"]["指标与计算"])
    assert "unused" not in str(output) and "不要输出" not in str(output)


@pytest.fixture
def client(monkeypatch):
    from core_engine.app.server import create_app
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        yield client


def test_template_upload_list_delete_and_example(client):
    response = client.post("/api/templates", files={"file": ("template.docx", make_template())})
    assert response.status_code == 200, response.text
    info = response.json()
    assert info["mode"] == "slots" and info["fields"] == ["作者"]
    assert client.get("/api/templates").json()["items"][0]["id"] == info["id"]
    assert client.get("/api/templates/example.docx").content[:2] == b"PK"
    assert client.delete(f"/api/templates/{info['id']}").status_code == 200
    assert client.get("/api/templates").json()["items"] == []


def test_oversized_template_is_rejected_before_spooling(client):
    response = client.post("/api/templates", content=b"", headers={"content-length": str(12 * 1024 * 1024)})
    assert response.status_code == 413


def test_export_files_download_after_restart_without_in_memory_job(client):
    path = templates.artifact_path("stored-job", "docx")
    path.write_bytes(make_template())
    assert client.get("/api/documents/stored-job/docx").content == path.read_bytes()
    assert client.get("/api/documents/stored-job/tex").status_code == 404


@pytest.fixture
def generation_client(monkeypatch):
    import json

    from core_engine.app import server, word_pdf
    from core_engine.config import get_settings
    from core_engine.report import llm, pipeline
    from core_engine.report.models import (
        PipelineResult,
        PipelineStatus,
        ReportData,
        ReportSection,
        ReportSentence,
    )

    class Writer:
        async def complete(self, system, user):
            if "Composition planner" in system:
                return '{"title":"固态电池文章","task":"完整成文","outline":[]}'
            if "conflict reviewer" in system:
                return '{"conflicts":[]}'
            if "section writer" in system:
                return "固态电池的课程讨论需要明确材料与结构边界，并在具体条件下分析实际应用。" * 5
            if "consistency reviewer" in system:
                return '{"edits":[],"notes":[]}'
            return json.dumps({"intent": "report_generation", "confidence": "high", "reasoning": "test"})
    runs = []
    class Pipeline:
        async def run(self, topic, **kwargs):
            runs.append(kwargs)
            report = ReportData(topic, "固态电池报告", "报告摘要。", sections=[ReportSection("行业概述",
                section_id="industry_overview", sentences=[ReportSentence("已审核的报告内容。", status="supported", citation_ids=["e1"])])])
            return PipelineResult(PipelineStatus.COMPLETED, topic, report=report)
    def convert(source, destination):
        destination.write_bytes(b"%PDF-1.7\nTesting only")
    monkeypatch.setenv("CE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    get_settings.cache_clear()
    monkeypatch.setattr(llm, "get_llm", lambda: Writer())
    monkeypatch.setattr(pipeline, "ReportPipeline", Pipeline)
    monkeypatch.setattr(word_pdf, "convert_docx_pdf", convert)
    try:
        with TestClient(server.create_app(), base_url="http://127.0.0.1") as client:
            yield client, runs
    finally:
        get_settings.cache_clear()


def test_article_generation_obeys_template_sections_and_restores_formatted_files(generation_client):
    from core_engine.app import server
    client, _ = generation_client
    info = client.post("/api/templates", files={"file": ("课程模板.docx", make_template("named"))}).json()
    response = client.post("/api/message", json={"message": "写一篇完整课程文章", "mode": "writing",
        "template_id": info["id"], "template_fields": {"作者": "王同学"}, "writing_options": {"length": 800, "use_web": False}})
    assert response.status_code == 200, response.text
    job = response.json()["job_id"]
    assert '"status": "completed"' in client.get(f"/api/report/{job}/events").text
    article = client.get(f"/api/writing/{job}").json()
    assert list(article["template_sections"]) == ["背景说明", "主要讨论", "结论与建议"]
    assert "## 引言" not in article["content"]
    assert article["template"]["has_docx"] and article["template"]["has_pdf"]
    assert article["factual_verification"] == "not_performed"
    server._JOBS.pop(job)
    from urllib.parse import unquote
    download = client.get(f"/api/documents/{job}/docx")
    assert "固态电池文章.docx" in unquote(download.headers["content-disposition"])
    doc = Document(io.BytesIO(download.content))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "王同学" in text and "{{" not in text and "固态电池" in text
    assert client.get(f"/api/documents/{job}/pdf").content.startswith(b"%PDF-")
    assert client.get("/api/history").json()["items"][0]["docx_ready"]


@pytest.mark.parametrize("endpoint", ["/api/message", "/api/report", "/api/database/session"])
def test_all_report_entrypoints_export_template_word_and_matching_pdf(generation_client, endpoint):
    client, runs = generation_client
    info = client.post("/api/templates", files={"file": ("报告模板.docx", make_template())}).json()
    response = client.post(endpoint, json={"topic": "固态电池", "message": "固态电池报告", "mode": "report",
        "template_id": info["id"], "template_fields": {"作者": "报告作者"}})
    assert response.status_code == 200, response.text
    data = response.json()
    job = data["job_id"]
    assert '"has_docx": true' in client.get(f"/api/report/{job}/events").text
    assert runs[-1]["compile_to_pdf"] is False
    doc = Document(io.BytesIO(client.get(f"/api/documents/{job}/docx").content))
    assert "已审核的报告内容" in "\n".join(p.text for p in doc.paragraphs)
    assert client.get(f"/api/report/{job}/pdf").content.startswith(b"%PDF-")


def test_comprehensive_report_inherits_template_selection(generation_client):
    from core_engine.app import database
    client, runs = generation_client
    info = templates.add_template("报告模板.docx", make_template())
    started = client.post("/api/message", json={"message": "报告", "mode": "report", "template_id": info["id"]}).json()
    client.get(f"/api/report/{started['job_id']}/events")
    sid = started["session_id"]
    database.add_article(sid, title="材料", filename="材料.txt", text="用户提供的研究材料。")
    response = client.post(f"/api/database/{sid}/comprehensive")
    assert response.status_code == 200, response.text
    job = response.json()["job_id"]
    assert '"has_docx": true' in client.get(f"/api/report/{job}/events").text
    assert runs[-1]["skip_search"] is True and runs[-1]["compile_to_pdf"] is False


def test_invalid_selected_template_fails_before_starting_generation(generation_client):
    client, runs = generation_client
    response = client.post("/api/message", json={"message": "报告", "mode": "report", "template_id": "missing"})
    assert response.status_code == 400 and runs == []


def test_word_report_preserves_review_status_and_question_coverage():
    from core_engine.report.models import ReportData
    report = ReportData("固态电池", "技术报告", "报告摘要。", quality_status="needs_review",
        research_audit={"questions": [{"question": "界面机制是什么？", "status": "partial", "missing": ["缺少同条件对照。", "c20仅有定性说明。"]},
                                       {"question": "如何定义全固态？", "status": "supported", "missing": []}]})
    payload = templates.report_payload(report, {"sections": []})
    assert "待复核" in payload["sections"]["摘要"][0]
    text = "\n".join(payload["sections"]["研究问题覆盖与证据缺口"])
    assert "1/2" in text and "缺少同条件对照" in text and "部分覆盖" in text
    assert "c20" not in text and "。；" not in text


def test_word_article_keeps_writing_notice_even_without_background_sources():
    article = {"title": "课程文章", "created_at": "2026-11-04", "content": "## 引言\n\n文章开头。",
               "notice": "材料仅作写作前提，未逐条核验。", "sources": []}
    payload = templates.article_payload(article)
    assert payload["sections"]["写作说明"] == [article["notice"]]
    assert payload["abstract"] == "文章开头。"


def test_word_analysis_keeps_assumptions_alternatives_and_uncertainty():
    from core_engine.report.models import ReportData
    report = ReportData("固态电池", "研究报告", "摘要", analysis=[{
        "conclusion": "若界面接触改善，则可能降低阻抗。", "assumptions": ["其他测试条件保持一致。", "电芯结构相同。"],
        "alternatives": ["副反应也可能影响阻抗"], "uncertainty": "缺少同条件对照", "citation_ids": []}])
    paragraphs = templates.report_payload(report, {"sections": []})["sections"]["分析与限制"]
    text = "\n".join(paragraphs)
    assert "分析推断" in text and "人工复核" in text
    assert "其他测试条件保持一致" in text and "副反应" in text and "缺少同条件对照" in text
    assert "。；" not in text


def test_article_technical_review_references_survive_export_without_duplicates():
    review_source = {"title": "复核依据", "url": "https://example.edu/principles"}
    article = {"title": "文章", "created_at": "2026-11-04", "content": "## 引言\n\n正文。",
        "sources": [{"title": "背景", "url": "https://example.edu/background"}],
        "manual_review": {"references": [review_source, review_source]}}
    references = templates.article_payload(article)["references"]
    assert len(references) == 2
    assert references[1].endswith("（内容复核资料）")


def test_analysis_research_citation_ids_get_real_reference_targets():
    from dataclasses import asdict

    from lxml import etree as ET

    from core_engine.report.models import ReportData, SentenceCitation, Source, SourceKind
    source = Source("https://example.edu/paper", "example.edu", "界面研究", SourceKind.ACADEMIC, "界面接触会影响阻抗。")
    citation = SentenceCitation("r1", "c1", source.url, source.text, 0, len(source.text), 1, None, source.content_sha256, "verbatim")
    report = ReportData("固态电池", "报告", "摘要", sources=[source],
        research_evidence={"citations": [asdict(citation)]}, analysis=[{"conclusion": "接触改善可能有利于界面。", "citation_ids": ["r1"]}])
    raw = make_template()
    info = templates.inspect_template(raw)
    payload = templates.report_payload(report, info)
    assert any(reference.startswith("[r1]") for reference in payload["references"])
    output_parts = package(templates.render_template(raw, info, payload, {}))
    doc = ET.fromstring(output_parts["word/document.xml"])
    targets = doc.xpath("//w:hyperlink/@w:anchor", namespaces=templates._NS)
    assert targets and set(targets) <= set(doc.xpath("//w:bookmarkStart/@w:name", namespaces=templates._NS))
    report.research_evidence["citations"][0]["quote"] = "并不存在的原文"
    with pytest.raises(templates.TemplateError, match="分析引用"):
        templates.report_payload(report, info)
