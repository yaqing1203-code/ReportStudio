"""Tests for Database Mode: document parsing, the session store, and the pipeline's
injected-source / skip-search path.

All offline — fake LLM/search, no network, no TeX. Run:
    pytest -q tests/test_database_mode.py
"""
from __future__ import annotations

import csv
import io

import pytest

from core_engine.report.documents import (
    SUPPORTED_EXTENSIONS,
    DocumentError,
    parse_document,
)
from core_engine.report.models import PipelineStatus, Source, SourceKind
from core_engine.report.pipeline import (
    ReportPipeline,
    _distinct_authoritative_domains,
    _merge_sources,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------
# Document parsing.
# --------------------------------------------------------------------------
def test_parse_txt():
    title, text = parse_document("notes.txt", b"Solar demand rose 30 percent in 2024.")
    assert title == "notes"
    assert "Solar demand rose" in text


def test_parse_markdown():
    raw = b"# Heading\n\nSolid-state batteries reached mass production.\n"
    _title, text = parse_document("brief.md", raw)
    assert "Solid-state batteries" in text


def test_parse_csv():
    buf = io.StringIO()
    csv.writer(buf).writerows([["Year", "Share"], ["2024", "12%"]])
    _title, text = parse_document("market.csv", buf.getvalue().encode("utf-8"))
    assert "Year | Share" in text
    assert "2024 | 12%" in text


def test_parse_docx_roundtrip():
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Lithium supply is concentrated in three countries.")
    bio = io.BytesIO()
    doc.save(bio)
    _title, text = parse_document("supply.docx", bio.getvalue())
    assert "Lithium supply is concentrated" in text


def test_parse_xlsx_roundtrip():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Metric", "Value"])
    ws.append(["TAM", "50 billion"])
    bio = io.BytesIO()
    wb.save(bio)
    _title, text = parse_document("sizing.xlsx", bio.getvalue())
    assert "Metric | Value" in text
    assert "TAM | 50 billion" in text


def test_empty_file_rejected():
    with pytest.raises(DocumentError):
        parse_document("empty.txt", b"")


def test_unsupported_extension_rejected():
    with pytest.raises(DocumentError) as ei:
        parse_document("archive.zip", b"PK\x03\x04...")
    assert "Unsupported" in str(ei.value)


def test_legacy_doc_rejected_with_guidance():
    with pytest.raises(DocumentError) as ei:
        parse_document("old.doc", b"\xd0\xcf\x11\xe0garbage")
    assert ".docx" in str(ei.value)


def test_supported_extensions_cover_requested_formats():
    for ext in (".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"):
        assert ext in SUPPORTED_EXTENSIONS


# --------------------------------------------------------------------------
# Source kind + pipeline helpers.
# --------------------------------------------------------------------------
def test_user_provided_is_high_trust_and_authoritative():
    src = Source("userdoc://s/a1", "user-document-a1", "My doc",
                 SourceKind.USER_PROVIDED, text="x")
    assert SourceKind.USER_PROVIDED.high_trust is True
    assert src.authoritative is True


def test_merge_sources_dedupes_by_url():
    a = Source("http://iea.org/x", "iea.org", "A", SourceKind.INDUSTRY_INSTITUTION, text="1")
    b = Source("userdoc://s/1", "user-document-1", "B", SourceKind.USER_PROVIDED, text="2")
    dup = Source("http://iea.org/x", "iea.org", "A2", SourceKind.INDUSTRY_INSTITUTION, text="3")
    merged = _merge_sources([a], [b, dup])
    urls = [s.url for s in merged]
    assert urls == ["http://iea.org/x", "userdoc://s/1"]  # dup dropped, base wins


def test_distinct_domains_counts_user_docs():
    # Synthetic user-doc URLs would be REJECTED by the allowlist classifier, but the
    # pipeline counts by the Source.kind — so they must still count here.
    srcs = [
        Source("userdoc://s/a1", "user-document-a1", "A", SourceKind.USER_PROVIDED, text="x"),
        Source("userdoc://s/a2", "user-document-a2", "B", SourceKind.USER_PROVIDED, text="y"),
    ]
    domains = _distinct_authoritative_domains(srcs)
    assert domains == {"user-document-a1", "user-document-a2"}


# --------------------------------------------------------------------------
# Pipeline: skip_search + extra_sources builds a report from injected docs alone.
# --------------------------------------------------------------------------
async def test_comprehensive_from_injected_sources_only(monkeypatch):
    """With skip_search=True the pipeline never touches the web; it must build a
    report purely from the injected user documents (fake LLM extracts claims)."""
    from core_engine.config import get_settings

    # Lower the gates so a couple of injected docs clear the scope check deterministically.
    monkeypatch.setenv("CE_MIN_AUTHORITATIVE_SOURCES", "1")
    monkeypatch.setenv("CE_MIN_TOTAL_CLAIMS", "1")
    monkeypatch.setenv("CE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    get_settings.cache_clear()

    docs = [
        Source("userdoc://s/a1", "user-document-a1", "User Doc 1",
               SourceKind.USER_PROVIDED,
               text="The global market grew to 50 billion dollars in 2024. "
                    "Adoption increased across three regions."),
        Source("userdoc://s/a2", "user-document-a2", "User Doc 2",
               SourceKind.USER_PROVIDED,
               text="Supply chains remain concentrated. Policy support expanded in 2023."),
    ]

    pipeline = ReportPipeline()
    result = await pipeline.run(
        "solar cell", compile_to_pdf=False,
        extra_sources=docs, skip_search=True,
    )
    get_settings.cache_clear()

    # It should NOT be out-of-scope: the injected docs are the source pool.
    assert result.status in (PipelineStatus.COMPLETED,), result.message
    assert result.report is not None
    # The user documents must appear as sources in the assembled report.
    kinds = {s.kind for s in result.report.sources}
    assert SourceKind.USER_PROVIDED in kinds


async def test_skip_search_with_no_sources_is_out_of_scope(monkeypatch):
    from core_engine.config import get_settings

    monkeypatch.setenv("CE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    get_settings.cache_clear()
    pipeline = ReportPipeline()
    result = await pipeline.run(
        "anything", compile_to_pdf=False, extra_sources=[], skip_search=True)
    get_settings.cache_clear()
    assert result.status is PipelineStatus.OUT_OF_SCOPE
