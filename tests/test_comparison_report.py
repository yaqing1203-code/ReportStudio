"""Tests for workflow E — the structured synthesis layer.

Covers: the comparison matrix (clusters -> booktabs), the two synthesis-layer
sections (source_comparison / limitations), the hedging rules for isolated
(孤证) claims, and the zh report locale. All offline: FakeLLM + FakeSearchProvider
+ FakeFetcher fixtures, no network, no API key, no TeX toolchain.

Run: pytest -q tests/test_comparison_report.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core_engine.report.models import PipelineStatus, SearchHit
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------
# E1 — comparison_matrix: clusters -> booktabs matrix
# --------------------------------------------------------------------------
def _clusters() -> list[dict]:
    """Two alignment clusters: a complement pair (same figure, different 口径) and
    a conflict pair. One cell rests on an L4 (weak-signal) source."""
    return [
        {"entity": "AlphaCo", "attribute": "revenue",
         "cells": [
             {"source_url": "https://www.iea.org/reports/a", "value": "10 USD bn",
              "qualifier": "domestic only", "credibility": 2,
              "claim_text": "AlphaCo revenue was 10 USD bn (domestic only)."},
             {"source_url": "https://blog.example.com/x", "value": "12 USD bn",
              "qualifier": "global", "credibility": 4,
              "claim_text": "AlphaCo revenue was 12 USD bn (global)."},
         ], "relation": "complement"},
        {"entity": "BetaCo", "attribute": "market_share",
         "cells": [
             {"source_url": "https://iea.org/reports/b", "value": "30%",
              "qualifier": None, "credibility": 1,
              "claim_text": "BetaCo market share was 30%."},
             {"source_url": "https://oecd.org/c", "value": "28%",
              "qualifier": None, "credibility": 2,
              "claim_text": "BetaCo market share was 28%."},
         ], "relation": "conflict"},
    ]


def test_comparison_matrix_booktabs_structure():
    from core_engine.report.charts import comparison_matrix

    tex = comparison_matrix(_clusters())
    # booktabs skeleton, content-only (no float environment)
    assert tex.startswith("\\begin{tabular}")
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert "\\begin{table}" not in tex and "\\begin{figure}" not in tex
    # rows: one per (entity, attribute); untrusted text is LaTeX-escaped
    assert "AlphaCo" in tex and "revenue" in tex
    assert "BetaCo" in tex and r"market\_share" in tex
    # columns: domain + qualifier (口径), domain extracted from the source URL
    assert "iea.org (domestic only)" in tex
    assert "blog.example.com (global)" in tex
    assert "oecd.org" in tex
    # values land in cells, escaped
    assert "10 USD bn" in tex and r"30\%" in tex and r"28\%" in tex
    # the L4 cell carries the dagger marker, followed by the footnote line
    assert r"12 USD bn\(^\dagger\)" in tex
    assert r"\footnotesize \(\dagger\) single low-credibility source, unverified." in tex


def test_comparison_matrix_empty_or_valueless_returns_empty():
    from core_engine.report.charts import comparison_matrix

    assert comparison_matrix([]) == ""
    # clusters exist but no cell carries a value -> still ""
    assert comparison_matrix([{
        "entity": "AlphaCo", "attribute": "revenue",
        "cells": [{"source_url": "https://iea.org/a", "value": None,
                   "qualifier": None, "credibility": 2, "claim_text": "x"}],
        "relation": "complement"}]) == ""


def test_build_all_aggregates_comparison_matrix():
    from core_engine.report.charts import build_all
    from core_engine.report.kg import ReportKnowledgeGraph

    kg = ReportKnowledgeGraph(topic="t")
    # signature back-compat: clusters optional, key always present
    assert build_all(kg)["comparison_matrix"] == ""
    with_clusters = build_all(kg, _clusters())
    assert "\\begin{tabular}" in with_clusters["comparison_matrix"]


# --------------------------------------------------------------------------
# End-to-end fixtures: 2 high-trust sources stating the same-metric claims
# (a complement cluster), optionally +1 general-web source (an isolated claim).
# --------------------------------------------------------------------------
def _pipeline_fixture(*, with_isolated: bool):
    from core_engine.report.llm import FakeLLM

    llm = FakeLLM()
    llm.set_structured_claims("https://iea.org/alphaco", [
        {"text": "AlphaCo revenue was 10 USD bn (domestic only).",
         "entity": "AlphaCo", "attribute": "revenue", "value": "10 USD bn",
         "qualifier": "domestic only", "time_scope": "FY2023"},
        {"text": "The AlphaCo market grew 15 percent in 2023."},
    ])
    llm.set_structured_claims("https://oecd.org/alphaco", [
        {"text": "AlphaCo revenue was 10 USD bn (global).",
         "entity": "AlphaCo", "attribute": "revenue", "value": "10 USD bn",
         "qualifier": "global", "time_scope": "FY2023"},
        {"text": "The AlphaCo market grew 15 percent in 2023."},
    ])
    pages = {
        "https://iea.org/alphaco": "IEA report on AlphaCo with revenue details. " * 8,
        "https://oecd.org/alphaco": "OECD report on AlphaCo with revenue details. " * 8,
    }
    if with_isolated:
        llm.set_structured_claims("https://blog.example.com/alphaco", [
            {"text": "AlphaCo plans a secret factory in 2026.",
             "entity": "AlphaCo", "attribute": "capacity", "value": "2 GW"},
        ])
        pages["https://blog.example.com/alphaco"] = ("Blog post about AlphaCo "
                                                     "plans and rumors. " * 8)
    search = FakeSearchProvider()
    search.add("alphaco", [SearchHit(u, "Report", "", u.split("/")[2]) for u in pages])
    return llm, search, FakeFetcher(pages)


@pytest.fixture
def pipeline_env(monkeypatch):
    """Offline-deterministic pipeline settings for the synthesis-layer tests."""
    from core_engine.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("CE_RELEVANCE_FILTER", "false")
    monkeypatch.setenv("CE_MULTI_QUERY_RESEARCH", "false")
    yield monkeypatch
    get_settings.cache_clear()


# --------------------------------------------------------------------------
# E2 — the two synthesis-layer sections, end to end
# --------------------------------------------------------------------------
async def test_pipeline_emits_source_comparison_and_matrix(pipeline_env):
    llm, search, fetcher = _pipeline_fixture(with_isolated=False)
    pipe = ReportPipeline(llm=llm, search=search, fetcher=fetcher)
    result = await pipe.run("AlphaCo industry", compile_to_pdf=False)
    assert result.status is PipelineStatus.COMPLETED, result.message

    ids = [s.section_id for s in result.report.sections]
    # source_comparison sits right after competitive_landscape; limitations closes.
    assert ids.index("source_comparison") == ids.index("competitive_landscape") + 1
    assert ids[-1] == "limitations"

    tex = Path(result.tex_path).read_text(encoding="utf-8")
    assert "Source Comparison" in tex            # the drafted section heading
    assert "\\begin{tabular}" in tex             # the comparison matrix block
    assert "10 USD bn" in tex                    # cluster values rendered
    # No isolated claims in this fixture -> full-corroboration statement.
    assert "All key claims were corroborated by multiple independent sources." in tex


async def test_pipeline_limitations_lists_isolated_claims(pipeline_env):
    llm, search, fetcher = _pipeline_fixture(with_isolated=True)
    pipe = ReportPipeline(llm=llm, search=search, fetcher=fetcher)
    result = await pipe.run("AlphaCo industry", compile_to_pdf=False)
    assert result.status is PipelineStatus.COMPLETED, result.message

    tex = Path(result.tex_path).read_text(encoding="utf-8")
    assert "Information Limitations" in tex
    # the isolated claim is listed with its source domain and credibility label
    assert "secret factory" in tex
    assert "blog.example.com" in tex
    assert "L3 Industry Consensus" in tex
    # ... and the full-corroboration sentence is NOT emitted in that case
    assert ("All key claims were corroborated by multiple independent sources."
            not in tex)


# --------------------------------------------------------------------------
# E3 — hedging rules: isolated claims reach drafting with the UNVERIFIED marker
# --------------------------------------------------------------------------
async def test_drafting_marks_isolated_claims_unverified(pipeline_env):
    from core_engine.report.llm import FakeLLM

    llm, search, fetcher = _pipeline_fixture(with_isolated=True)

    class RecordingLLM(type(llm)):
        def __init__(self, inner):
            self._inner = inner
            self.drafted: list[tuple[str, list[str]]] = []

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def draft_section(self, heading, claims, instructions=""):
            self.drafted.append((heading, claims))
            return await self._inner.draft_section(heading, claims, instructions)

    rec = RecordingLLM(llm)
    pipe = ReportPipeline(llm=rec, search=search, fetcher=fetcher)
    result = await pipe.run("AlphaCo industry", compile_to_pdf=False)
    assert result.status is PipelineStatus.COMPLETED, result.message

    marked = [line for _heading, claims in rec.drafted for line in claims
              if "[UNVERIFIED - single source]" in line]
    assert any("secret factory" in m for m in marked), (
        "isolated claims must reach draft_section with the UNVERIFIED marker")

    # FakeLLM preserves the marker verbatim in its deterministic output.
    out = await FakeLLM().draft_section(
        "H", ["[UNVERIFIED - single source] AlphaCo plans a secret factory in 2026."])
    assert "[UNVERIFIED - single source]" in out


# --------------------------------------------------------------------------
# E4 — section localization + zh LaTeX preamble
# --------------------------------------------------------------------------
def test_resolved_sections_localizes():
    from core_engine.config import Settings

    zh = dict(Settings(report_locale="zh").resolved_sections())
    assert zh["industry_overview"] == "行业概览"
    assert zh["policy_analysis"] == "政策分析"
    assert zh["industry_chain"] == "产业链图谱"
    assert zh["market_size"] == "市场规模"
    assert zh["competitive_landscape"] == "竞争格局"
    assert zh["source_comparison"] == "来源对比与分歧分析"
    assert zh["limitations"] == "信息局限性声明"

    en_ids = [sid for sid, _ in Settings(report_locale="en").resolved_sections()]
    assert en_ids[-2:] == ["source_comparison", "limitations"]
    assert dict(Settings(report_locale="en").resolved_sections())[
        "industry_overview"] == "Industry Overview"

    off = Settings(enable_synthesis_sections=False).resolved_sections()
    assert [sid for sid, _ in off] == [
        sid for sid, _ in Settings().report_sections]


def test_out_of_scope_message_localized():
    from core_engine.report.models import (
        OUT_OF_SCOPE_MESSAGE,
        OUT_OF_SCOPE_MESSAGE_ZH,
        out_of_scope_message,
    )
    assert out_of_scope_message("en") == OUT_OF_SCOPE_MESSAGE
    assert out_of_scope_message("zh") == OUT_OF_SCOPE_MESSAGE_ZH


async def test_zh_locale_renders_chinese_headings_and_ctex(pipeline_env):
    pipeline_env.setenv("CE_REPORT_LOCALE", "zh")
    llm, search, fetcher = _pipeline_fixture(with_isolated=False)
    pipe = ReportPipeline(llm=llm, search=search, fetcher=fetcher)
    result = await pipe.run("AlphaCo industry", compile_to_pdf=False)
    assert result.status is PipelineStatus.COMPLETED, result.message

    tex = Path(result.tex_path).read_text(encoding="utf-8")
    assert "\\usepackage{ctex}" in tex
    assert "行业概览" in tex
    assert "来源对比与分歧分析" in tex
    assert "信息局限性声明" in tex
    assert "行业简报" in tex                  # localized report title
