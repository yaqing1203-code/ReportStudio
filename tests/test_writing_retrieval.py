"""Regressions from the real solid-state battery course-article run."""
from __future__ import annotations

import asyncio
import json

import pytest

from core_engine.app import writing
from core_engine.report.models import CredibilityLevel, Source, SourceKind


@pytest.fixture
def anyio_backend():
    return "asyncio"


def source(url="https://example.edu/battery", text="固态电池的界面与制造问题需要研究。", **kwargs):
    return Source(url, "example.edu", "Solid-state battery research", SourceKind.ACADEMIC,
                  text, credibility=CredibilityLevel.L1, **kwargs)


def test_writing_instructions_cannot_become_the_subject_of_relevance_matching():
    request = "请以固态电池为主题，联网写一篇约1500字的完整课程文章，语言通顺。"
    assert writing._fallback_topic(request) == "固态电池"
    unrelated = Source("https://api.openalex.org/works/old", "api.openalex.org", "中国乡村建设",
                       SourceKind.ACADEMIC, "本文梳理理论实践的完整结构，并讨论不同观点与论述。")
    assert not writing._subject_match("固态电池", unrelated)
    assert writing._subject_match("固态电池", source(text="Solid-state batteries face interfacial challenges."))


def test_late_methods_and_limitations_are_read_with_exact_excerpt_offsets():
    body = "目录与前言。\n" * 800 + "\n固态电池的硫化物路线具有界面阻抗、压力和制造限制。\n" * 30
    material = writing._material(source(text=body), {"topic": "固态电池", "focuses": ["硫化物 界面 制造 压力"]})
    assert "硫化物路线" in material["text"]
    assert len(material["text"]) < len(body)
    for excerpt in material["excerpts"]:
        assert body[excerpt["start"]:excerpt["end"]] == excerpt["text"]


@pytest.mark.anyio
async def test_slow_and_failed_angles_cannot_discard_fast_relevant_sources(monkeypatch):
    from core_engine.report import scrape

    cancelled = asyncio.Event()
    async def gather(topic, provider, fetcher, *, queries, **kwargs):
        assert topic == "固态电池"
        if queries == ["slow"]:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        if queries == ["failed"]:
            raise TimeoutError("One unavailable angle")
        return [source()], []
    monkeypatch.setattr(scrape, "gather_sources", gather)
    monkeypatch.setattr(scrape, "get_fetcher", lambda: object())
    monkeypatch.setattr(writing, "_RETRIEVAL_TIMEOUT_S", .01)
    result = await writing.collect_materials("固态电池", lambda *_: None,
        research_plan={"topic": "固态电池", "queries": ["fast", "slow", "failed"]})
    assert [s["url"] for s in result] == ["https://example.edu/battery"]
    assert cancelled.is_set()


@pytest.mark.anyio
async def test_relevance_priority_and_url_deduplication(monkeypatch):
    from core_engine.report import scrape

    irrelevant = Source("https://example.edu/rural", "example.edu", "乡村建设", SourceKind.ACADEMIC,
                        "对历史与当代乡村建设展开讨论。", credibility=CredibilityLevel.L1)
    promo = Source("https://vendor.example/battery", "vendor.example", "固态电池宣传", SourceKind.GENERAL_WEB,
                   "固态电池的应用前景。", credibility=CredibilityLevel.L4)
    async def gather(*_, **kwargs):
        return [irrelevant, promo, source(), source()], []
    monkeypatch.setattr(scrape, "gather_sources", gather)
    monkeypatch.setattr(scrape, "get_fetcher", lambda: object())
    result = await writing.collect_materials("固态电池", lambda *_: None)
    assert [s["url"] for s in result] == ["https://example.edu/battery", "https://vendor.example/battery"]


@pytest.mark.anyio
async def test_model_query_plan_separates_subject_from_course_writing_instructions():
    # A complete engine run with distinct search and writing decisions.
    class Model:
        async def complete(self, system, user):
            if "Composition retrieval planner" in system:
                return json.dumps({"topic": "固态电池", "queries": ["固态电池 全固态 半固态", "solid-state batteries interfaces"],
                                   "focuses": ["界面问题"]})
            if "Composition planner" in system:
                return '{"title":"固态电池","task":"写完整课程文章","outline":[],"conflicts":[]}'
            if "section writer" in system:
                return "固态电池研究涉及电解质与电极之间的协同设计。" * 8
            return '{"edits":[],"notes":[]}'
    captured = []
    async def search(topic, emit, *, research_plan):
        captured.append((topic, research_plan))
        return []
    result = await writing.compose("以固态电池为主题写一篇课程文章", writing.WritingOptions(length=800),
                                   Model(), search_fn=search)
    assert captured[0][0] == "固态电池"
    assert "site:.edu" in captured[0][1]["queries"][0]
    assert "固态电池 全固态 半固态" in captured[0][1]["queries"]
    assert result["retrieval_plan"] == captured[0][1]


def test_staff_bio_with_incidental_topic_mentions_is_not_technical_background():
    profile = Source("https://www.example.edu.cn/people/professor", "example.edu.cn", "车辆学院-张三",
        SourceKind.GENERAL_WEB, "电话：123。E-mail：name@example.edu.cn。研究员，博士。发表论文多篇。"
        "研究固态电池，揭示了全固态电池相关机制。培养研究生多名。")
    assert not writing._subject_match("固态电池", profile)
    # Scientific prose and an author's name alone are not a staff profile.
    assert writing._subject_match("固态电池", source(text="研究员团队讨论了固态电池的界面问题。"))


def test_passing_mention_in_another_battery_chemistry_is_not_main_topic_coverage():
    unrelated = Source("https://api.openalex.org/works/lithium-air", "api.openalex.org",
        "锂空气电池的X射线成像", SourceKind.ACADEMIC,
        "本文讨论锂空气电池的成像。随着固态电池兴起，也出现新的界面思路。最后讨论多孔氧正极。")
    assert not writing._subject_match("固态电池", unrelated)
    topical_body = Source("https://example.edu/interfaces", "example.edu", "Interface engineering",
        SourceKind.ACADEMIC, "Solid-state batteries need stable interfaces. Solid-state batteries also face contact loss.")
    assert writing._subject_match("固态电池", topical_body)
