"""Writing depth is an explicit, bounded effort choice, never an evidence policy."""
from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from core_engine.app import writing
from core_engine.config import get_settings
from core_engine.report.models import CredibilityLevel, Source, SourceKind


@pytest.fixture
def anyio_backend():
    return "asyncio"


class DepthWriter:
    def __init__(self):
        self.calls = []
        self.conflicts = []
        self.fail_depth = False

    async def complete(self, system, user):
        payload = json.loads(user)
        self.calls.append((system, payload))
        if "Composition retrieval planner" in system:
            return json.dumps({"topic": "学习方法", "queries": [f"学习方法角度{i}" for i in range(7)],
                               "focuses": ["比较", "限制"]}, ensure_ascii=False)
        if "Composition depth planner" in system:
            if self.fail_depth:
                raise TimeoutError("optional analysis planning unavailable")
            return json.dumps({"sections": [{"heading": row["heading"],
                "mechanism": "解释方法如何影响理解。", "comparison": "比较适用条件。",
                "counterargument": "考虑工具依赖问题。", "limits": "避免把一种方法当作普遍保证。"}
                for row in payload["sections"]]}, ensure_ascii=False)
        if "Composition planner" in system:
            return json.dumps({"title": "学习方法", "task": "讨论学习方法", "outline": [
                {"heading": heading, "purpose": "展开论述"} for heading in ("基本认识", "方法比较", "适用边界")]},
                ensure_ascii=False)
        if "Composition conflict reviewer" in system:
            return json.dumps({"conflicts": self.conflicts}, ensure_ascii=False)
        if "Composition section writer" in system:
            target = payload.get("section", {}).get("target", payload.get("target_length", 100))
            sentence = "学习需要结合具体问题安排方法，并在练习中检查理解与适用条件。"
            return sentence * math.ceil(target / writing.length_of(sentence))
        return '{"edits":[],"notes":[]}'


def options(depth="standard", **kwargs):
    return writing.WritingOptions(length=800, depth=depth, use_web=False, **kwargs)


def test_depth_defaults_and_validation_are_independent_of_mode_or_length():
    assert writing.WritingOptions().depth == "standard"
    for depth in ("standard", "deep"):
        assert writing.WritingOptions(depth=depth, length=1500).length == 1500
    for invalid in ("auto", "report", "writing", "fast", "", None, True):
        with pytest.raises(ValidationError):
            writing.WritingOptions(depth=invalid)


@pytest.mark.anyio
@pytest.mark.parametrize("target", [400, 800, 4000])
async def test_depth_changes_planning_effort_but_never_the_requested_length(target):
    calls = {}
    for depth in ("standard", "deep"):
        llm = DepthWriter()
        result = await writing.compose("写一篇学习方法课程文章",
            writing.WritingOptions(length=target, use_web=False, depth=depth), llm)
        calls[depth] = llm.calls
        assert result["target_length"] == target
        assert result["depth"] == depth
        assert result["notice"] == writing.NOTICE
        assert result["factual_verification"] == "not_performed"
        assert result["basis_policy"] == "assumed_for_writing"
        planner = next(p for s, p in llm.calls if "Composition planner" in s)
        assert planner["target_length"] == target and planner["depth"] == depth
    standard = [p["section"]["target"] for s, p in calls["standard"] if "Composition section writer" in s]
    deep = [p["section"]["target"] for s, p in calls["deep"] if "Composition section writer" in s]
    assert standard == deep
    assert not any("Composition depth planner" in s for s, _ in calls["standard"])
    assert sum("Composition depth planner" in s for s, _ in calls["deep"]) == 1
    deep_sections = [p for s, p in calls["deep"] if "Composition section writer" in s]
    assert any(p["analysis_focus"].get("counterargument") for p in deep_sections)
    assert all("mechanisms" in s and "SAME requested target length" in s
               for s, _ in calls["deep"] if "Composition section writer" in s)


@pytest.mark.anyio
async def test_deep_wording_in_request_does_not_override_standard_selection():
    llm = DepthWriter()
    result = await writing.compose("深入讨论学习方法，写成课程文章", options(), llm)
    assert result["depth"] == "standard"
    assert result["depth_planning_status"] == "not_requested"
    assert not any("Composition depth planner" in s for s, _ in llm.calls)


@pytest.mark.anyio
async def test_depth_bounds_query_source_and_prompt_context_budgets_without_mutating_settings():
    original = get_settings().model_dump()
    results = {}
    for depth in ("standard", "deep"):
        llm = DepthWriter()
        captured = []
        async def search(topic, emit, *, research_plan, _captured=captured):
            _captured.append(research_plan)
            return [{"title": f"学习方法{i}", "url": f"https://example.edu/{i}",
                     "text": "讨论学习方法的适用条件与练习过程。" * 300} for i in range(10)]
        result = await writing.compose("以学习方法为主题写课程文章",
            writing.WritingOptions(length=800, depth=depth, materials="课堂学习与问题讨论。" * 500),
            llm, search_fn=search)
        planner = next(p for s, p in llm.calls if "Composition planner" in s)
        conflict = next(p for s, p in llm.calls if "Composition conflict reviewer" in s)
        draft = next(p for s, p in llm.calls if "Composition section writer" in s)
        results[depth] = (result, captured[0], planner, conflict, draft)
        assert captured[0]["depth"] == depth
        assert all(len(q) <= 120 for q in captured[0]["queries"])
    standard, deep = results["standard"], results["deep"]
    assert len(standard[1]["queries"]) == 3 and len(deep[1]["queries"]) == 4
    assert len(standard[0]["sources"]) == 4 and len(deep[0]["sources"]) == 6
    assert len(standard[2]["materials"]) == 2000 and len(deep[2]["materials"]) == 3000
    assert all(len(row["text"]) <= 1000 for row in standard[3]["background"])
    assert all(len(row["text"]) <= 1400 for row in deep[3]["background"])
    assert len(deep[4]["materials"]) > len(standard[4]["materials"])
    assert get_settings().model_dump() == original


@pytest.mark.anyio
@pytest.mark.parametrize("depth,queries,pages,sources", [("standard", 3, 4, 4), ("deep", 4, 5, 6)])
async def test_retrieval_applies_real_depth_limits_to_fetching(monkeypatch, depth, queries, pages, sources):
    from core_engine.report import scrape

    original = get_settings().model_dump()
    observed = []
    async def gather(topic, provider, fetcher, *, queries, max_pages, settings, **kwargs):
        observed.append((queries, max_pages, settings))
        return [Source(f"https://example.edu/{queries[0]}/{i}", "example.edu", "学习方法讨论",
            SourceKind.ACADEMIC, "学习方法需要关注理解、练习与适用条件。", credibility=CredibilityLevel.L1)
            for i in range(max_pages)], []
    monkeypatch.setattr(scrape, "gather_sources", gather)
    monkeypatch.setattr(scrape, "get_fetcher", lambda: object())
    result = await writing.collect_materials("学习方法", lambda *_: None,
        research_plan={"topic": "学习方法", "queries": [f"angle{i}" for i in range(8)], "depth": depth})
    assert len(observed) == queries
    assert len(result) == sources
    assert all(max_pages == pages and settings.scrape_max_pages == pages
               and settings.max_pages_per_query == pages for _, max_pages, settings in observed)
    assert all(settings.scrape_concurrency == 3 and settings.scrape_per_url_timeout_s == 15
               for _, _, settings in observed)
    assert get_settings().model_dump() == original


@pytest.mark.anyio
async def test_deep_selection_is_persisted_and_revision_honors_the_current_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(writing.runtime, "user_data_dir", lambda: tmp_path)
    first = await writing.compose("写一篇学习方法文章", options("deep"), DepthWriter())
    writing.save_article("depth-original", first)
    previous = writing.load_article("depth-original")
    assert previous["depth"] == "deep"
    assert previous["depth_budget"]["max_sources"] == 6
    llm = DepthWriter()
    revised = await writing.compose("改成简明的课程文章", options(previous_article_id="depth-original"),
                                    llm, previous=previous)
    planner = next(p for s, p in llm.calls if "Composition planner" in s)
    assert planner["depth"] == "standard" and planner["previous_depth"] == "deep"
    assert planner["previous_draft"]
    assert revised["depth"] == "standard" and revised["previous_depth"] == "deep"
    assert revised["previous_article_id"] == "depth-original"
    writing.save_article("depth-revised", revised)
    assert writing.load_article("depth-revised")["depth"] == "standard"


@pytest.mark.anyio
async def test_failed_deep_planning_keeps_complete_prose_and_does_not_certify_facts():
    llm = DepthWriter()
    llm.fail_depth = True
    result = await writing.compose("学习方法课程文章", options("deep"), llm)
    assert result["depth"] == "deep" and result["depth_planning_status"] == "unavailable"
    assert result["content"].endswith("。") and "## 结语" in result["content"]
    assert result["target_length"] == 800
    assert result["notice"] == writing.NOTICE
    assert result["factual_verification"] == "not_performed"
    assert any("深入讨论规划未完成" in note for note in result["notes"])


@pytest.mark.anyio
@pytest.mark.parametrize("depth", ["standard", "deep"])
async def test_both_depths_keep_conflicts_excluded_and_fabrication_guards(depth):
    llm = DepthWriter()
    llm.conflicts = [{"left": "本班人数为30人", "right": "本班人数为40人", "reason": "同一班级人数冲突"}]
    result = await writing.compose("学习方法课程文章",
        options(depth, materials="本班人数为30人。本班人数为40人。课堂讨论学习方法。"), llm)
    assert result["conflicts"] and "30人" not in result["content"] and "40人" not in result["content"]
    drafts = [(s, p) for s, p in llm.calls if "Composition section writer" in s]
    assert all("30人" not in p["materials"] and "40人" not in p["materials"] for _, p in drafts)
    assert all("Do not invent precise statistics" in s and "Do not claim research" in s
               and "omit them rather than selecting or averaging" in s for s, _ in drafts)
    assert result["basis_policy"] == "assumed_for_writing" and result["notice"] == writing.NOTICE
