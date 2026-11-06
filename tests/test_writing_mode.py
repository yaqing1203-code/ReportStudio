"""Composition is isolated from evidence-gated reports; never publish a partial draft."""
from __future__ import annotations

import json

import pytest

from core_engine.app import writing
from core_engine.report.llm import FakeLLM, LLMResponseError


@pytest.fixture
def anyio_backend():
    return "asyncio"


class EditorialLLM:
    """Scripted editorial decisions, including malformed/truncated responses."""

    def __init__(self):
        self.calls = []
        self.conflicts = []
        self.section_results = []
        self.review = {"edits": [], "notes": []}
        self.outline = ["背景与意义", "问题与讨论", "实践与思考"]

    async def complete(self, system, user):
        payload = json.loads(user) if user.startswith("{") else user
        self.calls.append((system, payload))
        if "Composition conflict reviewer" in system:
            return json.dumps({"conflicts": self.conflicts}, ensure_ascii=False)
        if "Composition planner" in system:
            return json.dumps({"title": "人工智能与学习", "task": "围绕学习方法写一篇完整文章",
                "outline": [{"heading": h, "purpose": "展开论述"} for h in self.outline],
                "conflicts": self.conflicts}, ensure_ascii=False)
        if "Composition section writer" in system:
            if self.section_results:
                result = self.section_results.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result
            return ("学习离不开主动思考，也需要合适的方法与工具。面对新的技术条件，学生应当围绕实际问题安排学习，"
                    "把理解知识、独立判断和日常练习结合起来，逐步形成适合自己的学习习惯。\n\n"
                    "因此，工具的价值需要落实到具体行动中。只有明确目标，认真检查思路，并在实践中不断调整方法，"
                    "才能让学习过程更加连贯，让获得的知识真正转化为解决问题的能力。")
        if "Composition consistency reviewer" in system:
            if isinstance(self.review, Exception):
                raise self.review
            return json.dumps(self.review, ensure_ascii=False)
        return '{"intent":"general_chat","confidence":"high","reasoning":"test"}'


def opts(**kwargs):
    return writing.WritingOptions(length=800, use_web=False, **kwargs)


@pytest.mark.anyio
async def test_complete_article_preserves_options_and_does_not_demand_evidence():
    llm = EditorialLLM()
    events = []
    result = await writing.compose("写一篇课程文章", opts(style="natural", allow_set_phrases=False,
        materials="课堂讨论强调主动学习。"), llm, on_progress=lambda stage, _: events.append(stage))
    assert result["content"].startswith("## 引言\n")
    assert result["content"].count("\n\n## ") == 4
    assert result["content"].endswith("。")
    assert result["factual_verification"] == "not_performed"
    assert result["conflicts"] == []
    assert "事实核验" in result["notice"]
    sections = [p for s, p in llm.calls if "section writer" in s]
    assert all("课堂讨论强调主动学习" in p["materials"] for p in sections)
    assert all(p["allow_set_phrases"] is False and "自然" in p["style"] for p in sections)
    assert events[-1] == "writing_review_done"


@pytest.mark.anyio
async def test_web_disabled_never_calls_search():
    async def forbidden(*_, **_kwargs):
        pytest.fail("Writing with web disabled must not access the network")
    result = await writing.compose("课程文章", opts(), EditorialLLM(), search_fn=forbidden)
    assert result["sources"] == []


@pytest.mark.anyio
@pytest.mark.parametrize("failed", [False, True])
async def test_empty_or_failed_search_does_not_block_complete_article(failed):
    async def search(*_, **_kwargs):
        if failed:
            raise TimeoutError("search unavailable")
        return []
    result = await writing.compose("课程文章", writing.WritingOptions(length=800), EditorialLLM(), search_fn=search)
    assert result["content"].endswith("。") and result["notes"]
    assert result["factual_verification"] == "not_performed"


@pytest.mark.anyio
async def test_background_sources_are_preserved_without_being_certified():
    sources = [{"title": "课堂背景", "url": "https://example.org/article", "text": "教育需要结合实践。"}]
    async def search(*_, **_kwargs):
        return sources
    result = await writing.compose("课程文章", writing.WritingOptions(length=800), EditorialLLM(), search_fn=search)
    assert result["sources"] == [{"title": "课堂背景", "url": "https://example.org/article"}]
    assert result["basis_policy"] == "assumed_for_writing"


@pytest.mark.anyio
async def test_conflicting_details_excluded_but_ordinary_material_retained():
    llm = EditorialLLM()
    llm.conflicts = [{"left": "本班人数为30人", "right": "本班人数为40人", "reason": "同一班级人数冲突"}]
    result = await writing.compose("写一篇文章", opts(materials="本班人数为30人。本班人数为40人。课程强调合作学习。"), llm)
    assert len(result["conflicts"]) == 1
    sections = [p for s, p in llm.calls if "section writer" in s]
    assert all("30人" not in p["materials"] and "40人" not in p["materials"] for p in sections)
    assert all("课程强调合作学习" in p["materials"] for p in sections)
    assert "35" not in result["content"]  # No invented compromise value.


@pytest.mark.anyio
async def test_different_years_and_fabricated_conflicts_are_not_dropped():
    llm = EditorialLLM()
    llm.conflicts = [{"left": "2024年人数为30人", "right": "2025年人数为40人"},
                     {"left": "材料不存在的观点甲", "right": "材料不存在的观点乙"}]
    result = await writing.compose("课程文章", opts(materials="2024年人数为30人。2025年人数为40人。"), llm)
    assert result["conflicts"] == []
    assert all("2024年人数为30人" in p["materials"] for s, p in llm.calls if "section writer" in s)


@pytest.mark.anyio
async def test_failed_conflict_review_does_not_block_structure_or_claim_it_was_completed():
    class Unavailable(EditorialLLM):
        async def complete(self, system, user):
            if "Composition conflict reviewer" in system:
                raise LLMResponseError("output_limit")
            return await super().complete(system, user)
    result = await writing.compose("写一篇完整课程文章", opts(), Unavailable())
    assert result["content"].endswith("。")
    assert result["conflict_review_status"] == "unavailable"
    assert any("材料分歧检查未完成" in note for note in result["notes"])


@pytest.mark.anyio
async def test_structure_planning_is_isolated_from_bulk_technical_comparison():
    llm = EditorialLLM()
    result = await writing.compose("写一篇课程文章", opts(materials="本班人数为30人。"), llm)
    planner = next(p for s, p in llm.calls if "Composition planner" in s)
    assert "background" not in planner and "background_titles" in planner
    assert any("Composition conflict reviewer" in s for s, _ in llm.calls)
    assert result["conflict_review_status"] == "checked"


@pytest.mark.anyio
async def test_earlier_review_chunks_cannot_starve_the_conclusion(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    clock = [0.0]
    budgets = []
    real_wait = asyncio.wait_for
    async def timed_wait(awaitable, *, timeout):
        if timeout <= 45:
            budgets.append(timeout)
        return await real_wait(awaitable, timeout=timeout)
    class SlowReviewer(EditorialLLM):
        async def complete(self, system, user):
            if "Composition consistency reviewer" in system:
                clock[0] += 40  # Simulate time spent by each previous chunk.
            return await super().complete(system, user)
    monkeypatch.setattr(writing, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(writing.asyncio, "wait_for", timed_wait)
    result = await writing.compose("课程文章", opts(), SlowReviewer())
    assert result["review_status"] == "checked"
    assert budgets[-3:] == [45, 45, 45]


@pytest.mark.anyio
async def test_editor_receives_bounded_relevant_context_instead_of_all_technical_material():
    llm = EditorialLLM()
    await writing.compose("完整中文课程文章", opts(materials="技术材料与界面讨论。" * 1000), llm)
    reviews = [p for s, p in llm.calls if "Composition consistency reviewer" in s]
    assert reviews and all(len(p["basis"]) <= 1810 for p in reviews)
    assert all(p["basis_scope"] == "selected original excerpts" for p in reviews)


@pytest.mark.anyio
async def test_reintroduced_conflict_cannot_be_published():
    llm = EditorialLLM()
    llm.conflicts = [{"left": "本班人数为30人", "right": "本班人数为40人"}]
    llm.section_results = ["本班人数为30人。" + "课程学习需要认真思考和充分交流。" * 12]
    with pytest.raises(writing.WritingError, match="争议细节"):
        await writing.compose("文章", opts(materials="本班人数为30人。本班人数为40人。"), llm)


@pytest.mark.anyio
@pytest.mark.parametrize("first", ["尚未完成", LLMResponseError("output_limit"), "- 学习\n- 思考"])
async def test_incomplete_section_gets_one_repair(first):
    llm = EditorialLLM()
    llm.section_results = [first]
    result = await writing.compose("课程文章", opts(), llm)
    assert result["content"].endswith("。")
    assert any("rewrite_reason" in p for s, p in llm.calls if "section writer" in s)


@pytest.mark.anyio
async def test_stray_english_word_in_chinese_article_is_rewritten():
    llm = EditorialLLM()
    llm.section_results = ["固态电池的 applications 场景值得讨论。" + "课程讨论需要明确材料边界和应用条件。" * 6]
    result = await writing.compose("完整中文课程文章", opts(), llm)
    assert "applications" not in result["content"]
    assert any("untranslated" in p.get("rewrite_reason", "") for s, p in llm.calls if "section writer" in s)


def test_chinese_language_check_preserves_formulas_acronyms_and_parenthetical_terms():
    text = "材料可采用 LLZO 电解质，并注意 H2S 风险。固态电解质（solid-state electrolytes）仍需研究。"
    assert writing._prose(text, 80, chinese=True) == text
    assert writing._mixed_words("讨论其可能的 applications 场景。") == ["applications"]


@pytest.mark.anyio
async def test_repeated_truncation_is_error_instead_of_completed_outline():
    llm = EditorialLLM()
    llm.section_results = ["提纲", "仍然只是提纲"]
    with pytest.raises(writing.WritingError):
        await writing.compose("课程文章", opts(), llm)


@pytest.mark.anyio
@pytest.mark.parametrize("review", [RuntimeError("provider failed"),
    {"edits": [{"excerpt": "## 结语", "replacement": ""}], "notes": None}])
async def test_failed_or_destructive_editor_retains_complete_draft(review):
    llm = EditorialLLM()
    llm.review = review
    result = await writing.compose("课程文章", opts(), llm)
    assert result["review_status"] in {"unavailable", "partial"}
    assert "## 结语\n" in result["content"] and result["content"].endswith("。")


@pytest.mark.anyio
async def test_duplicate_planned_headings_do_not_break_article():
    llm = EditorialLLM()
    llm.outline = ["背景", "背景", "引言"]
    result = await writing.compose("课程文章", opts(), llm)
    assert result["content"].count("## 引言\n") == 1
    assert result["content"].count("## ") == 5


@pytest.mark.anyio
async def test_intro_and_conclusion_with_subtitles_are_not_duplicated_in_body():
    llm = EditorialLLM()
    llm.outline = ["引言：为什么关注固态电池", "一、技术路线", "制造难题", "应用前景", "结论与展望"]
    result = await writing.compose("固态电池课程文章", opts(), llm)
    assert result["content"].count("## ") == 5
    assert "## 引言：" not in result["content"] and "## 结论与展望" not in result["content"]


@pytest.mark.parametrize("heading", ["1. Introduction: context", "二、引言：背景", "总结与展望", "结语：展望"])
def test_frame_heading_recognition(heading):
    assert writing._frame_heading(heading)


@pytest.mark.anyio
async def test_previous_article_is_context_for_revision():
    llm = EditorialLLM()
    result = await writing.compose("改为自然风格", opts(previous_article_id="old-id"), llm,
                                   previous={"content": "上一次文章讨论课堂合作。"})
    planner = next(p for s, p in llm.calls if "Composition planner" in s)
    assert "课堂合作" in planner["previous_draft"]
    assert result["previous_article_id"] == "old-id"


@pytest.mark.anyio
async def test_offline_demo_explains_need_for_real_writer():
    with pytest.raises(writing.WritingError, match="真实语言模型"):
        await writing.compose("课程文章", opts(), FakeLLM())


@pytest.mark.anyio
async def test_research_report_for_class_is_not_silently_downgraded_to_composition(monkeypatch):
    from core_engine.app.intent import IntentType, classify_intent
    from core_engine.report import llm

    model = FakeLLM()
    model.set_complete("intent classifier", '{"intent":"report_generation","confidence":"high","reasoning":"research"}')
    monkeypatch.setattr(llm, "get_llm", lambda: model)
    result = await classify_intent("请为课程作业写一篇固态电池研究报告，附逐句证据")
    assert result.intent == IntentType.REPORT_GENERATION


@pytest.mark.anyio
async def test_malformed_planner_does_not_publish_canned_or_partial_article():
    class BadPlanner(EditorialLLM):
        async def complete(self, system, user):
            return "计划如下：先写引言。"
    with pytest.raises(ValueError):
        await writing.compose("文章", opts(), BadPlanner())


@pytest.mark.anyio
async def test_truncated_review_retries_with_bounded_output_budget():
    class BudgetedWriter(EditorialLLM):
        def __init__(self):
            super().__init__()
            self.budgets = []
        async def complete(self, system, user):
            if "consistency reviewer" in system:
                raise LLMResponseError("output_limit")
            return await super().complete(system, user)
        async def complete_with_budget(self, system, user, *, max_tokens):
            self.budgets.append(max_tokens)
            return '{"edits":[],"notes":[]}'
    llm = BudgetedWriter()
    result = await writing.compose("课程文章", opts(), llm)
    assert result["review_status"] == "checked"
    assert len(llm.budgets) == 3 and all(8192 <= b <= 16384 for b in llm.budgets)


@pytest.mark.anyio
@pytest.mark.parametrize("host", ["api.minimaxi.com", "compatible.example"])
async def test_writing_budget_cannot_change_concurrent_or_later_report_calls(monkeypatch, host):
    import asyncio

    import httpx

    from core_engine.report.llm import OpenAILLM

    requests = []
    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["messages"][-1]["content"] == "fail":
            return httpx.Response(401, json={"error": "test failure"})
        await asyncio.sleep(.01)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]})
    actual = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: actual(transport=httpx.MockTransport(handler), **kwargs))
    model = OpenAILLM("test-key", "MiniMax-M3", 4096, "https://" + host + "/v1")
    await asyncio.gather(model.complete_with_budget("writing", "composition", max_tokens=8192),
                         model.complete("report", "report"))
    await model.complete("report", "later report")
    key = "max_completion_tokens" if host == "api.minimaxi.com" else "max_tokens"
    budgets = {p["messages"][-1]["content"]: p[key] for p in requests}
    assert budgets == {"composition": 8192, "report": 4096, "later report": 4096}
    with pytest.raises(httpx.HTTPStatusError):
        await model.complete_with_budget("writing", "fail", max_tokens=8192)
    await model.complete("report", "after failure")
    assert requests[-1][key] == 4096


def test_article_storage_and_path_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(writing.runtime, "user_data_dir", lambda: tmp_path)
    article = {"mode": "writing", "title": "中文文章", "content": "## 引言\n\n完整文章。"}
    writing.save_article("safe-id", article)
    assert writing.load_article("safe-id") == article
    assert writing.article_text_path("safe-id").read_text(encoding="utf-8") == "中文文章\n\n引言\n\n完整文章。\n"
    assert writing.load_article("../outside") is None
    assert writing.article_text_path("../outside") is None
    assert writing.load_article("missing") is None


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from core_engine.app import runtime, server
    from core_engine.config import get_settings
    from core_engine.report import llm

    monkeypatch.setenv("CE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    get_settings.cache_clear()
    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    writer = EditorialLLM()
    monkeypatch.setattr(llm, "get_llm", lambda: writer)
    async def no_report(job):
        job.done.set()
    monkeypatch.setattr(server, "_run_job", no_report)
    try:
        with TestClient(server.create_app(), base_url="http://127.0.0.1") as client:
            yield client, writer
    finally:
        get_settings.cache_clear()


def start_article(client):
    response = client.post("/api/message", json={"message": "写一篇课程文章", "mode": "writing",
                           "writing_options": {"length": 800, "use_web": False}})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["type"] == "writing" and data["notice"]
    events = client.get(f"/api/report/{data['job_id']}/events")
    assert '"status": "completed"' in events.text, events.text
    return data


def test_writing_api_persists_history_download_and_revision(api_client):
    client, writer = api_client
    data = start_article(client)
    article = client.get(f"/api/writing/{data['job_id']}").json()
    assert article["factual_verification"] == "not_performed"
    download = client.get(f"/api/writing/{data['job_id']}/download")
    assert download.status_code == 200
    assert "结语" in download.text and "attachment" in download.headers["content-disposition"]
    assert "事实核验" not in download.text  # Banner stays outside the article body.
    item = next(i for i in client.get("/api/history").json()["items"] if i["id"] == data["history_id"])
    assert item["kind"] == "writing" and item["status"] == "completed" and item["writing_ready"]
    response = client.post("/api/message", json={"message": "增加一段学习建议", "mode": "writing",
        "writing_options": {"length": 800, "use_web": False, "previous_article_id": data["job_id"]}})
    assert response.json()["type"] == "writing"
    client.get(f"/api/report/{response.json()['job_id']}/events")
    assert any(p.get("previous_draft") == article["content"] for s, p in writer.calls if "Composition planner" in s)


@pytest.mark.parametrize("mode,prompt,expected", [
    ("auto", "写一篇关于合作学习的课程作业", "writing"),
    ("chat", "生成研究报告", "chat"),
    ("report", "写一篇课程作业", "report"),
    ("auto", "请介绍软件有几个模式", "chat"),
])
def test_explicit_modes_and_automatic_essay_routing(api_client, mode, prompt, expected):
    client, _ = api_client
    response = client.post("/api/message", json={"message": prompt, "mode": mode,
                           "writing_options": {"use_web": False, "length": 800}})
    assert response.status_code == 200, response.text
    assert response.json()["type"] == expected


def test_failed_writing_job_is_terminal_and_not_downloadable(api_client):
    client, writer = api_client
    writer.section_results = ["提纲", "提纲"]
    response = client.post("/api/message", json={"message": "文章", "mode": "writing",
                           "writing_options": {"use_web": False, "length": 800}}).json()
    events = client.get(f"/api/report/{response['job_id']}/events")
    assert '"status": "error"' in events.text
    assert client.get(f"/api/writing/{response['job_id']}").status_code == 404
    entry = next(i for i in client.get("/api/history").json()["items"] if i["id"] == response["history_id"])
    assert entry["status"] == "error"


def test_completed_article_restores_without_in_memory_job(api_client):
    from core_engine.app import server

    client, _ = api_client
    data = start_article(client)
    server._JOBS.pop(data["job_id"])
    assert client.get(f"/api/writing/{data['job_id']}").status_code == 200
    assert client.get(f"/api/writing/{data['job_id']}/download").status_code == 200


def test_interrupted_article_is_not_left_generating_after_restart(api_client):
    from core_engine.app import history

    client, _ = api_client
    hid = history.add_entry(kind="writing", title="中断的文章", status="running", job_id="lost-job")
    entry = next(i for i in client.get("/api/history").json()["items"] if i["id"] == hid)
    assert entry["status"] == "error" and entry["writing_ready"] is False


def test_explicit_writing_reports_missing_model_configuration(api_client, monkeypatch):
    from core_engine.report import llm

    client, _ = api_client
    def no_model():
        raise RuntimeError("API key missing")
    monkeypatch.setattr(llm, "get_llm", no_model)
    response = client.post("/api/message", json={"message": "文章", "mode": "writing"})
    assert response.status_code == 400
    assert "Settings" in response.json()["detail"]


@pytest.mark.parametrize("payload", [{"mode": "unknown"}, {"writing_options": {"length": 399}},
    {"writing_options": {"length": 4001}}, {"writing_options": {"style": "invalid"}}])
def test_invalid_mode_and_options_are_actionable_validation_errors(api_client, payload):
    client, _ = api_client
    assert client.post("/api/message", json={"message": "文章", **payload}).status_code == 422


def test_missing_previous_article_does_not_start_job(api_client):
    client, _ = api_client
    response = client.post("/api/message", json={"message": "修改文章", "mode": "writing",
        "writing_options": {"previous_article_id": "missing"}})
    assert response.status_code == 404


@pytest.mark.anyio
async def test_overlong_introduction_is_rewritten_instead_of_publishing_whole_article_twice():
    llm = EditorialLLM()
    oversized = "固态电池的电解质路线与工程制造需要结合实际应用分析。" * 120
    llm.section_results = [oversized]
    result = await writing.compose("写一篇中文课程文章", opts(), llm)
    assert result["length"] <= 1350
    repairs = [p for s, p in llm.calls if "section writer" in s and "rewrite_reason" in p]
    assert len(repairs) == 1
    assert repairs[0]["previous_draft"] == oversized
    assert "length budget" in repairs[0]["rewrite_reason"]
    assert oversized not in result["content"]


@pytest.mark.anyio
async def test_repeated_length_violation_cannot_be_marked_complete():
    llm = EditorialLLM()
    llm.section_results = ["章节正文不能靠重复内容无限扩展。" * 100] * 2
    with pytest.raises(writing.WritingError, match="length budget"):
        await writing.compose("课程文章", opts(), llm)


@pytest.mark.anyio
async def test_total_length_is_checked_even_if_each_section_fits_its_limit():
    llm = EditorialLLM()
    llm.section_results = ["学习需要结合具体问题展开讨论。" * 18] * 5
    with pytest.raises(writing.WritingError, match="篇幅"):
        await writing.compose("课程文章", writing.WritingOptions(length=400, use_web=False), llm)
