"""Depth survives request routing, revision and document-only continuation."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core_engine.app import database, history, runtime, server, writing
from core_engine.config import get_settings

_REAL_REPORT_WORKER = server._run_job


@pytest.fixture
def depth_client(monkeypatch, tmp_path):
    monkeypatch.setenv("CE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    monkeypatch.delenv("CE_APP_SESSION_TOKEN", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    captured = {}

    async def report(job):
        captured[job.id] = {"job": job}

    async def article(job, options, previous):
        captured[job.id] = {"job": job, "options": options, "previous": previous}

    monkeypatch.setattr(server, "_run_job", report)
    monkeypatch.setattr(server, "_run_writing_job", article)
    with TestClient(server.create_app(), base_url="http://127.0.0.1") as client:
        yield client, captured
    get_settings.cache_clear()


@pytest.mark.parametrize("depth", ["standard", "deep"])
@pytest.mark.parametrize("mode", ["report", "writing"])
def test_explicit_depth_reaches_job_and_history(depth_client, mode, depth):
    client, captured = depth_client
    response = client.post("/api/message", json={"message": "固态电池", "mode": mode,
        "depth": depth, "writing_options": {"length": 1200, "use_web": False}})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["depth"] == depth
    job = captured[data["job_id"]]["job"]
    assert job.depth == depth
    entry = next(item for item in history.list_entries() if item["id"] == data["history_id"])
    assert entry["depth"] == depth
    if mode == "writing":
        options = captured[job.id]["options"]
        assert options.depth == depth and options.length == 1200
    else:
        session = client.get(f'/api/database/{data["session_id"]}').json()
        assert session["depth"] == depth


@pytest.mark.parametrize("endpoint", ["/api/message", "/api/report", "/api/database/session"])
def test_invalid_depth_is_rejected_before_job(depth_client, endpoint):
    client, captured = depth_client
    response = client.post(endpoint, json={"message": "测试", "topic": "测试", "mode": "report", "depth": "unverified"})
    assert response.status_code == 422
    assert not captured


@pytest.mark.parametrize("endpoint", ["/api/report", "/api/database/session"])
def test_legacy_entry_points_accept_depth(depth_client, endpoint):
    client, captured = depth_client
    data = client.post(endpoint, json={"topic": "固态电池", "depth": "deep"}).json()
    assert data["depth"] == "deep"
    assert captured[data["job_id"]]["job"].depth == "deep"


def test_comprehensive_keeps_session_depth_without_search(depth_client):
    client, captured = depth_client
    data = client.post("/api/message", json={"message": "研究固态电池", "mode": "report", "depth": "deep"}).json()
    sid = data["session_id"]
    database.add_article(sid, title="资料", filename="notes.txt", text="用户上传的资料正文。")
    response = client.post(f"/api/database/{sid}/comprehensive")
    assert response.status_code == 200, response.text
    result = response.json()
    job = captured[result["job_id"]]["job"]
    assert job.depth == result["depth"] == "deep"
    assert job.skip_search and job.extra_sources
    assert database.get_session(sid)["depth"] == "deep"
    response = client.post(f"/api/database/{sid}/comprehensive", json={"depth": "standard"})
    assert response.status_code == 200
    assert captured[response.json()["job_id"]]["job"].depth == "standard"
    assert database.get_session(sid)["depth"] == "standard"
    assert client.post(f"/api/database/{sid}/comprehensive", json={"depth": "invalid"}).status_code == 422


def test_revision_inherits_depth_only_when_selection_omitted(depth_client):
    client, captured = depth_client
    writing.save_article("previous-deep", {"mode": "writing", "title": "旧稿", "content": "旧稿正文", "depth": "deep"})
    payload = {"message": "修改这篇文章", "mode": "writing",
               "writing_options": {"previous_article_id": "previous-deep", "length": 1600}}
    response = client.post("/api/message", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["depth"] == "deep"
    assert captured[response.json()["job_id"]]["options"].length == 1600
    response = client.post("/api/message", json={**payload, "depth": "standard"})
    assert response.status_code == 200 and response.json()["depth"] == "standard"


def test_conflicting_duplicate_depth_does_not_silently_win(depth_client):
    client, _ = depth_client
    response = client.post("/api/message", json={"message": "课程文章", "mode": "writing", "depth": "standard",
        "writing_options": {"depth": "deep"}})
    assert response.status_code == 400


def test_legacy_nested_writing_depth_is_honored(depth_client):
    client, captured = depth_client
    response = client.post("/api/message", json={"message": "课程文章", "mode": "writing",
        "writing_options": {"depth": "deep", "length": 1800}})
    assert response.status_code == 200
    assert response.json()["depth"] == "deep"
    assert captured[response.json()["job_id"]]["options"].depth == "deep"


def test_depth_only_continuation_keeps_template_and_cover(depth_client):
    from core_engine.app import output_templates

    client, captured = depth_client
    template = output_templates.add_template("course.docx", output_templates.example_template())
    data = client.post("/api/message", json={"message": "固态电池报告", "mode": "report", "depth": "standard",
        "template_id": template["id"], "template_fields": {"作者": "作者甲"}}).json()
    sid = data["session_id"]
    database.add_article(sid, title="资料", filename="notes.txt", text="用户资料正文")
    response = client.post(f"/api/database/{sid}/comprehensive", json={"depth": "deep"})
    assert response.status_code == 200, response.text
    job = captured[response.json()["job_id"]]["job"]
    assert job.depth == "deep" and job.output_template[0]["id"] == template["id"]
    assert job.template_fields == {"作者": "作者甲"}
    response = client.post(f"/api/database/{sid}/comprehensive", json={"template_id": None})
    assert response.status_code == 200
    assert captured[response.json()["job_id"]]["job"].output_template is None


@pytest.mark.anyio
async def test_profile_initialization_error_still_finishes_and_records_depth(depth_client, monkeypatch):
    from core_engine.report import generation

    _, _ = depth_client
    entry = history.add_entry(kind="report", title="报告", depth="standard", job_id="old-job")
    job = server.Job(id="new-job", topic="报告", depth="deep", history_id=entry)

    def invalid(*_):
        raise ValueError("invalid local settings")

    base = get_settings()
    monkeypatch.setattr(generation, "report_profile", invalid)
    await _REAL_REPORT_WORKER(job)
    assert job.done.is_set() and job.terminal_event["depth"] == "deep"
    assert job.terminal_event["status"] == "error"
    assert get_settings() is base
    saved = next(item for item in history.list_entries() if item["id"] == entry)
    assert saved["depth"] == "deep" and saved["job_id"] == "new-job"
