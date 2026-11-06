"""Offline regressions for storage, compiler, upload and local API boundaries."""
from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from core_engine.app import database, runtime, server
from core_engine.report import documents
from core_engine.report.compile import CompileError, _build_cmd, available_engine
from core_engine.report.latex import render_tex, tex_escape, write_tex
from core_engine.report.models import Claim, Evidence, ReportData, Source, SourceKind
from core_engine.security.files import atomic_write_text


@pytest.fixture
def local_data(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def client(monkeypatch, local_data):
    monkeypatch.delenv("CE_APP_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(server, "_JOBS", {})
    with TestClient(server.create_app(), base_url="http://127.0.0.1") as client:
        yield client


@pytest.mark.parametrize("identifier", ["..", "../outside", "..\\outside", "C:\\outside", "/outside", "CON", "bad:id"])
def test_session_ids_cannot_escape_store(local_data, identifier):
    assert database.get_session(identifier) is None
    with pytest.raises(ValueError):
        database._write(identifier, {"topic": "invalid"})
    assert not (local_data / "outside").exists()


def test_unknown_session_read_does_not_create_directory(local_data):
    assert database.get_session("unknown") is None
    assert not (local_data / "database" / "unknown").exists()


def test_pdf_index_cannot_redirect_to_other_local_file(local_data):
    outside = local_data / "private.pdf"
    outside.write_bytes(b"private")
    (runtime.generated_pdfs_dir() / "index.json").write_text(
        json.dumps({"job1": {"pdf": str(outside)}}), encoding="utf-8")
    assert runtime.lookup_pdf("job1") is None
    assert runtime.lookup_pdf("../private") is None


def test_pdf_storage_roundtrip(local_data):
    source = local_data / "report.pdf"
    source.write_bytes(b"%PDF-test")
    stored = runtime.store_pdf("job1", source)
    assert runtime.lookup_pdf("job1") == stored
    assert stored.read_bytes() == source.read_bytes()


def test_evidence_download_survives_restart_and_rejects_path_traversal(client, local_data):
    path = local_data / "source.evidence.json"
    path.write_text(json.dumps({"schema_version": 1, "quote": "原文证据"}), encoding="utf-8")
    stored = runtime.store_evidence("evidence-job", path)
    assert runtime.lookup_evidence("evidence-job") == stored
    assert runtime.lookup_evidence("..\\source") is None
    # No in-memory job exists: the API must serve the persisted sidecar.
    response = client.get("/api/report/evidence-job/evidence")
    assert response.status_code == 200
    assert response.json()["quote"] == "原文证据"
    assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/api/report/missing/evidence").status_code == 404


def test_atomic_write_preserves_previous_file_on_replace_error(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("previous", encoding="utf-8")

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("core_engine.security.files.os.replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_text(path, "new")
    assert path.read_text(encoding="utf-8") == "previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["settings.json"]


def test_clearing_key_removes_applied_environment(monkeypatch, local_data):
    monkeypatch.delenv("CE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("CE_OUTPUT_DIR", raising=False)
    runtime.save_settings({"llm_api_key": "test-secret"})
    import os
    assert os.environ["CE_LLM_API_KEY"] == "test-secret"
    assert runtime.redacted_settings()["llm_api_key_set"] is True
    runtime.save_settings({"llm_api_key": ""})
    assert os.environ["CE_LLM_API_KEY"] == "test-secret"
    runtime.save_settings({"llm_api_key": None})
    assert "CE_LLM_API_KEY" not in os.environ
    assert "llm_api_key" not in runtime.load_settings()


@pytest.mark.parametrize("engine", ["cmd", "powershell", "C:\\Windows\\System32\\cmd.exe", "/bin/sh"])
def test_arbitrary_compiler_commands_rejected(engine):
    with pytest.raises(CompileError):
        available_engine(engine)


@pytest.mark.parametrize("engine", ["tectonic", "xelatex", "pdflatex", "latexmk"])
def test_compiler_disables_unsafe_features(engine, tmp_path):
    command, _passes = _build_cmd(engine, tmp_path / "a.tex", tmp_path)
    if engine == "tectonic":
        assert "--untrusted" in command
    else:
        assert "-no-shell-escape" in command
    if engine == "latexmk":
        assert "-norc" in command


def test_tex_escape_does_not_corrupt_its_own_commands():
    assert tex_escape("\\~^") == r"\textbackslash{}\textasciitilde{}\textasciicircum{}"
    assert tex_escape("\x00a\t") == " a "


def test_bibliography_url_cannot_inject_tex():
    url = r"https://example.org/}\input{private}%"
    report = ReportData(topic="topic", title="title", abstract="", sections=[],
                        claims=[Claim(id="c1", text="x", verified=True, candidate_source_urls=[url],
                                      evidence=[Evidence(source_url=url, quote="x", supports=True)])],
                        sources=[Source(url, "example.org", "source", SourceKind.GENERAL_WEB, text="x")])
    tex = render_tex(report)
    assert r"\input{private}" not in tex
    assert "%7D%5Cinput%7Bprivate%7D" in tex


def test_same_topic_runs_keep_distinct_outputs(tmp_path):
    report = ReportData(topic="same topic", title="title", abstract="", sections=[], claims=[], sources=[])
    first, second = write_tex(report, tmp_path), write_tex(report, tmp_path)
    assert first != second
    assert first.exists() and second.exists()


@pytest.mark.parametrize("strategy", ["aligned", "conflict_only", "cross_reference"])
def test_methodology_describes_actual_verification_contract(monkeypatch, strategy):
    from core_engine.config import get_settings
    monkeypatch.setenv("CE_VERIFY_STRATEGY", strategy)
    get_settings.cache_clear()
    try:
        report = ReportData(topic="topic", title="title", abstract="")
        tex = render_tex(report)
        assert "Every verified\nfinding below was cross-referenced" not in tex
        if strategy == "cross_reference":
            assert "Claims were checked for source support" in tex
        else:
            assert "does not independently check every claim" in tex
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("extension", ["docx", "xlsx"])
def test_office_archive_expansion_is_checked_before_parsing(monkeypatch, extension):
    monkeypatch.setattr(documents, "MAX_ARCHIVE_BYTES", 1024)
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload.xml", "A" * 2048)
    with pytest.raises(documents.DocumentError, match="expands"):
        documents.parse_document(f"bomb.{extension}", raw.getvalue())


def test_text_limit_stops_parser_iteration(monkeypatch):
    monkeypatch.setattr(documents, "MAX_TEXT_CHARS", 8)

    def parts():
        yield "abcdefgh"
        raise AssertionError("parser should stop once enough text has been collected")

    assert documents._join_bounded(parts()) == "abcdefgh"


def test_invalid_loopback_host_and_foreign_origins_rejected(client):
    assert client.get("/api/history", headers={"Host": "attacker.example"}).status_code == 403
    for origin in ("https://attacker.example", "null", "http://127.0.0.1:9999"):
        assert client.get("/api/history", headers={"Origin": origin}).status_code == 403
    assert client.get("/api/history", headers={"Origin": "http://127.0.0.1"}).status_code == 200


def test_session_token_required_and_secrets_not_cached(client, monkeypatch):
    monkeypatch.setenv("CE_APP_SESSION_TOKEN", "local-secret")
    assert client.get("/api/settings").status_code == 401
    response = client.get("/api/settings", headers={"X-Session-Token": "local-secret"})
    assert response.status_code == 200
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"


def test_invalid_compiler_setting_is_rejected_by_api(client):
    assert client.post("/api/settings", json={"latex_engine": "cmd.exe"}).status_code == 422


def test_upload_limit_checked_before_multipart_parse(client):
    sid = database.create_session("topic")
    response = client.post(f"/api/database/{sid}/articles", content=b"x",
                           headers={"Content-Length": str(30 * 1024 * 1024)})
    assert response.status_code == 413
    assert database.list_articles(sid) == []


def test_streamed_upload_limit_without_content_length(client):
    from core_engine.app.middleware import UploadLimitMiddleware
    app = server.create_app()
    for middleware in app.user_middleware:
        if middleware.cls is UploadLimitMiddleware:
            middleware.kwargs["max_bytes"] = 128
    sid = database.create_session("topic")
    boundary = "stream-boundary"

    def body():
        yield (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
               'filename="a.txt"\r\n\r\n').encode()
        yield b"x" * 129
        yield f"\r\n--{boundary}--\r\n".encode()

    with TestClient(app, base_url="http://127.0.0.1") as stream_client:
        response = stream_client.post(f"/api/database/{sid}/articles", content=body(),
                                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413, response.text
    assert database.list_articles(sid) == []


def test_finished_events_replay_for_multiple_viewers_and_resume(client):
    job = server.Job(id="stream-job", topic="topic")
    server._JOBS[job.id] = job
    job.publish({"stage": "starting"})
    job.publish({"stage": "completed", "final": True, "has_pdf": True})
    path = f"/api/report/{job.id}/events"
    first, second = client.get(path), client.get(path)
    assert first.text == second.text
    assert '"has_pdf": true' in first.text
    resumed = client.get(path, headers={"Last-Event-ID": "1"})
    assert '"starting"' not in resumed.text
    assert '"final": true' in resumed.text
    assert not job.subscribers


def test_event_buffers_stay_bounded_and_broadcast():
    import asyncio
    job = server.Job(id="bounded", topic="topic")
    viewers = [asyncio.Queue(maxsize=512), asyncio.Queue(maxsize=512)]
    job.subscribers.update(viewers)
    for i in range(600):
        job.publish({"stage": str(i)})
    job.publish({"final": True})
    assert len(job.events) == job.queue.qsize() == 512
    for viewer in viewers:
        assert viewer.qsize() == 512
        records = [viewer.get_nowait() for _ in range(512)]
        assert records[-1][1]["final"] is True
