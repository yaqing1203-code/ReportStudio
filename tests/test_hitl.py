"""Tests for workflow F: the human-in-the-loop (HITL) isolated-claim review node.

The pipeline checkpoint is OFF by default; when enabled (CE_HITL_ON_ISOLATED_CORE_
CLAIM=true) and a ConfirmationGate is supplied, a run pauses after verification if
any CORE isolated claim survived (isolated=True with credibility L3 or stronger,
numeric level <= 3), emits an 'awaiting_confirmation' progress event, and waits for
gate.decide("continue" | "abort"). A gate timeout defaults to "continue".

Server-side, POST /api/jobs/{id}/confirm delivers the decision: 404 unknown job,
409 when no review is pending, 200 + gate.decide otherwise.

All offline (FakeLLM / structured fixtures). Run: pytest -q tests/test_hitl.py
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core_engine.report.models import (
    CredibilityLevel,
    PipelineStatus,
    Source,
    SourceKind,
)
from core_engine.report.pipeline import ConfirmationGate, ReportPipeline

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------
# Pipeline fixtures: 2 general-web (L3) sources; a shared claim (corroborated)
# plus one unique claim per source (each a CORE isolated claim: single source,
# credibility L3).
# --------------------------------------------------------------------------
_URL_A = "https://tradejournal-a.example.com/report"
_URL_B = "https://tradejournal-b.example.com/report"
_SHARED = "The sector market grew 10 percent in 2024."
_SHARED_2 = "Sector exports reached 40 billion dollars in 2023."
_SHARED_3 = "The regulator issued 12 new licenses in 2022."
_UNIQUE_A = "Factory alpha output was 500 units in 2023."
_UNIQUE_B = "Employment in the sector rose 8 percent in 2022."


def _fixture(shared_only: bool = False):
    """(FakeLLM, extra_sources) driving a skip_search run. With shared_only=True
    every claim appears in both sources, so nothing is isolated."""
    from core_engine.report.llm import FakeLLM

    llm = FakeLLM()
    shared = [_SHARED, _SHARED_2, _SHARED_3]
    claims_a = shared if shared_only else shared + [_UNIQUE_A]
    claims_b = shared if shared_only else shared + [_UNIQUE_B]
    llm.set_structured_claims(_URL_A, [{"text": t} for t in claims_a])
    llm.set_structured_claims(_URL_B, [{"text": t} for t in claims_b])
    sources = [
        Source(_URL_A, "tradejournal-a.example.com", "Journal A",
               SourceKind.GENERAL_WEB, text="body A",
               credibility=CredibilityLevel.L3),
        Source(_URL_B, "tradejournal-b.example.com", "Journal B",
               SourceKind.GENERAL_WEB, text="body B",
               credibility=CredibilityLevel.L3),
    ]
    return llm, sources


def _enable_hitl(monkeypatch, tmp_path, enabled: bool):
    from core_engine.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("CE_HITL_ON_ISOLATED_CORE_CLAIM", "true" if enabled else "false")
    monkeypatch.setenv("CE_OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    return get_settings


async def _await_event(events, stage, timeout_s=5.0):
    """Poll until a progress event with `stage` arrives (the run emits it before
    blocking on the gate)."""
    for _ in range(int(timeout_s / 0.01)):
        hit = next((d for s, d in events if s == stage), None)
        if hit is not None:
            return hit
        await asyncio.sleep(0.01)
    return None


# --------------------------------------------------------------------------
# ConfirmationGate unit behavior
# --------------------------------------------------------------------------
async def test_gate_timeout_defaults_to_continue():
    gate = ConfirmationGate()
    assert await gate.wait(timeout=0.05) == "continue"


async def test_gate_decide_roundtrip_and_validation():
    gate = ConfirmationGate()
    with pytest.raises(ValueError):
        gate.decide("bogus")
    gate.decide("abort")
    assert await gate.wait(timeout=1.0) == "abort"


# --------------------------------------------------------------------------
# Pipeline checkpoint
# --------------------------------------------------------------------------
async def test_switch_off_never_pauses(monkeypatch, tmp_path):
    """Default (switch OFF): a gate may be passed but the run never waits and no
    awaiting_confirmation event is emitted — behavior identical to before."""
    _enable_hitl(monkeypatch, tmp_path, False)
    from core_engine.config import get_settings
    llm, sources = _fixture()
    gate = ConfirmationGate()
    events: list[tuple[str, str]] = []
    try:
        pipe = ReportPipeline(llm=llm, fetcher=None)
        result = await pipe.run(
            "sector test", compile_to_pdf=False,
            on_progress=lambda s, d: events.append((s, d)),
            extra_sources=sources, skip_search=True,
            confirmation_gate=gate)
    finally:
        get_settings.cache_clear()
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert not any(s == "awaiting_confirmation" for s, _ in events)
    assert gate.decision is None


async def test_switch_on_pauses_then_continue(monkeypatch, tmp_path):
    """Switch ON + core isolated claims: the run emits awaiting_confirmation with a
    JSON summary (truncated text, domain, credibility label) and SUSPENDS until
    gate.decide('continue'), after which it completes normally."""
    _enable_hitl(monkeypatch, tmp_path, True)
    from core_engine.config import get_settings
    llm, sources = _fixture()
    gate = ConfirmationGate()
    events: list[tuple[str, str]] = []
    try:
        pipe = ReportPipeline(llm=llm, fetcher=None)
        task = asyncio.create_task(pipe.run(
            "sector test", compile_to_pdf=False,
            on_progress=lambda s, d: events.append((s, d)),
            extra_sources=sources, skip_search=True,
            confirmation_gate=gate))

        detail = await _await_event(events, "awaiting_confirmation")
        assert detail is not None, "run never reached the HITL checkpoint"
        summary = json.loads(detail)
        assert len(summary["claims"]) == 2
        by_text = {c["text"]: c for c in summary["claims"]}
        assert by_text[_UNIQUE_A]["domain"] == "tradejournal-a.example.com"
        assert by_text[_UNIQUE_A]["credibility"] == CredibilityLevel.L3.label
        assert by_text[_UNIQUE_B]["domain"] == "tradejournal-b.example.com"

        await asyncio.sleep(0.05)
        assert not task.done(), "run must stay suspended until a human decides"

        gate.decide("continue")
        result = await asyncio.wait_for(task, timeout=15)
    finally:
        get_settings.cache_clear()
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert result.report is not None
    assert any(s == "confirmation_resolved" for s, _ in events)


async def test_switch_on_abort_blocks_run(monkeypatch, tmp_path):
    """decide('abort') at the checkpoint halts the run with a terminal status whose
    trace records 'aborted by user at isolated-claim review'."""
    _enable_hitl(monkeypatch, tmp_path, True)
    from core_engine.config import get_settings
    llm, sources = _fixture()
    gate = ConfirmationGate()
    events: list[tuple[str, str]] = []
    try:
        pipe = ReportPipeline(llm=llm, fetcher=None)
        task = asyncio.create_task(pipe.run(
            "sector test", compile_to_pdf=False,
            on_progress=lambda s, d: events.append((s, d)),
            extra_sources=sources, skip_search=True,
            confirmation_gate=gate))
        assert await _await_event(events, "awaiting_confirmation") is not None
        gate.decide("abort")
        result = await asyncio.wait_for(task, timeout=15)
    finally:
        get_settings.cache_clear()
    assert result.status is PipelineStatus.BLOCKED
    assert result.report is None               # assembly never ran
    assert result.tex_path is None
    assert any(entry.get("reason") == "aborted by user at isolated-claim review"
               for entry in result.trace)


async def test_switch_on_without_core_isolated_never_pauses(monkeypatch, tmp_path):
    """Switch ON but every claim is corroborated across both sources -> no core
    isolated claim -> no pause, no event, gate untouched."""
    _enable_hitl(monkeypatch, tmp_path, True)
    from core_engine.config import get_settings
    llm, sources = _fixture(shared_only=True)
    gate = ConfirmationGate()
    events: list[tuple[str, str]] = []
    try:
        pipe = ReportPipeline(llm=llm, fetcher=None)
        result = await pipe.run(
            "sector test", compile_to_pdf=False,
            on_progress=lambda s, d: events.append((s, d)),
            extra_sources=sources, skip_search=True,
            confirmation_gate=gate)
    finally:
        get_settings.cache_clear()
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert not any(s == "awaiting_confirmation" for s, _ in events)
    assert gate.decision is None


# --------------------------------------------------------------------------
# Server: POST /api/jobs/{job_id}/confirm (FastAPI TestClient wiring)
# --------------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    from core_engine.config import get_settings
    get_settings.cache_clear()

    from core_engine.app import runtime
    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)

    from fastapi.testclient import TestClient

    from core_engine.app.server import create_app

    return TestClient(create_app())


def _make_job(job_id: str):
    from core_engine.app import server
    job = server.Job(id=job_id, topic="hitl wiring")
    server._JOBS[job_id] = job
    return job


def test_confirm_unknown_job_is_404(client):
    r = client.post("/api/jobs/no-such-job/confirm", json={"decision": "continue"})
    assert r.status_code == 404


def test_confirm_without_pending_review_is_409(client):
    from core_engine.app import server
    job = _make_job("hitl-409")
    try:
        r = client.post(f"/api/jobs/{job.id}/confirm", json={"decision": "continue"})
        assert r.status_code == 409
    finally:
        server._JOBS.pop(job.id, None)


def test_confirm_pending_review_is_200_and_delivers_decision(client):
    from core_engine.app import server
    job = _make_job("hitl-200")
    job.pending_confirmation = {
        "claims": [{"text": "t", "domain": "d", "credibility": "L3 Industry Consensus"}],
        "timeout_s": 600,
    }
    try:
        r = client.post(f"/api/jobs/{job.id}/confirm", json={"decision": "abort"})
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True}
        assert job.gate.decision == "abort"
        assert job.pending_confirmation is None
        # The review is consumed: a repeat confirm is a conflict, not a replay.
        r2 = client.post(f"/api/jobs/{job.id}/confirm", json={"decision": "abort"})
        assert r2.status_code == 409
    finally:
        server._JOBS.pop(job.id, None)


def test_confirm_rejects_invalid_decision(client):
    from core_engine.app import server
    job = _make_job("hitl-400")
    job.pending_confirmation = {"claims": [], "timeout_s": 600}
    try:
        r = client.post(f"/api/jobs/{job.id}/confirm", json={"decision": "maybe"})
        assert r.status_code == 400
        assert job.gate.decision is None
    finally:
        server._JOBS.pop(job.id, None)


async def test_run_job_stashes_pending_confirmation_and_resumes():
    """_run_job wires the gate into the pipeline: an 'awaiting_confirmation' progress
    event is stored on job.pending_confirmation AND pushed to the SSE queue; a gate
    decision lets the run reach its final event."""
    from core_engine.app import server
    from core_engine.report import pipeline as pipeline_mod
    from core_engine.report.models import PipelineResult

    summary = {"claims": [{"text": "claim", "domain": "d.example.com",
                           "credibility": "L3 Industry Consensus"}],
               "timeout_s": 600}

    class StubPipeline:
        async def run(self, topic, *, on_progress=None, confirmation_gate=None, **_kw):
            assert confirmation_gate is not None, "job.gate must reach the pipeline"
            on_progress("awaiting_confirmation", json.dumps(summary))
            decision = await confirmation_gate.wait(timeout=5)
            status = (PipelineStatus.COMPLETED if decision == "continue"
                      else PipelineStatus.BLOCKED)
            return PipelineResult(status=status, topic=topic,
                                  message=f"decision={decision}")

    job = server.Job(id="hitl-run-job", topic="wiring")
    server._JOBS[job.id] = job
    original = pipeline_mod.ReportPipeline
    pipeline_mod.ReportPipeline = StubPipeline
    try:
        task = asyncio.create_task(server._run_job(job))
        for _ in range(500):
            if job.pending_confirmation is not None:
                break
            await asyncio.sleep(0.01)
        assert job.pending_confirmation == summary

        # The same event was queued for SSE consumers.
        await asyncio.sleep(0.05)
        queued = []
        while not job.queue.empty():
            queued.append(job.queue.get_nowait())
        assert any(e.get("stage") == "awaiting_confirmation" for e in queued)

        job.gate.decide("continue")
        await asyncio.wait_for(task, timeout=10)
        assert job.status == "completed"
        assert job.pending_confirmation is None
    finally:
        pipeline_mod.ReportPipeline = original
        server._JOBS.pop(job.id, None)
