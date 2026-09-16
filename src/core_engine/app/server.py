"""FastAPI backend for the desktop app.

The pywebview shell (shell.py) loads a local web UI (static/index.html) that talks
to THIS server over http://127.0.0.1:<port>. The server is bound to loopback only —
it is never exposed off the machine — so there is no auth layer by design; the trust
boundary is the OS user session, same as any desktop app.

Endpoints:
  GET  /api/health              -> liveness + resolved engine/provider info
  GET  /api/settings            -> redacted settings (secrets shown only as *_set flags)
  POST /api/settings            -> persist settings (API protocol, base URL, key, ...)
  POST /api/report              -> start a report run; returns {job_id}
  GET  /api/report/{id}/events  -> Server-Sent Events stream of live progress
  GET  /api/report/{id}/pdf     -> download the finished PDF
  GET  /                        -> the chat UI (static files)

Runs are executed on the server's asyncio loop; progress is pushed onto a per-job
asyncio.Queue that the SSE endpoint drains. Jobs live in-memory (single-user desktop
app) — no database.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core_engine.app import runtime

log = logging.getLogger(__name__)


# ----- request/response models --------------------------------------------
# MUST be at module level, not nested in create_app(). With `from __future__ import
# annotations` active, FastAPI resolves endpoint annotations as STRINGS via
# get_type_hints() against MODULE globals. Models defined locally inside create_app()
# are invisible to that lookup, so FastAPI can't tell they are BaseModel subclasses
# and wrongly treats them as query params -> 422 {"loc":["query","..."]}.
class ReportRequest(BaseModel):
    topic: str


class MessageRequest(BaseModel):
    """Dual-mode message: can be chat or report request."""
    message: str


class ChatResponse(BaseModel):
    """Response for general chat mode."""
    type: str = "chat"
    content: str


class ReportResponse(BaseModel):
    """Response for report generation mode.

    Every report now runs the unified brief-first flow: `session_id` links to the
    database session so the UI can offer document upload and (auto) comprehensive
    generation without the user picking a mode. `history_id` links the persistent
    history entry so the UI can reflect server-side history.
    """
    type: str = "report"
    job_id: str
    session_id: str = ""
    history_id: str = ""
    intent_reasoning: str = ""


class DatabaseSessionRequest(BaseModel):
    """Start a Database Mode session: kicks off the initial brief report."""
    topic: str


class SettingsUpdate(BaseModel):
    # All optional: the panel sends only what changed. Secrets sent as "" mean
    # 'unchanged'. Unknown fields are ignored by runtime.save_settings.
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    anthropic_api_key: str | None = None
    search_provider: str | None = None
    search_api_key: str | None = None
    verify_mode: str | None = None
    latex_engine: str | None = None


class ConfirmRequest(BaseModel):
    """Human decision at the HITL isolated-claim review checkpoint (workflow F)."""
    decision: str                     # "continue" | "abort"


def _make_gate():
    """Lazy factory for Job.gate: importing the pipeline at module import time would
    pull in the whole report stack; defer it to the first job creation."""
    from core_engine.report.pipeline import ConfirmationGate
    return ConfirmationGate()


@dataclass
class Job:
    id: str
    topic: str
    queue: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)
    status: str = "running"          # running | completed | out_of_scope | blocked | error
    pdf_path: str | None = None
    message: str = ""
    done: asyncio.Event = field(default_factory=asyncio.Event)
    # Database Mode extensions. `extra_sources` are pre-built Source objects injected
    # into the run (user documents + a brief's sources); `skip_search` builds the report
    # from those alone. `on_complete(result)` is an optional callback fired after the run
    # so the caller (e.g. the database session) can snapshot the outcome/sources.
    extra_sources: list | None = None
    skip_search: bool = False
    on_complete: Any | None = None
    # Persistent-history link. When set, _run_job updates this history entry on
    # completion (status, pdf_ready, and job_id -> whichever job just finished, so the
    # history item always downloads the best available PDF for the topic).
    history_id: str | None = None
    # HITL review (workflow F). `gate` is handed to the pipeline run so it can pause at
    # the isolated-claim checkpoint; `pending_confirmation` holds the parsed
    # 'awaiting_confirmation' summary while the run waits for a human decision (None
    # when no review is outstanding — the confirm endpoint answers 409 then).
    gate: Any = field(default_factory=_make_gate)
    pending_confirmation: dict | None = None


_JOBS: dict[str, Job] = {}

# Single-user desktop app: one in-memory conversation for the whole session.
# Imported lazily-safe (conversation.py has no heavy deps).
from core_engine.app.conversation import Conversation

_CONVERSATION = Conversation()


def create_app():
    """Build the FastAPI app. Imported lazily by the shell so importing this module
    (e.g. in tests) does not require fastapi to be installed until it's actually run."""
    logging.basicConfig(level=logging.INFO)

    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    from core_engine import __version__

    # `from __future__ import annotations` (top of file) stringifies every endpoint
    # annotation; FastAPI resolves them via get_type_hints against THIS MODULE's
    # globals. Module-level request models resolve fine, but FastAPI types imported
    # locally here (UploadFile) would not — publish it to module globals so the file
    # upload endpoint's `file: UploadFile` annotation resolves. See the note on the
    # request/response models above for the same class of bug.
    globals().setdefault("UploadFile", UploadFile)

    app = FastAPI(title="Core Engine Report Studio", version=__version__)

    # Loopback session token: the pywebview shell generates a random token per launch
    # and passes it via CE_APP_SESSION_TOKEN. When set, every API request must present
    # it (?token= or X-Session-Token header) except the health probe (the shell polls
    # it before the window exists). Static assets are exempt: <script>/<link> tags
    # can't carry the token, and the attack surface that matters is the state-mutating
    # /api/* surface. When unset (CLI runs, tests) all requests pass — preserving the
    # no-auth desktop default.
    @app.middleware("http")
    async def session_token_guard(request, call_next):
        import os

        token = os.environ.get("CE_APP_SESSION_TOKEN")
        if not token:
            return await call_next(request)
        path = request.url.path
        if not path.startswith("/api/") or path == "/api/health":
            return await call_next(request)
        presented = request.query_params.get("token") or request.headers.get("X-Session-Token")
        if presented != token:
            return JSONResponse({"detail": "invalid session token"}, status_code=401)
        return await call_next(request)

    # Request/response models are defined at MODULE level (see top of file) — they
    # must NOT be nested here or FastAPI mis-resolves them as query params under
    # `from __future__ import annotations`.

    # ----- meta -----------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        from core_engine.config import get_settings
        from core_engine.report.compile import available_engine

        s = get_settings()
        engine_path = available_engine()
        return {
            "ok": True,
            "llm_provider": s.llm_provider,
            "search_provider": s.search_provider,
            "verify_mode": s.verify_mode,
            "latex_engine": s.latex_engine,
            "latex_available": engine_path is not None,
            "frozen": runtime.is_frozen(),
            "output_dir": str(runtime.output_dir()),
        }

    @app.get("/api/settings")
    async def get_settings_endpoint() -> dict[str, Any]:
        return runtime.redacted_settings()

    @app.post("/api/settings")
    async def post_settings(update: SettingsUpdate) -> dict[str, Any]:
        # exclude_unset so a field the panel didn't touch is not sent as null.
        return runtime.save_settings(update.model_dump(exclude_unset=True))

    # ----- persistent history (survives restarts; stored on disk) ---------
    @app.get("/api/history")
    async def get_history() -> dict[str, Any]:
        from core_engine.app import database, history

        items = history.list_entries()
        # Enrich each report entry with the documents the user uploaded for it, so the
        # sidebar can show a "Brief" badge (no uploads) or a hover list of file names.
        # The session store owns the article list; join on session_id.
        for it in items:
            if it.get("kind") != "report":
                continue
            sid = it.get("session_id")
            files: list[str] = []
            if sid:
                try:
                    files = [a.get("filename") or a.get("title") or "document"
                             for a in database.list_articles(sid)]
                except Exception:
                    files = []
            it["files"] = files
            it["file_count"] = len(files)
        return {"items": items}

    @app.delete("/api/history")
    async def delete_history() -> dict[str, Any]:
        from core_engine.app import history
        history.clear()
        return {"ok": True, "items": []}

    # ----- dual-mode message endpoint -------------------------------------
    @app.post("/api/message")
    async def send_message(req: MessageRequest) -> ChatResponse | ReportResponse:
        """Dual-mode endpoint: classifies intent and routes to chat or report generation."""
        import logging as _logging

        from core_engine.app.intent import (
            IntentType,
            LLMNotConfiguredError,
            classify_intent,
            handle_chat,
        )

        _log = _logging.getLogger("core_engine.app.server")
        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required.")

        _log.info("message received: %r", message)

        # Step 1: Classify intent. A missing/invalid provider is a CONFIG problem —
        # return a clear, actionable 400 the UI can show verbatim (no [object Object]).
        try:
            classification = await classify_intent(message)
        except LLMNotConfiguredError as e:
            _log.warning("LLM not configured: %s", e)
            raise HTTPException(
                status_code=400,
                detail=(
                    "No language model is configured. Open Settings and choose a "
                    "provider, then enter your API key (and Base URL for OpenAI-"
                    f"compatible endpoints). Details: {e}"
                ),
            )
        except Exception as e:  # unexpected — surface the real reason, not a generic 500
            _log.exception("intent classification crashed")
            raise HTTPException(status_code=500, detail=f"Intent classification failed: {e}")

        _log.info("intent=%s confidence=%s reasoning=%s",
                  classification.intent.value, classification.confidence,
                  classification.reasoning)

        # Step 2: Route based on intent.
        if classification.intent == IntentType.GENERAL_CHAT:
            _log.info("routing to CHAT")
            from core_engine.app.conversation import maybe_summarize
            from core_engine.config import get_settings
            from core_engine.report.llm import get_llm

            try:
                # Context guard: compress older turns if we're over the word limit,
                # so long sessions never blow the model's context window.
                s = get_settings()
                try:
                    await maybe_summarize(
                        _CONVERSATION, get_llm(),
                        word_limit=s.chat_summarize_word_limit,
                    )
                except LLMNotConfiguredError:
                    raise
                except Exception:
                    _log.warning("summarization skipped", exc_info=True)

                history = _CONVERSATION.transcript()
                response_text = await handle_chat(message, history=history)

                # Record the turn only after a successful reply.
                _CONVERSATION.add_user(message)
                _CONVERSATION.add_assistant(response_text)
                # Persist a chat entry so it survives restarts too.
                try:
                    from core_engine.app import history
                    history.add_entry(kind="chat", title=message, status="completed")
                except Exception:
                    _log.warning("could not record chat history", exc_info=True)
                return ChatResponse(content=response_text)
            except LLMNotConfiguredError as e:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No language model is configured. Open Settings and enter "
                        f"your API key. Details: {e}"
                    ),
                )
            except Exception as e:
                _log.exception("chat handler crashed")
                raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

        # REPORT_GENERATION — unified brief-first flow. Every report starts a database
        # SESSION and runs the brief autonomously. The UI then offers document upload and
        # auto-generates the comprehensive report if the user adds any. No mode toggle.
        # We record the request in the conversation (so a later chat turn has context)
        # and in the PERSISTENT history (so it survives restarts).
        _log.info("routing to REPORT (brief-first session, silent)")
        _CONVERSATION.add_user(message)
        _CONVERSATION.add_assistant(f"[Started generating a report on: {message}]")

        from core_engine.app import database, history

        session_id = database.create_session(message)
        job = Job(id=uuid.uuid4().hex, topic=message)
        hist_id = history.add_entry(
            kind="report", title=message, status="running",
            job_id=job.id, session_id=session_id,
        )
        job.history_id = hist_id
        database.set_history_id(session_id, hist_id)

        def _snapshot(result) -> None:
            srcs = result.report.sources if result.report else []
            database.save_brief(
                session_id, job_id=job.id, status=result.status.value, sources=srcs)

        job.on_complete = _snapshot
        _JOBS[job.id] = job
        asyncio.create_task(_run_job(job))
        return ReportResponse(
            job_id=job.id, session_id=session_id, history_id=hist_id,
            intent_reasoning=classification.reasoning)

    # ----- report runs (legacy endpoint, kept for backwards compatibility) ---
    @app.post("/api/report")
    async def start_report(req: ReportRequest) -> dict[str, str]:
        """Legacy endpoint: directly starts a report without intent classification."""
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required.")
        job = Job(id=uuid.uuid4().hex, topic=topic)
        _JOBS[job.id] = job
        asyncio.create_task(_run_job(job))
        return {"job_id": job.id}

    # ----- Database Mode (two-step: brief -> inject docs -> comprehensive) ----
    @app.post("/api/database/session")
    async def db_create_session(req: DatabaseSessionRequest) -> dict[str, str]:
        """Create a Database Mode session and start its initial BRIEF report.

        The brief runs the normal pipeline. When it finishes, on_complete snapshots the
        brief's sources into the session so the comprehensive step can reuse them."""
        from core_engine.app import database

        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required.")

        session_id = database.create_session(topic)
        job = Job(id=uuid.uuid4().hex, topic=topic)

        def _snapshot(result) -> None:
            # Persist the brief's outcome + the sources it used (verified-claim sources
            # only would be too narrow; snapshot the full pool so the comprehensive run
            # has the same footing plus the user's docs).
            srcs = result.report.sources if result.report else []
            database.save_brief(
                session_id, job_id=job.id, status=result.status.value, sources=srcs)

        job.on_complete = _snapshot
        _JOBS[job.id] = job
        asyncio.create_task(_run_job(job))
        return {"session_id": session_id, "job_id": job.id}

    @app.get("/api/database/{session_id}")
    async def db_get_session(session_id: str) -> dict[str, Any]:
        from core_engine.app import database

        data = database.get_session(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Unknown database session.")
        # Return a light view (no source/article body text).
        return {
            "id": data["id"],
            "topic": data["topic"],
            "brief_status": data.get("brief_status"),
            "brief_job_id": data.get("brief_job_id"),
            "brief_source_count": len(data.get("brief_sources", [])),
            "articles": database.list_articles(session_id),
            "comprehensive_job_id": data.get("comprehensive_job_id"),
        }

    @app.post("/api/database/{session_id}/articles")
    async def db_add_article(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        """Upload one document (pdf/docx/xlsx/csv/txt/md). Parsed to text and stored as a
        USER_PROVIDED source in the session."""
        from core_engine.app import database
        from core_engine.report.documents import DocumentError, parse_document

        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Unknown database session.")

        filename = file.filename or "upload"
        raw = await file.read()
        try:
            title, text = parse_document(filename, raw)
        except DocumentError as e:
            # A parse problem is the user's to fix (wrong type, scanned PDF, etc.) — 400
            # with the actionable message, not a 500.
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read the file: {e}")

        record = database.add_article(
            session_id, title=title, filename=filename, text=text)
        return {"ok": True, "article": record,
                "articles": database.list_articles(session_id)}

    @app.get("/api/database/{session_id}/articles")
    async def db_list_articles(session_id: str) -> dict[str, Any]:
        from core_engine.app import database

        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Unknown database session.")
        return {"articles": database.list_articles(session_id)}

    @app.delete("/api/database/{session_id}/articles/{article_id}")
    async def db_remove_article(session_id: str, article_id: str) -> dict[str, Any]:
        from core_engine.app import database

        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Unknown database session.")
        removed = database.remove_article(session_id, article_id)
        return {"ok": removed, "articles": database.list_articles(session_id)}

    @app.post("/api/database/{session_id}/comprehensive")
    async def db_comprehensive(session_id: str) -> dict[str, str]:
        """Generate the comprehensive report: re-run the pipeline over the brief's
        sources + the user's uploaded documents, with NO new web search."""
        from core_engine.app import database

        data = database.get_session(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Unknown database session.")
        if not data.get("articles"):
            raise HTTPException(
                status_code=400,
                detail="Add at least one document before generating the comprehensive report.",
            )

        sources = database.sources_for_comprehensive(session_id)
        if not sources:
            raise HTTPException(
                status_code=400,
                detail=("No sources available. The brief report may not have finished, "
                        "or produced no sources — try adding more documents."),
            )

        job = Job(
            id=uuid.uuid4().hex, topic=data["topic"],
            extra_sources=sources, skip_search=True,
        )
        # Reuse the session's history entry so the comprehensive result updates the
        # SAME sidebar item (its Download then points at the richer comprehensive PDF).
        job.history_id = data.get("history_id")
        _JOBS[job.id] = job
        database.set_comprehensive_job(session_id, job.id)
        asyncio.create_task(_run_job(job))
        return {"job_id": job.id}

    # ----- HITL review (workflow F) -----------------------------------------
    @app.post("/api/jobs/{job_id}/confirm")
    async def confirm_job(job_id: str, req: ConfirmRequest) -> dict[str, bool]:
        """Deliver the human decision for a job paused at the isolated-claim review
        checkpoint. 404 unknown job; 409 when the job is not waiting for a decision."""
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        if job.pending_confirmation is None:
            raise HTTPException(
                status_code=409,
                detail="This job is not waiting for a confirmation.")
        decision = req.decision.strip().lower()
        if decision not in ("continue", "abort"):
            raise HTTPException(
                status_code=400, detail="decision must be 'continue' or 'abort'.")
        job.gate.decide(decision)
        job.pending_confirmation = None
        return {"ok": True}

    @app.get("/api/report/{job_id}/events")
    async def report_events(job_id: str):
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job.")

        async def event_stream():
            # Replay nothing; stream from now. Each item is one SSE 'message'. The
            # stream terminates ONLY on the explicit final sentinel (`final: true`)
            # that _run_job pushes — NOT on a stage name, because the pipeline also
            # emits an intermediate "completed" progress line that must not close the
            # stream before the client gets the download info.
            #
            # HEARTBEAT: some phases (bulk claim extraction, a slow single verify call)
            # legitimately run longer than the client's inactivity watchdog without
            # producing a stage event. If we simply blocked on queue.get(), the client
            # would wrongly declare the engine "stuck". So we wait with a timeout and, on
            # each idle tick, emit a lightweight heartbeat that keeps the connection warm
            # and resets the client watchdog without rendering a status line.
            HEARTBEAT_S = 10.0
            while True:
                try:
                    item = await asyncio.wait_for(job.queue.get(), timeout=HEARTBEAT_S)
                except TimeoutError:
                    # No real event yet — send a keep-alive tick and keep waiting.
                    yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                    continue
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("final"):
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/report/{job_id}/pdf")
    async def report_pdf(job_id: str):
        # Serve from PERMANENT disk storage first — this survives backend restarts,
        # so a history 'Download' works even after the in-memory job is gone.
        stored = runtime.lookup_pdf(job_id)
        if stored is not None:
            return FileResponse(str(stored), media_type="application/pdf",
                                filename=stored.name)
        # Fall back to the in-memory job (same-session, before persistence completed).
        job = _JOBS.get(job_id)
        if job is not None and job.pdf_path and Path(job.pdf_path).exists():
            filename = Path(job.pdf_path).name
            return FileResponse(job.pdf_path, media_type="application/pdf",
                                filename=filename)
        raise HTTPException(
            status_code=404,
            detail="PDF not found. It may have been cleared; use Re-run to regenerate.",
        )

    # ----- static UI (mounted last so /api/* wins) ------------------------
    static_dir = runtime.bundled_resource("web")
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="web")

    return app


async def _run_job(job: Job) -> None:
    """Execute one pipeline run, pushing progress events onto the job queue."""
    from core_engine.config import get_settings
    from core_engine.report.pipeline import ReportPipeline

    loop = asyncio.get_running_loop()

    def on_progress(stage: str, detail: str) -> None:
        # Called from within the async pipeline (same loop). Use call_soon_threadsafe
        # defensively in case a provider ever hops threads.
        if stage == "awaiting_confirmation":
            # HITL checkpoint (workflow F): the pipeline is now blocked on job.gate.
            # Stash the parsed summary so /api/jobs/{id}/confirm can tell a real
            # pending review (200) from a stale/duplicate confirm (409).
            try:
                job.pending_confirmation = json.loads(detail)
            except Exception:
                job.pending_confirmation = {"claims": [], "raw": detail}
        elif stage == "confirmation_resolved":
            job.pending_confirmation = None
        loop.call_soon_threadsafe(
            job.queue.put_nowait, {"stage": stage, "detail": detail}
        )

    try:
        log.debug("Starting report job: %s for topic: %s", job.id, job.topic)

        # Check settings before creating pipeline
        settings = get_settings()
        log.debug("LLM Provider: %s", settings.llm_provider)
        log.debug("Search Provider: %s", settings.search_provider)
        log.debug("Has Anthropic Key: %s", bool(settings.anthropic_api_key))
        log.debug("Has LLM Key: %s", bool(settings.llm_api_key))
        log.debug("Has Search Key: %s", bool(settings.search_api_key))

        # Rebuild the pipeline per run so the latest saved settings (provider/key)
        # are picked up — get_settings cache is cleared on save.
        log.debug("Creating ReportPipeline...")
        pipeline = ReportPipeline()

        log.debug("Starting pipeline.run()...")
        # Overall run cap — a hard ceiling so the whole job can never hang the UI even
        # if some stage misbehaves beyond its own timeouts. Generous headroom over the
        # verification deadline + assembly + compile time. Doubled base minimum from 300s.
        run_budget = max(600.0, settings.verify_deadline_s + settings.assemble_deadline_s + 180.0)
        result = await asyncio.wait_for(
            pipeline.run(
                job.topic, on_progress=on_progress,
                extra_sources=job.extra_sources, skip_search=job.skip_search,
                confirmation_gate=job.gate,
            ),
            timeout=run_budget)
        # The run is past any review checkpoint now — a confirm after this is a 409.
        job.pending_confirmation = None

        log.debug("Pipeline completed with status: %s", result.status.value)
        job.status = result.status.value
        job.message = result.message
        job.pdf_path = result.pdf_path

        # Database Mode: let the caller snapshot the run outcome (brief sources, etc.).
        if job.on_complete is not None:
            try:
                job.on_complete(result)
            except Exception as e:
                log.warning("job on_complete hook failed: %s", e)

        # Persist the PDF to permanent storage keyed by the job id, so it survives a
        # backend restart and history 'Download' can serve it from disk later.
        if result.pdf_path:
            try:
                stored = runtime.store_pdf(job.id, Path(result.pdf_path), topic=job.topic)
                job.pdf_path = str(stored)
                log.debug("PDF persisted to %s", stored)
            except Exception as e:
                log.warning("could not persist PDF: %s", e)

        # Update the persistent history entry so it survives restarts. The job that just
        # finished (brief OR comprehensive) becomes the one this history item downloads,
        # so the history always points at the latest/best PDF for the topic.
        if job.history_id:
            try:
                from core_engine.app import history
                history.update_entry(
                    job.history_id,
                    status=result.status.value,
                    job_id=job.id,
                    pdf_ready=bool(result.pdf_path),
                )
            except Exception as e:
                log.warning("could not update history: %s", e)

        final = {
            "stage": _terminal_stage(result.status),
            "detail": result.message,
            "status": result.status.value,
            "has_pdf": bool(result.pdf_path),
            # Absolute path where the finished PDF was saved on this machine, so the UI
            # can tell the user exactly where to find it (not just offer a download).
            "pdf_path": str(job.pdf_path) if job.pdf_path else None,
            "tex_path": result.tex_path,
            "final": True,          # sentinel: closes the SSE stream (see event_stream)
        }
        await job.queue.put(final)
    except TimeoutError:
        # The overall run cap fired — always send a terminal event so the UI clears.
        log.error("Job %s exceeded the overall run time limit.", job.id)
        job.status = "error"
        job.message = "The run exceeded the time limit and was stopped."
        _mark_history_error(job)
        await job.queue.put({
            "stage": "error",
            "detail": ("The run took too long and was stopped. Try a narrower topic, "
                       "or check that your LLM/search providers are responding."),
            "status": "error", "has_pdf": False, "final": True,
        })
    except Exception as e:  # never let a crash hang the SSE stream
        import traceback
        error_detail = traceback.format_exc()
        log.error("Job %s failed with exception:\n%s", job.id, error_detail)

        job.status = "error"
        job.message = str(e)
        _mark_history_error(job)
        await job.queue.put({"stage": "error", "detail": f"Unexpected error: {e}",
                             "status": "error", "has_pdf": False, "final": True})
    finally:
        job.pending_confirmation = None
        job.done.set()


def _mark_history_error(job: Job) -> None:
    """Best-effort: flag this job's history entry as failed so the sidebar reflects it."""
    if not job.history_id:
        return
    try:
        from core_engine.app import history
        history.update_entry(job.history_id, status="error", pdf_ready=False)
    except Exception:
        pass


def _terminal_stage(status) -> str:
    from core_engine.report.models import PipelineStatus

    return {
        PipelineStatus.COMPLETED: "completed",
        PipelineStatus.OUT_OF_SCOPE: "out_of_scope",
        PipelineStatus.BLOCKED: "blocked",
        PipelineStatus.ERROR: "error",
    }.get(status, "error")


def run_server(host: str = "127.0.0.1", port: int = 8765, *, log_level: str = "warning"):
    """Blocking entrypoint: start uvicorn on loopback. Called by the pywebview shell
    (on a background thread) and usable directly via `python -m core_engine.app.server`."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    run_server()
