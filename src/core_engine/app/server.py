"""FastAPI backend for the desktop app.

The pywebview shell (shell.py) loads a local web UI (static/index.html) that talks
to THIS server over http://127.0.0.1:<port>. The shell supplies a per-launch token.
Loopback host/origin checks also protect direct development launches against
cross-site browser requests.

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
import hmac
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from core_engine.app import runtime
from core_engine.app.writing import NOTICE as WRITING_NOTICE
from core_engine.app.writing import WritingOptions
from core_engine.report.generation import GenerationDepth

log = logging.getLogger(__name__)


# ----- request/response models --------------------------------------------
# MUST be at module level, not nested in create_app(). With `from __future__ import
# annotations` active, FastAPI resolves endpoint annotations as STRINGS via
# get_type_hints() against MODULE globals. Models defined locally inside create_app()
# are invisible to that lookup, so FastAPI can't tell they are BaseModel subclasses
# and wrongly treats them as query params -> 422 {"loc":["query","..."]}.
class OutputSelection(BaseModel):
    template_id: str | None = Field(default=None, max_length=128)
    template_fields: dict[str, str] = Field(default_factory=dict, max_length=30)
    depth: GenerationDepth = "standard"


class ReportRequest(OutputSelection):
    topic: str


class MessageRequest(OutputSelection):
    """Three functional modes, with optional automatic routing."""
    message: str
    mode: Literal["auto", "chat", "report", "writing"] = "auto"
    writing_options: WritingOptions = Field(default_factory=WritingOptions)


class WritingResponse(BaseModel):
    type: Literal["writing"] = "writing"
    job_id: str
    history_id: str
    notice: str = WRITING_NOTICE
    depth: GenerationDepth = "standard"


class ChatResponse(BaseModel):
    """Response for general chat mode."""
    type: Literal["chat"] = "chat"
    content: str


class ReportResponse(BaseModel):
    """Response for report generation mode.

    Every report now runs the unified brief-first flow: `session_id` links to the
    database session so the UI can offer document upload and (auto) comprehensive
    generation without the user picking a mode. `history_id` links the persistent
    history entry so the UI can reflect server-side history.
    """
    type: Literal["report"] = "report"
    job_id: str
    session_id: str = ""
    history_id: str = ""
    intent_reasoning: str = ""
    depth: GenerationDepth = "standard"


class DatabaseSessionRequest(OutputSelection):
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
    openalex_api_key: str | None = None
    openalex_search_enabled: bool | None = None
    verify_mode: str | None = None
    latex_engine: Literal["auto", "tectonic", "xelatex", "pdflatex", "latexmk"] | None = None


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
    queue: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=512))
    events: deque[tuple[int, dict]] = field(default_factory=lambda: deque(maxlen=512))
    subscribers: set[asyncio.Queue[tuple[int, dict]]] = field(default_factory=set)
    event_sequence: int = 0
    terminal_event: dict | None = None
    status: str = "running"          # running | completed | out_of_scope | blocked | error
    pdf_path: str | None = None
    evidence_path: str | None = None
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
    output_template: tuple[dict, bytes] | None = None
    template_fields: dict[str, str] = field(default_factory=dict)
    docx_path: str | None = None
    export_info: dict = field(default_factory=dict)
    depth: GenerationDepth = "standard"

    def publish(self, item: dict) -> None:
        """Retain a bounded replay log and broadcast independently to each viewer."""
        self.event_sequence += 1
        record = (self.event_sequence, item)
        self.events.append(record)
        if item.get("final"):
            self.terminal_event = item
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(item)
        for subscriber in tuple(self.subscribers):
            if subscriber.full():
                subscriber.get_nowait()
            subscriber.put_nowait(record)


_JOBS: dict[str, Job] = {}

# Single-user desktop app: one in-memory conversation for the whole session.
# Imported lazily-safe (conversation.py has no heavy deps).
from core_engine.app.conversation import Conversation

_CONVERSATION = Conversation()


def create_app():
    """Build the FastAPI app. Imported lazily by the shell so importing this module
    (e.g. in tests) does not require fastapi to be installed until it's actually run."""
    logging.basicConfig(level=logging.INFO)

    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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
    globals().setdefault("Request", Request)

    app = FastAPI(title="Core Engine Report Studio", version=__version__)
    from core_engine.app.middleware import UploadLimitMiddleware
    app.add_middleware(UploadLimitMiddleware)

    def resolve_template(req: OutputSelection):
        from core_engine.app.output_templates import TemplateError, selection
        try:
            return selection(req.template_id, req.template_fields)
        except TemplateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

        host = request.url.hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return JSONResponse({"detail": "invalid loopback host"}, status_code=403)
        path = request.url.path
        if path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                    same_origin = (parsed.scheme == request.url.scheme
                                   and parsed.hostname == host
                                   and (parsed.port or (443 if parsed.scheme == "https" else 80))
                                   == (request.url.port or (443 if request.url.scheme == "https" else 80)))
                except ValueError:
                    same_origin = False
                if not same_origin:
                    return JSONResponse({"detail": "cross-origin API request rejected"}, status_code=403)
            token = os.environ.get("CE_APP_SESSION_TOKEN")
            if token and path != "/api/health":
                presented = request.headers.get("X-Session-Token") or request.query_params.get("token", "")
                if not hmac.compare_digest(presented.encode(), token.encode()):
                    return JSONResponse({"detail": "invalid session token"}, status_code=401)
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if path.startswith("/api/") or path in {"/", "/index.html"}:
            response.headers["Cache-Control"] = "no-store"
        return response

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
        from core_engine.config import get_settings

        data = runtime.redacted_settings()
        settings = get_settings()
        data["openalex_search_enabled"] = settings.openalex_search_enabled
        data["openalex_api_key_set"] = bool(settings.openalex_api_key)
        return data

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
            if (it.get("kind") == "writing" and it.get("status") == "running"
                    and it.get("job_id") not in _JOBS):
                # A restart loses active tasks, while completed articles live on
                # disk. Never leave an interrupted draft labelled "generating".
                history.update_entry(it["id"], status="error", writing_ready=False)
                it.update(status="error", writing_ready=False)
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

    # Output templates are separate from research documents: never index them as evidence.
    @app.get("/api/templates")
    async def templates_list():
        from core_engine.app.output_templates import list_templates
        return {"items": await asyncio.to_thread(list_templates)}

    @app.post("/api/templates")
    async def template_upload(file: UploadFile = File(...)):
        from core_engine.app.output_templates import MAX_TEMPLATE_BYTES, TemplateError, add_template
        try:
            raw = await file.read(MAX_TEMPLATE_BYTES + 1)
            return await asyncio.to_thread(add_template, file.filename or "template.docx", raw)
        except TemplateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await file.close()

    @app.get("/api/templates/example.docx")
    async def template_example():
        import io

        from core_engine.app.output_templates import example_template
        return StreamingResponse(io.BytesIO(await asyncio.to_thread(example_template)),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="report-template.docx"'})

    @app.delete("/api/templates/{template_id}")
    async def template_delete(template_id: str):
        from core_engine.app.output_templates import TemplateError, delete_template
        try:
            delete_template(template_id)
        except TemplateError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True}

    @app.get("/api/documents/{job_id}/{extension}")
    async def exported_document(job_id: str, extension: str):
        import re

        from core_engine.app import history
        from core_engine.app.output_templates import TemplateError, artifact_path
        from core_engine.app.writing import load_article
        try:
            path = artifact_path(job_id, extension)
        except (TemplateError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="导出文件不存在。") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="导出文件不存在。")
        article = load_article(job_id)
        title = article.get("title") if article else next((e.get("title") for e in history.list_entries() if e.get("job_id") == job_id), "模板文档")
        title = re.sub(r'[\x00-\x1f<>:"/\\|?*]', " ", str(title or "模板文档")).strip(" .")[:100] or "模板文档"
        return FileResponse(str(path), media_type="application/pdf" if extension == "pdf" else
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename=title + "." + extension)

    @app.delete("/api/history")
    async def delete_history() -> dict[str, Any]:
        from core_engine.app import history
        history.clear()
        return {"ok": True, "items": []}

    # ----- three-mode message endpoint ------------------------------------
    @app.post("/api/message")
    async def send_message(req: MessageRequest) -> ChatResponse | ReportResponse | WritingResponse:
        """Route explicitly or classify chat, research report and complete writing."""
        import logging as _logging

        from core_engine.app.intent import (
            IntentClassification,
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
            if req.mode == "auto":
                classification = await classify_intent(message)
            else:
                from core_engine.report.llm import get_llm

                try:
                    get_llm()  # validate configuration without a classifier request
                except RuntimeError as exc:
                    raise LLMNotConfiguredError(str(exc)) from exc
                classification = IntentClassification(intent={"chat": IntentType.GENERAL_CHAT,
                    "report": IntentType.REPORT_GENERATION, "writing": IntentType.WRITING}[req.mode],
                    reasoning="Explicit mode selection")
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
        if classification.intent == IntentType.WRITING:
            from core_engine.app import history
            from core_engine.app.writing import load_article

            if len(message) > 12000:
                raise HTTPException(status_code=400, detail="写作要求过长，请控制在 12000 字符以内。")
            previous = None
            if req.writing_options.previous_article_id:
                previous = load_article(req.writing_options.previous_article_id)
                if previous is None:
                    raise HTTPException(status_code=404, detail="上一稿不存在，请从历史记录打开文章或重新粘贴原文。")
            # Top-level selection is canonical. Legacy clients revising an article
            # without a selection retain that article's depth instead of silently
            # switching a deep draft to standard; an explicit nested option still
            # works for clients that only know WritingOptions.
            depth = req.depth
            if "depth" not in req.model_fields_set:
                if "depth" in req.writing_options.model_fields_set:
                    depth = req.writing_options.depth
                elif previous and previous.get("depth") in {"standard", "deep"}:
                    depth = previous["depth"]
            elif ("depth" in req.writing_options.model_fields_set
                  and req.writing_options.depth != depth):
                raise HTTPException(status_code=400, detail="内容深度设置不一致，请只在主选项中选择简洁或深入。")
            job = Job(id=uuid.uuid4().hex, topic=message, depth=depth)
            job.output_template = resolve_template(req)
            job.template_fields = req.template_fields
            job.history_id = history.add_entry(kind="writing", title=message[:160], status="running", job_id=job.id, depth=depth)
            _JOBS[job.id] = job
            options = req.writing_options.model_copy(update={"depth": depth})
            asyncio.create_task(_run_writing_job(job, options, previous))
            return WritingResponse(job_id=job.id, history_id=job.history_id, depth=depth)
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

                transcript = _CONVERSATION.transcript()
                response_text = await handle_chat(message, history=transcript)

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
        template = resolve_template(req)
        _CONVERSATION.add_user(message)
        _CONVERSATION.add_assistant(f"[Started generating a report on: {message}]")

        from core_engine.app import database, history

        session_id = database.create_session(message, depth=req.depth)
        database.set_output_template(session_id, req.template_id, req.template_fields)
        job = Job(id=uuid.uuid4().hex, topic=message, output_template=template, template_fields=req.template_fields, depth=req.depth)
        hist_id = history.add_entry(
            kind="report", title=message, status="running",
            job_id=job.id, session_id=session_id, depth=req.depth,
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
            intent_reasoning=classification.reasoning, depth=req.depth)

    # ----- report runs (legacy endpoint, kept for backwards compatibility) ---
    @app.post("/api/report")
    async def start_report(req: ReportRequest) -> dict[str, str]:
        """Legacy endpoint: directly starts a report without intent classification."""
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required.")
        job = Job(id=uuid.uuid4().hex, topic=topic, depth=req.depth)
        job.output_template = resolve_template(req)
        job.template_fields = req.template_fields
        _JOBS[job.id] = job
        asyncio.create_task(_run_job(job))
        return {"job_id": job.id, "depth": job.depth}

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

        template = resolve_template(req)
        session_id = database.create_session(topic, depth=req.depth)
        database.set_output_template(session_id, req.template_id, req.template_fields)
        job = Job(id=uuid.uuid4().hex, topic=topic, output_template=template, template_fields=req.template_fields, depth=req.depth)

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
        return {"session_id": session_id, "job_id": job.id, "depth": job.depth}

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
            "depth": data.get("depth", "standard"),
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
        from core_engine.report.documents import MAX_FILE_BYTES, DocumentError, parse_document

        if database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Unknown database session.")

        filename = file.filename or "upload"
        raw = await file.read(MAX_FILE_BYTES + 1)
        try:
            if len(raw) > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds the 25 MB upload limit.")
            title, text = await asyncio.to_thread(parse_document, filename, raw)
        except DocumentError as e:
            # A parse problem is the user's to fix (wrong type, scanned PDF, etc.) — 400
            # with the actionable message, not a 500.
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Could not read the file: {e}")
        finally:
            await file.close()

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
    async def db_comprehensive(session_id: str, req: OutputSelection | None = None) -> dict[str, str]:
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
            depth=req.depth if req is not None and "depth" in req.model_fields_set else data.get("depth", "standard"),
        )
        # A depth-only request patches the session instead of discarding its
        # template and cover fields. Explicit template_id=None still clears it.
        selection_data = {"template_id": data.get("template_id"),
                          "template_fields": data.get("template_fields", {}), "depth": job.depth}
        if req is not None:
            for key in ("template_id", "template_fields"):
                if key in req.model_fields_set:
                    selection_data[key] = getattr(req, key)
            if ("template_id" in req.model_fields_set and req.template_id != data.get("template_id")
                    and "template_fields" not in req.model_fields_set):
                selection_data["template_fields"] = {}
        req = OutputSelection(**selection_data)
        job.output_template = resolve_template(req)
        job.template_fields = req.template_fields
        database.set_output_template(session_id, req.template_id, req.template_fields)
        database.set_depth(session_id, job.depth)
        # Reuse the session's history entry so the comprehensive result updates the
        # SAME sidebar item (its Download then points at the richer comprehensive PDF).
        job.history_id = data.get("history_id")
        _JOBS[job.id] = job
        database.set_comprehensive_job(session_id, job.id)
        asyncio.create_task(_run_job(job))
        return {"job_id": job.id, "depth": job.depth}

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
    async def report_events(job_id: str, request: Request):
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job.")

        try:
            last_id = max(0, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            last_id = 0

        async def event_stream():
            subscriber = asyncio.Queue(maxsize=512)
            replay = list(job.events)
            job.subscribers.add(subscriber)
            try:
                for sequence, item in replay:
                    if sequence > last_id:
                        yield f"id: {sequence}\ndata: {json.dumps(item)}\n\n"
                        if item.get("final"):
                            return
                # Always close a completed stream, even if the client already saw
                # its terminal ID or sends a future/invalid resume position.
                if job.terminal_event is not None:
                    yield f"data: {json.dumps(job.terminal_event)}\n\n"
                    return
                while True:
                    try:
                        sequence, item = await asyncio.wait_for(subscriber.get(), timeout=10.0)
                    except TimeoutError:
                        yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                        continue
                    yield f"id: {sequence}\ndata: {json.dumps(item)}\n\n"
                    if item.get("final"):
                        return
            finally:
                job.subscribers.discard(subscriber)

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

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

    @app.get("/api/report/{job_id}/evidence")
    async def report_evidence(job_id: str):
        path = runtime.lookup_evidence(job_id)
        if path is None:
            job = _JOBS.get(job_id)
            path = Path(job.evidence_path) if job and job.evidence_path else None
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Evidence audit not available.")
        return FileResponse(str(path), media_type="application/json", filename=path.name)

    @app.get("/api/writing/{article_id}")
    async def writing_article(article_id: str):
        from core_engine.app.writing import load_article

        article = load_article(article_id)
        if article is None:
            raise HTTPException(status_code=404, detail="文章不存在或已不可用。")
        return article

    @app.get("/api/writing/{article_id}/download")
    async def writing_download(article_id: str):
        from core_engine.app.writing import article_text_path

        path = article_text_path(article_id)
        if path is None:
            raise HTTPException(status_code=404, detail="文章文件不可用。")
        return FileResponse(str(path), media_type="text/plain; charset=utf-8", filename="成文文章.txt")

    # ----- static UI (mounted last so /api/* wins) ------------------------
    static_dir = runtime.bundled_resource("web")
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="web")

    return app


async def _run_writing_job(job: Job, options: WritingOptions, previous: dict | None) -> None:
    from core_engine.app import history
    from core_engine.app.writing import compose, save_article
    from core_engine.report.llm import get_llm

    def progress(stage, detail):
        job.publish({"stage": stage, "detail": detail})

    try:
        compose_args: dict[str, Any] = {"previous": previous, "on_progress": progress}
        if job.output_template:
            compose_args["template_info"] = job.output_template[0]
        article = await asyncio.wait_for(compose(job.topic, options, get_llm(), **compose_args), timeout=600)
        article["depth"] = job.depth
        if job.output_template:
            from core_engine.app.output_templates import article_payload, export_document
            article["template_sections"] = job.output_template[0]["sections"]
            progress("template_export_start", "按 Word 模板填充并导出文件…")
            job.export_info = await asyncio.to_thread(export_document, job.id, job.output_template,
                article_payload(article), job.template_fields)
            article["template"] = {**job.export_info, "fields": job.template_fields}
            progress("template_export_done", "模板 Word 已生成。")
        await asyncio.to_thread(save_article, job.id, article)
        if job.history_id:
            history.update_entry(job.history_id, status="completed", title=article["title"], writing_ready=True,
                                 factual_verification="not_performed", docx_ready=job.export_info.get("has_docx", False),
                                 template_name=job.export_info.get("template_name"), depth=job.depth)
        job.status = "completed"
        job.publish({"stage": "completed", "status": "completed", "kind": "writing", "article_id": job.id,
                     "detail": "文章已完成；材料作为写作前提采用，未作事实核验。", "depth": job.depth, "final": True})
    except Exception as exc:
        log.exception("Composition job failed")
        job.status = "error"
        _mark_history_error(job)
        job.publish({"stage": "error", "status": "error", "kind": "writing", "final": True,
                     "depth": job.depth,
                     "detail": "成文未完成，请检查模型连接或缩短要求后重试。" +
                               (f" {exc}" if isinstance(exc, ValueError) else "")})
    finally:
        job.done.set()


async def _run_job(job: Job) -> None:
    """Apply a task-local profile, shared by nested model/search/audit components."""
    from core_engine.config import get_settings, settings_for_run
    from core_engine.report.generation import report_profile

    try:
        with settings_for_run(report_profile(get_settings(), job.depth)):
            await _run_report_job(job)
    except Exception:
        # Initialization may fail before the inner worker's guarded block.
        log.exception("Could not initialize report generation")
        if not job.done.is_set():
            job.status = "error"
            job.message = "无法初始化生成参数，请检查设置后重试。"
            _mark_history_error(job)
            job.publish({"stage": "error", "status": "error", "detail": job.message,
                         "depth": job.depth, "has_pdf": False, "final": True})
            job.done.set()


async def _run_report_job(job: Job) -> None:
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
        item = {"stage": stage, "detail": detail}
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            job.publish(item)
        else:
            loop.call_soon_threadsafe(job.publish, item)

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
        run_budget = max(600.0, settings.verify_deadline_s + settings.assemble_deadline_s
                         + settings.grounding_deadline_s + 180.0
                         + (settings.research_deadline_s + settings.research_plan_timeout_s if settings.research_enabled else 0)
                         + (settings.reasoning_deadline_s if settings.reasoning_enabled else 0))
        render_options: dict[str, Any] = {"compile_to_pdf": False} if job.output_template else {}
        result = await asyncio.wait_for(
            pipeline.run(
                job.topic, on_progress=on_progress,
                extra_sources=job.extra_sources, skip_search=job.skip_search,
                confirmation_gate=job.gate,
                **render_options,
            ),
            timeout=run_budget)
        # The run is past any review checkpoint now — a confirm after this is a 409.
        job.pending_confirmation = None

        if job.output_template and result.ok and result.report is not None:
            from core_engine.app.output_templates import (
                artifact_path,
                export_document,
                report_payload,
            )
            on_progress("template_export_start", "按 Word 模板排版已审核报告…")
            job.export_info = await asyncio.to_thread(export_document, job.id, job.output_template,
                report_payload(result.report, job.output_template[0]), job.template_fields)
            job.docx_path = str(artifact_path(job.id, "docx"))
            result.pdf_path = str(artifact_path(job.id, "pdf")) if job.export_info["has_pdf"] else None
            if job.export_info["warnings"]:
                result.message += " " + " ".join(job.export_info["warnings"])
            on_progress("template_export_done", "模板文件已生成。")

        log.debug("Pipeline completed with status: %s", result.status.value)
        job.status = result.status.value
        job.message = result.message
        job.pdf_path = result.pdf_path
        job.evidence_path = result.evidence_path
        if result.evidence_path:
            try:
                job.evidence_path = str(runtime.store_evidence(job.id, Path(result.evidence_path)))
            except OSError:
                log.warning("could not persist evidence audit", exc_info=True)

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
                    evidence_ready=bool(job.evidence_path),
                    quality_status=result.report.quality_status if result.report else "not_audited",
                    docx_ready=bool(job.docx_path), template_name=job.export_info.get("template_name"),
                    depth=job.depth,
                )
            except Exception as e:
                log.warning("could not update history: %s", e)

        final = {
            "stage": _terminal_stage(result.status),
            "depth": job.depth,
            "detail": result.message,
            "status": result.status.value,
            "has_pdf": bool(result.pdf_path),
            "has_evidence": bool(job.evidence_path),
            "has_docx": bool(job.docx_path),
            "docx_path": job.docx_path,
            "template": job.export_info,
            "quality_status": result.report.quality_status if result.report else "not_audited",
            "grounding_audit": result.report.grounding_audit if result.report else {},
            "answerability": result.answerability,
            "research_audit": result.research_audit,
            "analysis_audit": result.report.analysis_audit if result.report else {},
            "source_quality": result.report.source_quality if result.report else {},
            "evidence_policy": result.evidence_policy,
            # Absolute path where the finished PDF was saved on this machine, so the UI
            # can tell the user exactly where to find it (not just offer a download).
            "pdf_path": str(job.pdf_path) if job.pdf_path else None,
            "tex_path": result.tex_path,
            "final": True,          # sentinel: closes the SSE stream (see event_stream)
        }
        job.publish(final)
    except TimeoutError:
        # The overall run cap fired — always send a terminal event so the UI clears.
        log.error("Job %s exceeded the overall run time limit.", job.id)
        job.status = "error"
        job.message = "The run exceeded the time limit and was stopped."
        _mark_history_error(job)
        job.publish({
            "stage": "error",
            "detail": ("The run took too long and was stopped. Try a narrower topic, "
                       "or check that your LLM/search providers are responding."),
            "status": "error", "has_pdf": False, "final": True, "depth": job.depth,
        })
    except Exception as e:  # never let a crash hang the SSE stream
        import traceback
        error_detail = traceback.format_exc()
        log.error("Job %s failed with exception:\n%s", job.id, error_detail)

        job.status = "error"
        job.message = str(e)
        _mark_history_error(job)
        job.publish({"stage": "error", "detail": f"Unexpected error: {e}",
                             "status": "error", "has_pdf": False, "final": True, "depth": job.depth})
    finally:
        job.pending_confirmation = None
        job.done.set()


def _mark_history_error(job: Job) -> None:
    """Best-effort: flag this job's history entry as failed so the sidebar reflects it."""
    if not job.history_id:
        return
    try:
        from core_engine.app import history
        history.update_entry(job.history_id, status="error", job_id=job.id, depth=job.depth,
                             pdf_ready=bool(job.pdf_path), evidence_ready=bool(job.evidence_path),
                             docx_ready=bool(job.docx_path), writing_ready=False)
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
