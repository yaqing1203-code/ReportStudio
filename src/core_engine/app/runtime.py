"""Runtime paths + user-settings persistence for the packaged desktop app.

Two problems this module solves, both specific to shipping a standalone .exe:

  1. WRITABLE PATHS. In a PyInstaller onefile build, the code lives in a read-only
     temp dir (sys._MEIPASS) and REPO_ROOT does not exist. Output PDFs, the settings
     file, and the Tectonic cache must go somewhere the user can actually write —
     a per-user app-data directory. `user_data_dir()` picks the right place whether
     we're frozen or running from source.

  2. SETTINGS THAT SURVIVE RESTARTS. The GUI Settings panel lets the user choose an
     API protocol (Anthropic / OpenAI-compatible), a base URL, and an API key. Those
     are persisted to settings.json here and re-applied as CE_* env vars on launch,
     BEFORE the first get_settings() call, so the existing pydantic config picks them
     up with zero changes to config.py. The API key is stored in a user-only-readable
     file — see save_settings().

Nothing here imports the pipeline, so it is import-cheap and safe to call at startup.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC
from pathlib import Path

APP_DIR_NAME = "CoreEngineReports"

# Fields the GUI is allowed to set, mapped to the CE_ env var the config reads.
# Keeping this explicit (rather than accepting arbitrary keys) means the frontend
# can never set an unexpected setting by POSTing extra fields.
_SETTINGS_TO_ENV = {
    "llm_provider": "CE_LLM_PROVIDER",
    "llm_model": "CE_LLM_MODEL",
    "llm_base_url": "CE_LLM_BASE_URL",
    "llm_api_key": "CE_LLM_API_KEY",
    "anthropic_api_key": "CE_ANTHROPIC_API_KEY",
    "search_provider": "CE_SEARCH_PROVIDER",
    "search_api_key": "CE_SEARCH_API_KEY",
    "firecrawl_api_key": "CE_FIRECRAWL_API_KEY",
    "jina_api_key": "CE_JINA_API_KEY",
    "search_router_enabled": "CE_SEARCH_ROUTER_ENABLED",
    "verify_mode": "CE_VERIFY_MODE",
    "latex_engine": "CE_LATEX_ENGINE",
    "hitl_on_isolated_core_claim": "CE_HITL_ON_ISOLATED_CORE_CLAIM",
}

# Keys we treat as secret: never returned to the frontend in cleartext, stored in a
# user-only file. The GUI shows whether a key is SET, not the value itself.
_SECRET_KEYS = {"llm_api_key", "anthropic_api_key", "search_api_key",
                "firecrawl_api_key", "jina_api_key"}


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) bundle."""
    return getattr(sys, "frozen", False)


def user_data_dir() -> Path:
    """Per-user, writable app-data directory. Created on first use.

    Windows : %LOCALAPPDATA%\\CoreEngineReports
    macOS   : ~/Library/Application Support/CoreEngineReports
    Linux   : $XDG_DATA_HOME/CoreEngineReports or ~/.local/share/CoreEngineReports
    """
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / APP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir() -> Path:
    """Where generated .tex / .pdf land. User-writable even in a frozen build."""
    d = user_data_dir() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_dir() -> Path:
    """Session-specific directory for logs and temporary files. User-writable."""
    d = user_data_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def generated_pdfs_dir() -> Path:
    """Permanent storage for generated report PDFs, keyed by report id. Survives
    restarts, unlike the in-memory job store."""
    d = user_data_dir() / "generated_pdfs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pdf_index_path() -> Path:
    return generated_pdfs_dir() / "index.json"


def _load_pdf_index() -> dict:
    p = _pdf_index_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def store_pdf(report_id: str, source_pdf: Path, topic: str = "") -> Path:
    """Copy a freshly-compiled PDF into permanent storage as <report_id>.pdf and
    record it in the index. Returns the stored path. Idempotent per report_id."""
    import shutil

    source_pdf = Path(source_pdf)
    dest = generated_pdfs_dir() / f"{report_id}.pdf"
    if source_pdf.resolve() != dest.resolve():
        shutil.copyfile(source_pdf, dest)
    index = _load_pdf_index()
    index[report_id] = {
        "pdf": str(dest),
        "topic": topic,
        "created": _now_iso(),
    }
    _pdf_index_path().write_text(json.dumps(index, indent=2), encoding="utf-8")
    return dest


def lookup_pdf(report_id: str) -> Path | None:
    """Return the stored PDF path for a report id if it exists on disk, else None.
    Checks the index first, then falls back to the conventional <id>.pdf filename."""
    index = _load_pdf_index()
    entry = index.get(report_id)
    if entry:
        p = Path(entry["pdf"])
        if p.exists():
            return p
    # Fallback: the file may exist even if the index was lost.
    cand = generated_pdfs_dir() / f"{report_id}.pdf"
    return cand if cand.exists() else None


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now(UTC).isoformat()


def bundled_resource(rel: str) -> Path:
    """Resolve a path to a resource that was bundled into the .exe (templates, the
    web UI, the tectonic binary). Works both frozen (sys._MEIPASS) and from source."""
    if is_frozen():
        base = Path(sys._MEIPASS)
    else:
        # src/core_engine/app/runtime.py -> repo root is parents[3]
        base = Path(__file__).resolve().parents[3]
    return base / rel


def _settings_path() -> Path:
    return user_data_dir() / "settings.json"


def load_settings() -> dict:
    """Read persisted GUI settings (or {} if none yet). Secrets ARE included here —
    this is the trusted server side; use redacted_settings() for anything sent to UI."""
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def redacted_settings() -> dict:
    """Settings safe to send to the frontend: secret values replaced by a boolean
    '<key>_set' flag so the UI can show 'key configured' without ever holding it."""
    data = load_settings()
    out: dict = {}
    for k, v in data.items():
        if k in _SECRET_KEYS:
            out[f"{k}_set"] = bool(v)
        else:
            out[k] = v
    return out


def save_settings(update: dict) -> dict:
    """Merge + persist GUI settings, then apply them to the environment so the next
    get_settings() sees them. Only whitelisted keys are accepted. A secret sent as an
    empty string is treated as 'leave unchanged' (the UI sends '' when the user didn't
    retype the key); send an explicit null to clear it.

    Returns the redacted settings for the UI.
    """
    current = load_settings()
    for key, value in update.items():
        if key not in _SETTINGS_TO_ENV:
            continue  # ignore unknown keys — the frontend cannot set arbitrary config
        if key in _SECRET_KEYS and value == "":
            continue  # blank secret = keep existing
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value

    p = _settings_path()
    p.write_text(json.dumps(current, indent=2), encoding="utf-8")
    _restrict_permissions(p)
    apply_settings_to_env(current)
    return redacted_settings()


def apply_settings_to_env(data: dict | None = None) -> None:
    """Push persisted settings into os.environ as CE_* vars, then clear the cached
    Settings so they take effect. Call once at startup and after every save."""
    data = data if data is not None else load_settings()
    for key, env in _SETTINGS_TO_ENV.items():
        if key in data and data[key] is not None and data[key] != "":
            os.environ[env] = str(data[key])
    # Always route output to the user-writable dir in the packaged app.
    os.environ.setdefault("CE_OUTPUT_DIR", str(output_dir()))

    # In a frozen build REPO_ROOT (used by the default latex_template_dir) does not
    # exist — the templates were bundled under _MEIPASS. Point config at the bundled
    # copy so the renderer finds report.tex.j2. Harmless from source (path matches).
    if is_frozen():
        os.environ.setdefault(
            "CE_LATEX_TEMPLATE_DIR",
            str(bundled_resource("core_engine/report/templates")),
        )

    # Drop the lru_cache on get_settings so the new env is read next call.
    try:
        from core_engine.config import get_settings

        get_settings.cache_clear()
    except Exception:
        pass


def _restrict_permissions(path: Path) -> None:
    """Best-effort: make the settings file (which holds API keys) user-only readable.
    On POSIX we chmod 600; on Windows the per-user LOCALAPPDATA dir is already ACL'd
    to the user, so we skip (a full icacls lockdown is overkill here)."""
    if not sys.platform.startswith("win"):
        try:
            path.chmod(0o600)
        except OSError:
            pass
