"""Desktop shell — the .exe entrypoint.

Boots the FastAPI server on a background thread bound to loopback, waits until it
answers, then opens a native pywebview window pointed at it. pywebview uses the OS
web renderer (WebView2 on Windows, WebKit on macOS/Linux) so there is no bundled
browser to ship — the whole app is one process.

Flow:
  1. apply persisted user settings (API keys etc.) to the environment.
  2. pick a free loopback port.
  3. start uvicorn in a daemon thread.
  4. poll /api/health until it responds (or time out with a clear error).
  5. create the webview window; block until the user closes it.

Run from source:  python -m core_engine.app.shell
Frozen exe:       ReportStudio.exe   (see packaging/ReportStudio.spec)
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.request

from core_engine.app import runtime

WINDOW_TITLE = "Report Studio — Verified Industry Reports"


def _free_port() -> int:
    """Port to bind. Honours CE_APP_PORT (fixed port, useful for debugging/verifying a
    frozen build); otherwise asks the OS for an unused loopback port so two instances
    don't collide."""
    import os

    override = os.environ.get("CE_APP_PORT")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(host: str, port: int) -> None:
    import uvicorn

    from core_engine.app.server import create_app

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


def _wait_for_health(base: str, timeout: float = 30.0) -> bool:
    """Poll the health endpoint until the server is ready. Returns False on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=1.0) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


def main() -> int:
    # 0. Fix stdout/stderr for frozen GUI apps (they're None in windowed .exe builds).
    #    Redirect to a log file so print() calls and error traces don't crash.
    if sys.stdout is None or sys.stderr is None:
        import tempfile
        from pathlib import Path

        # Use a temp directory fallback if session_dir fails
        try:
            log_dir = runtime.session_dir()
        except Exception:
            log_dir = Path(tempfile.gettempdir()) / "reportstudio"
            log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / "app.log"
        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
        # 长寿命日志句柄：用于重定向 stdout/stderr，存活整个进程生命周期，不能用 with。

        if sys.stdout is None:
            sys.stdout = log_file
        if sys.stderr is None:
            sys.stderr = log_file

    # 1. Persisted settings -> env, BEFORE anything reads config.
    runtime.apply_settings_to_env()

    # 1b. Per-launch session token: the loopback server is only reachable from this
    # machine, but any local process (or a malicious page opened in the same user
    # session via the system browser) could otherwise drive the API. The token is
    # random per launch and passed to the server (same process) via the environment;
    # the web UI receives it in the URL and echoes it on every request.
    import os
    import secrets

    token = secrets.token_urlsafe(32)
    os.environ["CE_APP_SESSION_TOKEN"] = token

    # 2/3. Start the server thread.
    host, port = "127.0.0.1", _free_port()
    base = f"http://{host}:{port}"
    threading.Thread(target=_serve, args=(host, port), daemon=True).start()

    # 4. Wait for readiness so the window never opens on a blank/errored page.
    if not _wait_for_health(base):
        sys.stderr.write("Report Studio: backend failed to start.\n")
        return 1

    # 5. Native window. Imported here so importing this module stays cheap/testable.
    import webview

    webview.create_window(WINDOW_TITLE, f"{base}/?token={token}",
                          width=1180, height=820, min_size=(900, 640))
    # Give the window the SAME "RS" icon as the exe so the taskbar, title bar, and
    # Alt+Tab thumbnail all match. pywebview>=5 accepts icon= on start(); if the file
    # is missing or the platform ignores it, we still start normally.
    icon = _window_icon()
    try:
        if icon:
            webview.start(icon=icon)
        else:
            webview.start()
    except TypeError:
        # Older pywebview without icon= support — start without it rather than crash.
        webview.start()
    return 0


def _window_icon() -> str | None:
    """Absolute path to the bundled 'RS' .ico for the window/taskbar icon, or None.

    Frozen: bundled at the app root (see ReportStudio.spec datas). From source: the
    packaging/ dir. Returns None if not found so the app still runs with a default icon."""
    from pathlib import Path

    candidates = [
        runtime.bundled_resource("reportstudio.ico"),
    ]
    if not runtime.is_frozen():
        # src/core_engine/app/shell.py -> repo root is parents[3]
        candidates.append(
            Path(__file__).resolve().parents[3] / "packaging" / "reportstudio.ico")
    for c in candidates:
        try:
            if Path(c).exists():
                return str(c)
        except Exception:
            continue
    return None


if __name__ == "__main__":
    raise SystemExit(main())
