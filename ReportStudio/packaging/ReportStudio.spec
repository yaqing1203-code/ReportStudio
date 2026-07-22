# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Report Studio — the standalone desktop .exe.

Build:  pyinstaller packaging/ReportStudio.spec        (from the repo root)
Output: dist/ReportStudio/ReportStudio.exe             (onedir; see EXE/COLLECT)

What this bundles so the app is self-contained out-of-the-box:
  - web/                     the chat UI (index.html, styles.css, app.js)
  - report/templates/        the LaTeX template
  - packaging/bin/           OPTIONAL: drop tectonic(.exe) here and it ships inside
                             the exe; compile.py resolves it from sys._MEIPASS/bin
                             first, so PDF generation works with zero user setup.

Notes:
  - onedir (not onefile): faster startup and avoids re-extracting a large bundle to
    temp on every launch. Ship the whole dist/ReportStudio/ folder (or wrap it in an
    installer). Flip `onefile=True` below if you specifically need a single file.
  - hiddenimports cover packages PyInstaller's static analysis can miss: uvicorn's
    per-request imports, the anthropic/openai SDKs (optional providers), and
    pywebview's platform backend.
  - The heavy dormant KG/RAG deps (psycopg, pgvector, torch, …) are deliberately
    NOT collected — the report app does not import them. Keeps the exe lean.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# spec files don't get __file__; PyInstaller injects SPECPATH (this dir).
ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"

onefile = False

# --- application icon: "RS" white-on-orange (packaging/reportstudio.ico) ----------
# Regenerate with: conda run -n py311 python packaging/make_icon.py
# Passed to EXE(icon=...) so the .exe shows it in Explorer and the taskbar. None if
# the file is missing (build still succeeds with the default icon).
_icon = ROOT / "packaging" / "reportstudio.ico"
icon_path = str(_icon) if _icon.exists() else None

# --- data files bundled next to the code (resolved via runtime.bundled_resource) ---
datas = [
    (str(ROOT / "web"), "web"),
    (str(SRC / "core_engine" / "report" / "templates"),
     "core_engine/report/templates"),
]
# Ship the icon as a bundled resource too, so the running window (taskbar / title bar /
# Alt+Tab) can load the SAME .ico the exe uses — resolved via runtime.bundled_resource.
if icon_path:
    datas.append((icon_path, "."))

# --- optional bundled Tectonic binary ------------------------------------------
# If you place tectonic.exe (Windows) or tectonic (macOS/Linux) in packaging/bin/,
# it gets shipped inside the app and used automatically. If absent, the app still
# builds and falls back to any system TeX engine at runtime.
_bin = ROOT / "packaging" / "bin"
binaries = []
if _bin.exists():
    for f in _bin.iterdir():
        if f.is_file():
            binaries.append((str(f), "bin"))

# --- libffi runtime DLL (REQUIRED) ---------------------------------------------
# On a conda Python, _ctypes.pyd depends on libffi (ffi-8.dll), which lives in the
# env's Library/bin — a directory PyInstaller does NOT scan. Without it, importing
# _ctypes fails at runtime, which cascades into click -> uvicorn -> the whole server
# thread dying (the GUI opens but the API never starts). Bundle every ffi*.dll from
# the env next to the exe so _ctypes loads. sys.prefix is the active env root.
import sys as _sys
_libdir = Path(_sys.prefix) / "Library" / "bin"
for _pat in ("ffi*.dll", "libffi*.dll"):
    for _dll in _libdir.glob(_pat):
        binaries.append((str(_dll), "."))

hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("fastapi")
    + collect_submodules("pypdf")
    # python-multipart: FastAPI checks BOTH `python_multipart` and `multipart` (and
    # `multipart.multipart`). Miss any and create_app() raises when the upload route is
    # registered, silently killing the server thread. Collect all submodules of both.
    + collect_submodules("python_multipart")
    + collect_submodules("multipart")
    + collect_submodules("openpyxl")
    + [
        "anthropic",          # optional LLM provider (guarded import at runtime)
        "openai",             # optional OpenAI-compatible provider
        "duckduckgo_search",  # keyless-search fallback (lazy import; static analysis misses it)
        "webview",
        "core_engine.app.server",
        "core_engine.report.pipeline",
        # --- Database Mode: document parsers (lazy-imported in documents.py) ---
        "python_multipart",   # canonical name FastAPI imports for multipart uploads
        "multipart",          # legacy shim name FastAPI also imports
        "multipart.multipart",
        "fitz",               # PyMuPDF PDF fallback
        "docx",               # python-docx (.docx)
        "openpyxl",           # .xlsx spreadsheets
        "core_engine.app.database",
        "core_engine.app.history",    # persistent, disk-backed report/chat history
        "core_engine.report.documents",
    ]
)

block_cipher = None

a = Analysis(
    [str(SRC / "core_engine" / "app" / "shell.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude the dormant data-agent stack + heavy ML deps the report app never imports.
    excludes=[
        "psycopg", "psycopg_pool", "pgvector", "sqlalchemy",
        "torch", "sentence_transformers", "FlagEmbedding", "transformers",
        "tkinter", "matplotlib",
        # Exclude Qt bindings — pywebview uses the OS webview (Edge/Chromium on Windows,
        # WebKit on macOS), not Qt. Some transitive dependency pulls these in; block them.
        "PyQt5", "PyQt6", "PySide2", "PySide6",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if onefile:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
        name="ReportStudio",
        debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
        runtime_tmpdir=None, console=False, disable_windowed_traceback=False,
        icon=icon_path,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="ReportStudio",
        debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
        console=False, disable_windowed_traceback=False,
        icon=icon_path,
    )
    coll = COLLECT(
        exe, a.binaries, a.zipfiles, a.datas,
        strip=False, upx=True, upx_exclude=[], name="ReportStudio",
    )
