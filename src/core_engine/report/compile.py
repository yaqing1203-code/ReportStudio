"""LaTeX -> PDF compilation.

Prefers **Tectonic** — a single self-contained LaTeX binary that bundles the TeX
engine and auto-fetches (then caches) only the packages a document needs. That is
what makes "compile a PDF out-of-the-box" work inside a frozen .exe: we ship one
~15 MB binary instead of a multi-GB TeX distribution, and Tectonic does its own
rerun-until-stable passes (no manual two-pass loop, no latexmk).

Resolution order (first available wins), configurable via CE_LATEX_ENGINE:
  1. tectonic  — bundled, zero user setup. The packaged-exe default.
  2. xelatex / pdflatex — a system TeX install, if the user already has one.
  3. latexmk   — convenience wrapper around a system engine.

Failure modes are explicit, because a missing toolchain is the likeliest problem:
  - no engine at all  -> CompileError with an actionable hint (the pipeline turns
    this into a graceful "PDF skipped, .tex written" result, never a crash).
  - compile error     -> CompileError carrying the tail of the .log for triage.

Compilation runs in a temp dir so aux files (.aux/.log/.out) don't litter output/,
then only the .pdf is copied back next to the .tex.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from core_engine.config import get_settings

log = logging.getLogger(__name__)

# Engines we know how to drive, in preference order when the configured one is
# "auto" or unavailable. Tectonic first — it is the self-contained default.
_ENGINE_PREFERENCE = ("tectonic", "xelatex", "pdflatex", "latexmk")


class CompileError(RuntimeError):
    """Raised when the PDF cannot be produced. Message is user-actionable."""


def _bundled_dir() -> Path | None:
    """When frozen by PyInstaller, binaries we ship live under sys._MEIPASS. We add
    a `bin/` there so a bundled tectonic(.exe) is found without touching system PATH."""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) / "bin" if base else None


def _resolve(engine: str) -> str | None:
    """Return the absolute path to `engine`, checking the bundled bin/ dir first
    (so the packaged Tectonic wins) then the system PATH. None if not found."""
    bundled = _bundled_dir()
    if bundled:
        for name in (engine, f"{engine}.exe"):
            cand = bundled / name
            if cand.exists():
                return str(cand)
    return shutil.which(engine)


def available_engine(preferred: str | None = None) -> str | None:
    """Pick the engine to use. If `preferred` (or CE_LATEX_ENGINE) is a concrete
    engine that resolves, use it; if it's 'auto' or missing, fall back through the
    preference list. Returns the resolved command path, or None if nothing is found."""
    preferred = preferred or get_settings().latex_engine
    order: tuple[str, ...]
    if preferred and preferred != "auto":
        # Try the requested engine first, then the rest as fallback.
        order = (preferred,) + tuple(e for e in _ENGINE_PREFERENCE if e != preferred)
    else:
        order = _ENGINE_PREFERENCE
    for engine in order:
        resolved = _resolve(engine)
        if resolved:
            return resolved
    return None


def engine_available(engine: str | None = None) -> bool:
    """True if SOME usable LaTeX engine is available (bundled or on PATH). Lets the
    pipeline degrade gracefully (write .tex, skip PDF) instead of crashing."""
    return available_engine(engine) is not None


def _is_tectonic(cmd_path: str) -> bool:
    return "tectonic" in Path(cmd_path).name.lower()


def _build_cmd(cmd_path: str, work_tex: Path, out_dir: Path) -> tuple[list[str], int]:
    """Return (argv, passes) for the resolved engine. Tectonic self-reruns, so it
    needs a single invocation; the classic engines need two passes for TOC/refs."""
    if _is_tectonic(cmd_path):
        # Tectonic: keep it hermetic and quiet; it fetches+caches packages itself.
        # Note: --synctex option removed for compatibility with Tectonic 0.16.x
        return (
            [cmd_path, "--outdir", str(out_dir), "--keep-logs", str(work_tex)],
            1,
        )
    if Path(cmd_path).name.lower().startswith("latexmk"):
        return ([cmd_path, "-xelatex", "-interaction=nonstopmode", "-halt-on-error",
                 f"-outdir={out_dir}", str(work_tex)], 1)
    # xelatex / pdflatex
    return (
        [cmd_path, "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={out_dir}", str(work_tex)],
        2,
    )


def compile_pdf(tex_path: Path, *, engine: str | None = None) -> Path:
    """Compile a .tex file to PDF next to it. Returns the PDF path.

    Raises CompileError with actionable detail on any failure.
    """
    s = get_settings()
    tex_path = Path(tex_path)
    if not tex_path.exists():
        raise CompileError(f"tex file not found: {tex_path}")

    cmd_path = available_engine(engine)
    if not cmd_path:
        raise CompileError(
            "No LaTeX engine available. The packaged app ships Tectonic; if you are "
            "running from source, install Tectonic (https://tectonic-typesetting.github.io) "
            "or a TeX distribution (TeX Live / MiKTeX), or set CE_LATEX_ENGINE."
        )

    with tempfile.TemporaryDirectory(prefix="ce_latex_") as tmp:
        tmp_dir = Path(tmp)
        work_tex = tmp_dir / tex_path.name
        work_tex.write_text(tex_path.read_text(encoding="utf-8"), encoding="utf-8")

        argv, passes = _build_cmd(cmd_path, work_tex, tmp_dir)
        # Tectonic caches downloaded packages here; keep it stable across runs so the
        # first compile pays the fetch cost once and later ones are offline+fast.
        env = dict(os.environ)
        env.setdefault("TECTONIC_CACHE_DIR", str(s.output_dir / ".tectonic-cache"))

        last_log = ""
        for i in range(passes):
            try:
                proc = subprocess.run(
                    argv, cwd=tmp_dir, capture_output=True, text=True,
                    timeout=max(120.0, s.scrape_timeout_s * 12), env=env,
                )
            except subprocess.TimeoutExpired:
                raise CompileError(
                    f"{Path(cmd_path).name} timed out. First compile with Tectonic may "
                    f"need to download packages; retry once the cache is warm."
                )
            log_file = tmp_dir / (work_tex.stem + ".log")
            last_log = (log_file.read_text(errors="ignore")
                        if log_file.exists() else (proc.stdout + proc.stderr))
            if proc.returncode != 0:
                raise CompileError(
                    f"{Path(cmd_path).name} failed on pass {i + 1}. Log tail:\n"
                    + "\n".join(last_log.splitlines()[-25:])
                )

        produced = tmp_dir / (work_tex.stem + ".pdf")
        if not produced.exists():
            raise CompileError(
                f"{Path(cmd_path).name} exited cleanly but produced no PDF. Log tail:\n"
                + "\n".join(last_log.splitlines()[-25:])
            )
        final_pdf = tex_path.with_suffix(".pdf")
        shutil.copyfile(produced, final_pdf)
        log.info("compiled %s -> %s (via %s)", tex_path.name, final_pdf.name,
                 Path(cmd_path).name)
        return final_pdf
