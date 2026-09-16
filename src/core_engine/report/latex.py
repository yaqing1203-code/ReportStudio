"""LaTeX rendering — verified ReportData -> .tex string.

Two things make this safe and robust:

  1. ESCAPING. Scraped source text is UNTRUSTED. If a source title contains '$',
     '\\', '&', '%', or '{', naive interpolation either breaks compilation or (worse)
     injects LaTeX commands. The `tex` filter escapes every special character, so
     the renderer cannot be made to execute arbitrary LaTeX from page content.

  2. CUSTOM JINJA DELIMITERS. LaTeX is full of { } and %, which collide with
     Jinja's default {{ }} / {% %}. We use << >> and <% %> instead (see template),
     so the .tex template stays readable and valid LaTeX on its own.

The renderer only ever receives verified ReportData (the pipeline enforces this),
so it structurally cannot typeset an unverified claim.
"""
from __future__ import annotations

import re
from pathlib import Path

from core_engine.config import get_settings
from core_engine.report.models import ReportData

# Order matters: backslash first, or we double-escape the escapes.
_TEX_REPLACEMENTS = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]


def tex_escape(value: object) -> str:
    """Escape a value for safe inclusion in LaTeX. Applied to ALL interpolated
    strings via the `tex` Jinja filter."""
    s = str(value)
    for target, repl in _TEX_REPLACEMENTS:
        s = s.replace(target, repl)
    # Collapse control chars that would confuse the compiler.
    s = s.replace("\r", " ").replace("\t", " ")
    return s


def _truncate(value: str, length: int = 60) -> str:
    s = str(value)
    return s if len(s) <= length else s[: length - 1].rstrip() + "…"


def _make_env():
    """Build the Jinja environment with LaTeX-safe delimiters. Imported lazily so
    the package works without Jinja installed until you actually render."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    s = get_settings()
    env = Environment(
        loader=FileSystemLoader(str(s.latex_template_dir)),
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
        autoescape=select_autoescape(enabled_extensions=(), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["tex"] = tex_escape
    env.filters["truncate"] = _truncate
    return env


def render_tex(report: ReportData, *, template: str = "report.tex.j2") -> str:
    """Render verified ReportData to a LaTeX source string."""
    env = _make_env()
    tmpl = env.get_template(template)
    s = get_settings()
    bib = report.bibliography()
    return tmpl.render(
        title=report.title,
        abstract=report.abstract,
        generated_at=report.generated_at,
        sections=report.sections,
        rumors=report.rumors,
        sources=bib,
        source_count=len(bib),
        min_sources=s.min_sources_per_claim,
        verify_rounds=s.verify_rounds,
        verify_mode=s.verify_mode,
    )


def write_tex(report: ReportData, out_dir: Path | None = None) -> Path:
    """Render and write the .tex file, returning its path."""
    s = get_settings()
    out_dir = out_dir or s.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(report.topic)
    tex_path = out_dir / f"{slug}.tex"
    tex_path.write_text(render_tex(report), encoding="utf-8")
    return tex_path


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "report")[:60]
