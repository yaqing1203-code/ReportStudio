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
from urllib.parse import quote
from uuid import uuid4

from core_engine.config import Settings, get_settings
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
    # One pass: replacement commands' braces must not themselves be escaped.
    replacements = dict(_TEX_REPLACEMENTS)
    s = "".join(replacements.get(char, char) for char in str(value))
    # Collapse control chars that would confuse the compiler.
    s = "".join(char if char == "\n" or ord(char) >= 32 else " " for char in s)
    return s


def tex_url(value: object) -> str:
    """Encode URL delimiters that could close a \\url argument or inject commands."""
    return quote(str(value), safe=":/?#[]@!$&'()*+,;=%~_-.")


def tex_quote(value: object) -> str:
    """Typeset PDF extraction artifacts without clipping or changing source characters."""
    long_word = r"[A-Za-z0-9\ufb00-\ufb06()./_-]{24,}"
    parts = re.split("(" + long_word + ")", str(value))
    return "".join(r"\allowbreak{}".join(tex_escape(part[i:i + 12]) for i in range(0, len(part), 12))
                   if re.fullmatch(long_word, part) else tex_escape(part) for part in parts)


def _truncate(value: str, length: int = 60) -> str:
    s = str(value)
    return s if len(s) <= length else s[: length - 1].rstrip() + "…"


def _make_env(settings: Settings | None = None):
    """Build the Jinja environment with LaTeX-safe delimiters. Imported lazily so
    the package works without Jinja installed until you actually render."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    s = settings or get_settings()
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
    env.filters["tex_url"] = tex_url
    env.filters["tex_quote"] = tex_quote
    env.filters["truncate"] = _truncate
    return env


def render_tex(report: ReportData, *, template: str = "report.tex.j2",
               settings: Settings | None = None) -> str:
    """Render verified ReportData to a LaTeX source string."""
    env = _make_env(settings)
    tmpl = env.get_template(template)
    s = settings or get_settings()
    bib = report.bibliography()
    return tmpl.render(
        title=report.title,
        abstract=report.abstract,
        generated_at=report.generated_at,
        sections=report.sections,
        rumors=report.rumors,
        sources=bib,
        source_attribution={source.url: {
            "authors": [a for a in (source.metadata.get("authors") or []) if isinstance(a, str)][:12],
            "licenses": [url for url in (source.metadata.get("licenses") or []) if isinstance(url, str)
                         and url.startswith("https://creativecommons.org/")],
        } for source in bib},
        source_count=len(bib),
        min_sources=s.min_sources_per_claim,
        verify_rounds=s.verify_rounds,
        verify_mode=s.verify_mode,
        verify_strategy=s.verify_strategy,
        locale=s.report_locale,
        citations=report.citations,
        quality_status=report.quality_status,
        grounding_audit=report.grounding_audit,
        rumor_sentences=report.rumor_sentences,
        dispute_sentences=report.dispute_sentences, leads=report.leads,
        evidence_policy=report.evidence_policy,
        source_by_url={source.url: source for source in report.sources},
        metrics=report.metrics,
        scientific_audit=report.research_audit.get("scientific_comparisons", {}),
        calculations=report.calculations,
        chart_audit=report.chart_audit,
        research_plan=report.research_plan, research_audit=report.research_audit,
        research_evidence={**report.research_evidence, "citations": [c for c in report.research_evidence.get("citations", [])
            if c["id"] in {eid for row in report.research_audit.get("questions", []) + report.analysis for eid in row.get("citation_ids", [])}]},
        analysis=report.analysis, analysis_audit=report.analysis_audit,
        source_quality=report.source_quality,
    )


def write_tex(report: ReportData, out_dir: Path | None = None, *,
              settings: Settings | None = None) -> Path:
    """Render and write the .tex file, returning its path."""
    import hashlib

    s = settings or get_settings()
    out_dir = out_dir or s.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(report.topic)
    # Append a short topic hash so two topics that slug-collide (e.g. differing only
    # in punctuation/casing, or truncated at the 60-char cap) never overwrite each other.
    hash8 = hashlib.sha1(report.topic.encode("utf-8")).hexdigest()[:8]
    # Repeated/concurrent runs of the SAME topic must not overwrite each other.
    tex_path = out_dir / f"{slug}-{hash8}-{uuid4().hex[:12]}.tex"
    tex_path.write_text(render_tex(report, settings=s), encoding="utf-8")
    return tex_path


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "report")[:60]
