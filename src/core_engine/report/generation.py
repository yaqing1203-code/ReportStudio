"""User-selected output scope; evidence requirements are never preset controls."""
from __future__ import annotations

from typing import Literal

from core_engine.config import Settings

GenerationDepth = Literal["standard", "deep"]

# Standard gives a focused treatment; deep keeps the configured research ceiling.
# min() respects smaller operator-defined ceilings, including disabled budgets.
# Research question count/requirements and every verification/publication switch
# are deliberately absent: smaller output must not manufacture apparent coverage.
_STANDARD_CAPS: dict[str, int | float] = {
    "scrape_max_pages": 12,
    "scrape_max_results": 8,
    "extraction_max_claims": 60,
    "extraction_max_chunks_per_source": 2,
    "research_max_rounds": 1,
    "research_max_queries": 3,
    "research_deadline_s": 120.0,
    "research_fulltext_max_documents": 3,
    "research_local_max_chunks": 3,
    "reasoning_max_findings": 2,
    "max_deep_dives": 2,
    "deep_dive_max_claims": 15,
    "deep_dive_min_paragraphs": 1,
    "deep_dive_target_words": 180,
    "section_min_paragraphs": 1,
    "section_target_words": 150,
}


def report_profile(base: Settings, depth: GenerationDepth) -> Settings:
    """Copy a run's bounded scope, without editing environment or saved settings."""
    if depth not in {"standard", "deep"}:
        raise ValueError("内容深度必须是 standard 或 deep")
    update = {key: min(getattr(base, key), cap) for key, cap in _STANDARD_CAPS.items()} if depth == "standard" else {}
    return base.model_copy(update={**update, "generation_depth": depth})


def profile_audit(settings: Settings) -> dict:
    return {"depth": settings.generation_depth,
            "budgets": {key: getattr(settings, key) for key in _STANDARD_CAPS}}
