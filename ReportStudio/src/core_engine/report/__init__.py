"""Report-generation pipeline (current project objective).

A strict linear flow with a hard verification gate:

    topic -> search -> STRICT source filter -> scrape
          -> extract claims -> VERIFY (triple-check + cross-reference)
          -> scope gate -> LaTeX render -> PDF compile

Public entry point: ReportPipeline.run(topic) -> PipelineResult.
"""

from core_engine.report.models import (  # noqa: F401
    OUT_OF_SCOPE_MESSAGE,
    PipelineResult,
    PipelineStatus,
    ReportData,
)
from core_engine.report.pipeline import ReportPipeline  # noqa: F401
