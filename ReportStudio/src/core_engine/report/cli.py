"""CLI entry point for the report pipeline.

    python -m core_engine.report.cli "impact of carbon pricing on EU industry"

By default runs with the REAL providers (whatever CE_SEARCH_PROVIDER / CE_LLM_PROVIDER
are set to). For an offline smoke run with fixtures, set CE_SEARCH_PROVIDER=fake and
CE_LLM_PROVIDER=fake, or use the demo flag which wires deterministic fakes in-process.

Exit codes let this be scripted:
  0  COMPLETED
  2  OUT_OF_SCOPE
  3  BLOCKED
  4  ERROR
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from core_engine.report.models import PipelineStatus
from core_engine.report.pipeline import ReportPipeline

_EXIT = {
    PipelineStatus.COMPLETED: 0,
    PipelineStatus.OUT_OF_SCOPE: 2,
    PipelineStatus.BLOCKED: 3,
    PipelineStatus.ERROR: 4,
}


async def _run(topic: str, *, no_pdf: bool, show_trace: bool) -> int:
    pipeline = ReportPipeline()
    result = await pipeline.run(topic, compile_to_pdf=not no_pdf)

    print(f"\nstatus : {result.status.value}")
    print(f"topic  : {result.topic}")
    if result.message:
        print(f"message: {result.message}")
    if result.tex_path:
        print(f"tex    : {result.tex_path}")
    if result.pdf_path:
        print(f"pdf    : {result.pdf_path}")

    if show_trace:
        print("\n--- trace ---")
        for step in result.trace:
            print(step)

    return _EXIT[result.status]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a verified LaTeX report from a topic.")
    parser.add_argument("topic", help="the report topic")
    parser.add_argument("--no-pdf", action="store_true", help="render .tex only, skip compilation")
    parser.add_argument("--trace", action="store_true", help="print the full audit trace")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.topic, no_pdf=args.no_pdf, show_trace=args.trace))


if __name__ == "__main__":
    sys.exit(main())
