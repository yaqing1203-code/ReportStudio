"""Offline integration demo; synthetic values exercise charts, real excerpts exercise refusal.

python evals/report_quality/phase2_demo.py --compile-pdf
No external model, retrieval or billable service is used.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from core_engine.config import get_settings
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import FakeLLM
from core_engine.report.models import Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet, load_dataset, score_case
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider


async def run(output: Path, compile_pdf: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    llm = FakeLLM()
    url = "userdoc://phase2-synthetic-market"
    rows = []
    for year, amount in ((2023, "90"), (2024, "120")):
        text = f"In {year} Solar passenger-cars market size in China at current prices was {amount} USD bn (actual)."
        rows.append({"text": text, "metric": {"name": "market_size", "value": amount, "unit": "USD bn",
                     "entity": "Solar", "period": str(year), "geography": "China", "scope": "passenger-cars",
                     "price_basis": "current", "value_kind": "actual"}})
    llm.set_structured_claims(url, rows)
    source = Source(url, "uploaded.local", "合成测试数据（非真实市场统计）", SourceKind.USER_PROVIDED,
                    " ".join(row["text"] for row in rows))
    settings = get_settings().model_copy(update={
        "min_authoritative_sources": 1, "min_total_claims": 1, "verify_strategy": "cross_reference",
        "verify_mode": "strict", "report_locale": "zh", "enable_charts": True, "latex_engine": "xelatex",
        "enable_deep_dive": False, "enable_synthesis_sections": False, "active_verify_enabled": False,
        "max_reverify_attempts": 0, "output_dir": output,
    })

    def pipeline():
        pipe = ReportPipeline(llm=llm, search=FakeSearchProvider(), fetcher=FakeFetcher())
        pipe._s = settings
        pipe._harness._s = settings
        return pipe

    result = await pipeline().run("合成测试：market size in 2023 and 2024", compile_to_pdf=compile_pdf,
                                  extra_sources=[source], skip_search=True)
    if result.report is None or result.status.value != "completed" or (compile_pdf and not result.pdf_path):
        raise RuntimeError(result.message)
    checks = check_packet(evidence_packet(result.report))
    if checks["integrity_errors"] or checks["calculation_replay_rate"] != 1 or checks["checked_charts"] < 1:
        raise RuntimeError(f"Numeric integration check failed: {checks}")
    data = load_dataset()
    row = data["sources"][0]
    case = next(c for c in data["cases"] if c["id"] == "insufficient-profit-evidence")
    refused = await pipeline().run(case["question"], compile_to_pdf=compile_pdf, skip_search=True,
        extra_sources=[Source(row["url"], "stats.gov.cn", row["title"], SourceKind.GOVERNMENT, row["text"])])
    packet = json.loads(Path(refused.evidence_path).read_text(encoding="utf-8"))
    refusal_score = score_case(packet, case, data)
    if not refusal_score["automatic_checks_passed"] or refused.tex_path or refused.pdf_path:
        raise RuntimeError("Unanswerable input produced a report or lacked explicit refusal evidence")
    summary = {"synthetic_values": True, "live_model_evaluated": False,
               "report_pdf": result.pdf_path, "report_tex": result.tex_path, "evidence": result.evidence_path,
               "checks": checks, "refusal_evidence": refused.evidence_path, "refusal_score": refusal_score}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/phase2-demo"))
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(run(args.output, args.compile_pdf))
    print(json.dumps({"pdf": summary["report_pdf"], "replay_rate": summary["checks"]["calculation_replay_rate"],
                      "refusal_passed": summary["refusal_score"]["automatic_checks_passed"]}, ensure_ascii=False))
