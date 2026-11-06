"""A separate Methods span can support conditions of an exact table observation."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from core_engine.report.fulltext import jats_document
from core_engine.report.metrics import catalog_from_claims, comparison_errors
from core_engine.report.models import Claim, SentenceCitation, Source, SourceKind
from core_engine.report.scientific import table_candidates


def measurement(
    url: str = "https://example.edu/experiment", *, method: str = "EIS", row_method: str = "",
) -> tuple[Source, Claim, SentenceCitation]:
    filler = "This experiment studied prepared electrolytes in separately described samples. " * 8
    method_header = "<th>Test method</th>" if row_method else ""
    method_cell = f"<td>{row_method}</td>" if row_method else ""
    raw = f"""<article article-type="research-article"><body>
    <sec><title>Results</title><p>{filler}</p><table-wrap id="t1"><label>Table 1</label>
    <caption><p>Ionic conductivity of prepared electrolytes.</p></caption><table>
    <tr><th>Electrolyte</th><th>Ionic conductivity at 25 °C [mS cm−1]</th>{method_header}</tr>
    <tr><td>LLZO</td><td>1.2</td>{method_cell}</tr></table></table-wrap></sec>
    <sec sec-type="methods"><title>Methods</title><p>The ionic conductivity was measured using {method}.
    Samples were cold-pressed at 566 MPa, and measurements were performed at 188 MPa and 25 °C.</p></sec>
    </body></article>"""
    document = jats_document(raw)
    assert document is not None
    source = Source(url, "example.edu", "Measurement", SourceKind.ACADEMIC, document.text,
                    document_format="jats", metadata=document.metadata)
    candidates, _ = table_candidates(source)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.metric is not None and candidate.source_quote
    start, end = candidate.source_windows[url][0]
    claim = Claim("c1", candidate.text, entity=candidate.entity, qualifier=candidate.qualifier,
                  metric=deepcopy(candidate.metric), reporting_status="source_reported")
    citation = SentenceCitation("e1", claim.id, url, candidate.source_quote, start, end,
                                1, None, source.content_sha256, "verbatim")
    return source, claim, citation


def test_catalog_keeps_measurement_value_and_methods_as_separate_evidence():
    source, claim, citation = measurement()
    assert claim.metric is not None
    assert claim.metric["conditions"] == {"temperature": "25 °C"}
    assert "188 MPa" not in citation.quote and "188 MPa" not in claim.text
    metrics, issues = catalog_from_claims([claim], [citation], sources=[source])
    assert len(metrics) == 1 and not issues
    metric = metrics[0]
    assert metric.conditions == {"temperature": "25 °C", "pressure": "188 MPa", "test_method": "EIS"}
    assert metric.evidence_ids == [citation.id]
    assert metric.condition_evidence == claim.metric["condition_evidence"]
    for witness in metric.condition_evidence.values():
        assert source.text[witness["start"]:witness["end"]] == witness["quote"]
        assert witness["end"] <= citation.start_char
        assert witness["source_version"] == citation.source_sha256
    assert comparison_errors(metric, metric) == ["missing_condition_sample_form"]


@pytest.mark.parametrize(("field", "replacement"), [
    ("source_url", "https://example.edu/other"), ("source_version", "0" * 64),
    ("start", 0), ("end", 0), ("quote", "Measurements were performed at 566 MPa."),
    ("table_id", "other-table"), ("row_index", 999), ("value", "566 MPa"),
])
def test_invalid_condition_proposal_is_withheld_without_losing_row_value(field, replacement):
    source, claim, citation = measurement()
    assert claim.metric is not None
    claim.metric["condition_evidence"]["pressure"][field] = replacement
    metrics, issues = catalog_from_claims([claim], [citation], sources=[source])
    assert len(metrics) == 1 and not issues
    assert metrics[0].evidence_ids == [citation.id]
    assert metrics[0].conditions == {"temperature": "25 °C", "test_method": "EIS"}
    assert set(metrics[0].condition_evidence) == {"test_method"}


def test_replay_requires_the_value_row_and_its_exact_source_version():
    source, claim, citation = measurement()
    amended = replace(source, text=source.text + "\nThe article snapshot has changed.")
    broader = replace(citation, start_char=0, end_char=len(source.text), quote=source.text)
    for records, sources in [([citation], []), ([citation], [amended]), ([broader], [source])]:
        metrics, issues = catalog_from_claims([claim], records, sources=sources)
        assert len(metrics) == 1 and not issues
        assert metrics[0].conditions == {"temperature": "25 °C"}
        assert metrics[0].condition_evidence == {}


def test_equal_values_in_different_papers_cannot_pool_methods_evidence():
    source, claim, citation = measurement()
    other, other_claim, other_citation = measurement("https://example.edu/other", method="DC polarization")
    assert claim.metric is not None and other_claim.metric is not None
    claim.metric["condition_evidence"]["test_method"] = other_claim.metric["condition_evidence"]["test_method"]
    records = [citation, replace(other_citation, id="e2")]
    metrics, issues = catalog_from_claims([claim], records, sources=[source, other])
    assert len(metrics) == 1 and not issues
    assert metrics[0].conditions == {"temperature": "25 °C", "pressure": "188 MPa"}
    assert metrics[0].evidence_ids == [citation.id]
    assert metrics[0].source_versions == [source.content_sha256]


def test_methods_cannot_replace_a_condition_reported_on_the_row():
    _, original, _ = measurement()
    source, claim, citation = measurement(row_method="DC polarization")
    assert original.metric is not None and claim.metric is not None
    claim.metric["condition_evidence"] = original.metric["condition_evidence"]
    metrics, issues = catalog_from_claims([claim], [citation], sources=[source])
    assert len(metrics) == 1 and not issues
    assert metrics[0].conditions == {"temperature": "25 °C", "test_method": "DC polarization"}
    assert metrics[0].condition_evidence == {}


@pytest.mark.anyio
async def test_report_pipeline_exports_and_replays_separate_methods_evidence(tmp_path):
    from core_engine.app.output_templates import report_payload
    from core_engine.config import get_settings
    from core_engine.report.llm import EntailmentJudgement, FakeLLM
    from core_engine.report.pipeline import ReportPipeline
    from core_engine.report.quality_eval import check_packet

    source, claim, row = measurement()
    quote = next(line for line in row.quote.splitlines() if line.startswith("Electrolyte:"))

    class Reviewer(FakeLLM):
        async def extract_claims(self, source_url, text, **kwargs):
            return []

        async def check_entailment(self, text, context):
            return EntailmentJudgement(text == claim.text and quote in context, quote)

    pipeline = ReportPipeline(llm=Reviewer())
    pipeline._s = get_settings().model_copy(update={
        "output_dir": tmp_path, "report_locale": "zh", "min_authoritative_sources": 1,
        "min_total_claims": 1, "research_enabled": False, "enable_charts": False,
        "enable_deep_dive": False, "enable_synthesis_sections": False,
        "verify_strategy": "aligned", "active_verify_enabled": False,
    })
    result = await pipeline.run("LLZO", extra_sources=[source], skip_search=True, compile_to_pdf=False)
    assert result.report is not None, result.message
    assert result.evidence_path and result.tex_path
    packet = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
    assert check_packet(packet)["integrity_errors"] == []
    assert packet["citations"][0]["quote"] == quote
    assert packet["metrics"][0]["conditions"]["pressure"] == "188 MPa"
    scientific = packet["research_audit"]["scientific_comparisons"]
    assert len(scientific["methods"]) == 1
    assert scientific["observations"][0]["method_ids"] == {"pressure": "M1", "test_method": "M1"}
    tex = Path(result.tex_path).read_text(encoding="utf-8")
    assert r"\hyperref[methods:M1]{[M1]}" in tex and r"\label{methods:M1}" in tex
    word_text = "\n".join(report_payload(result.report, {"sections": []})["sections"]["实验条件与比较缺口"])
    assert "pressure 188 MPa [M1]" in word_text and scientific["methods"][0]["quote"] in word_text

    changed_display = deepcopy(packet)
    changed_display["research_audit"]["scientific_comparisons"]["methods"][0]["quote"] = "Altered witness"
    assert "invalid_scientific_condition_audit" in check_packet(changed_display)["integrity_errors"]
    packet["metrics"][0]["conditions"]["pressure"] = "566 MPa"
    packet["metrics"][0]["condition_evidence"]["pressure"]["value"] = "566 MPa"
    assert check_packet(packet)["metric_integrity_errors"]
