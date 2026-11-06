"""A fetched table must retain units/conditions all the way to anchored metrics."""
from __future__ import annotations

import pytest

from core_engine.report.metrics import catalog_from_claims
from core_engine.report.models import Claim, SentenceCitation, Source, SourceKind
from core_engine.report.scientific import comparison_audit, scalar, table_candidates
from core_engine.report.tables import append_tables, extract_tables

HTML = """<table id="t1"><caption>Measured ionic conductivity of solid electrolytes.</caption>
<tr><th>Electrolyte</th><th>Ionic conductivity (mS cm−1)</th><th>Temperature (°C)</th>
<th>Stack pressure (MPa)</th><th>Test method</th><th>Sample form</th></tr>
<tr><td>LLZO</td><td>1.2</td><td>25</td><td>5</td><td>EIS</td><td>pellet</td></tr>
<tr><td>LATP</td><td>0.8</td><td>25</td><td>5</td><td>EIS</td><td>pellet</td></tr></table>"""


def source(html=HTML):
    text, tables = append_tables("Original article prose.", extract_tables(html, "html"))
    return Source("https://example.edu/paper", "example.edu", "Scientific measurements", SourceKind.ACADEMIC,
                  text=text, document_format="html", metadata={"tables": tables, "full_text_accessed": True})


def anchored_candidates(document):
    extracted, audit = table_candidates(document)
    claims, citations = [], []
    for index, item in enumerate(extracted, 1):
        identifier = f"c{index}"
        claim = Claim(identifier, item.text, candidate_source_urls=[document.url], entity=item.entity,
                      attribute=item.attribute, value=item.value, metric=item.metric, reporting_status="source_reported")
        claims.append(claim)
        start = document.text.index(item.source_quote)
        citations.append(SentenceCitation(f"e{index}", identifier, document.url, item.source_quote,
            start, start + len(item.source_quote), 1, None, document.content_sha256, "fixture_verbatim"))
    return claims, citations, audit


def test_table_rows_register_source_bound_scientific_metrics_and_conditions():
    document = source()
    claims, citations, extracted = anchored_candidates(document)
    assert extracted["table_candidates"] == 2
    metrics, issues = catalog_from_claims(claims, citations)
    assert not issues, issues
    assert [metric.entity for metric in metrics] == ["LLZO", "LATP"]
    assert all(metric.name == "ionic_conductivity" for metric in metrics)
    audit = comparison_audit(metrics)
    assert audit["registered"] == 2 and audit["comparable_pairs"] == 1, audit
    assert all(not row["missing"] for row in audit["observations"])


def test_missing_conditions_are_not_imputed_from_another_table_or_sample():
    document = source(HTML.replace("<td>25</td>", "<td>not reported</td>", 1))
    claims, citations, _ = anchored_candidates(document)
    metrics, issues = catalog_from_claims(claims, citations)
    assert not issues, issues
    audit = comparison_audit(metrics)
    assert audit["registered"] == 2 and audit["comparable_pairs"] == 0
    assert audit["condition_scope"] == "anchored_observation_only"
    assert "missing_condition_temperature" in audit["observations"][0]["missing"]
    assert "温度" in audit["observations"][0]["missing_labels"]


@pytest.mark.parametrize("cell", ["~1.2", "1–2", "1 to 2", "1.2±0.1", "<1.2", "1 to 2 mS/cm", "1.2 ± 0.1 mS/cm"])
def test_non_scalar_measurements_are_not_silently_flattened(cell):
    document = source(HTML.replace("<td>1.2</td>", f"<td>{cell}</td>"))
    candidates, _ = table_candidates(document)
    assert len(candidates) == 1 and candidates[0].entity == "LATP"


def test_bad_row_offset_or_spanned_data_never_produces_a_fact():
    document = source()
    document.metadata["tables"][0]["row_records"][0]["start"] += 1
    document.metadata["tables"][0]["row_records"][1]["structure_status"] = "ambiguous_data_span"
    candidates, audit = table_candidates(document)
    assert candidates == [] and len(audit["skipped_rows"]) == 2
    assert table_candidates(source(), 0)[0] == []


def test_scalar_handles_scientific_notation_without_treating_ranges_as_values():
    assert str(scalar("1.2 × 10−3")) == "0.0012"
    assert scalar("1 to 2") is None


def test_published_table_measurement_temperature_is_not_drying_temperature():
    # Numeric excerpt from Table 1, DOI 10.1002/smtd.202401046, CC-BY.
    # https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11926510/fullTextXML
    html = """<table><caption>Ionic conductivity of various electrolytes synthesized in different conditions.</caption>
    <tr><th>Electrolyte</th><th>Amount of solvent [cm3]</th><th>Solid loading [wt%]</th><th>Ball size [mm]</th>
    <th>Drying temperature [°C]</th><th>Ionic conductivity at 25 °C [mS cm−1]</th></tr>
    <tr><td>Li3PS4 + 2LiBH4</td><td>10</td><td>20.3</td><td>1</td><td>90</td><td>8.6</td></tr></table>"""
    document = source(html)
    extracted, _ = table_candidates(document)
    assert len(extracted) == 1
    assert extracted[0].metric["conditions"]["temperature"] == "25 °C"
    assert extracted[0].metric["conditions"]["drying_temperature"] == "90 °C"
    claims, citations, _ = anchored_candidates(document)
    metrics, issues = catalog_from_claims(claims, citations)
    assert len(metrics) == 1 and not issues, issues


def test_published_compound_capacity_column_does_not_replace_conductivity_temperature():
    # Numeric excerpt from Table 1, DOI 10.3390/nano14221773, CC-BY.
    html = """<table><caption>The ionic conductivity and test conditions of polymers.</caption>
    <tr><th>Polymer Matrix</th><th>Li Salt</th><th>Temp. (°C)</th><th>Ionic Conductivity (S cm−1)</th>
    <th>Capacity mA h g−1 (Current Density, Cycle Index) (Test Temperature)</th></tr>
    <tr><td>CA/PEGMEA/SN</td><td>LiTFSI</td><td>25</td><td>1.9 × 10−3</td><td>171 (0.05C, 150) (25 °C)</td></tr></table>"""
    extracted, _ = table_candidates(source(html))
    assert len(extracted) == 1
    assert extracted[0].metric["conditions"]["temperature"] == "25 °C"
    assert extracted[0].metric["value"] == "0.0019"


def test_appended_pdf_table_evidence_uses_original_physical_page():
    from core_engine.report.grounding import _citation_page

    document = source()
    document.document_format = "pdf"
    table = document.metadata["tables"][0]
    table["locator"] = {"page": 2, "page_count": 8}
    assert _citation_page(document, table["row_records"][0]["start"]) == 2


def test_independent_numeric_replay_retains_and_rechecks_scientific_conditions():
    from dataclasses import asdict

    from core_engine.report.quality_eval import check_numeric_layer

    claims, citations, _ = anchored_candidates(source())
    metrics, issues = catalog_from_claims(claims, citations)
    assert not issues
    packet = {"claims": [asdict(claim) for claim in claims], "metrics": [metric.to_dict() for metric in metrics]}
    records = {record.id: asdict(record) for record in citations}
    checked = check_numeric_layer(packet, records)
    assert checked["registered_metrics"] == 2 and not checked["metric_integrity_errors"], checked
    packet["metrics"][0]["conditions"]["temperature"] = "90 °C"
    checked = check_numeric_layer(packet, records)
    assert checked["metric_integrity_errors"], "A rendered condition differing from the source must fail replay"


def test_scientific_observation_export_keeps_missing_conditions_and_citations():
    from core_engine.app.output_templates import report_payload
    from core_engine.report.latex import render_tex
    from core_engine.report.models import ReportData

    document = source(HTML.replace("<td>25</td>", "<td>not reported</td>", 1))
    claims, citations, _ = anchored_candidates(document)
    metrics, _ = catalog_from_claims(claims, citations)
    report = ReportData("技术测试", "实验条件报告", "摘要", claims=claims, citations=citations,
                        sources=[document], metrics=[metric.to_dict() for metric in metrics],
                        research_audit={"scientific_comparisons": comparison_audit(metrics)})
    payload = report_payload(report, {"sections": []})
    text = "\n".join(payload["sections"]["实验条件与比较缺口"])
    assert "温度" in text and "[e1]" in text and "性能排名" in text
    assert "缺口仅针对当前绑定的原文" in text
    tex = render_tex(report)
    assert "Experimental Observations and Comparability" in tex or "实验观测与可比性边界" in tex
    assert "LLZO" in tex and "e1" in tex


@pytest.mark.parametrize("spelling", ["1.9 × 10<sup>−3</sup>", "1.9 × 10^−3", "1.9e-3"])
def test_normalized_table_value_passes_numeric_guard_without_loosening_units(spelling):
    from core_engine.report.text import numbers_supported

    document = source(HTML.replace("mS cm−1", "S cm−1").replace("<td>1.2</td>", f"<td>{spelling}</td>"))
    candidates, _ = table_candidates(document)
    claim = candidates[0]
    assert numbers_supported(claim.text, claim.source_quote)
    assert not numbers_supported(claim.text.replace("0.0019", "0.019"), claim.source_quote)
    assert not numbers_supported(claim.text.replace("S/cm", "mS/cm"), claim.source_quote)
    assert not numbers_supported(claim.text.replace("S/cm", "MS/cm"), claim.source_quote)
    assert not numbers_supported(claim.text.replace("25 °C", "90 °C"), claim.source_quote)


@pytest.mark.parametrize("cell", ["1–2", "~1.9", "1.9±0.1", "<1.9"])
def test_table_numeric_guard_cannot_reconstruct_a_scalar_from_uncertain_cell(cell):
    from core_engine.report.text import numbers_supported

    assert not numbers_supported("Conductivity is 1.9 S/cm.", f"Ionic conductivity (S/cm): {cell}")
    assert not numbers_supported("Revenue is CNY 1.9.", "Revenue (CNY): 1.9")


@pytest.mark.anyio
async def test_scientific_table_reaches_real_grounding_and_publication_policy():
    from core_engine.config import get_settings
    from core_engine.report.grounding import SentenceGrounder
    from core_engine.report.llm import EntailmentJudgement
    from core_engine.report.publication import apply_policy

    document = source(HTML.replace("mS cm−1", "S cm−1").replace("<td>1.2</td>", "<td>1.9 × 10<sup>−3</sup></td>"))
    item = table_candidates(document)[0][0]

    class TableEntailment:
        async def check_entailment(self, text, context):
            assert item.source_quote in context
            return EntailmentJudgement(True, item.source_quote)

    claim = Claim("c1", item.text, candidate_source_urls=[document.url], entity=item.entity,
                  attribute=item.attribute, value=item.value, metric=item.metric,
                  source_windows={document.url: [[document.text.index(item.source_quote),
                                                document.text.index(item.source_quote) + len(item.source_quote)]]})
    grounder = SentenceGrounder(TableEntailment(), get_settings())
    anchored, citations = await grounder.anchor_claims([claim], [document])
    assert anchored == [claim] and len(citations) == 1
    assert document.text[citations[0].start_char:citations[0].end_char] == item.source_quote
    decision = apply_policy([claim], [document], citations)
    assert decision["claims"]["c1"]["status"] == "source_reported"
    metrics, issues = catalog_from_claims(anchored, citations)
    assert len(metrics) == 1 and not issues, issues
    assert str(metrics[0].value) == "0.0019"
