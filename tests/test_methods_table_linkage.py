"""Methods evidence needs an exact, compatible measurement and table row."""
from __future__ import annotations

from decimal import Decimal

import pytest

from core_engine.report.fulltext import jats_document
from core_engine.report.metrics import catalog_from_claims
from core_engine.report.models import Claim, SentenceCitation, Source, SourceKind
from core_engine.report.scientific import (
    anchored_table_candidate,
    comparison_audit,
    table_candidates,
)

# These compact protocols preserve the distinctions in DOI 10.1002/smtd.202401046
# without depending on a downloaded fixture or running any network calls.
IONIC = ("The ionic conductivity was measured using EIS. Samples were cold-pressed at 566 MPa, "
         "and measurements were performed at 188 MPa and 25 °C.")
ELECTRONIC = ("For the electronic conductivity measurements, a sample was cold-pressed at 100 MPa. "
              "Afterward, measurements were conducted at 25 °C under 20 MPa.")
TABLE = """<table-wrap id="T1"><label>Table 1</label><caption><p>Ionic conductivity of prepared electrolytes.</p></caption>
<table id="T1-grid"><thead><tr><th>Electrolyte</th><th>Drying temperature [°C]</th>
<th>Ionic conductivity at 25 °C [mS cm<sup>−1</sup>]</th>
<th>Electronic conductivity at 25 °C [mS cm<sup>−1</sup>]</th></tr></thead>
<tbody><tr><td>Sample A</td><td>90</td><td>8.6</td><td>0.001</td></tr>
<tr><td>Sample B</td><td>110</td><td>7.2</td><td>0.002</td></tr></tbody></table></table-wrap>"""


def article(methods=f"<p>{IONIC} {ELECTRONIC}</p>", table=TABLE, article_type="research-article"):
    prose = "The authors prepared solid electrolyte samples with several drying temperatures. " * 8
    doc = jats_document(f"""<article xmlns="urn:jats" article-type="{article_type}"><body>
    <sec><title>Results</title><p>{prose}</p>{table}</sec>
    <sec id="methods"><title>Experimental Section</title>
    <sec id="conductivity"><title>Conductivity measurement</title>{methods}</sec></sec></body></article>""")
    assert doc is not None
    return Source("https://example.org/paper", "example.org", "Solid electrolyte experiment", SourceKind.ACADEMIC,
                  text=doc.text, document_format=doc.document_format, metadata=doc.metadata)


def test_joint_methods_paragraph_keeps_measurement_protocols_and_quotes_separate():
    source = article()
    rows, _ = table_candidates(source)
    assert len(rows) == 4
    for item in rows:
        metric = item.metric
        assert metric["conditions"]["temperature"] == "25 °C"
        assert metric["conditions"]["drying_temperature"] in {"90 °C", "110 °C"}
        assert "pressure" not in metric["conditions"] and "MPa" not in item.text
        row_start, row_end = item.source_windows[source.url][0]
        assert source.text[row_start:row_end] == item.source_quote
        methods = metric["condition_evidence"]
        pressure = methods["pressure"]
        assert pressure["source_version"] == source.content_sha256
        assert pressure["source_url"] == source.url
        assert pressure["end"] <= row_start
        assert pressure["locator"]["section_id"] == "conductivity"
        assert pressure["quote"] == source.text[pressure["start"]:pressure["end"]]
        if item.attribute == "ionic_conductivity":
            assert pressure["value"] == "188 MPa" and pressure["quote"] == IONIC
            assert methods["test_method"]["value"] == "EIS"
        else:
            assert pressure["value"] == "20 MPa" and pressure["quote"] == ELECTRONIC
            assert set(methods) == {"pressure"}
        assert anchored_table_candidate(
            source, row_start, row_end, metric["name"], item.entity, Decimal(metric["value"]), metric["unit"],
        ) == item


def test_explicit_table_scope_can_supply_temperature_and_uses_real_table_label_and_id():
    methods = f'<p>For the samples in <xref ref-type="table" rid="T1">Table 1</xref>, {IONIC}</p>'
    table = TABLE.replace(" at 25 °C", "")
    rows, _ = table_candidates(article(methods, table))
    ionic = rows[0].metric
    assert ionic["condition_evidence"]["temperature"]["value"] == "25 °C"
    assert ionic["condition_evidence"]["pressure"]["applicability"] == "explicit_table_reference"
    assert "temperature" not in ionic["conditions"]
    assert "condition_evidence" not in rows[1].metric
    for mismatched in (methods.replace('rid="T1"', 'rid="T2"'), methods.replace("Table 1", "Table 2")):
        assert "condition_evidence" not in table_candidates(article(mismatched, table))[0][0].metric


@pytest.mark.parametrize("methods", [
    f"<p>For Sample A, {IONIC}</p>",
    f"<p>{IONIC} This was only for Sample A.</p>",
    f"<p>{IONIC}</p><p>{IONIC.replace('188 MPa', '20 MPa')}</p>",
    f"<p>{IONIC}</p><p>The conductivity was measured at 30 MPa and 25 °C.</p>",
    f"<p>{IONIC.replace('188 MPa', '188–200 MPa')}</p>",
    f"<p>{IONIC.replace('188 MPa', '188 MPa and under 200 MPa')}</p>",
    f"<p>{IONIC.replace('25 °C', '35 °C')}</p>",
    "<p>The ionic conductivity was measured using EIS on pellets pressed at 566 MPa and dried at 25 °C.</p>",
    "<p>The ionic conductivity was measured using a cell rated at 188 MPa and 25 °C.</p>",
])
def test_ambiguous_or_preparation_only_conditions_leave_row_unknown(methods):
    rows, _ = table_candidates(article(methods))
    assert rows[0].metric["conditions"]["temperature"] == "25 °C"
    assert "condition_evidence" not in rows[0].metric


def test_secondary_tables_and_malformed_method_offsets_cannot_add_conditions():
    literature_table = TABLE.replace("Sample A", 'Sample A<xref ref-type="bibr" rid="ref1">†</xref>')
    for source in (article(article_type="review-article"), article(table=literature_table)):
        rows, _ = table_candidates(source)
        assert rows and all("condition_evidence" not in row.metric for row in rows)
    for anchor in ("paragraph", "heading"):
        source = article()
        span = source.metadata["methods_spans"][0]
        location = span if anchor == "paragraph" else span["locator"]["methods_heading"]
        location["start"] += 1
        rows, _ = table_candidates(source)
        assert rows and all("condition_evidence" not in row.metric for row in rows)


def test_comparison_audit_reuses_one_methods_span_for_multiple_rows_and_fields():
    source = article()
    claims, citations = [], []
    for index, item in enumerate(table_candidates(source)[0]):
        claim_id = f"c{index}"
        claims.append(Claim(claim_id, item.text, candidate_source_urls=[source.url], entity=item.entity,
                            attribute=item.attribute, value=item.value, metric=item.metric,
                            reporting_status="source_reported"))
        start, end = item.source_windows[source.url][0]
        citations.append(SentenceCitation(f"e{index}", claim_id, source.url, item.source_quote,
            start, end, 1, None, source.content_sha256, "fixture_verbatim"))
    metrics, issues = catalog_from_claims(claims, citations, sources=[source])
    assert len(metrics) == 4 and not issues
    audit = comparison_audit(metrics)
    assert audit["condition_scope"] == "anchored_observation_and_methods"
    assert len(audit["methods"]) == 2
    assert audit["observations"][0]["method_ids"] == audit["observations"][2]["method_ids"] == {
        "pressure": "M1", "test_method": "M1"}
    assert audit["observations"][1]["method_ids"] == audit["observations"][3]["method_ids"] == {
        "pressure": "M2"}
    assert audit["methods"][0]["quote"] == IONIC and audit["methods"][1]["quote"] == ELECTRONIC
