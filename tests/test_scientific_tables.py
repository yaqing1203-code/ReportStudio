"""Scientific measurements must retain the fetched table's units and conditions."""
from __future__ import annotations

import copy

import httpx
import pytest

from core_engine.report.fulltext import jats_document
from core_engine.report.models import SearchHit
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider, HttpFetcher, gather_sources
from core_engine.report.tables import (
    MAX_CELL_CHARS,
    MAX_RAW_CHARS,
    MAX_ROWS,
    MAX_TABLES,
    append_tables,
    extract_tables,
    strip_extracted_tables,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


HTML = '''<figure class="table-wrap" id="T1">
<figcaption>Table 1. Electrolyte conductivity measured at 25 °C.</figcaption>
<table><thead><tr><th rowspan="2">Sample</th><th colspan="2">Conductivity</th></tr>
<tr><th>mS cm<sup>−1</sup></th><th>S cm<sup>−1</sup></th></tr></thead>
<tbody><tr><td rowspan="2">Li<sub>6</sub>PS<sub>5</sub>Cl</td><td>4.0</td><td>0.004</td></tr>
<tr><td>5.0</td><td>0.005</td></tr></tbody>
<tfoot><tr><td colspan="3">* Stack pressure 10 MPa; pellet thickness 1 mm; impedance spectroscopy.</td></tr></tfoot>
</table></figure>'''


def _assert_offsets(text, tables):
    for table in tables:
        assert 0 <= table["start"] < table["end"] <= len(text)
        for row in table["row_records"]:
            assert text[row["start"]:row["end"]] == row["text"]
            assert table["start"] <= row["start"] < row["end"] <= table["end"]


def test_multirow_headers_units_rowspans_and_footnotes_stay_together():
    tables = extract_tables(HTML, "html")
    assert len(tables) == 1 and tables[0]["id"] == "T1"
    assert tables[0]["headers"] == ["Sample", "Conductivity / mS cm^−1", "Conductivity / S cm^−1"]
    assert tables[0]["rows"] == [["Li6PS5Cl", "4.0", "0.004"], ["Li6PS5Cl", "5.0", "0.005"]]
    text, attached = append_tables("Original prose.", tables)
    _assert_offsets(text, attached)
    for row in attached[0]["row_records"]:
        assert "25 °C" in row["text"] and "10 MPa" in row["text"] and "impedance spectroscopy" in row["text"]
        assert "Conductivity / mS cm^−1:" in row["text"]
    assert attached[0]["row_records"][1]["inherited_columns"] == [0]
    assert attached[0]["conditions_status"] == "unknown"


def test_table_context_does_not_bleed_from_adjacent_table():
    other = HTML.replace('id="T1"', 'id="T2"').replace("25 °C", "60 °C").replace("10 MPa", "50 MPa")
    text, tables = append_tables("", extract_tables(HTML + other, "html"))
    assert len(tables) == 2
    assert "25 °C" in tables[0]["row_records"][0]["text"]
    assert "60 °C" not in tables[0]["row_records"][0]["text"]
    assert "50 MPa" in tables[1]["row_records"][0]["text"]
    assert "10 MPa" not in tables[1]["row_records"][0]["text"]
    _assert_offsets(text, tables)


def test_publisher_inner_scroll_wrapper_does_not_hide_outer_caption_and_notes():
    html = '''<figure id="Tab1" class="c-article-table"><figcaption>Table 1. Tested at 25 °C.</figcaption>
    <div class="c-article-table-container"><div><table><tr><th>Sample</th><th>Conductivity (mS/cm)</th></tr>
    <tr><td>A</td><td>4.0</td></tr></table></div></div>
    <div class="c-article-table-note">Pressure was 5 MPa.</div></figure>'''
    text, tables = append_tables("", extract_tables(html, "html"))
    assert tables[0]["id"] == "Tab1"
    assert "25 °C" in tables[0]["row_records"][0]["text"]
    assert "5 MPa" in tables[0]["row_records"][0]["text"]
    _assert_offsets(text, tables)


def test_duplicate_table_markup_is_extracted_once_and_both_raw_copies_removed():
    html = "<p>Before</p>" + HTML + HTML + "<p>After</p>"
    tables = extract_tables(html, "html")
    assert len(tables) == 1
    stripped = strip_extracted_tables(html, tables)
    assert "<table>" not in stripped and "0.004" not in stripped
    text, attached = append_tables("Before. After.", tables)
    second, repeated = append_tables(text, attached)
    assert second == text
    assert repeated[0]["row_records"][1]["inherited_columns"] == [0]
    assert text.count("Conductivity / mS cm^−1: 4.0") == 1


def test_numeric_data_colspan_does_not_manufacture_two_measurements():
    html = '<table><tr><th>Sample</th><th>σ (mS/cm)</th><th>σ (S/cm)</th></tr><tr><td>A</td><td colspan="2">4.0</td></tr></table>'
    text, tables = append_tables("", extract_tables(html, "html"))
    assert tables[0]["rows"] == [["A", "4.0", ""]]
    assert tables[0]["row_records"][0]["structure_status"] == "ambiguous_data_span"
    assert "σ (S/cm): 4.0" not in text


def test_unknown_headers_and_conditions_are_preserved_without_guessing():
    html = '<table><tr><td>A</td><td>4.0</td></tr><tr><td>B</td><td>5.0</td></tr></table>'
    text, tables = append_tables("", extract_tables(html, "html"))
    assert tables[0]["headers"] == ["", ""]
    assert tables[0]["header_status"] == tables[0]["conditions_status"] == "unknown"
    assert tables[0]["caption"] == "" and tables[0]["footnotes"] == []
    assert "25" not in text and "MPa" not in text


def test_jats_caption_external_footnotes_namespace_and_doi_are_retained():
    xml = '''<article xmlns="urn:jats" article-type="research-article"><front><article-meta><article-id pub-id-type="doi">10.1234/example</article-id>
    <title-group><article-title>Solid-state electrolyte conditions</article-title></title-group></article-meta></front>
    <body><sec><title>Results</title><p>''' + "Measured solid-state electrolyte specimens under explicitly recorded conditions. " * 10 + '''</p>
    <table-wrap id="T2"><label>Table 2.</label><caption><title>Conductivity at 25 °C.</title></caption>
    <table><thead><tr><th>Sample</th><th>Conductivity (mS/cm)</th></tr></thead>
    <tbody><tr><td>Sulfide<xref ref-type="table-fn" rid="TF1">a</xref></td><td>4.0</td></tr></tbody></table>
    <table-wrap-foot><fn id="TF1"><p>a: Tested at 5 MPa using impedance spectroscopy.</p></fn></table-wrap-foot>
    </table-wrap></sec></body></article>'''
    document = jats_document(xml)
    assert document is not None and document.metadata["full_text_accessed"]
    assert document.metadata["doi"] == "10.1234/example"
    assert document.metadata["title"] == "Solid-state electrolyte conditions"
    assert document.metadata["article_type_declared"] == "research-article"
    table = document.metadata["tables"][0]
    assert table["id"] == "T2" and table["caption"] == "Table 2. Conductivity at 25 °C."
    assert "5 MPa" in table["row_records"][0]["text"] and "25 °C" in table["row_records"][0]["text"]
    assert document.text.count("Conductivity (mS/cm): 4.0") == 1
    _assert_offsets(document.text, document.metadata["tables"])


def test_real_publisher_tbody_sublabels_belong_to_grouped_headers_not_measurements():
    # Minimal structural excerpt from Table 2, DOI:10.1002/smtd.202401046,
    # PMCID PMC11926510. The publisher places 4a/4d sublabels in tbody.
    xml = '''<table-wrap id="smtd202401046-tbl-0002"><label>Table 2</label><caption><p>Proportion (in %).</p></caption>
    <table><thead><tr><th>Drying temperature [°C]</th><th colspan="2">Proportion from XRD [Rietveld refinement]</th>
    <th colspan="2">Proportion from TGA‐MS [Deconvolution]</th></tr></thead><tbody>
    <tr><td/><td>4a</td><td>4d</td><td>4a</td><td>4d</td></tr>
    <tr><td>25</td><td>52</td><td>48</td><td>59</td><td>41</td></tr></tbody></table></table-wrap>'''
    text, tables = append_tables("", extract_tables(xml, "jats"))
    assert tables[0]["headers"] == ["Drying temperature [°C]",
        "Proportion from XRD [Rietveld refinement] / 4a", "Proportion from XRD [Rietveld refinement] / 4d",
        "Proportion from TGA‐MS [Deconvolution] / 4a", "Proportion from TGA‐MS [Deconvolution] / 4d"]
    assert tables[0]["rows"] == [["25", "52", "48", "59", "41"]]
    assert tables[0]["header_status"] == "explicit_with_body_subheaders"
    _assert_offsets(text, tables)


def test_jats_break_elements_separate_listed_properties_without_changing_values():
    xml = '<table-wrap><table><tr><th>Type</th><th>Conductivity</th><th>Properties</th></tr><tr><td>Sulfide</td><td>1.0</td><td>High conductivity<break/>Moisture sensitive</td></tr></table></table-wrap>'
    tables = extract_tables(xml, "jats")
    assert tables[0]["rows"][0] == ["Sulfide", "1.0", "High conductivity Moisture sensitive"]


@pytest.mark.parametrize("payload", [
    '<article><body><table><tr><th>A</th><th>B</th></tr><tr><td>X</td><td>4.0</td></tr></table>',
    '<!DOCTYPE article [<!ENTITY xx SYSTEM "file:///private">]><article><body>&xx;</body></article>',
    '<!DOCTYPE article [<!ENTITY xx "4.0">]><article><body>&xx;</body></article>',
])
def test_malformed_or_entity_declaring_jats_never_becomes_a_table(payload):
    assert extract_tables(payload, "jats") == []
    assert jats_document(payload) is None


def test_omitted_html_cell_end_tags_are_understood_but_unclosed_table_is_not():
    html = '<table><tr><th>Sample<th>σ (mS/cm)<tr><td>A<td>4.0</table>'
    tables = extract_tables(html, "html")
    assert tables[0]["headers"] == ["Sample", "σ (mS/cm)"] and tables[0]["rows"] == [["A", "4.0"]]
    assert extract_tables(html.replace("</table>", ""), "html") == []


@pytest.mark.parametrize("span", ['colspan="999999999"', 'rowspan="999999999"', 'colspan="-1"', 'rowspan="0"'])
def test_malformed_or_explosive_spans_are_rejected(span):
    html = '<table><tr><th>A</th><th>B</th></tr><tr><td ' + span + '>A</td><td>4.0</td></tr></table>'
    assert extract_tables(html, "html") == []


def test_table_counts_rows_raw_padding_and_cell_lengths_are_bounded():
    unique = [HTML.replace("25 °C", f"{i} °C") for i in range(MAX_TABLES + 10)]
    assert len(extract_tables("".join(unique), "html")) == MAX_TABLES
    rows = ''.join(f'<tr><td>Sample {i}</td><td>{i}.0</td></tr>' for i in range(MAX_ROWS + 10))
    tables = extract_tables('<table><tr><th>Sample</th><th>σ (mS/cm)</th></tr>' + rows + '</table>', "html")
    assert len(tables[0]["rows"]) <= MAX_ROWS and tables[0]["truncated"]
    assert extract_tables(" " * MAX_RAW_CHARS + HTML, "html") == []
    assert extract_tables(HTML.replace("4.0", "4.0" + "x" * MAX_CELL_CHARS), "html") == []


def test_append_budget_never_slices_a_numeric_row_or_mutates_input():
    tables = extract_tables(HTML, "html")
    before = copy.deepcopy(tables)
    row_text = append_tables("", tables)[1][0]["row_records"][0]["text"]
    text, attached = append_tables("Body.", tables, max_chars=len("Body.\n\n") + len(row_text) + 1)
    assert len(attached[0]["row_records"]) == 1 and attached[0]["truncated"]
    _assert_offsets(text, attached)
    assert "0.004" in text and "10 MPa" in text
    assert append_tables("Already too long.", tables, max_chars=3) == ("Already too long.", [])
    assert tables == before


async def _fetch_html(monkeypatch, html):
    async def read(client, url, **kwargs):
        return httpx.Response(200, text=html, headers={"content-type": "text/html"}, request=httpx.Request("GET", url))
    async def allowed(*args):
        return True
    fetcher = HttpFetcher()
    monkeypatch.setattr(fetcher, "_allowed", allowed)
    monkeypatch.setattr("core_engine.report.scrape.read_public_text", read)
    return await fetcher.fetch("https://example.edu/paper")


async def test_fetched_html_and_gather_preserve_final_source_offsets(monkeypatch):
    html = '<html><body><h1>Solid-state electrolyte measurements</h1><p>' + "Solid-state electrolytes were measured under reported test conditions. " * 12 + '</p>' + HTML + '</body></html>'
    document = await _fetch_html(monkeypatch, html)
    assert document.document_format == "html"
    provider = FakeSearchProvider()
    provider.add("", [SearchHit("https://example.edu/paper", "Solid-state electrolyte measurements", "", "example.edu")])
    sources, _ = await gather_sources("solid-state electrolyte", provider,
        FakeFetcher({"https://example.edu/paper": document}), queries=["solid-state electrolyte"])
    assert len(sources) == 1
    _assert_offsets(sources[0].text, sources[0].metadata["tables"])
    assert sources[0].text.count("Conductivity / mS cm^−1: 4.0") == 1


async def test_html_landing_abstract_does_not_gain_table_or_fulltext_status(monkeypatch):
    abstract = "Solid-state batteries use solid electrolytes under specific test conditions. " * 3
    html = '<meta name="citation_doi" content="10.1234/example"><meta name="citation_title" content="Paper"><meta name="citation_abstract" content="' + abstract + '">' + HTML
    document = await _fetch_html(monkeypatch, html)
    assert document.document_format == "abstract"
    assert not document.metadata["full_text_accessed"] and "tables" not in document.metadata


def _pdf(*, caption=True, ruled=True, page_number=1):
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for _ in range(page_number - 1):
        document.new_page(width=500, height=400).insert_text((50, 50), "Original introduction page.")
    page = document.new_page(width=500, height=400)
    if caption:
        page.insert_text((50, 50), "Table 1. Conductivity at 25 C", fontsize=10)
    if ruled:
        for x in [50, 180, 320]:
            page.draw_line((x, 80), (x, 170))
        for y in [80, 110, 140, 170]:
            page.draw_line((50, y), (320, y))
    for x, y, value in [(58, 100, "Sample"), (188, 100, "Conductivity(mS/cm)"),
                         (58, 130, "A"), (188, 130, "4.0"), (58, 160, "B"), (188, 160, "5.0"),
                         (50, 190, "Note: Stack pressure 10 MPa.")]:
        page.insert_text((x, y), value, fontsize=9)
    raw = document.tobytes()
    document.close()
    return raw


def test_pdf_grid_preserves_scalar_decimal_units_caption_notes_and_page():
    text, tables = append_tables("Original PDF text.", extract_tables(_pdf(), "pdf"))
    assert len(tables) == 1
    assert tables[0]["headers"] == ["Sample", "Conductivity(mS/cm)"]
    assert tables[0]["rows"] == [["A", "4.0"], ["B", "5.0"]]
    assert tables[0]["locator"]["page"] == 1
    assert "25 C" in tables[0]["row_records"][0]["text"] and "10 MPa" in tables[0]["row_records"][0]["text"]
    _assert_offsets(text, tables)


def test_pdf_tables_keep_the_physical_one_based_page_separate_from_appended_offsets():
    text, tables = append_tables("First page.\fSecond page.", extract_tables(_pdf(page_number=2), "pdf"))
    assert tables[0]["locator"]["page"] == 2
    assert tables[0]["locator"]["page_count"] == tables[0]["physical_page_count"] == 2
    _assert_offsets(text, tables)


@pytest.mark.parametrize(("caption", "ruled"), [(False, True), (True, False)])
def test_pdf_without_caption_or_explicit_grid_is_not_guessed(caption, ruled):
    assert extract_tables(_pdf(caption=caption, ruled=ruled), "pdf") == []


def test_corrupt_pdf_and_unknown_format_degrade_without_new_access():
    assert extract_tables(b"%PDF-corrupt", "pdf") == []
    assert extract_tables(HTML, "spreadsheet") == []


def test_numeric_superscript_exponents_are_distinct_from_ranges_and_citations():
    html = '''<table><tr><th>Material</th><th>Conductivity (S cm<sup>−1</sup>)</th></tr>
    <tr><td>Li<sub>3</sub>PS<sub>4</sub></td><td>1.9 × 10<sup>−3</sup></td></tr>
    <tr><td>Range</td><td>10−5–10−3</td></tr>
    <tr><td>Footnote</td><td>4.0<sup><a href="#fn1">1</a></sup></td></tr>
    <tr><td>Letter note</td><td>5.0<sup>a</sup></td></tr></table>'''
    tables = extract_tables(html, "html")
    assert tables[0]["headers"][1] == "Conductivity (S cm^−1)"
    assert tables[0]["rows"][0] == ["Li3PS4", "1.9 × 10^−3"]
    assert tables[0]["rows"][1][1] == "10−5–10−3"
    assert tables[0]["rows"][2][1] == "4.0 [1]"
    assert tables[0]["rows"][3][1] == "5.0a"


def test_unparsed_jats_table_keeps_plain_text_fallback_instead_of_losing_original_cells():
    xml = '<article><body><p>' + "Solid-state electrolyte test protocol. " * 20 + '</p><table-wrap><caption><p>Unsupported geometry.</p></caption><table><tr><th>Sample</th><th>Conductivity (mS/cm)</th></tr><tr><td colspan="9999999">Original sample</td><td>4.0</td></tr></table></table-wrap></body></article>'
    document = jats_document(xml)
    assert document is not None and "tables" not in document.metadata
    assert "Original sample" in document.text and "4.0" in document.text and "mS/cm" in document.text


async def test_optional_table_adapter_failure_does_not_discard_readable_html(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("optional structure parser failed")
    monkeypatch.setattr("core_engine.report.tables.extract_tables", broken)
    result = await _fetch_html(monkeypatch, '<p>Original solid-state measurements.</p>' + HTML)
    assert isinstance(result, str) and "Original solid-state" in result and "4.0" in result
