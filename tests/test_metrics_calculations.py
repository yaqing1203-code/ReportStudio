from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, replace
from decimal import Decimal, Inexact, localcontext

import pytest

from core_engine.report.calculations import calculate, replay
from core_engine.report.chart_audit import validate_charts
from core_engine.report.charts import build_all, comparison_matrix
from core_engine.report.kg import Competitor, MarketDatum, MarketSize, ReportKnowledgeGraph
from core_engine.report.metrics import (
    Metric,
    MetricError,
    catalog_from_claims,
    comparison_errors,
    convert_value,
    decimal_value,
    observed_quantities,
    parse_unit,
    quantities,
)
from core_engine.report.models import Claim, SentenceCitation
from core_engine.report.quality_eval import check_numeric_layer, load_dataset


def observation(identifier="m1", value="90", period="2023", **kwargs):
    fields = {"id": identifier, "name": "market_size", "entity": "Solar", "value": Decimal(value), "unit": "USD bn",
              "period": period, "geography": "China", "scope": "passenger-cars", "price_basis": "current",
              "value_kind": "actual", "claim_ids": ["c" + identifier], "evidence_ids": ["e" + identifier],
              "source_versions": ["version"]}
    fields.update(kwargs)
    return Metric(**fields)


def source_pair():
    claims, citations = [], []
    for i, (year, amount) in enumerate(((2023, "90"), (2024, "120")), 1):
        text = (f"In {year} the market was {amount} USD bn "
                "[entity=Solar; geography=China; scope=passenger-cars; price_basis=current; value_kind=actual].")
        claims.append(Claim(f"c{i}", text, candidate_source_urls=["https://example.org/market"], verified=True))
        citations.append(SentenceCitation(f"e{i}", f"c{i}", "https://example.org/market", text, 0, len(text),
                                         1, None, hashlib.sha256(text.encode()).hexdigest(), "verbatim"))
    metrics, issues = catalog_from_claims(claims, citations)
    assert not issues and len(metrics) == 2
    return claims, citations, metrics


@pytest.mark.parametrize(("value", "source", "target", "expected"), [
    ("1.25", "亿元", "万元", "12500"), ("1000", "USD mn", "USD bn", "1"),
    ("0.125", "ratio", "%", "12.5"), ("1", "pp", "bp", "100"),
    ("1234.56", "MW", "GW", "1.23456"), ("2.5", "GWh", "kWh", "2500000"),
])
def test_exact_unit_conversion(value, source, target, expected):
    assert convert_value(Decimal(value), source, target) == Decimal(expected)


@pytest.mark.parametrize(("source", "target"), [("USD", "CNY"), ("USD", "EUR"), ("%", "pp"), ("GW", "GWh")])
def test_currency_and_dimension_confusion_is_rejected(source, target):
    with pytest.raises(MetricError):
        convert_value(Decimal(1), source, target)


@pytest.mark.parametrize("value", [True, 0.1, "NaN", "Infinity", "1e1000", "1e-1000", "9" * 60, "1,23"])
def test_invalid_decimal_inputs_are_bounded(value):
    with pytest.raises(MetricError):
        decimal_value(value)


@pytest.mark.parametrize("unit", ["mW", "mw", "gwh", "bn", "dollars", "unknown"])
def test_ambiguous_or_wrong_case_units_are_rejected(unit):
    with pytest.raises(MetricError):
        parse_unit(unit)


def test_currency_prefix_and_chinese_quantities_preserve_decimal_values():
    assert quantities("营收１，２３４．５６亿元，份额１２．５％，同比提高１．２个百分点") == [
        (Decimal("1234.56"), "CNY 100m"), (Decimal("12.5"), "%"), (Decimal("1.2"), "pp")]
    assert quantities("revenue $5.25 billion and USD 12 million") == [
        (Decimal("5.25"), "USD bn"), (Decimal(12), "USD mn")]
    assert quantities("capacity 1.2e3 MW") == [(Decimal(1200), "MW")]
    assert quantities("市场规模为1,23亿元") == []


def test_oversized_unstructured_metric_is_withheld_instead_of_crashing_catalog():
    metrics, issues = catalog_from_claims([Claim("invalid", "The market size was 1e1000 USD bn.")], [])
    assert not metrics and issues


def test_metric_name_is_bound_to_its_own_amount_not_any_number_in_quote():
    text = "TAM is 100 USD bn and revenue is 200 USD bn."
    assert observed_quantities(text, "tam") == [(Decimal(100), "USD bn")]
    claim = Claim("c1", text, metric={"name": "tam", "value": "200", "unit": "USD bn"})
    citation = SentenceCitation("e1", "c1", "url", text, 0, len(text), 1, None, "hash", "verbatim")
    metrics, issues = catalog_from_claims([claim], [citation])
    assert not metrics and "No anchored" in issues[0]["reason"]


def test_invented_metadata_is_rejected_and_unknown_metadata_stays_unknown():
    claims, citations, _ = source_pair()
    claims[0].metric = {"name": "market_size", "value": "90", "unit": "USD bn", "geography": "Mars"}
    assert catalog_from_claims(claims[:1], citations[:1])[1]
    text = "In 2023 the market was 90 USD bn."
    claim = Claim("plain", text)
    citation = SentenceCitation("plain-e", "plain", "url", text, 0, len(text), 1, None, "hash", "verbatim")
    metric = catalog_from_claims([claim], [citation])[0][0]
    assert metric.geography is None and metric.scope is None and metric.value_kind is None
    assert "missing_scope" in comparison_errors(metric, metric)


def test_pooled_quotes_cannot_attach_another_observations_period_or_scope():
    text = "In 2023 the market was 90 USD bn in China. In 2024 the market was 120 USD bn in Europe."
    claim = Claim("c1", text, metric={"name": "market_size", "value": "90", "unit": "USD bn", "period": "2024"})
    citation = SentenceCitation("e1", "c1", "url", text, 0, len(text), 1, None, "hash", "verbatim")
    assert not catalog_from_claims([claim], [citation])[0]
    claim.metric = {"name": "market_size", "value": "90", "unit": "USD bn", "period": "2023", "geography": "Europe"}
    assert not catalog_from_claims([claim], [citation])[0]


def test_pinned_real_gdp_release_supports_exact_conversion_and_replay():
    row = next(s for s in load_dataset()["sources"] if s["id"] == "nbs-final")
    text = row["text"]
    claim = Claim("gdp", text, candidate_source_urls=[row["url"]], metric={
        "name": "gdp", "value": "1348066", "unit": "亿元", "period": "2024",
        "price_basis": "current", "value_kind": "final"})
    citation = SentenceCitation("real-e", "gdp", row["url"], text, 0, len(text), 1, None, row["sha256"], "verbatim")
    metrics, issues = catalog_from_claims([claim], [citation])
    assert not issues and len(metrics) == 1
    result, record = calculate("unit_conversion", metrics, target_unit="CNY bn", output_id="converted")
    assert result.value == Decimal("134806.6")
    assert replay(record, {metrics[0].id: metrics[0]}).value == result.value


def test_deterministic_growth_cagr_and_replay_do_not_depend_on_ambient_precision():
    a, b = observation(), observation("m2", "120", "2024")
    with localcontext() as context:
        context.prec = 5
        context.Emax = 3
        context.traps[Inexact] = True
        result, record = calculate("growth_rate", [a, b], output_id="d1")
        assert convert_value(Decimal(120000), "USD mn", "USD bn") == 120
    assert result.value == Decimal("33.333333")
    assert replay(record, {a.id: a, b.id: b}).to_dict() == result.to_dict()
    end = observation("m3", "108.9", "2025")
    result, _ = calculate("cagr", [a, end], output_id="d2")
    assert result.value == Decimal("10.000000")
    record["rounding"] = "ROUND_UP"
    with pytest.raises(MetricError, match="replay"):
        replay(record, {a.id: a, b.id: b})


@pytest.mark.parametrize(("field", "value"), [
    ("geography", "US"), ("scope", "all-cars"), ("price_basis", "constant"),
    ("value_kind", "forecast"), ("entity", "Wind"), ("geography", None), ("unit", "CNY bn"),
])
def test_growth_rejects_incompatible_or_unknown_scope(field, value):
    a, b = observation(), replace(observation("m2", "120", "2024"), **{field: value})
    with pytest.raises(MetricError):
        calculate("growth_rate", [a, b], output_id="bad")


@pytest.mark.parametrize(("operation", "before", "after", "period"), [
    ("growth_rate", "0", "100", "2024"), ("growth_rate", "90", "120", "2025"),
    ("cagr", "90", "0", "2025"), ("cagr", "90", "120", "2022"),
    ("growth_rate", "90", "120", "2024Q1"), ("growth_rate", "1e-40", "1e40", "2024"),
])
def test_invalid_growth_inputs_are_domain_errors(operation, before, after, period):
    with pytest.raises(MetricError):
        calculate(operation, [observation(value=before), observation("m2", after, period)], output_id="bad")


def test_percentage_points_are_not_relative_growth_and_share_needs_a_denominator():
    a = observation(name="market_share", value="20", unit="%", rate_basis="passenger-cars sales")
    b = replace(a, id="m2", value=Decimal(25), period="2024")
    difference, _ = calculate("percentage_point_change", [a, b], output_id="d1")
    assert difference.value == 5 and difference.unit == "pp"
    relative, _ = calculate("growth_rate", [a, b], output_id="d2")
    assert relative.value == 25 and relative.unit == "%"
    part = observation("part", "30", entity="A", part_of="Solar")
    total = observation()
    share, _ = calculate("share", [part, total], output_id="share")
    assert share.value == Decimal("33.333333") and share.rate_basis == "part_of:Solar"
    with pytest.raises(MetricError):
        calculate("share", [replace(part, part_of=None), total], output_id="bad")


def test_percentage_point_and_explicit_part_whole_calculations_are_wired_into_report_audit():
    before = observation("before", "20", name="market_share", unit="%", rate_basis="sales")
    after = replace(before, id="after", value=Decimal(25), period="2024")
    part = observation("part", "30", entity="A", part_of="Solar")
    total = observation("total")
    _, _, derived, records, _, issues = validate_charts(ReportKnowledgeGraph("Solar"),
                                                       [before, after, part, total], [], [])
    assert not issues
    assert {r["operation"] for r in records} == {"share", "percentage_point_change"}
    assert any(m.unit == "pp" and m.value == 5 for m in derived)


def test_equivalent_explicit_price_and_rate_labels_compare_but_annualized_and_yoy_do_not():
    a = observation()
    assert not comparison_errors(a, replace(a, price_basis="nominal"))
    ratio = replace(a, name="growth_rate", unit="%", rate_basis="yoy")
    assert not comparison_errors(ratio, replace(ratio, rate_basis="同比"))
    assert comparison_errors(ratio, replace(ratio, rate_basis="annualized"))


def trend_graph(metrics):
    return ReportKnowledgeGraph("Solar", market=MarketSize(series=[
        MarketDatum(int(m.period), m.value, m.unit, m.claim_ids) for m in metrics]))


def test_valid_chart_is_normalized_and_growth_is_computed_from_exact_evidence():
    claims, citations, metrics = source_pair()
    graph = trend_graph(metrics)
    assert not build_all(graph)["market_plot"]  # proposed values cannot render
    clean, _, derived, records, audit, issues = validate_charts(graph, metrics, claims, citations)
    assert not issues
    assert "\\addplot" in build_all(clean)["market_plot"]
    assert {r["operation"] for r in records} == {"growth_rate", "cagr"}
    assert {m.value for m in derived} == {Decimal("33.333333")}
    packet = {"claims": [asdict(c) for c in claims], "metrics": [m.to_dict() for m in metrics + derived],
              "calculations": records, "chart_audit": audit}
    valid = {c.id: asdict(c) for c in citations}
    checked = check_numeric_layer(packet, valid)
    assert not checked["metric_integrity_errors"]
    assert checked["calculation_replay_rate"] == 1 and checked["checked_charts"] == 1
    for target in ("metric", "calculation", "chart"):
        forged = copy.deepcopy(packet)
        if target == "metric":
            forged["metrics"][0]["value"] = "900"
        elif target == "calculation":
            forged["calculations"][0]["value"] = "99"
        else:
            forged["chart_audit"][0]["context"][0]["geography"] = "US"
        assert check_numeric_layer(forged, valid)["metric_integrity_errors"]


@pytest.mark.parametrize(("field", "value"), [("geography", "US"), ("price_basis", "constant"),
    ("value_kind", "forecast"), ("scope", None), ("unit", "CNY bn"), ("period", "2023")])
def test_chart_withholds_incomparable_conflicting_or_incomplete_inputs(field, value):
    claims, citations, metrics = source_pair()
    metrics[1] = replace(metrics[1], **{field: value})
    clean, _, _, _, audit, _ = validate_charts(trend_graph(metrics), metrics, claims, citations)
    assert not clean.market.series and not build_all(clean)["market_plot"]
    assert audit[0]["status"] == "withheld"


def test_chart_converts_scale_and_rejects_unrelated_claim_ids():
    claims, citations, metrics = source_pair()
    metrics[1] = replace(metrics[1], value=Decimal(120000), unit="USD mn")
    clean, _, _, records, _, _ = validate_charts(trend_graph(metrics), metrics, claims, citations)
    assert clean.market.series[1].value == 120 and clean.market.series[1].unit == "USD bn"
    assert any(r["operation"] == "unit_conversion" for r in records)
    bad = trend_graph(metrics)
    bad.market.series[0].claim_ids = ["c2"]
    clean, *_ = validate_charts(bad, metrics, claims, citations)
    assert not build_all(clean)["market_plot"]


def test_tam_hierarchy_and_share_sum_are_checked_before_rendering():
    tam = observation(name="tam", value="100")
    sam = observation("m2", "120", name="sam")
    graph = ReportKnowledgeGraph("Solar", market=MarketSize(tam=100, sam=120, unit="USD bn",
                                                         claim_ids=tam.claim_ids + sam.claim_ids))
    clean, _, _, _, audit, _ = validate_charts(graph, [tam, sam], [], [])
    assert audit[0]["status"] == "withheld" and not build_all(clean)["market_callout"]
    shares = [observation("s1", "70", name="market_share", entity="A", unit="%", rate_basis="sales"),
              observation("s2", "40", name="market_share", entity="B", unit="%", rate_basis="sales")]
    graph = ReportKnowledgeGraph("Solar", competitors=[Competitor(m.entity, m.value, claim_ids=m.claim_ids) for m in shares])
    clean, *_ = validate_charts(graph, shares, [], [])
    assert not build_all(clean)["competitive_bar"]


def test_share_total_cannot_round_away_an_excess_under_a_low_precision_context():
    shares = [observation("s1", "50.0000000000000000000000001", name="market_share", entity="A", unit="%", rate_basis="sales"),
              observation("s2", "50", name="market_share", entity="B", unit="%", rate_basis="sales")]
    graph = ReportKnowledgeGraph("Solar", competitors=[Competitor(m.entity, m.value, claim_ids=m.claim_ids) for m in shares])
    with localcontext() as context:
        context.prec = 5
        clean, *_ = validate_charts(graph, shares, [], [])
    assert not build_all(clean)["competitive_bar"]


def test_comparison_preserves_periods_sources_and_duplicate_values():
    cells = [{"source_url": "https://example.org/a", "time_scope": "2023", "value": "90 USD bn"},
             {"source_url": "https://example.org/a", "time_scope": "2024", "value": "120 USD bn"},
             {"source_url": "https://example.org/b", "time_scope": "2024", "value": "130 USD bn"},
             {"source_url": "https://example.org/b", "time_scope": "2024", "value": "140 USD bn"}]
    rendered = comparison_matrix([{"entity": "Solar", "attribute": "market_size", "cells": cells}])
    assert all(value in rendered for value in ("90 USD bn", "120 USD bn", "130 USD bn / 140 USD bn", "2023", "2024"))


@pytest.mark.parametrize("name,unit", [("specific_capacity", "mAh/g"), ("areal_capacity", "mAh/cm2"),
    ("ionic_conductivity", "mS/cm"), ("capacity_retention", "%"),
    ("gravimetric_energy_density", "Wh/kg"), ("cycle_life", "cycles"),
    ("specific_capacity.growth_rate", "%")])
def test_scientific_part_of_observations_never_become_market_share(name, unit):
    conditions = {"temperature": "25 °C", "pressure": "5 MPa", "test_method": "EIS",
        "sample_form": "pellet", "cell_type": "full_cell", "mass_basis": "active material",
        "c_rate": "1 C", "voltage_window": "2.5-4.2 V", "cycle_count": "100 cycles",
        "retention_threshold": "80%"}
    part = observation("part", "40", name=name, unit=unit, entity="Part", part_of="Total",
        boundary="cell", conditions=conditions, rate_basis="test")
    total = replace(part, id="total", entity="Total", part_of=None, value=Decimal(100))
    with pytest.raises(MetricError, match="Scientific observations cannot be converted into market share"):
        calculate("share", [part, total], output_id="forbidden-share")


def test_financial_market_share_and_scientific_unit_conversion_still_work():
    part = observation("part", "40", entity="Part", part_of="Total")
    total = observation("total", "100", entity="Total")
    share, record = calculate("share", [part, total], output_id="share")
    assert share.name == "market_share" and share.value == 40 and share.unit == "%"
    assert replay(record, {part.id: part, total.id: total}).value == 40
    scientific = Metric("science", "ionic_conductivity", "LLZO", Decimal("1.2"), "mS/cm")
    converted, conversion = calculate("unit_conversion", [scientific], output_id="converted", target_unit="S/m")
    assert converted.name == "ionic_conductivity" and converted.value == Decimal("0.12")
    assert replay(conversion, {scientific.id: scientific}).value == Decimal("0.12")
