"""Scientific values retain dimensions, source-local conditions and unknowns."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal, Inexact, localcontext

import pytest

from core_engine.report.metrics import (
    Metric,
    MetricError,
    catalog_from_claims,
    comparison_errors,
    convert_value,
    decimal_value,
    header_observation,
    labelled_observations,
    observed_quantities,
    parse_unit,
    quantities,
)
from core_engine.report.models import Claim, SentenceCitation


@pytest.mark.parametrize("value,source,target,expected", [
    ("1", "mS/cm", "S/m", "0.1"), ("0.01", "S/cm", "mS/cm", "10"),
    ("100", "μS cm⁻¹", "mS/cm", "0.1"), ("100", "uS/cm", "S/m", "0.01"),
    ("300", "Wh kg⁻¹", "Wh/g", "0.3"), ("700", "Wh/L", "mWh/cm³", "700"),
    ("150", "mAh g⁻¹", "Ah/kg", "150"), ("3", "mAh/cm²", "Ah/m²", "30"),
    ("0.5", "mA cm⁻²", "A/m²", "5"), ("0.1", "A/cm²", "mA/cm2", "100"),
    ("10", "ohm cm²", "Ω m²", "0.001"), ("1000", "mΩ cm²", "Ω cm2", "1"),
    ("1", "MPa", "kPa", "1000"), ("25", "°C", "K", "298.15"),
    ("1000", "cycle", "cycles", "1000"),
])
def test_scientific_unit_conversion_is_exact(value, source, target, expected):
    with localcontext() as context:
        context.prec = 3
        context.traps[Inexact] = True
        assert convert_value(decimal_value(value), source, target) == Decimal(expected)


@pytest.mark.parametrize("unit", ["MS/cm", "ms/cm", "S/CM", "MAh/g", "mah/g", "wh/kg", "Wh/Kg", "mA/CM2", "mPa"])
def test_case_sensitive_scientific_units_do_not_confuse_milli_and_mega(unit):
    with pytest.raises(MetricError):
        parse_unit(unit)


@pytest.mark.parametrize("left,right", [("Wh/kg", "Wh/L"), ("mAh/g", "mAh/cm2"),
    ("mS/cm", "mA/cm2"), ("Ω", "Ω cm2"), ("cycles", "%")])
def test_scientific_dimensions_are_not_silently_interchanged(left, right):
    with pytest.raises(MetricError):
        convert_value(Decimal(1), left, right)


@pytest.mark.parametrize("text,expected", [("1.2 × 10⁻³", "0.0012"), ("1.2×10−3", "0.0012"),
    ("1.2 × 10^{-3}", "0.0012"), ("10⁻³", "0.001"), ("1.2E-3", "0.0012"), (".5", "0.5")])
def test_scientific_notation_keeps_unicode_exponent_signs(text, expected):
    assert decimal_value(text) == Decimal(expected)
    assert quantities(text + " S/cm") == [(Decimal(expected), "S/cm")]


@pytest.mark.parametrize("text", ["1.2×10^{3", "1.2×10^3}", "10^{3", "1.2×10^1000", "10⁻⁹⁹⁹"])
def test_malformed_or_oversized_scientific_notation_is_rejected(text):
    with pytest.raises(MetricError):
        decimal_value(text)


@pytest.mark.parametrize("text", ["1–3 mS/cm", "1 to 3 mS/cm", "1.2±0.1 mS/cm", ">1.2 mS/cm",
    "≤1.2 mS/cm", "1.2 mS/cm ± 0.1", "1 mS/cm to 3 mS/cm"])
def test_ranges_limits_and_uncertainty_are_not_unlabelled_point_estimates(text):
    assert quantities(text) == []
    assert header_observation("Ionic conductivity (mS/cm)", text) is None


@pytest.mark.parametrize("header,cell,value,unit", [
    ("Ionic conductivity (mS cm−1)", "1.2", "1.2", "mS/cm"),
    ("Specific energy (Wh kg⁻¹)", "300", "300", "Wh/kg"),
    ("Volumetric energy density (Wh/L)", "700", "700", "Wh/L"),
    ("Temperature (°C)", "25", "25", "°C"),
    ("Stack pressure (MPa)", "5", "5", "MPa"),
])
def test_unit_in_header_is_explicitly_bound_to_its_own_scalar_cell(header, cell, value, unit):
    assert header_observation(header, cell) == (Decimal(value), unit)


def test_ambiguous_or_missing_header_units_remain_unknown():
    assert header_observation("Ionic conductivity", "1.2") is None
    assert header_observation("Ionic conductivity (mS/cm or S/m)", "1.2") is None


def conductivity(identifier="m1", **kwargs):
    fields = {"id": identifier, "name": "ionic_conductivity", "entity": "LLZO", "value": "1.2",
              "unit": "mS/cm", "boundary": "electrolyte", "conditions": {"temperature": "25 °C",
              "pressure": "5 MPa", "test_method": "EIS", "sample_form": "pellet"}}
    fields.update(kwargs)
    return Metric(**fields)


def test_science_comparisons_use_experimental_conditions_instead_of_financial_geography():
    a = conductivity(geography="China")
    b = conductivity("m2", value="0.12", unit="S/m", geography="Europe", conditions={
        "temperature": "298.15 K", "pressure": "5000 kPa",
        "test_method": "electrochemical impedance spectroscopy", "sample_form": "pellet"})
    assert not comparison_errors(a, b)
    assert not comparison_errors(a, b, same_period=True)
    assert a.period is None and a.value_kind is None and a.rate_basis is None


@pytest.mark.parametrize("key,value,error", [
    ("temperature", None, "missing_condition_temperature"),
    ("temperature", "room temperature", "missing_condition_temperature"),
    ("temperature", "60 °C", "different_condition_temperature"),
    ("pressure", "10 MPa", "different_condition_pressure"),
    ("test_method", "DC polarization", "different_condition_test_method"),
    ("sample_form", "thin film", "different_condition_sample_form"),
])
def test_unknown_or_mismatched_test_conditions_block_comparisons(key, value, error):
    a = conductivity()
    b = conductivity("m2", conditions={**a.conditions, key: value})
    assert error in comparison_errors(a, b)


def test_unknown_boundaries_and_unmatched_optional_conditions_are_not_filled_from_memory():
    a = conductivity()
    assert "missing_boundary" in comparison_errors(a, replace(a, boundary=None))
    assert "different_boundary" in comparison_errors(a, replace(a, boundary="cell"))
    b = replace(a, conditions={**a.conditions, "electrolyte_thickness": "100 μm"})
    assert "missing_condition_electrolyte_thickness" in comparison_errors(a, b)
    assert "different_entity" in comparison_errors(a, replace(a, entity="LATP"))
    assert not comparison_errors(a, replace(a, entity="LATP"), allow_entity=True)


def energy(cell_type="full_cell", **kwargs):
    conditions = {"temperature": "25 °C", "pressure": "5 MPa", "test_method": "galvanostatic cycling",
                  "cell_type": cell_type, "mass_basis": "total cell mass", "c_rate": "0.1 C"}
    return Metric(id="energy", name="gravimetric_energy_density", entity="cell A", value="300",
                  unit="Wh/kg", boundary="cell", conditions=conditions, **kwargs)


def test_full_half_and_material_cell_boundaries_cannot_be_ranked_as_equivalent():
    a = energy()
    assert "different_condition_cell_type" in comparison_errors(a, energy("half_cell"))
    assert "different_boundary" in comparison_errors(a, replace(a, boundary="electrode"))
    b = replace(a, conditions={**a.conditions, "mass_basis": "active material"})
    assert "different_condition_mass_basis" in comparison_errors(a, b)
    assert not comparison_errors(a, replace(a, conditions={**a.conditions, "c_rate": "C/10"}))
    assert "missing_condition_rate" in comparison_errors(a, replace(a, conditions={
        key: value for key, value in a.conditions.items() if key != "c_rate"}))


def test_capacity_retention_is_not_market_share_and_does_not_use_a_sales_denominator():
    metric = Metric(id="retention", name="capacity_retention", entity="cell A", value="105", unit="%",
        boundary="cell", conditions={"temperature": "25 °C", "pressure": "5 MPa", "test_method": "cycling",
            "cell_type": "full_cell", "cycle_count": "100 cycles", "voltage_window": "2.5-4.2 V", "c_rate": "1 C"})
    assert not comparison_errors(metric, metric)
    assert metric.rate_basis is None  # No invented market-share/sales basis.
    with pytest.raises(MetricError, match="Market share"):
        Metric(id="share", name="market_share", entity="A", value="105", unit="%")
    assert "different_metric" in comparison_errors(metric,
        Metric(id="share", name="market_share", entity="A", value="80", unit="%"))


def evidence(text, metric=None, *, entity="LLZO", quote=None):
    claim = Claim("c1", text, entity=entity, metric=metric, reporting_status="source_reported")
    excerpt = quote or text
    citation = SentenceCitation("e1", "c1", "https://example.edu/paper", excerpt, 0, len(excerpt),
        1, None, hashlib.sha256(excerpt.encode()).hexdigest(), "verbatim")
    return catalog_from_claims([claim], [citation])


def row_quote():
    return ("Measured ionic conductivity of solid electrolytes.\n"
        "Electrolyte: LLZO | Ionic conductivity (mS cm−1): 1.2 | Temperature (°C): 25 | "
        "Stack pressure (MPa): 5 | Test method: EIS | Sample form: pellet")


def test_catalog_binds_table_header_units_conditions_and_equivalent_temperature_units():
    proposal = {"name": "ionic_conductivity", "value": "0.12", "unit": "S/m", "entity": "LLZO",
        "boundary": "electrolyte", "conditions": {"temperature": "298.15 K", "pressure": "5000 kPa",
        "test_method": "EIS", "sample_form": "pellet"}}
    metrics, issues = evidence(row_quote(), proposal)
    assert not issues and len(metrics) == 1
    assert metrics[0].evidence_ids == ["e1"] and metrics[0].source_versions
    assert metrics[0].conditions["temperature"] == "298.15 K"
    assert metrics[0].to_dict()["boundary"] == "electrolyte"
    assert not comparison_errors(metrics[0], metrics[0])


def test_unknown_conditions_do_not_block_metric_retention_or_get_guessed():
    text = "LLZO ionic conductivity was 1.2 mS/cm."
    metrics, issues = evidence(text, {"name": "ionic_conductivity", "value": "1.2", "unit": "mS/cm",
                                     "entity": "LLZO", "conditions": {"temperature": None}})
    assert not issues and metrics[0].boundary is None
    assert metrics[0].conditions == {"temperature": None}
    assert "missing_condition_temperature" in comparison_errors(metrics[0], metrics[0])
    assert "missing_boundary" in comparison_errors(metrics[0], metrics[0])


@pytest.mark.parametrize("condition,value", [("temperature", "80 °C"), ("pressure", "100 MPa"),
    ("cell_type", "full_cell"), ("test_method", "DC polarization")])
def test_catalog_rejects_invented_experimental_metadata(condition, value):
    proposal = {"name": "ionic_conductivity", "value": "1.2", "unit": "mS/cm", "entity": "LLZO",
                "conditions": {condition: value}}
    metrics, issues = evidence(row_quote(), proposal)
    assert not metrics and "No anchored" in issues[0]["reason"]


def test_pooled_samples_cannot_borrow_a_condition_from_another_row():
    quote = ("Sample: LLZO; Ionic conductivity (mS/cm): 1.2; Temperature (°C): 25; "
             "Sample: LATP; Ionic conductivity (mS/cm): 1.2; Temperature (°C): 80")
    proposal = {"name": "ionic_conductivity", "value": "1.2", "unit": "mS/cm", "entity": "LLZO",
                "conditions": {"temperature": "80 °C"}}
    assert not evidence(quote, proposal)[0]


def test_ionic_electronic_and_different_metric_values_remain_distinct():
    text = "LLZO ionic conductivity was 1 mS/cm and electronic conductivity was 0.01 mS/cm."
    assert observed_quantities(text, "ionic_conductivity") == [(Decimal(1), "mS/cm")]
    assert observed_quantities(text, "electronic_conductivity") == [(Decimal("0.01"), "mS/cm")]
    assert observed_quantities(text, "conductivity") == []
    forged = {"name": "ionic_conductivity", "value": "0.01", "unit": "mS/cm", "entity": "LLZO"}
    assert not evidence(text, forged)[0]
    assert "different_metric" in comparison_errors(conductivity(),
        conductivity("m2", name="electronic_conductivity"))


@pytest.mark.parametrize("label,unit,name", [("specific energy", "Wh/kg", "gravimetric_energy_density"),
    ("energy density", "Wh/kg", "gravimetric_energy_density"),
    ("energy density", "Wh/L", "volumetric_energy_density")])
def test_scientific_alias_and_explicit_units_resolve_mass_vs_volume(label, unit, name):
    metrics, issues = evidence(f"Cell A {label} was 300 {unit}.", entity="Cell A")
    assert not issues and metrics[0].name == name


def test_scientific_notation_catalog_is_bound_to_the_anchored_numeric_amount():
    text = "LLZO ionic conductivity was 1.2 × 10⁻³ S cm⁻¹."
    metric = {"name": "ionic_conductivity", "value": "1.2", "unit": "mS/cm", "entity": "LLZO"}
    assert evidence(text, metric)[0]
    metric["value"] = "12"
    assert not evidence(text, metric)[0]


@pytest.mark.parametrize("setting", ["room temperature", "RT", "ambient pressure", "25–40 °C", "~25 °C"])
def test_reported_qualitative_or_ranged_conditions_are_retained_but_never_imputed(setting):
    key = "pressure" if "pressure" in setting else "temperature"
    text = f"LLZO ionic conductivity was 1.2 mS/cm at {setting}."
    proposal = {"name": "ionic_conductivity", "value": "1.2", "unit": "mS/cm", "entity": "LLZO",
                "conditions": {key: setting}}
    metrics, issues = evidence(text, proposal)
    assert not issues and metrics[0].conditions[key] == setting
    assert f"missing_condition_{key}" in comparison_errors(metrics[0], metrics[0])


def test_preparation_temperature_cannot_support_measurement_temperature():
    text = ("Electrolyte: LLZO | Drying temperature [°C]: 90 | "
            "Ionic conductivity at 25 °C [mS cm−1]: 8.6 | Amount of solvent [cm3]: 10 | "
            "Solid loading [wt%]: 20.3 | Ball size [mm]: 1")
    proposal = {"name": "ionic_conductivity", "value": "8.6", "unit": "mS/cm", "entity": "LLZO",
        "conditions": {"temperature": "25 °C", "drying_temperature": "90 °C", "solvent_amount": "10 cm3",
                       "solid_loading": "20.3 wt%", "ball_size": "1 mm"}}
    metrics, issues = evidence(text, proposal)
    assert len(metrics) == 1 and not issues
    proposal["conditions"]["temperature"] = "90 °C"
    assert not evidence(text, proposal)[0]


def test_explicit_lithium_conductivity_symbol_does_not_become_electronic_conductivity():
    text = "Halide SEs: LLZO | σLi (mS cm−1) @RT: 0.446"
    proposal = {"name": "ionic_conductivity", "value": "0.446", "unit": "mS/cm", "entity": "LLZO"}
    assert evidence(text, proposal)[0]
    proposal["name"] = "electronic_conductivity"
    assert not evidence(text, proposal)[0]


def test_compound_capacity_temperature_does_not_poison_conductivity_measurement_binding():
    text = ("Polymer Matrix: PEO | Li Salt: LiTFSI | Temp. (°C): 25 | Ionic Conductivity (S cm^−1): "
            "3.08 × 10^−4 | Capacity mA h g^−1 (Current Density, Cycle Index) (Test Temperature): "
            "~118 (0.1C, 350) (80 °C)")
    proposal = {"name": "ionic_conductivity", "value": "0.000308", "unit": "S/cm", "entity": "PEO",
                "conditions": {"temperature": "25 °C", "lithium_salt": "LiTFSI"}}
    metrics, issues = evidence(text, proposal, entity="PEO")
    assert len(metrics) == 1 and not issues
    proposal["conditions"]["temperature"] = "80 °C"
    assert not evidence(text, proposal, entity="PEO")[0]


def test_fabrication_pressure_cannot_be_borrowed_as_measurement_stack_pressure():
    text = ("Electrolyte: LLZO | Ionic conductivity (mS/cm): 1.2 | "
            "Pellet pressing pressure (MPa): 300 | Stack pressure (MPa): 5")
    proposal = {"name": "ionic_conductivity", "value": "1.2", "unit": "mS/cm", "entity": "LLZO",
                "conditions": {"pressure": "5 MPa", "fabrication_pressure": "300 MPa"}}
    assert evidence(text, proposal)[0]
    proposal["conditions"]["pressure"] = "300 MPa"
    assert not evidence(text, proposal)[0]


@pytest.mark.parametrize("unit,name", [("mAh/g", "specific_capacity"), ("mAh/cm2", "areal_capacity")])
def test_generic_capacity_label_requires_an_explicit_dimension(unit, name):
    text = f"Sample: LLZO | Capacity ({unit}): 150"
    metrics, issues = evidence(text, {"name": name, "unit": unit, "value": "150", "entity": "LLZO"})
    assert not issues and metrics[0].name == name
    assert not evidence(text, {"name": "specific_capacity" if name == "areal_capacity" else "areal_capacity",
                               "unit": unit, "value": "150", "entity": "LLZO"})[0]


def test_labelled_scientific_values_stay_bound_to_their_own_field_and_unit():
    text = ("Ionic conductivity (mS/cm): 1.2 | Temperature (°C): 25 | Stack pressure (MPa): 5\n"
            "Ionic conductivity (S/m): 0.9 | Temperature (K): 298.15 | Cell: LLZO")
    assert labelled_observations(text) == [(Decimal("1.2"), "mS/cm"), (Decimal(25), "°C"),
        (Decimal(5), "MPa"), (Decimal("0.9"), "S/m"), (Decimal("298.15"), "K")]


def test_labelled_numeric_helper_does_not_supply_missing_units_or_financial_inference():
    assert labelled_observations("Ionic conductivity: 1.2 | Temperature: 25 | Revenue (USD bn): 90") == []
    assert labelled_observations("Market share (%): 80 | Capacity retention (%): 90") == [(Decimal(90), "%")]
    assert labelled_observations("Ionic conductivity (mS/cm): 1–2 | Temperature (°C): ~25") == []
