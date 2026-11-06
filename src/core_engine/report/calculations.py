"""Deterministic metric calculations with replayable inputs and rounding policy."""
from __future__ import annotations

import re
from dataclasses import replace
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation, localcontext

from core_engine.report.metrics import (
    SCIENTIFIC_METRICS,
    Metric,
    MetricError,
    comparison_errors,
    convert_value,
    decimal_text,
    parse_unit,
)

PRECISION = 64
QUANTUM = Decimal("0.000001")


def _year(metric: Metric) -> int:
    if not metric.period or not re.fullmatch(r"(?:19|20|21)\d{2}", metric.period):
        raise MetricError("Annual calculations require an explicit calendar-year period")
    return int(metric.period)


def calculate(operation: str, inputs: list[Metric], *, output_id: str,
              target_unit: str | None = None) -> tuple[Metric, dict]:
    if not inputs or len({metric.id for metric in inputs}) != len(inputs):
        raise MetricError("Missing or duplicate calculation inputs")
    first = inputs[0]
    with localcontext(Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)):
        formula = ""
        period = first.period
        if operation == "unit_conversion":
            if len(inputs) != 1 or not target_unit:
                raise MetricError("Unit conversion needs one input and a target unit")
            value = convert_value(first.value, first.unit, target_unit)
            unit = parse_unit(target_unit).symbol
            formula = "input * source_unit_factor / target_unit_factor"
            name = first.name
        elif operation in {"growth_rate", "cagr", "percentage_point_change"}:
            if len(inputs) != 2:
                raise MetricError("This calculation requires [earlier, later] observations")
            earlier, later = inputs
            errors = comparison_errors(earlier, later)
            if errors:
                raise MetricError(", ".join(errors))
            years = _year(later) - _year(earlier)
            if not 0 < years <= 100:
                raise MetricError("Periods must be increasing, bounded calendar years")
            before = earlier.value
            after = convert_value(later.value, later.unit, earlier.unit)
            if operation == "percentage_point_change":
                if parse_unit(earlier.unit).dimension != "ratio":
                    raise MetricError("Percentage-point change requires ratios, not amounts or pp inputs")
                value = (convert_value(after, earlier.unit, "%")
                         - convert_value(before, earlier.unit, "%"))
                unit = "pp"
                formula = "later_in_percent - earlier_in_percent"
            else:
                if before <= 0 or after < 0:
                    raise MetricError("Growth/CAGR needs a positive base and nonnegative end value")
                if operation == "growth_rate":
                    if years != 1:
                        raise MetricError("Year-on-year growth requires consecutive years; use CAGR for gaps")
                    value = (after / before - 1) * 100
                    formula = "(later_in_base_unit / earlier - 1) * 100"
                else:
                    if after <= 0:
                        raise MetricError("CAGR requires strictly positive endpoints")
                    value = ((after / before) ** (Decimal(1) / Decimal(years)) - 1) * 100
                    formula = "((later_in_base_unit / earlier) ** (1 / elapsed_years) - 1) * 100"
                unit = "%"
            name = first.name + "." + operation
            period = f"{earlier.period}..{later.period}"
            # Reject excessive outputs before quantization can exceed context
            # precision; callers receive a domain error rather than a crash.
            if abs(value.adjusted()) > 50:
                raise MetricError("Calculated value exceeds the supported magnitude")
            try:
                value = value.quantize(QUANTUM)
            except InvalidOperation as exc:
                raise MetricError("Calculated value exceeds the supported precision") from exc
        elif operation == "share":
            if len(inputs) != 2:
                raise MetricError("Share requires [part, total]")
            if any(metric.name.split(".", 1)[0] in SCIENTIFIC_METRICS for metric in inputs):
                raise MetricError("Scientific observations cannot be converted into market share")
            part, total = inputs
            errors = comparison_errors(part, total, allow_entity=True, same_period=True)
            if errors or not part.part_of or part.part_of != total.entity:
                raise MetricError("Share requires comparable observations and an explicit part_of denominator")
            denominator = convert_value(total.value, total.unit, part.unit)
            if denominator <= 0 or part.value < 0 or part.value > denominator:
                raise MetricError("Share must have 0 <= part <= positive total")
            value = (part.value / denominator * 100).quantize(QUANTUM)
            unit, formula, name = "%", "part / total_in_part_unit * 100", "market_share"
        else:
            raise MetricError("Unknown calculation operation")
    output = replace(
        first, id=output_id, name=name, value=value, unit=unit, period=period, origin="calculated",
        rate_basis=(f"part_of:{inputs[1].entity}" if operation == "share" else first.rate_basis),
        claim_ids=sorted({cid for metric in inputs for cid in metric.claim_ids}),
        evidence_ids=sorted({eid for metric in inputs for eid in metric.evidence_ids}),
        source_versions=sorted({version for metric in inputs for version in metric.source_versions}))
    record = {
        "id": f"calc-{output_id}", "operation": operation,
        "input_ids": [metric.id for metric in inputs], "output_id": output_id,
        "inputs": [{"id": metric.id, "value": decimal_text(metric.value), "unit": metric.unit} for metric in inputs],
        "value": decimal_text(output.value), "unit": output.unit, "period": output.period,
        "formula": formula, "precision": PRECISION, "rounding": "ROUND_HALF_EVEN",
        "quantum": None if operation == "unit_conversion" else str(QUANTUM),
        "target_unit": target_unit, "evidence_ids": output.evidence_ids,
    }
    return output, record


def replay(record: dict, metrics: dict[str, Metric]) -> Metric:
    inputs = [metrics[identifier] for identifier in record["input_ids"]]
    output, expected = calculate(record["operation"], inputs, output_id=record["output_id"],
                                 target_unit=record.get("target_unit"))
    if record != expected:
        raise MetricError("Calculation record does not match deterministic replay")
    return output
