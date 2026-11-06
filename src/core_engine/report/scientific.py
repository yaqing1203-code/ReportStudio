"""Conservative table observations and explicit experimental-comparison gaps.

Table cells are candidates, not certified facts. They pass through the same
publication and source-grounding pipeline as model-extracted observations.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from core_engine.report.llm import ExtractedClaim
from core_engine.report.methods import method_condition_evidence
from core_engine.report.metrics import (
    METRIC_ALIASES,
    Metric,
    MetricError,
    comparison_errors,
    decimal_text,
    decimal_value,
    header_observation,
    mentions_metric,
    parse_unit,
    quantities,
)
from core_engine.report.models import Source

SCIENTIFIC_NAMES = {
    "ionic_conductivity", "electronic_conductivity", "conductivity",
    "gravimetric_energy_density", "volumetric_energy_density", "specific_capacity",
    "areal_capacity", "current_density", "critical_current_density", "cycle_life",
    "capacity_retention", "coulombic_efficiency", "interfacial_impedance", "impedance",
}
_FIELD_LABELS = {"boundary": "测量边界", "temperature": "温度", "pressure": "压力", "test_method": "测试方法",
    "sample_form": "样品形态", "cell_type": "电池构型", "current_density": "电流密度", "c_rate": "倍率",
    "rate": "电流或倍率", "voltage_window": "电压窗口", "cycle_count": "循环次数",
    "retention_threshold": "容量保持阈值", "mass_basis": "质量口径", "volume_basis": "体积口径",
    "electrolyte_thickness": "电解质厚度", "electrode_loading": "电极载量"}

_CONDITIONS = {
    "drying_temperature": ("drying temperature", "干燥温度"),
    "sintering_temperature": ("sintering temperature", "烧结温度"),
    "fabrication_pressure": ("fabrication pressure", "pellet pressing pressure", "制备压力", "压片压力"),
    "solvent_amount": ("amount of solvent", "solvent amount", "溶剂量"),
    "solid_loading": ("solid loading", "固体含量"),
    "ball_size": ("ball size", "球径"),
    "lithium_salt": ("li salt", "lithium salt", "锂盐"),
    "temperature": ("temperature", "temp.", "温度"),
    "pressure": ("stack pressure", "operating pressure", "pressure", "压力"),
    "test_method": ("test method", "measurement method", "测试方法", "测量方法"),
    "sample_form": ("sample form", "sample type", "样品形态"),
    "cell_type": ("cell type", "cell configuration", "电池类型", "电池构型"),
    "current_density": ("current density", "电流密度"),
    "c_rate": ("c-rate", "c rate", "倍率"),
    "voltage_window": ("voltage window", "voltage range", "电压窗口", "电压范围"),
    "cycle_count": ("cycle count", "cycles", "循环次数"),
    "retention_threshold": ("retention threshold", "保持率阈值"),
    "mass_basis": ("mass basis", "质量口径"),
    "volume_basis": ("volume basis", "体积口径"),
    "electrolyte_thickness": ("electrolyte thickness", "电解质厚度"),
    "electrode_loading": ("electrode loading", "cathode loading", "areal loading", "载量"),
}
_ENTITY = re.compile(r"\b(?:sample|materials?|electrolytes?|composition|cell|specimen|polymer matrix|matrix|compound|sse|name)\b|样品|材料|电解质|电池", re.IGNORECASE)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def scalar(value: str) -> Decimal | None:
    """Exact scalars only: ranges, errors, inequalities and approximations stay data."""
    text = unicodedata.normalize("NFKC", value).strip().replace("−", "-").replace("–", "-")
    text = re.sub(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[×x]\s*10\s*\^?\s*([+-]?\d+)", r"\1e\2", text)
    if not _NUMBER.fullmatch(text):
        return None
    try:
        return decimal_value(text)
    except MetricError:
        return None


def _unit(header: str) -> str | None:
    # Keep case: mS and MS are physically different. Only units explicitly in
    # this cell's header are candidates; another row/column cannot donate one.
    candidates = re.findall(r"[([]([^()\[\]]+)[)\]]", header)
    candidates += [header.rsplit(" / ", 1)[-1]]
    for candidate in candidates:
        try:
            return parse_unit(candidate.strip()).symbol
        except MetricError:
            continue
    return None


def _name(header: str, unit: str) -> str | None:
    # Match the most specific metric before its generic suffix.
    matched = [name for name in SCIENTIFIC_NAMES if name in METRIC_ALIASES and mentions_metric(header, name)]
    if "electronic_conductivity" in matched:
        return "electronic_conductivity"
    if "ionic_conductivity" in matched:
        return "ionic_conductivity"
    if "critical_current_density" in matched:
        return "critical_current_density"
    if "interfacial_impedance" in matched:
        return "interfacial_impedance"
    if re.search(r"energy density|specific energy|能量密度|比能量", header, re.IGNORECASE):
        for name in ("gravimetric_energy_density", "volumetric_energy_density"):
            try:
                target = "Wh/kg" if name == "gravimetric_energy_density" else "Wh/L"
                if parse_unit(unit).dimension == parse_unit(target).dimension:
                    return name
            except MetricError:
                continue
    if re.search(r"\bcapacity\b|容量", header, re.IGNORECASE):
        for name, target in (("specific_capacity", "mAh/g"), ("areal_capacity", "mAh/cm2")):
            try:
                if parse_unit(unit).dimension == parse_unit(target).dimension:
                    return name
            except MetricError:
                continue
    return max(matched, key=len) if matched else None


def _condition(header: str, cell: str) -> tuple[str, str] | None:
    label = header.strip().casefold()
    for name, aliases in _CONDITIONS.items():
        # A compound Capacity(...Test Temperature...) column is not a dedicated
        # temperature observation. Processing/drying temperature stays separate.
        if any(re.match(r"^" + re.escape(alias.casefold()) + r"(?:\b|\s|[.([:/])", label)
               for alias in aliases):
            value = cell.strip()
            if not value or value.casefold() in {"-", "–", "−", "—", "/", "n/a", "na", "unknown", "not reported"}:
                return None
            # Explicit header units may qualify a scalar condition. Never supply
            # a unit from general knowledge or treat a missing cell as ambient.
            unit = re.findall(r"[([]([^()\[\]]+)[)\]]", header)
            if scalar(value) is not None and unit:
                value += " " + unit[-1].strip()
            return name, value
    return None


def table_candidates(
    source: Source, max_claims: int = 12, *, _row_span: tuple[int, int] | None = None,
) -> tuple[list[ExtractedClaim], dict]:
    """Only valid, anchored scalar cells produce candidates; verification follows."""
    candidates, skipped = [], []
    seen = set()
    if max_claims <= 0:
        return [], {"table_candidates": 0, "skipped_rows": [], "budget_exhausted": True}
    tables = source.metadata.get("tables", [])
    for table in tables[:30] if isinstance(tables, list) else []:
        if not isinstance(table, dict):
            continue
        headers = table.get("headers", [])
        records = table.get("row_records", [])
        if not isinstance(headers, list) or not all(isinstance(header, str) for header in headers):
            continue
        for row in records[:200] if isinstance(records, list) else []:
            if not isinstance(row, dict):
                continue
            cells, quote = row.get("cells", []), row.get("text", "")
            start, end = row.get("start"), row.get("end")
            if _row_span is not None and (start, end) != _row_span:
                continue
            if (not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start
                    or not isinstance(quote, str) or source.text[start:end] != quote or not quote
                    or not isinstance(cells, list) or len(cells) != len(headers)
                    or row.get("structure_status") in {"ambiguous_data_span", "ambiguous"}):
                skipped.append({"table_id": table.get("id"), "row": row.get("row_index"), "reason": "invalid_or_ambiguous_row"})
                continue
            entity_column = next(((header, str(cell).strip()) for header, cell in zip(headers, cells, strict=True)
                                  if _ENTITY.search(header) and str(cell).strip()
                                  and _condition(header, str(cell)) is None
                                  and not _unit(header) and not quantities(str(cell))
                                  and scalar(str(cell)) is None), None)
            entity = entity_column[1] if entity_column else None
            conditions = dict(item for header, cell in zip(headers, cells, strict=True)
                              if (item := _condition(header, str(cell))) is not None)
            for header, cell in zip(headers, cells, strict=True):
                unit = _unit(header)
                amount = scalar(str(cell))
                observation = header_observation(header, str(cell))
                if observation is not None:
                    amount, unit = observation
                # Also accept an exact scalar followed by an explicit cell unit.
                if amount is None:
                    normalized = unicodedata.normalize("NFKC", str(cell)).strip().replace("−", "-")
                    match = re.fullmatch(r"(" + _NUMBER.pattern + r")\s*(\S.*)", normalized)
                    if match:
                        try:
                            amount, unit = decimal_value(match[1]), parse_unit(match[2]).symbol
                        except MetricError:
                            pass
                name = _name(header, unit) if unit else None
                if amount is None or not unit or not name or entity is None:
                    continue
                identity = (table.get("id"), row.get("row_index"), header, decimal_text(amount), unit)
                if identity in seen:
                    continue
                seen.add(identity)
                metric: dict[str, Any] = {"name": name, "value": decimal_text(amount), "unit": unit, "entity": entity,
                          "conditions": {key: value for key, value in conditions.items()
                                         if key != name and (key != "cycle_count" or name != "cycle_life")}}
                conflicting_temperature = False
                measured_temp = re.search(r"(?:\bat\s*|@\s*)([+-]?\d+(?:\.\d+)?)\s*(°\s*C|K)\b", header, re.IGNORECASE)
                if measured_temp:
                    explicit = measured_temp[1] + " " + re.sub(r"\s+", "", measured_temp[2])
                    existing = metric["conditions"].get("temperature")
                    if existing and existing.replace(" ", "") != explicit.replace(" ", ""):
                        # Conflicting table contexts require review, not a guessed
                        # winner; keep the value but leave measurement temp unknown.
                        metric["conditions"].pop("temperature", None)
                        conflicting_temperature = True
                        skipped.append({"table_id": table.get("id"), "row": row.get("row_index"),
                                        "reason": "conflicting_measurement_temperature"})
                    else:
                        metric["conditions"]["temperature"] = explicit
                if not conflicting_temperature:
                    condition_evidence = method_condition_evidence(source, table, row, name, metric["conditions"])
                    if condition_evidence:
                        metric["condition_evidence"] = condition_evidence
                # Boundary can only come from explicit source wording; no full-cell
                # assertion is supplied merely because the topic is batteries.
                if entity_column and re.search(r"\belectrolyte\b|电解质", entity_column[0], re.IGNORECASE):
                    metric["boundary"] = "electrolyte"
                elif entity_column and re.search(r"\bcell\b|电芯", entity_column[0], re.IGNORECASE):
                    metric["boundary"] = "cell"
                context = "; ".join(f"{_CONDITIONS[key][0]}: {value}" for key, value in metric["conditions"].items())
                label = METRIC_ALIASES[name][0]
                text = f"{entity}: {label} = {decimal_text(amount)} {unit}" + (f"; {context}" if context else "") + "."
                candidates.append(ExtractedClaim(text=text, candidate_source_urls=[source.url], source_quote=quote,
                    entity=entity, attribute=name, value=f"{decimal_text(amount)} {unit}",
                    qualifier=f"table={table.get('id')}; row={row.get('row_index')}", metric=metric,
                    source_windows={source.url: [[start, end]]}))
                if len(candidates) >= max_claims:
                    return candidates, {"table_candidates": len(candidates), "skipped_rows": skipped, "budget_exhausted": True}
    return candidates, {"table_candidates": len(candidates), "skipped_rows": skipped, "budget_exhausted": False}


def anchored_table_candidate(
    source: Source, start: int, end: int, name: str, entity: str | None, value: Decimal, unit: str,
) -> ExtractedClaim | None:
    """Recover an observation from a citation contained by exactly one table row."""
    if not 0 <= start < end <= len(source.text):
        return None
    covering_span = None
    tables = source.metadata.get("tables", [])
    for table in tables[:30] if isinstance(tables, list) else []:
        rows = table.get("row_records", []) if isinstance(table, dict) else []
        for row in rows[:200] if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            left, right = row.get("start"), row.get("end")
            if type(left) is int and type(right) is int and left <= start < end <= right:
                if covering_span is not None:
                    return None
                covering_span = (left, right)
    if covering_span is None:
        return None
    candidates, audit = table_candidates(source, max_claims=200, _row_span=covering_span)
    if audit["budget_exhausted"]:
        return None
    matching = [item for item in candidates if item.metric is not None and item.attribute == name
                and item.entity == entity and item.metric["value"] == decimal_text(value) and item.metric["unit"] == unit]
    return matching[0] if len(matching) == 1 else None


def comparison_audit(metrics: Iterable[Metric]) -> dict:
    observations = [metric for metric in metrics if metric.name in SCIENTIFIC_NAMES]
    rows, pairs = [], []
    methods: list[dict] = []
    method_ids: dict[tuple, str] = {}
    for metric in observations:
        missing = comparison_errors(metric, metric, allow_entity=True, same_period=True)
        row_methods = {}
        for key, witness in getattr(metric, "condition_evidence", {}).items():
            span = (witness["source_url"], witness["source_version"], witness["start"], witness["end"])
            if span not in method_ids:
                method_id = f"M{len(methods) + 1}"
                method_ids[span] = method_id
                methods.append({"id": method_id, **{key: witness[key] for key in (
                    "source_url", "source_version", "start", "end", "quote", "locator", "applicability")}})
            row_methods[key] = method_ids[span]
        rows.append({"id": metric.id, "name": metric.name, "entity": metric.entity,
                     "value": decimal_text(metric.value), "unit": metric.unit,
                     "boundary": getattr(metric, "boundary", None), "conditions": getattr(metric, "conditions", {}),
                     "missing": missing,
                     "missing_labels": [_FIELD_LABELS.get(key.removeprefix("missing_condition_").removeprefix("missing_"), key)
                                        for key in missing],
                     "evidence_ids": metric.evidence_ids, "claim_ids": metric.claim_ids, "method_ids": row_methods})
    for index, left in enumerate(observations):
        for right in observations[index + 1:]:
            if left.name != right.name:
                continue
            errors = comparison_errors(left, right, allow_entity=True, same_period=True)
            pairs.append({"left": left.id, "right": right.id, "comparable": not errors, "reasons": errors})
    return {"observations": rows, "pairs": pairs, "methods": methods,
            "registered": len(rows), "comparable_pairs": sum(pair["comparable"] for pair in pairs),
            "condition_scope": "anchored_observation_and_methods" if methods else "anchored_observation_only",
            "note": "Comparability applies only to explicit recorded conditions; it does not establish independent replication. "
                    "Missing conditions refer to the anchored observation, not proof that the full article omits them."}
