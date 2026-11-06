"""Approve chart inputs against source-backed metrics before any numeric rendering."""
from __future__ import annotations

import copy
import re
from dataclasses import replace
from itertools import pairwise

from core_engine.report.calculations import calculate
from core_engine.report.kg import MarketDatum, ReportKnowledgeGraph
from core_engine.report.metrics import (
    Metric,
    MetricError,
    comparison_errors,
    convert_value,
    decimal_sum,
    decimal_value,
    parse_unit,
    quantities,
)
from core_engine.report.models import Claim, SentenceCitation
from core_engine.report.text import normalize_text


def _same_provenance(candidates: list[Metric]) -> bool:
    keys = {(m.entity, m.period, m.geography, m.scope, m.price_basis, m.rate_basis, m.value_kind)
            for m in candidates}
    return len(keys) <= 1


def _find(metrics: list[Metric], ids: list[str], name: str, value, unit: str = "",
          *, entity: str | None = None, period: str | None = None) -> Metric:
    amount = decimal_value(str(value))
    matches = []
    for metric in metrics:
        if metric.name != name or not metric.claim_ids or not set(metric.claim_ids).issubset(ids):
            continue
        if entity is not None and normalize_text(metric.entity or "") != normalize_text(entity):
            continue
        if period is not None and metric.period != period:
            continue
        try:
            if convert_value(amount, unit or metric.unit, metric.unit) == metric.value:
                matches.append(metric)
        except MetricError:
            continue
    if not matches or not _same_provenance(matches):
        raise MetricError("No unambiguous metric supports the proposed value, entity and period")
    return matches[0]


def validate_charts(kg: ReportKnowledgeGraph, metrics: list[Metric], claims: list[Claim],
                    citations: list[SentenceCitation], clusters=None):
    cleaned = copy.deepcopy(kg)
    cleaned.numeric_validated = True
    cleaned.validated_blocks = set()
    audit: list[dict] = []
    calculations: list[dict] = []
    calculated: list[Metric] = []
    issues: list[dict] = []
    by_claim = {claim.id: claim for claim in claims}
    grounded = {citation.claim_id for citation in citations}

    def record(block: str, status: str, reason="", observations=None):
        observation_list = observations or []
        audit.append({"block": block, "status": status, "reason": reason,
                      "metric_ids": [m.id for m in observation_list],
                      "evidence_ids": sorted({eid for m in observation_list for eid in m.evidence_ids}),
                      "context": [{"entity": m.entity, "period": m.period, "geography": m.geography,
                                   "scope": m.scope, "price_basis": m.price_basis,
                                   "rate_basis": m.rate_basis, "value_kind": m.value_kind,
                                   "unit": m.unit} for m in observation_list]})
        if status == "validated":
            cleaned.validated_blocks.add(block)

    def compute(operation, inputs, **kwargs):
        output, calculation = calculate(operation, inputs, output_id=f"d{len(calculated) + 1}", **kwargs)
        calculated.append(output)
        calculations.append(calculation)
        return output

    # Explicit part/whole relationships may produce shares, never inferred totals.
    for part in metrics:
        if not part.part_of:
            continue
        totals = [m for m in metrics if m.entity == part.part_of and m.period == part.period and m.name == part.name]
        if len(totals) == 1:
            try:
                compute("share", [part, totals[0]])
            except MetricError as exc:
                issues.append({"metric_id": part.id, "calculation": "share", "reason": str(exc)})

    # Percentage-point deltas retain their own unit; do not label them as growth.
    ratios = [metric for metric in metrics if parse_unit(metric.unit).dimension == "ratio"
              and re.fullmatch(r"(?:19|20|21)\d{2}", metric.period or "")]
    for earlier in ratios:
        candidates = [later for later in ratios if int(later.period or "0") == int(earlier.period or "0") + 1
                      and not comparison_errors(earlier, later)]
        earlier_count = sum(m.period == earlier.period and not comparison_errors(earlier, m) for m in ratios)
        if len(candidates) == earlier_count == 1:
            try:
                compute("percentage_point_change", [earlier, candidates[0]])
            except MetricError as exc:
                issues.append({"calculation": "percentage_point_change", "reason": str(exc)})

    # Validate chain names and direction as well; ID existence alone is insufficient.
    node_ok = all(any(cid in grounded and cid in by_claim and normalize_text(node.name) in normalize_text(by_claim[cid].text)
                      and node.tier.value in normalize_text(by_claim[cid].text)
                      for cid in node.claim_ids) for node in cleaned.chain_nodes)
    edge_ok = all(any(cid in grounded and cid in by_claim and re.search(
        re.escape(normalize_text(edge.src)) + r"\s*(?:supplies|供应)\s*" + re.escape(normalize_text(edge.dst)),
        normalize_text(by_claim[cid].text)) for cid in edge.claim_ids) for edge in cleaned.chain_edges)
    if cleaned.chain_nodes:
        if node_ok and edge_ok:
            record("chain_tikz", "validated")
        else:
            record("chain_tikz", "withheld", "Chain tier or edge direction lacks anchored support")
            cleaned.chain_nodes, cleaned.chain_edges = [], []

    observations = []
    if kg.market.series:
        try:
            for point in kg.market.series:
                metric = _find(metrics, point.claim_ids, "market_size", point.value,
                               point.unit, period=str(point.year))
                errors = comparison_errors(metric, metric, same_period=True)
                if errors:
                    raise MetricError(", ".join(errors))
                observations.append(metric)
            observations.sort(key=lambda m: m.period or "")
            if len({m.period for m in observations}) != len(observations):
                raise MetricError("Duplicate/conflicting observations for one period; no silent overwrite")
            if len(observations) < 2:
                raise MetricError("A trend requires at least two observations")
            for metric in observations[1:]:
                errors = comparison_errors(observations[0], metric)
                if errors:
                    raise MetricError(", ".join(errors))
            normalized = []
            for metric in observations:
                if not re.fullmatch(r"(?:19|20|21)\d{2}", metric.period or ""):
                    raise MetricError("Annual chart cannot silently combine quarterly/fiscal periods")
                chart_metric = metric if metric.unit == observations[0].unit else compute(
                    "unit_conversion", [metric], target_unit=observations[0].unit)
                normalized.append(MarketDatum(int(metric.period or ""), chart_metric.value, chart_metric.unit,
                                              metric.claim_ids, chart_metric.id))
            cleaned.market.series = normalized
            record("market_plot", "validated", observations=observations)
            for earlier, later in pairwise(observations):
                if int(later.period or "") - int(earlier.period or "") == 1:
                    try:
                        compute("growth_rate", [earlier, later])
                    except MetricError as exc:
                        issues.append({"calculation": "growth_rate", "reason": str(exc)})
            try:
                cagr = compute("cagr", [observations[0], observations[-1]])
                cleaned.market.cagr_pct = cagr.value
                cleaned.market.cagr_period = cagr.period
            except MetricError as exc:
                cleaned.market.cagr_pct = None
                issues.append({"calculation": "cagr", "reason": str(exc)})
        except MetricError as exc:
            cleaned.market.series, cleaned.market.cagr_pct = [], None
            record("market_plot", "withheld", str(exc), observations)
    else:
        cleaned.market.cagr_pct = None

    # Headline values require their own metric witnesses, not a union of arbitrary IDs.
    headline = []
    proposed = [(name, getattr(kg.market, name)) for name in ("tam", "sam", "som")
                if getattr(kg.market, name) is not None]
    for name in ("tam", "sam", "som"):
        setattr(cleaned.market, name, None)
    if proposed:
        try:
            for name, amount in proposed:
                metric = _find(metrics, kg.market.claim_ids, name, amount, kg.market.unit)
                errors = comparison_errors(metric, metric, same_period=True)
                if errors:
                    raise MetricError(", ".join(errors))
                headline.append(metric)
            for metric in headline[1:]:
                errors = comparison_errors(headline[0], replace(metric, name=headline[0].name), same_period=True)
                if errors:
                    raise MetricError(", ".join(errors))
            amounts = {m.name: convert_value(m.value, m.unit, headline[0].unit) for m in headline}
            for larger, smaller in (("tam", "sam"), ("sam", "som"), ("tam", "som")):
                if larger in amounts and smaller in amounts and amounts[smaller] > amounts[larger]:
                    raise MetricError("TAM/SAM/SOM hierarchy is numerically inconsistent")
            for metric in headline:
                setattr(cleaned.market, metric.name, amounts[metric.name])
                cleaned.market.metric_ids[metric.name] = metric.id
            cleaned.market.unit, cleaned.market.period = headline[0].unit, headline[0].period
            cleaned.market.scope_label = f"{headline[0].entity} / {headline[0].geography} / {headline[0].scope}"
            record("market_callout", "validated", observations=headline)
        except MetricError as exc:
            record("market_callout", "withheld", str(exc), headline)

    # CAGR is computed from approved endpoints; never copy a model's claimed percentage.
    if kg.market.cagr_pct is not None:
        if cleaned.market.cagr_pct is None:
            issues.append({"field": "reported_cagr", "reason": "No comparable endpoints to recompute CAGR"})
        elif decimal_value(str(kg.market.cagr_pct)) != cleaned.market.cagr_pct:
            issues.append({"field": "reported_cagr", "reason": "Reported/model CAGR differs from deterministic result"})

    competitors = []
    shares: list[Metric] = []
    if kg.competitors:
        try:
            for competitor in kg.competitors:
                if competitor.market_share_pct is None:
                    raise MetricError("Missing share; no numeric ranking rendered")
                metric = _find(metrics + calculated, competitor.claim_ids, "market_share",
                               competitor.market_share_pct, "%", entity=competitor.name)
                errors = comparison_errors(metric, metric, same_period=True)
                if errors:
                    raise MetricError(", ".join(errors))
                if shares:
                    errors = comparison_errors(shares[0], metric, allow_entity=True, same_period=True)
                    if errors:
                        raise MetricError(", ".join(errors))
                shares.append(metric)
                copy_competitor = copy.deepcopy(competitor)
                copy_competitor.market_share_pct = convert_value(metric.value, metric.unit, "%")
                copy_competitor.metric_id = metric.id
                # Competitive advantages have not received a chart-specific semantic audit.
                copy_competitor.advantage = ""
                competitors.append(copy_competitor)
            if len({normalize_text(m.entity or "") for m in shares}) != len(shares):
                raise MetricError("Duplicate entities cannot be counted twice in a share chart")
            if decimal_sum(convert_value(m.value, m.unit, "%") for m in shares) > 100:
                raise MetricError("Selected shares exceed 100% for the same market denominator")
            record("competitive_table", "validated", observations=shares)
            if len(shares) >= 2:
                record("competitive_bar", "validated", observations=shares)
        except MetricError as exc:
            competitors = []
            record("competitive_table", "withheld", str(exc), shares)
            record("competitive_bar", "withheld", str(exc), shares)
    cleaned.competitors = competitors

    accepted_clusters = []
    comparison_metrics = []
    for cluster in clusters or []:
        accepted = copy.deepcopy(cluster)
        valid_cells = []
        try:
            for cell in cluster.get("cells", []):
                cid = cell.get("claim_id") or next((c.id for c in claims if c.text == cell.get("claim_text")), None)
                found = quantities(str(cell.get("value", "")))
                if len(found) != 1 or cid is None:
                    raise MetricError("Comparison cell lacks an unambiguous value/unit or claim")
                amount, unit = found[0]
                metric = _find(metrics, [cid], normalize_text(cluster.get("attribute", "")), amount,
                               unit, entity=cluster.get("entity"))
                if comparison_errors(metric, metric, same_period=True):
                    raise MetricError("Comparison scope/period is incomplete")
                if not any(c.claim_id == cid and c.source_url == cell.get("source_url") for c in citations):
                    raise MetricError("Comparison cell source is not an evidence witness")
                cell_copy = copy.deepcopy(cell)
                cell_copy.update({"value": f"{metric.value} {metric.unit}", "time_scope": metric.period,
                                  "qualifier": f"{metric.geography}; {metric.scope}; {metric.value_kind}",
                                  "metric_id": metric.id})
                valid_cells.append(cell_copy)
                comparison_metrics.append(metric)
            accepted["cells"] = valid_cells
            accepted_clusters.append(accepted)
        except MetricError as exc:
            record("comparison_matrix", "withheld", str(exc))
    if accepted_clusters:
        record("comparison_matrix", "validated", observations=comparison_metrics)
    return cleaned, accepted_clusters, calculated, calculations, audit, issues
