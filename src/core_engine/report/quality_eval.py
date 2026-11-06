"""Reproducible report-quality evaluation over pinned, annotated public-source excerpts.

baseline exercises the real pipeline with the extractive FakeLLM; it is explicitly
not a live-model benchmark. live uses the configured provider. score grades saved
evidence packets without trusting their self-reported audit statistics.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from core_engine.report.answerability import assess_coverage
from core_engine.report.calculations import replay
from core_engine.report.discovery import equivalent_terms
from core_engine.report.evidence_quality import assess_evidence
from core_engine.report.grounding import evidence_packet
from core_engine.report.metrics import (
    Metric,
    MetricError,
    catalog_from_claims,
    comparison_errors,
    convert_value,
    decimal_sum,
)
from core_engine.report.models import Claim, CredibilityLevel, SentenceCitation, Source, SourceKind
from core_engine.report.publication import apply_policy, attribution
from core_engine.report.reasoning import counterevidence_status, validate_finding
from core_engine.report.scientific import comparison_audit
from core_engine.report.text import normalize_text, numbers_supported, numeric_tokens
from core_engine.security.files import atomic_write_text

DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "evals" / "report_quality" / "cases.json"


def packet_sentences(packet: dict) -> list[dict]:
    sentences = list(packet.get("rumor_sentences", [])) + list(packet.get("dispute_sentences", []))
    for section in packet.get("sections", []):
        sentences.extend(section.get("sentences", []))
        for dive in section.get("deep_dives", []):
            sentences.extend(dive.get("sentences", []))
    return sentences


def check_numeric_layer(packet: dict, valid_citations: dict) -> dict:
    """Rebind observations to exact passages and replay every derived result.

    Self-reported validation status, results and precision are never sufficient.
    Semantic scope annotation still requires human review.
    """
    registry: dict[str, Metric] = {}
    errors = []
    claims = {claim["id"]: claim for claim in packet.get("claims", [])}
    citations = [SentenceCitation(**row) for row in valid_citations.values()]
    records = packet.get("metrics", [])
    derived = {}
    seen = set()
    method_sources: list[Source] | None = None
    for row in records:
        identifier = row.get("id")
        try:
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise MetricError("Missing or duplicate metric ID")
            seen.add(identifier)
            metric = Metric(**row)
            if metric.origin == "calculated":
                derived[identifier] = metric
                continue
            if metric.origin != "source" or len(metric.claim_ids) != 1:
                raise MetricError("Source metrics require one atomic claim")
            claim = claims[metric.claim_ids[0]]
            if claim.get("reporting_status") in {"disputed", "lead_only"}:
                raise MetricError("Unsettled or unanchored claims cannot be calculation inputs")
            # Bind the serialized observation itself, rather than a separate model
            # proposal whose value or scope could differ from what is rendered.
            proposal = {key: row.get(key) for key in (
                "name", "value", "unit", "entity", "period", "geography", "scope",
                "price_basis", "rate_basis", "value_kind", "part_of")}
            proposal["boundary"] = row.get("boundary")
            proposal["conditions"] = row.get("conditions", {})
            original = claim.get("metric")
            premise = Claim(claim["id"], claim["text"], metric=proposal)
            if isinstance(original, dict) and original.get("condition_evidence"):
                # Enriched conditions cannot be replayed as if they came from the
                # table row. Rebuild the link from the original row-only proposal.
                premise.metric = original
                premise.qualifier = claim.get("qualifier")
                premise.source_windows = claim.get("source_windows", {})
                premise.candidate_source_urls = claim.get("candidate_source_urls", [])
                if method_sources is None:
                    method_sources = [Source(s["url"], s["domain"], s["title"], SourceKind(s["kind"]), s["text"],
                                             document_format=s.get("document_format", ""), metadata=s.get("metadata", {}))
                                      for s in packet.get("sources", [])]
            rebuilt, issues = catalog_from_claims([premise], citations, sources=method_sources or ())
            if issues or len(rebuilt) != 1:
                raise MetricError("Metric amount/name/scope is not bound to source evidence")
            expected = rebuilt[0].to_dict()
            expected["id"] = identifier
            if metric.to_dict() != expected:
                raise MetricError("Metric provenance or value was modified")
            registry[identifier] = metric
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid_metric:{identifier}:{exc}")
    replayed = 0
    outputs = set()
    for record in packet.get("calculations", []):
        try:
            output = replay(record, registry)
            if output.id in outputs or output.id not in derived or output.to_dict() != derived[output.id].to_dict():
                raise MetricError("Calculated output is missing, duplicated or differs from replay")
            registry[output.id] = output
            outputs.add(output.id)
            replayed += 1
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            errors.append(f"invalid_calculation:{record.get('id')}:{exc}")
    for identifier in derived.keys() - outputs:
        errors.append(f"unreplayed_metric:{identifier}")
    scientific = packet.get("research_audit", {}).get("scientific_comparisons", {})
    if not isinstance(scientific, dict):
        errors.append("invalid_scientific_condition_audit")
    elif "methods" in scientific or (scientific and any(m.condition_evidence for m in registry.values())):
        rendered_metrics = [registry[identifier] for row in records
                            if isinstance(identifier := row.get("id"), str) and identifier in registry]
        if scientific != comparison_audit(rendered_metrics):
            errors.append("invalid_scientific_condition_audit")
    checked_charts = 0
    for entry in packet.get("chart_audit", []):
        if entry.get("status") != "validated":
            continue
        try:
            ids = entry.get("metric_ids", [])
            if not ids and entry.get("block") != "chain_tikz":
                raise MetricError("Numeric charts need explicit metric inputs")
            inputs = [registry[identifier] for identifier in ids]
            context = [{key: getattr(metric, key) for key in (
                "entity", "period", "geography", "scope", "price_basis", "rate_basis", "value_kind", "unit")}
                       for metric in inputs]
            evidence = sorted({eid for metric in inputs for eid in metric.evidence_ids})
            if entry.get("context") != context or entry.get("evidence_ids") != evidence:
                raise MetricError("Chart scope/evidence differs from registered inputs")
            for metric in inputs:
                if comparison_errors(metric, metric, same_period=True):
                    raise MetricError("Chart has incomplete input scope")
            block = entry["block"]
            if block == "market_plot" and (len(inputs) < 2 or any(m.name != "market_size" for m in inputs)
                                          or len({m.period for m in inputs}) != len(inputs)
                                          or any(not re.fullmatch(r"(?:19|20|21)\d{2}", m.period or "") for m in inputs)):
                raise MetricError("Invalid annual trend inputs")
            if (block in {"competitive_bar", "competitive_table"}
                    and (any(m.name != "market_share" for m in inputs)
                         or len({m.entity for m in inputs}) != len(inputs)
                         or decimal_sum(convert_value(m.value, m.unit, "%") for m in inputs) > 100)):
                raise MetricError("Invalid share denominator, identities or total")
            if block == "market_callout" and inputs:
                if any(m.name not in {"tam", "sam", "som"} for m in inputs):
                    raise MetricError("Invalid market headline metric")
                for metric in inputs[1:]:
                    if comparison_errors(inputs[0], replace(metric, name=inputs[0].name), same_period=True):
                        raise MetricError("Incomparable market headline inputs")
                amounts = {m.name: convert_value(m.value, m.unit, inputs[0].unit) for m in inputs}
                if any(amounts[small] > amounts[large] for large, small in (("tam", "sam"), ("sam", "som"), ("tam", "som"))
                       if large in amounts and small in amounts):
                    raise MetricError("Invalid TAM/SAM/SOM hierarchy")
            if inputs and entry["block"] in {"market_plot", "competitive_bar", "competitive_table"}:
                for metric in inputs[1:]:
                    if comparison_errors(inputs[0], metric, allow_entity=entry["block"] != "market_plot",
                                         same_period=entry["block"] != "market_plot"):
                        raise MetricError("Chart combines incomparable observations")
            checked_charts += 1
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid_chart:{entry.get('block')}:{exc}")
    count = len(packet.get("calculations", []))
    return {"metric_integrity_errors": errors, "registered_metrics": len(registry),
            "replayed_calculations": replayed, "calculation_replay_rate": replayed / count if count else None,
            "checked_charts": checked_charts}


def check_source_quality(packet: dict) -> list[str]:
    claims = packet.get("claims", [])
    if not packet.get("source_quality") and not any(c.get("evidence_quality") for c in claims):
        return []
    try:
        sources = [Source(s["url"], s["domain"], s["title"], SourceKind(s["kind"]), s["text"],
                          document_format=s.get("document_format", ""), metadata=s.get("metadata", {}))
                   for s in packet.get("sources", [])]
        observations = [Claim(c["id"], c["text"], evidence_quality=c.get("evidence_quality", {})) for c in claims]
        expected = assess_evidence(observations, sources, [SentenceCitation(**c) for c in packet.get("citations", [])])
        errors = [f"invalid_evidence_quality:{c['id']}" for c in claims if c.get("evidence_quality")
                  and c["evidence_quality"] != expected["claims"][c["id"]]]
        profile = packet.get("source_quality", {})
        if profile and any(profile.get(key) != value for key, value in expected.items()):
            errors.append("invalid_source_quality_summary")
        return errors
    except (ValueError, TypeError, KeyError):
        return ["invalid_source_quality_schema"]


def check_research_layer(packet: dict, expected_sources=None) -> dict:
    errors = []
    plan, audit, evidence = (packet.get(key, {}) for key in ("research_plan", "research_audit", "research_evidence"))
    findings = packet.get("analysis", [])
    if not plan:
        return {"research_integrity_errors": ["analysis_without_research_plan"] if findings else [],
                "research_question_coverage": None, "checked_analyses": 0}
    if (plan.get("objective") != packet.get("topic") or not plan.get("questions")
            or plan["questions"][0].get("question") != packet.get("topic")
            or plan["questions"][0].get("required") is not True):
        errors.append("research_plan_changed_original_task")
    nested = {"sources": packet.get("sources", []), "claims": evidence.get("claims", []),
              "citations": evidence.get("citations", []), "evidence_policy": audit.get("evidence_policy", {})}
    checked = check_packet(nested, expected_sources)
    errors.extend("research:" + error for error in checked["integrity_errors"])
    claims = {c["id"]: c for c in evidence.get("claims", [])}
    anchored = {c["claim_id"] for c in evidence.get("citations", [])}
    citations = {c["id"]: c for c in evidence.get("citations", [])}
    questions = {q["id"]: q for q in plan.get("questions", [])}
    if len(questions) != len(plan.get("questions", [])):
        errors.append("duplicate_planned_question_ids")
    seen = set()
    covered = 0
    for row in audit.get("questions", []):
        try:
            question = questions[row["id"]]
            ids = row.get("claim_ids", [])
            if row["id"] in seen or row["question"] != question["question"] or row["required"] != question["required"]:
                raise ValueError("Changed or duplicate research question")
            seen.add(row["id"])
            if row["status"] not in {"supported", "partial", "missing", "undetermined"}:
                raise ValueError("Unknown research coverage status")
            if not isinstance(ids, list) or any(cid not in claims or cid not in anchored for cid in ids):
                raise ValueError("Research coverage has unanchored witness IDs")
            expected = {c["id"] for c in citations.values() if c["claim_id"] in ids}
            if set(row.get("citation_ids", [])) != expected:
                raise ValueError("Research coverage citation IDs differ from its premises")
            if row["status"] in {"supported", "partial"}:
                if row["status"] == "supported" and any(claims[cid].get("reporting_status") == "disputed" for cid in ids):
                    raise ValueError("Disputed evidence cannot complete a research question")
                if not ids:
                    raise ValueError("Research coverage has unanchored witness IDs")
                if row.get("method") == "deterministic_coverage" or row["status"] == "supported":
                    result = assess_coverage(question["question"], [Claim(c["id"], c["text"],
                        candidate_source_urls=c.get("candidate_source_urls", [])) for c in claims.values() if c["id"] in ids],
                        [Source(s["url"], s["domain"], s["title"], SourceKind(s["kind"]), s["text"]) for s in packet.get("sources", [])],
                        topic_aliases=equivalent_terms(question["question"]))
                    if ((row["status"] == "supported" and result.status == "insufficient")
                            or (row.get("method") == "deterministic_coverage" and (result.status != "answerable"
                                or question["kind"] in {"explanation", "counterevidence", "risk"}))):
                        raise ValueError("Self-reported research coverage could not be reproduced")
            covered += row["status"] == "supported"
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid_research_coverage:{row.get('id')}:{exc}")
    if seen != questions.keys():
        errors.append("missing_research_question_assessment")
    if audit.get("coverage_status") != ("complete" if covered == len(questions) and questions else "partial"):
        errors.append("invalid_research_coverage_summary")
    if audit.get("search_policy") == "document_only" and (audit.get("searches") or audit.get("initial_queries")):
        errors.append("document_only_research_searched_web")
    for search in audit.get("searches", []):
        if search.get("question_id") not in questions or not search.get("gap"):
            errors.append("unplanned_gap_search")
    budgets = audit.get("budgets", {})
    if len(audit.get("searches", [])) > budgets.get("queries", 0) or len(audit.get("rounds", [])) > budgets.get("rounds", 0):
        errors.append("research_budget_exceeded")
    checked_analyses = 0
    for finding in findings:
        try:
            expected_finding = validate_finding(finding, plan, evidence, packet.get("calculations", []), packet.get("metrics", []))
            if any(finding.get(key) != value for key, value in expected_finding.items()):
                raise ValueError("Analysis fields differ from provenance/numeric validation")
            if finding.get("counterevidence_status") != counterevidence_status(finding, audit):
                raise ValueError("Analysis counterevidence status differs from completed searches")
            checked_analyses += 1
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid_analysis:{finding.get('id')}:{exc}")
    return {"research_integrity_errors": errors, "checked_analyses": checked_analyses,
            "research_question_coverage": covered / len(questions) if questions else None}


def check_publication_policy(packet: dict, valid: dict) -> list[str]:
    """Replay publication levels from source snapshots, never from audit labels."""
    policy = packet.get("evidence_policy")
    if not policy:
        return []  # historical packets predate explicit publication levels
    try:
        from dataclasses import fields

        names = {f.name for f in fields(Claim)} - {"evidence", "confidence", "credibility"}
        claims = [Claim(**{key: value for key, value in row.items() if key in names})
                  for row in packet.get("claims", [])]
        ids = {c.id for c in claims}
        leads = [Claim(row["claim_id"], row["text"], candidate_source_urls=row.get("candidate_source_urls", []))
                 for row in packet.get("leads", []) if row["claim_id"] not in ids]
        sources = [Source(s["url"], s["domain"], s["title"], SourceKind(s["kind"]), s["text"],
                          document_format=s.get("document_format", ""), metadata=s.get("metadata", {}))
                   for s in packet.get("sources", [])]
        expected = apply_policy(claims + leads, sources, [SentenceCitation(**c) for c in valid.values()])
        errors = []
        if any(policy.get(key) != value for key, value in expected.items()):
            errors.append("invalid_publication_policy")
        by_id = {c.id: c for c in claims}
        for row in packet.get("claims", []):
            claim = by_id[row["id"]]
            if row.get("reporting_status") != claim.reporting_status or row.get("verified") != claim.verified:
                errors.append(f"invalid_publication_status:{claim.id}")
        disputes = packet.get("dispute_sentences", [])
        for sentence in packet_sentences(packet):
            if sentence.get("status") != "supported":
                continue
            selected = [by_id[cid] for cid in sentence.get("claim_ids", []) if cid in by_id]
            records = [SentenceCitation(**valid[eid]) for eid in sentence.get("citation_ids", []) if eid in valid]
            if not selected or any(c.reporting_status == "lead_only" for c in selected):
                errors.append("unanchored_publication_premise")
                continue
            state, _ = attribution(selected, sources, records, "en")
            if ((state == "disputed") != (sentence in disputes)
                    or sentence.get("evidence_status") != state
                    or sentence.get("attribution") not in {attribution(selected, sources, records, lang)[1] for lang in ("en", "zh")}):
                errors.append("invalid_sentence_evidence_attribution")
        return errors
    except (ValueError, TypeError, KeyError, AttributeError):
        return ["invalid_publication_policy_schema"]


def check_packet(packet: dict, expected_sources: list[dict] | None = None) -> dict:
    """Check provenance integrity and numeric traceability, not semantic truth."""
    sources = {source["url"]: source for source in packet.get("sources", [])}
    claims = {claim["id"]: claim for claim in packet.get("claims", [])}
    expected = {source["url"]: source for source in expected_sources or []}
    citations = packet.get("citations", [])
    valid = {}
    errors = []
    for source in packet.get("sources", []):
        digest = hashlib.sha256(source.get("text", "").encode("utf-8")).hexdigest()
        if source.get("content_sha256") != digest or (expected and (
                source.get("url") not in expected or expected[source["url"]]["sha256"] != digest)):
            errors.append(f"invalid_source_snapshot:{source.get('url')}")
    seen = set()
    for citation in citations:
        cid = citation.get("id")
        source = sources.get(citation.get("source_url"))
        claim = claims.get(citation.get("claim_id"))
        duplicate = cid in seen
        seen.add(cid)
        ok = False
        if source is not None and claim is not None and isinstance(cid, str) and cid and not duplicate:
            text = source.get("text", "")
            start, end = citation.get("start_char"), citation.get("end_char")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            ok = (type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                  and bool(citation.get("quote")) and text[start:end] == citation["quote"]
                  and digest == citation.get("source_sha256") == source.get("content_sha256")
                  and citation["source_url"] in claim.get("candidate_source_urls", [])
                  and numbers_supported(claim["text"], citation["quote"]))
            if expected:
                pinned = expected.get(source["url"])
                ok = ok and pinned is not None and pinned["sha256"] == digest
        if ok:
            valid[cid] = citation
        else:
            errors.append(f"invalid_citation:{cid}")
    all_sentences = packet_sentences(packet)
    published = [s for s in all_sentences if s.get("status") == "supported"]
    attributed = 0
    numeric = 0
    numeric_ok = 0
    for index, sentence in enumerate(published):
        ids = sentence.get("citation_ids", [])
        links = [valid[eid] for eid in ids if eid in valid]
        claim_ids = set(sentence.get("claim_ids", []))
        citation_claims = {link["claim_id"] for link in links}
        complete = (bool(ids) and len(links) == len(set(ids)) == len(ids) and bool(claim_ids)
                    and citation_claims == claim_ids and all(cid in claims for cid in claim_ids))
        if complete:
            attributed += 1
        else:
            errors.append(f"unattributed_published_sentence:{index}")
        if numeric_tokens(sentence.get("text", "")):
            numeric += 1
            if complete and numbers_supported(sentence["text"], "\n".join(link["quote"] for link in links)):
                numeric_ok += 1
            else:
                errors.append(f"untraceable_numeric_sentence:{index}")
    numeric_layer = check_numeric_layer(packet, valid)
    errors.extend(numeric_layer["metric_integrity_errors"])
    research_layer = check_research_layer(packet, expected_sources)
    errors.extend(research_layer["research_integrity_errors"])
    errors.extend(check_source_quality(packet))
    errors.extend(check_publication_policy(packet, valid))
    return {
        **numeric_layer,
        **research_layer,
        "published_sentences": len(published), "withheld_sentences": sum(
            sentence.get("status") == "withheld" for sentence in all_sentences),
        "anchored_citations": len(valid), "total_citations": len(citations),
        "anchored_citation_rate": len(valid) / len(citations) if citations else None,
        "sentence_citation_coverage": attributed / len(published) if published else None,
        "numeric_traceability": numeric_ok / numeric if numeric else None,
        "integrity_errors": errors,
        "semantic_citation_correctness": None, "analysis_quality": None,
        "human_review_required": True,
    }


def load_dataset(path: Path = DEFAULT_DATASET) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported quality dataset schema")
    source_ids = set()
    for source in data["sources"]:
        if source["id"] in source_ids:
            raise ValueError("Duplicate source ID")
        source_ids.add(source["id"])
        if hashlib.sha256(source["text"].encode("utf-8")).hexdigest() != source["sha256"]:
            raise ValueError(f"Source snapshot hash mismatch: {source['id']}")
        if not source["url"].startswith("https://") or not source.get("published_at"):
            raise ValueError("Source provenance metadata missing")
        if any(not isinstance(unit, str) or unit not in source["text"] for unit in source.get("extractive_units", [])):
            raise ValueError("Annotated extractive unit is not present in the pinned source")
    cases = set()
    for case in data["cases"]:
        if case["id"] in cases or not set(case["source_ids"]).issubset(source_ids):
            raise ValueError("Duplicate case or unknown source")
        cases.add(case["id"])
        for fact in case["required_facts"]:
            if not set(fact["source_ids"]).issubset(case["source_ids"]):
                raise ValueError("Gold fact references an unavailable source")
            for pattern in fact["required_patterns"]:
                re.compile(pattern)
    return data


def score_case(packet: dict, case: dict, dataset: dict) -> dict:
    selected = [s for s in dataset["sources"] if s["id"] in case["source_ids"]]
    metrics = check_packet(packet, selected)
    source_by_id = {s["id"]: s for s in selected}
    citations = {c["id"]: c for c in packet.get("citations", [])}
    published = [s for s in packet_sentences(packet) if s.get("status") == "supported"]
    covered = []
    for fact in case["required_facts"]:
        allowed_urls = {source_by_id[sid]["url"] for sid in fact["source_ids"]}
        for sentence in published:
            text = normalize_text(sentence["text"])
            links = [citations[eid] for eid in sentence.get("citation_ids", []) if eid in citations]
            if (all(re.search(pattern, text) for pattern in fact["required_patterns"])
                    and numeric_tokens(fact["numeric_reference"]).issubset(numeric_tokens(text))
                    and any(link["source_url"] in allowed_urls for link in links)
                    and not metrics["integrity_errors"]):
                covered.append(fact["id"])
                break
    corpus = "\n".join(s["text"] for s in published)
    forbidden = [p for p in case.get("forbidden_patterns", []) if re.search(p, corpus, re.IGNORECASE)]
    total = len(case["required_facts"])
    metrics.update({"case_id": case["id"], "gold_facts_covered": covered,
                    "gold_facts_missing": [f["id"] for f in case["required_facts"] if f["id"] not in covered],
                    "gold_fact_coverage": len(covered) / total if total else None,
                    "forbidden_assertions": forbidden,
                    "automatic_checks_passed": bool(total) and len(covered) == total
                                                and not forbidden and not metrics["integrity_errors"]})
    if case.get("expected_abstention"):
        coverage = packet.get("answerability", {})
        metrics["automatic_checks_passed"] = (not published and not metrics["integrity_errors"]
                                               and packet.get("kind") == "insufficient_evidence"
                                               and coverage.get("status") == "insufficient"
                                               and bool(coverage.get("missing")))
    profile = packet.get("source_quality", {})
    if case.get("expected_evidence_mode") and profile.get("evidence_mode") != case["expected_evidence_mode"]:
        metrics["automatic_checks_passed"] = False
        metrics["integrity_errors"].append("expected_evidence_mode_missing")
    if "expected_origin_groups" in case and profile.get("origin_group_count") != case["expected_origin_groups"]:
        metrics["automatic_checks_passed"] = False
        metrics["integrity_errors"].append("unexpected_origin_group_count")
    return metrics


async def run_cases(dataset: dict, mode: str, output: Path, case_ids=None, trials: int = 1) -> dict:
    from core_engine.config import get_settings
    from core_engine.report.llm import FakeLLM, get_llm
    from core_engine.report.pipeline import ReportPipeline
    from core_engine.report.scrape import FakeFetcher, FakeSearchProvider

    settings = get_settings()
    if mode == "live" and settings.llm_provider == "fake":
        raise ValueError("Live evaluation requires CE_LLM_PROVIDER=anthropic or openai and configured credentials")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output.parent / (output.stem + "-artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in dataset["cases"]:
        if case_ids and case["id"] not in case_ids:
            continue
        selected = [s for s in dataset["sources"] if s["id"] in case["source_ids"]]
        for trial in range(trials):
            llm = FakeLLM() if mode == "baseline" else get_llm()
            if mode == "baseline" and isinstance(llm, FakeLLM):
                for source in selected:
                    if source.get("extractive_units"):
                        llm.set_structured_claims(source["url"], [{"text": unit} for unit in source["extractive_units"]])
            pipeline = ReportPipeline(llm=llm, search=FakeSearchProvider(), fetcher=FakeFetcher())
            # Isolate per-case settings; no process-global environment rewrites.
            pipeline._s = settings.model_copy(update={
                "min_authoritative_sources": 1, "min_total_claims": 1, "verify_strategy": "cross_reference",
                "verify_mode": "strict", "report_locale": case["locale"], "enable_charts": False,
                "enable_deep_dive": False, "enable_synthesis_sections": True, "active_verify_enabled": False,
                "max_reverify_attempts": 0, "output_dir": artifact_dir,
            })
            pipeline._harness._s = pipeline._s
            thresholds = case.get("evidence_thresholds", {})
            allowed_thresholds = {key: value for key, value in thresholds.items()
                                  if key in {"min_authoritative_sources", "min_total_claims", "sparse_evidence_enabled"}}
            if allowed_thresholds:
                pipeline._s = pipeline._s.model_copy(update=allowed_thresholds)
                pipeline._harness._s = pipeline._s
            sources = [Source(url=s["url"], domain=s["url"].split("/")[2], title=s["title"],
                              kind=(SourceKind(s["kind"]) if s.get("kind") else SourceKind.INDUSTRY_INSTITUTION if "irena.org" in s["url"]
                                    else SourceKind.GOVERNMENT), text=s["text"],
                              fetched_at=dataset["captured_at"], published_at=s["published_at"],
                              credibility=CredibilityLevel.L1, document_format=s.get("document_format", ""),
                              metadata=s.get("metadata", {})) for s in selected]
            started = time.monotonic()
            result = await pipeline.run(case["question"], compile_to_pdf=False, extra_sources=sources, skip_search=True)
            packet = (evidence_packet(result.report) if result.report else
                      json.loads(Path(result.evidence_path).read_text(encoding="utf-8")) if result.evidence_path else {})
            metrics = score_case(packet, case, dataset)
            metrics.update({"trial": trial + 1, "pipeline_status": result.status.value,
                            "elapsed_s": round(time.monotonic() - started, 3),
                            "evidence_path": result.evidence_path, "trace": result.trace})
            if result.status.value == "error":
                metrics["automatic_checks_passed"] = False
                metrics["error"] = result.message
            results.append(metrics)
    if not results:
        raise ValueError("No evaluation cases selected")
    summary = {
        "schema_version": 1, "dataset_version": dataset["version"],
        "mode": mode, "live_model_evaluated": mode == "live",
        "provider": settings.llm_provider if mode == "live" else "extractive_fake",
        "model": settings.llm_model if mode == "live" else "deterministic_baseline",
        "cases_run": len(results), "automatic_checks_passed": sum(r["automatic_checks_passed"] for r in results),
        "human_review_required": True,
        "limitations": ["Fixed source excerpts; web retrieval quality is not evaluated.",
                        "Gold coverage rules and anchor checks are not a semantic correctness or professional-quality score.",
                        "Source versions are pinned historical releases, not claims about current statistics."],
        "results": results,
    }
    # Keep human judgments separate from machine metrics. Unique run forms never
    # overwrite a review a person has already filled out.
    run_id = uuid4().hex[:12]
    review_path = output.with_name(f"{output.stem}-{run_id}.review.json")
    review = {
        "schema_version": 1, "run_id": run_id, "dataset_version": dataset["version"],
        "provider": summary["provider"], "model": summary["model"],
        "scale": "0=unacceptable, 1=major issues, 2=partial, 3=good, 4=meets task; null=not reviewed",
        "rubric": dataset["rubric"],
        "reviews": [{"case_id": r["case_id"], "trial": r["trial"],
                     "evidence_path": r["evidence_path"],
                     "scores": {dimension: None for dimension in dataset["rubric"]},
                     "reviewer": "", "notes": ""} for r in results],
    }
    atomic_write_text(review_path, json.dumps(review, ensure_ascii=False, indent=2))
    summary["run_id"] = run_id
    summary["human_review_form"] = str(review_path.resolve())
    atomic_write_text(output, json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "baseline", "live", "score"], default="check")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("output/quality-evaluation.json"))
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--trials", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--packet", type=Path)
    args = parser.parse_args(argv)
    try:
        data = load_dataset(args.dataset)
        if args.mode == "check":
            print(f"Dataset {data['version']}: {len(data['sources'])} snapshots, {len(data['cases'])} annotated cases; hashes valid.")
            return 0
        if args.mode == "score":
            if not args.packet or not args.case_ids or len(args.case_ids) != 1:
                raise ValueError("score requires --packet and one --case")
            case = next(c for c in data["cases"] if c["id"] == args.case_ids[0])
            metrics = score_case(json.loads(args.packet.read_text(encoding="utf-8")), case, data)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(args.output, json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0 if metrics["automatic_checks_passed"] else 1
        summary = asyncio.run(run_cases(data, args.mode, args.output, args.case_ids, args.trials))
        print(f"{args.mode}: {summary['automatic_checks_passed']}/{summary['cases_run']} automatic checks passed. "
              f"Human review required. Results: {args.output.resolve()}")
        return 0 if summary["automatic_checks_passed"] == summary["cases_run"] else 1
    except (ValueError, StopIteration, OSError, KeyError, TypeError) as exc:
        print(f"Evaluation configuration error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
