from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from core_engine.report.grounding import SentenceGrounder, evidence_packet
from core_engine.report.llm import FakeLLM
from core_engine.report.models import Claim, ReportData, ReportSection, Source, SourceKind
from core_engine.report.quality_eval import (
    DEFAULT_DATASET,
    check_packet,
    load_dataset,
    main,
    run_cases,
    score_case,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def pinned_packet():
    from core_engine.config import get_settings
    data = load_dataset()
    source = data["sources"][0]
    claim = Claim("c1", source["text"], candidate_source_urls=[source["url"]])
    report = ReportData("GDP", "GDP", "", claims=[claim],
                        sources=[Source(source["url"], "stats.gov.cn", source["title"],
                                        SourceKind.GOVERNMENT, source["text"])],
                        sections=[ReportSection("GDP", paragraphs=[source["text"]], claim_ids=["c1"])])
    report = await SentenceGrounder(FakeLLM(), get_settings()).audit_report(report)
    return evidence_packet(report), data


def test_real_source_dataset_has_pinned_versions_and_human_rubric():
    data = load_dataset()
    assert len(data["cases"]) == 20
    assert len(data["sources"]) == 5
    assert "analysis_quality" in data["rubric"]
    assert len({case["id"] for case in data["cases"]}) == 20
    assert all(source["published_at"] for source in data["sources"])


async def test_gold_facts_and_anchors_are_checked_independently():
    packet, data = await pinned_packet()
    score = score_case(packet, data["cases"][0], data)
    assert score["gold_fact_coverage"] == 1
    assert score["sentence_citation_coverage"] == 1
    assert score["numeric_traceability"] == 1
    assert score["semantic_citation_correctness"] is None
    assert score["human_review_required"] is True


@pytest.mark.parametrize("mutation", ["wrong_decimal", "bad_offset", "invented_quote", "wrong_claim", "missing_citation", "forged_source"])
async def test_adversarial_evidence_packets_do_not_pass_quality_checks(mutation):
    packet, data = await pinned_packet()
    packet = copy.deepcopy(packet)
    sentence = packet["sections"][0]["sentences"][0]
    if mutation == "wrong_decimal":
        sentence["text"] = sentence["text"].replace("5.0%", "50%")
    elif mutation == "bad_offset":
        packet["citations"][0]["start_char"] += 1
    elif mutation == "invented_quote":
        packet["citations"][0]["quote"] = "not a real excerpt"
    elif mutation == "wrong_claim":
        sentence["claim_ids"] = ["invented-claim"]
    elif mutation == "missing_citation":
        sentence["citation_ids"] = []
    else:
        packet["sources"][0]["text"] += "forged material"
        # Updating both hashes can fool a self-consistency check, but not the gold snapshot.
        digest = hashlib.sha256(packet["sources"][0]["text"].encode()).hexdigest()
        packet["sources"][0]["content_sha256"] = digest
        packet["citations"][0]["source_sha256"] = digest
    score = score_case(packet, data["cases"][0], data)
    assert score["integrity_errors"]
    assert score["automatic_checks_passed"] is False


async def test_valid_citation_to_wrong_gold_source_does_not_count_as_coverage():
    packet, data = await pinned_packet()
    score = score_case(packet, data["cases"][2], data)
    assert score["gold_fact_coverage"] == 0


def test_cli_validates_dataset_and_rejects_invalid_score_arguments(tmp_path):
    assert main(["--mode", "check"]) == 0
    assert main(["--mode", "score", "--output", str(tmp_path / "score.json")]) == 2
    data = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    data["sources"][0]["text"] += "changed"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(path)


async def test_baseline_runs_real_pipeline_without_claiming_live_model_quality(tmp_path):
    summary = await run_cases(load_dataset(), "baseline", tmp_path / "baseline.json")
    assert summary["cases_run"] == 20
    assert summary["live_model_evaluated"] is False
    assert summary["human_review_required"] is True
    assert all(r["automatic_checks_passed"] for r in summary["results"])
    assert all(not r["integrity_errors"] for r in summary["results"])
    assert (tmp_path / "baseline.json").exists()
    review = json.loads(Path(summary["human_review_form"]).read_text(encoding="utf-8"))
    assert len(review["reviews"]) == 20
    assert all(score is None for item in review["reviews"] for score in item["scores"].values())


async def test_correct_but_irrelevant_report_does_not_pass_abstention_case():
    packet, data = await pinned_packet()
    case = next(c for c in data["cases"] if c["id"] == "insufficient-profit-evidence")
    score = score_case(packet, case, data)
    assert not score["integrity_errors"]
    assert score["automatic_checks_passed"] is False


async def test_live_eval_cannot_silently_use_fake_provider(monkeypatch, tmp_path):
    from core_engine.config import get_settings
    monkeypatch.setenv("CE_LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="Live evaluation requires"):
            await run_cases(load_dataset(), "live", tmp_path / "live.json")
    finally:
        get_settings.cache_clear()


async def test_self_reported_perfect_score_does_not_override_grader():
    packet, _data = await pinned_packet()
    packet["audit"] = {"supported_ratio": 1, "citation_count": 999999}
    packet["sections"][0]["sentences"][0]["citation_ids"] = []
    assert check_packet(packet)["sentence_citation_coverage"] == 0
