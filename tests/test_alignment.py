"""Tests for the "aligned" verification strategy (alignment/fusion layer).

Fully offline: FakeLLM (with structured-claim and complete() injection), FakeFetcher,
and a stub active-search function. Run: pytest -q tests/test_alignment.py
"""
from __future__ import annotations

import json

import pytest

from core_engine.report.llm import ContradictionJudgement, FakeLLM
from core_engine.report.models import (
    Claim,
    CredibilityLevel,
    SearchHit,
    Source,
    SourceKind,
)
from core_engine.report.scrape import FakeFetcher
from core_engine.report.verify import VerificationHarness

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _clean_settings():
    from core_engine.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _src(url: str, kind: SourceKind, credibility: CredibilityLevel,
         text: str = "body") -> Source:
    return Source(url, url.split("/")[2], "T", kind, text=text,
                  credibility=credibility)


class _NeverContradictLLM(FakeLLM):
    """check_contradiction always 'no conflict' — isolates corroborate/complement
    classification from the numeric-divergence heuristic."""
    async def check_contradiction(self, a, b):
        return ContradictionJudgement(contradict=False, note="[stub] no conflict")


# --------------------------------------------------------------------------
# 1. Entity alias normalization -> same cluster
# --------------------------------------------------------------------------
async def test_entity_normalization_merges_aliases_into_one_cluster():
    fake = FakeLLM()
    fake.set_complete("entity alias normalizer", json.dumps(
        {"mapping": {"CATL": "宁德时代", "宁德时代": "宁德时代"}}))
    a = Claim(id="c1", text="CATL held 35% of the market in 2024.",
              candidate_source_urls=["https://iea.org/a"],
              entity="CATL", attribute="market_share", value="35%")
    b = Claim(id="c2", text="宁德时代 held 35% of the market in 2024.",
              candidate_source_urls=["https://reuters.com/b"],
              entity="宁德时代", attribute="market_share", value="35%")
    sources = [
        _src("https://iea.org/a", SourceKind.INDUSTRY_INSTITUTION, CredibilityLevel.L2),
        _src("https://reuters.com/b", SourceKind.AUTHORITATIVE_MEDIA,
             CredibilityLevel.L2),
    ]
    report = await VerificationHarness(fake).verify([a, b], sources)
    # Alias rewritten to the canonical name on BOTH claims.
    assert a.entity == "宁德时代" and b.entity == "宁德时代"
    # Both land in ONE (entity, attribute) cluster.
    assert len(report.clusters) == 1
    cluster = report.clusters[0]
    assert cluster["entity"] == "宁德时代"
    assert cluster["attribute"] == "market_share"
    assert {cell["claim_text"] for cell in cluster["cells"]} == {a.text, b.text}


# --------------------------------------------------------------------------
# 2. Intra-cluster tri-classification
# --------------------------------------------------------------------------
async def test_same_calibre_same_value_corroborates_and_merges_evidence():
    a = Claim(id="c1", text="Acme market share was 35% under GAAP in 2024.",
              candidate_source_urls=["https://iea.org/a"],
              entity="Acme", attribute="market_share", value="35%",
              qualifier="GAAP", time_scope="2024")
    b = Claim(id="c2", text="Acme market share was 35% under GAAP in 2024.",
              candidate_source_urls=["https://reuters.com/b"],
              entity="Acme", attribute="market_share", value="35%",
              qualifier="GAAP", time_scope="2024")
    sources = [
        _src("https://iea.org/a", SourceKind.INDUSTRY_INSTITUTION, CredibilityLevel.L2),
        _src("https://reuters.com/b", SourceKind.AUTHORITATIVE_MEDIA,
             CredibilityLevel.L2),
    ]
    report = await VerificationHarness(_NeverContradictLLM()).verify([a, b], sources)
    assert report.clusters[0]["relation"] == "corroborate"
    assert report.rejected_claims == []
    # Corroboration merged the support sets onto BOTH claims.
    for c in (a, b):
        assert set(c.candidate_source_urls) == {
            "https://iea.org/a", "https://reuters.com/b"}
        assert {e.source_url for e in c.evidence} == set(c.candidate_source_urls)


async def test_different_qualifier_is_complement_both_kept():
    a = Claim(id="c1", text="Acme revenue was 50 billion under GAAP.",
              candidate_source_urls=["https://iea.org/a"],
              entity="Acme", attribute="revenue", value="50 billion",
              qualifier="GAAP")
    b = Claim(id="c2", text="Acme revenue was 50 billion under non-GAAP.",
              candidate_source_urls=["https://reuters.com/b"],
              entity="Acme", attribute="revenue", value="50 billion",
              qualifier="non-GAAP")
    sources = [
        _src("https://iea.org/a", SourceKind.INDUSTRY_INSTITUTION, CredibilityLevel.L2),
        _src("https://reuters.com/b", SourceKind.AUTHORITATIVE_MEDIA,
             CredibilityLevel.L2),
    ]
    report = await VerificationHarness(_NeverContradictLLM()).verify([a, b], sources)
    assert report.clusters[0]["relation"] == "complement"
    # BOTH claims kept — complement never drops.
    assert {c.id for c in report.kept_claims} == {"c1", "c2"}
    assert report.rejected_claims == []
    assert report.reasons.get("pair:c1|c2", "").startswith("complement")


async def test_contradiction_drops_lower_authority_claim():
    gov = _src("https://bls.gov/a", SourceKind.GOVERNMENT, CredibilityLevel.L1)
    blog = _src("https://someblog.net/b", SourceKind.GENERAL_WEB, CredibilityLevel.L4)
    hi = Claim(id="gov1", text="Acme market share was 35% in 2024.",
               candidate_source_urls=[gov.url],
               entity="Acme", attribute="market_share", value="35%",
               time_scope="2024")
    lo = Claim(id="blog1", text="Acme market share was 60% in 2024.",
               candidate_source_urls=[blog.url],
               entity="Acme", attribute="market_share", value="60%",
               time_scope="2024")
    # FakeLLM's heuristic: same subject, diverging numbers -> contradiction.
    report = await VerificationHarness(FakeLLM()).verify([lo, hi], [gov, blog])
    assert {c.id for c in report.kept_claims} == {"gov1"}
    assert [c.id for c in report.rejected_claims] == ["blog1"]
    assert "contradicted" in report.reasons["blog1"]
    assert report.clusters[0]["relation"] == "conflict"


# --------------------------------------------------------------------------
# 3. Slot-less claims fall back to the lexical conflict path (conflict_only parity)
# --------------------------------------------------------------------------
async def test_unstructured_claims_use_lexical_conflict_path():
    gov = _src("https://bls.gov/a", SourceKind.GOVERNMENT, CredibilityLevel.L1)
    blog = _src("https://someblog.net/b", SourceKind.GENERAL_WEB, CredibilityLevel.L4)
    hi = Claim(id="gov1", text="The national unemployment rate for 2024 was 4 percent.",
               candidate_source_urls=[gov.url])
    lo = Claim(id="blog1", text="The national unemployment rate for 2024 was 9 percent.",
               candidate_source_urls=[blog.url])
    report = await VerificationHarness(FakeLLM()).verify([lo, hi], [gov, blog])
    # Identical outcome to conflict_only: low-authority side dropped.
    assert {c.id for c in report.kept_claims} == {"gov1"}
    assert "contradicted" in report.reasons["blog1"]
    assert report.clusters == []          # no structured clusters involved


# --------------------------------------------------------------------------
# 4. Isolated (孤证) marking
# --------------------------------------------------------------------------
async def test_isolated_flag_single_vs_multi_source():
    lone_src = _src("https://trade-assoc.example.org/x",
                    SourceKind.INDUSTRY_INSTITUTION, CredibilityLevel.L3)
    s1 = _src("https://iea.org/a", SourceKind.INDUSTRY_INSTITUTION, CredibilityLevel.L2)
    s2 = _src("https://reuters.com/b", SourceKind.AUTHORITATIVE_MEDIA,
              CredibilityLevel.L2)
    lone = Claim(id="c1", text="Adoption grew 30 percent in 2025.",
                 candidate_source_urls=[lone_src.url])
    multi = Claim(id="c2", text="Capacity reached 8 GW in 2024.",
                  candidate_source_urls=[s1.url, s2.url])
    report = await VerificationHarness(FakeLLM()).verify(
        [lone, multi], [lone_src, s1, s2])
    by_id = {c.id: c for c in report.kept_claims}
    assert by_id["c1"].isolated is True     # single source, L3
    assert by_id["c2"].isolated is False    # multi-source
    # The isolated claim got a provenance/timeliness source note in the audit trail.
    assert "source note" in report.reasons.get("c1", "")


# --------------------------------------------------------------------------
# 5. Internal consistency review downgrades credibility
# --------------------------------------------------------------------------
async def test_internal_inconsistency_downgrades_credibility():
    fake = FakeLLM()
    fake.set_complete("internal consistency reviewer",
                      json.dumps({"consistent": False, "note": "totals do not add up"}))
    src = _src("https://trade-assoc.example.org/x", SourceKind.INDUSTRY_INSTITUTION,
               CredibilityLevel.L3,
               text="The market was 50 billion. The market was 90 billion in total.")
    c = Claim(id="c1", text="The market was 50 billion dollars.",
              candidate_source_urls=[src.url])
    report = await VerificationHarness(fake).verify([c], [src])
    kept = report.kept_claims[0]
    assert kept.isolated is True
    assert kept.credibility is CredibilityLevel.L4      # L3 downgraded one level
    assert "downgraded" in report.reasons["c1"]
    assert "totals do not add up" in report.reasons["c1"]


# --------------------------------------------------------------------------
# 6. Baseline deviation above threshold downgrades an isolated claim
# --------------------------------------------------------------------------
async def test_deviation_above_threshold_downgrades():
    fake = FakeLLM()
    fake.set_complete("deviation assessor",
                      json.dumps({"deviation": 0.9, "note": "far above field baseline"}))
    src = _src("https://trade-assoc.example.org/x", SourceKind.INDUSTRY_INSTITUTION,
               CredibilityLevel.L3)
    c = Claim(id="c1", text="The niche market grew 900 percent in one year.",
              candidate_source_urls=[src.url])
    report = await VerificationHarness(fake).verify([c], [src])
    kept = report.kept_claims[0]
    assert kept.credibility is CredibilityLevel.L4
    assert "deviation 0.90" in report.reasons["c1"]


# --------------------------------------------------------------------------
# 7. Active verification: query budget respected; corroboration lifts isolation
# --------------------------------------------------------------------------
async def test_active_verify_query_budget_and_isolation_lift():
    fake = FakeLLM()
    # Generator proposes FIVE queries; only active_verify_max_queries (3) may run.
    fake.set_complete("verification query generator", json.dumps([
        "q1 proxy shipments", "q2 proxy capacity", "q3 proxy revenue",
        "q4 extra", "q5 extra",
    ]))
    new_url = "https://oecd.org/proxy-report"
    fake.set_structured_claims(new_url, [{
        "text": "Acme shipments imply a market share near 35%.",
        "entity": "Acme", "attribute": "market_share", "value": "35%",
    }])

    searched: list[str] = []

    async def stub_search(query: str):
        searched.append(query)
        return [SearchHit(new_url, "Proxy Report", "", "oecd.org")]

    fetcher = FakeFetcher({new_url: "Acme shipments imply a market share near 35% "
                                    "for the sector overall."})
    src = _src("https://trade-assoc.example.org/x", SourceKind.INDUSTRY_INSTITUTION,
               CredibilityLevel.L3)
    c = Claim(id="c1", text="Acme market share was 35% in 2024.",
              candidate_source_urls=[src.url],
              entity="Acme", attribute="market_share", value="35%",
              time_scope="2024")
    harness = VerificationHarness(fake, fetcher=fetcher, search_fn=stub_search)
    report = await harness.verify([c], [src])

    from core_engine.config import get_settings
    assert len(searched) <= get_settings().active_verify_max_queries
    assert len(searched) == 3                     # budget fully used, never exceeded
    kept_by_id = {k.id: k for k in report.kept_claims}
    # Corroborating source found -> isolation lifted, new claim merged into the pool.
    assert kept_by_id["c1"].isolated is False
    assert any(k.id.startswith("av") for k in report.kept_claims)
    assert "isolation lifted" in report.reasons["c1"]
    # The actively-found claim traces to the newly fetched source.
    av = next(k for k in report.kept_claims if k.id.startswith("av"))
    assert av.candidate_source_urls == [new_url]
