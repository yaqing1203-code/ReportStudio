"""Tests for source credibility grading (L1-L4) and the credibility-weighted
ranking in gather_sources. Fully offline (fake provider/fetcher).

    pytest -q tests/test_credibility.py
"""
from __future__ import annotations

import pytest

from core_engine.report.models import (
    Claim,
    CredibilityLevel,
    SearchHit,
    Source,
    SourceKind,
)
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider, gather_sources
from core_engine.report.sources import classify, credibility_for

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --------------------------------------------------------------------------
# credibility_for — domain -> level mapping
# --------------------------------------------------------------------------
def test_credibility_l1_anchors():
    # Government / regulators.
    assert credibility_for("https://energy.gov/report") is CredibilityLevel.L1
    assert credibility_for("https://ec.europa.eu/info") is CredibilityLevel.L1
    # Top academic venues.
    assert credibility_for("https://www.nature.com/articles/x") is CredibilityLevel.L1
    # Standards bodies within the institution tier.
    assert credibility_for("https://www.iso.org/standard/1.html") is CredibilityLevel.L1
    assert credibility_for("https://www.ieee.org/x") is CredibilityLevel.L1


def test_credibility_l2_professional():
    assert credibility_for("https://www.reuters.com/world/x") is CredibilityLevel.L2
    assert credibility_for("https://www.mckinsey.com/insights") is CredibilityLevel.L2
    assert credibility_for("https://www.pewresearch.org/x") is CredibilityLevel.L2
    # IGO research / trade associations (institution tier, not standards bodies).
    assert credibility_for("https://www.iea.org/reports/x") is CredibilityLevel.L2
    assert credibility_for("https://www.oecd.org/x") is CredibilityLevel.L2


def test_credibility_l3_default_and_l4_weak():
    # Unrecognized but non-denylisted host -> L3 (industry consensus default).
    assert classify("https://someblog.example.net/x") is SourceKind.GENERAL_WEB
    assert credibility_for("https://someblog.example.net/x") is CredibilityLevel.L3
    # Known self-published / social-adjacent platforms -> L4 weak signal.
    assert credibility_for("https://www.zhihu.com/answer/1") is CredibilityLevel.L4
    assert credibility_for("https://blog.csdn.net/post/1") is CredibilityLevel.L4
    # Denylist still rejected by classify(); credibility grades it L4 (never used).
    assert classify("https://www.reddit.com/r/x") is SourceKind.REJECTED
    assert credibility_for("https://www.reddit.com/r/x") is CredibilityLevel.L4
    assert credibility_for("") is CredibilityLevel.L4


def test_weights_and_labels():
    assert CredibilityLevel.L1.weight == 1.0
    assert CredibilityLevel.L2.weight == 0.8
    assert CredibilityLevel.L3.weight == 0.5
    assert CredibilityLevel.L4.weight == 0.2
    assert "L1" in CredibilityLevel.L1.label
    assert CredibilityLevel.L1 < CredibilityLevel.L4  # IntEnum ordering by level number


def test_source_and_claim_defaults():
    src = Source("https://x.example/a", "x.example", "T", SourceKind.GENERAL_WEB,
                 text="body")
    assert src.credibility is CredibilityLevel.L3       # neutral default
    claim = Claim(id="c1", text="t")
    assert claim.credibility is None
    assert claim.isolated is False
    assert claim.entity is None and claim.attribute is None and claim.value is None
    assert claim.qualifier is None and claim.time_scope is None


# --------------------------------------------------------------------------
# Credibility-weighted ranking in gather_sources
# --------------------------------------------------------------------------
async def test_l4_high_relevance_never_outranks_l1(monkeypatch):
    """An L4 weak-signal page with HIGH topical relevance must still rank behind an
    L1 anchor with moderate relevance after the weighted sort."""
    from core_engine.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CE_MULTI_QUERY_RESEARCH", "false")
    # Keep the relevance filter ON so both pages earn a real density score.

    # L4 page: topic-dense (density ~0.29). L1 page: sparser (density ~0.09).
    zhihu_body = ("clean energy " + "some other filler words here ") * 60
    cdc_body = ("clean energy " + "filler " * 20) * 15

    search = FakeSearchProvider()
    # Zhihu listed FIRST by the provider — only the ranking can demote it.
    search.add("clean energy", [
        SearchHit("https://www.zhihu.com/answer/1", "L4 post", "", "zhihu.com"),
        SearchHit("https://cdc.gov/clean-energy", "CDC report", "", "cdc.gov"),
    ])
    fetcher = FakeFetcher({
        "https://www.zhihu.com/answer/1": zhihu_body,
        "https://cdc.gov/clean-energy": cdc_body,
    })
    try:
        sources, _rejected = await gather_sources("clean energy", search, fetcher)
    finally:
        get_settings.cache_clear()

    assert {s.url for s in sources} == {
        "https://www.zhihu.com/answer/1", "https://cdc.gov/clean-energy"}
    # Gather filled credibility from the domain mapping.
    by_url = {s.url: s for s in sources}
    assert by_url["https://cdc.gov/clean-energy"].credibility is CredibilityLevel.L1
    assert by_url["https://www.zhihu.com/answer/1"].credibility is CredibilityLevel.L4
    # Weighted ranking: L1 anchor leads despite the L4 page's higher relevance density.
    assert sources[0].url == "https://cdc.gov/clean-energy"
    assert sources[1].url == "https://www.zhihu.com/answer/1"


async def test_ranking_weights_are_configurable(monkeypatch):
    """With rank_w_source=0 the ordering falls back to relevance alone, proving the
    credibility term (not insertion order) drove the ranking above."""
    from core_engine.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CE_MULTI_QUERY_RESEARCH", "false")
    monkeypatch.setenv("CE_RANK_W_SOURCE", "0")
    monkeypatch.setenv("CE_RANK_W_FRESHNESS", "0")
    monkeypatch.setenv("CE_RANK_W_RELEVANCE", "1")

    zhihu_body = ("clean energy " + "some other filler words here ") * 60
    cdc_body = ("clean energy " + "filler " * 20) * 15
    search = FakeSearchProvider()
    search.add("clean energy", [
        SearchHit("https://www.zhihu.com/answer/1", "L4 post", "", "zhihu.com"),
        SearchHit("https://cdc.gov/clean-energy", "CDC report", "", "cdc.gov"),
    ])
    fetcher = FakeFetcher({
        "https://www.zhihu.com/answer/1": zhihu_body,
        "https://cdc.gov/clean-energy": cdc_body,
    })
    try:
        sources, _ = await gather_sources("clean energy", search, fetcher)
    finally:
        get_settings.cache_clear()

    assert len(sources) == 2
    # Pure relevance ordering: the denser zhihu page leads.
    assert sources[0].url == "https://www.zhihu.com/answer/1"
