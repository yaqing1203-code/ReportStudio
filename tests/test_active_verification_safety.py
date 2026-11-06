"""A second page is corroboration only if its claim agrees and is independent."""
from __future__ import annotations

import pytest

from core_engine.report.llm import FakeLLM
from core_engine.report.models import Claim, Source, SourceKind
from core_engine.report.verify import VerificationHarness, VerificationReport


@pytest.mark.parametrize("value,time_scope,domain,expected", [
    ("35%", "2024", "independent.org", True),
    ("45%", "2024", "independent.org", False),
    ("3.5%", "2024", "independent.org", False),
    ("35", "2024", "independent.org", False),
    ("-35%", "2024", "independent.org", False),
    ("35%", "2023", "independent.org", False),
    ("35%", None, "independent.org", False),
    ("35%", "2024", "trade.example.org", False),
])
def test_active_corroboration_requires_matching_value_scope_and_independent_domain(value, time_scope, domain, expected):
    original = Source("https://trade.example.org/a", "trade.example.org", "a", SourceKind.GENERAL_WEB, text="source text")
    other = Source(f"https://{domain}/b", domain, "b", SourceKind.GENERAL_WEB, text="source text")
    target = Claim(id="target", text="Acme market share was 35% in 2024.",
                   candidate_source_urls=[original.url], entity="Acme", attribute="market_share",
                   value="35%", time_scope="2024", isolated=True)
    extracted = Claim(id="new", text=f"Acme shipments imply {value} in {time_scope}.",
                      candidate_source_urls=[other.url], entity="Acme", attribute="market_share",
                      value=value, time_scope=time_scope)
    report = VerificationReport(kept_claims=[target])
    corroborated = VerificationHarness(FakeLLM())._merge_active_claim(
        extracted, other, target, report, {s.url: s for s in [original, other]})
    assert bool(corroborated) is expected
    if expected:
        assert other.url in target.supporting_sources()


def test_distinct_chinese_claims_are_not_collapsed_to_empty_text():
    first = Source("https://trade.example.org/a", "trade.example.org", "a", SourceKind.GENERAL_WEB, text="市场增长")
    second = Source("https://other.example.org/b", "other.example.org", "b", SourceKind.GENERAL_WEB, text="市场下降")
    target = Claim(id="target", text="市场增长", candidate_source_urls=[first.url])
    extracted = Claim(id="new", text="市场下降", candidate_source_urls=[second.url])
    report = VerificationReport(kept_claims=[target])
    assert not VerificationHarness(FakeLLM())._merge_active_claim(
        extracted, second, target, report, {s.url: s for s in [first, second]})
    assert len(report.kept_claims) == 2
    assert second.url not in target.candidate_source_urls
