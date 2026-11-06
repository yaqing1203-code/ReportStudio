from __future__ import annotations

import pytest

from core_engine.report.llm import FakeLLM
from core_engine.report.models import Claim, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline, _normalize
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider, relevance_score
from core_engine.report.text import normalize_text, numeric_tokens, split_sentences
from core_engine.report.verify import VerificationHarness, claim_source_overlap

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("left,right", [
    ("市场增长", "市场下降"), ("增长3.5%", "增长35%"), ("利润-3.5亿元", "利润3.5亿元"),
    ("增长5%", "增长5个百分点"), ("营收5亿元", "营收5万元"),
    ("2023年收入", "2024年收入"), ("没有增长", "增长"),
])
def test_identity_never_erases_material_differences(left, right):
    assert _normalize(left) != _normalize(right)


@pytest.mark.parametrize("left,right", [
    ("３．５０％", "3.5 percent"), ("−3.5%", "-3.50%"),
    ("1,000.00", "1000"), ("4 448 GW", "4448 gigawatts"),
    ("5个百分点", "5 percentage points"), ("百分之5.0", "5%"),
    ("9007199254740993", "9007199254740993.0"),
])
def test_numeric_equivalence_is_decimal_safe(left, right):
    assert numeric_tokens(left) == numeric_tokens(right)


@pytest.mark.parametrize("left,right", [("3.5%", "35%"), ("-5%", "5%"), ("5%", "5个百分点"),
                                        ("5亿元", "5万元"), ("5GW", "5GWh"),
                                        ("9007199254740992", "9007199254740993")])
def test_numeric_signatures_preserve_value_unit_and_sign(left, right):
    assert numeric_tokens(left) != numeric_tokens(right)


def test_unicode_sentence_boundaries_keep_decimals_and_short_chinese_claims():
    assert split_sentences("增长3.5%。利润下降。亏损-2.2亿元！") == ["增长3.5%。", "利润下降。", "亏损-2.2亿元！"]
    assert split_sentences("U.S. GDP rose 2.3 percent. 2024 growth was 2.8 percent.") == [
        "U.S. GDP rose 2.3 percent.", "2024 growth was 2.8 percent."]
    assert normalize_text("全角３．５％") == "全角3.5%"


async def test_main_extraction_keeps_chinese_and_numeric_claims_distinct():
    source = Source("https://example.gov/a", "example.gov", "统计", SourceKind.GOVERNMENT,
                    "市场增长。市场下降。利润3.5亿元。利润35亿元。利润-3.5亿元。")
    pipeline = ReportPipeline(llm=FakeLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    claims = await pipeline._extract_claims([source])
    assert len(claims) == 5
    assert {c.text for c in claims} == set(split_sentences(source.text))


async def test_same_words_with_conflicting_structured_scope_are_not_deduped():
    llm = FakeLLM()
    sources = []
    for i, value in enumerate(["3.5%", "35%"]):
        url = f"https://example.gov/{i}"
        sources.append(Source(url, "example.gov", "a", SourceKind.GOVERNMENT, "市场增长。"))
        llm.set_structured_claims(url, [{"text": "市场增长。", "entity": "企业", "attribute": "增速", "value": value}])
    pipeline = ReportPipeline(llm=llm, search=FakeSearchProvider(), fetcher=FakeFetcher())
    assert len(await pipeline._extract_claims(sources)) == 2


def test_chinese_relevance_overlap_and_section_routing():
    relevant = relevance_score("固态电池市场", "固态电池市场需求持续增长。" * 30)
    irrelevant = relevance_score("固态电池市场", "今天的天气晴朗，适合户外活动。" * 30)
    assert relevant[0] > irrelevant[0]
    assert irrelevant == (0.0, 0)
    assert claim_source_overlap("固态电池市场规模增长", "固态电池市场规模已有统计") > 0
    pipeline = ReportPipeline(llm=FakeLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    buckets = pipeline._bucket_claims([Claim("c1", "政府出台监管政策。"), Claim("c2", "市场规模达5亿元。")])
    assert [c.id for c in buckets["policy_analysis"]] == ["c1"]
    assert [c.id for c in buckets["market_size"]] == ["c2"]


def test_chinese_claims_enter_conflict_candidate_checks():
    harness = VerificationHarness(FakeLLM())
    assert harness._candidate_conflict_pairs([
        Claim("c1", "公司市场份额为3.5%。"), Claim("c2", "公司市场份额为35%。")]) == [(0, 1)]


def test_year_and_separate_amount_are_not_mistaken_for_grouped_thousands():
    assert numeric_tokens("2024 123") == numeric_tokens("2024; 123")
    assert numeric_tokens("4 448 GW") == numeric_tokens("4448 GW")
    assert numeric_tokens("5个基点") == numeric_tokens("5 basis points")


async def test_failed_numeric_conflict_check_never_creates_corroboration():
    class FailingLLM(FakeLLM):
        async def check_contradiction(self, claim_a, claim_b):
            raise TimeoutError("simulated provider timeout")
    from core_engine.config import get_settings
    from core_engine.report.verify import VerificationReport
    sources = [Source(f"https://{domain}/a", domain, "统计", SourceKind.GOVERNMENT, "公司增长3.5%。")
               for domain in ["first.gov", "second.gov"]]
    claims = [Claim(f"c{i}", "公司增长3.5%。", candidate_source_urls=[source.url],
                    entity="公司", attribute="增长率", value="3.5%") for i, source in enumerate(sources)]
    harness = VerificationHarness(FailingLLM())
    report = VerificationReport()
    import time
    await harness._classify_clusters(claims, {s.url: s for s in sources}, report,
                                     time.monotonic() + get_settings().verify_deadline_s, lambda *args: None)
    assert report.clusters[0]["relation"] == "unresolved"
    assert all(len(claim.candidate_source_urls) == 1 for claim in claims)


@pytest.mark.parametrize("claim,evidence", [
    ("Impedance is 5 Ω cm2.", "Impedance (mΩ cm2): 5"),
    ("Areal capacity is 3 mAh/cm2.", "Areal capacity (Ah/m2): 3"),
    ("Energy density is 3 Wh/m3.", "Energy density (mWh/cm3): 3"),
    ("Conductivity is 1 MS/cm.", "Ionic conductivity (mS/cm): 1"),
    ("Current density is 1 A/cm2.", "Current density (mA/cm2): 1"),
    ("Capacity is 1 MAh/g.", "Specific capacity (mAh/g): 1"),
    ("Cycle life is 1000 cycles.", "A bare number 1000."),
])
def test_scientific_numeric_guard_never_drops_units_or_changes_prefix_scale(claim, evidence):
    from core_engine.report.text import numbers_supported

    assert not numbers_supported(claim, evidence)


@pytest.mark.parametrize("claim,evidence", [
    ("Impedance is 5 Ω cm².", "Impedance (ohm cm2): 5"),
    ("Impedance is 5 mΩ cm2.", "Impedance (mΩ cm²): 5"),
    ("Areal capacity is 3 mAh/cm2.", "Areal capacity (mAh cm−2): 3"),
    ("Current density is .5 A/m².", "Current density (A/m2): .5"),
    ("Energy density is 700 mWh/cm3.", "Volumetric energy density (mWh/cm³): 700"),
    ("Energy density is 700 Wh/m3.", "Volumetric energy density (Wh/m³): 700"),
    ("Cycle life is 1000 cycles.", "Cycle life (cycles): 1000"),
])
def test_new_scientific_dimensions_have_complete_explicit_numeric_signatures(claim, evidence):
    from core_engine.report.text import numbers_supported

    assert numbers_supported(claim, evidence)


@pytest.mark.parametrize("bad_cell", ["Ionic conductivity (MS/cm): 1.2",
    "Ionic conductivity (S/cm): 1e1000", "Ionic conductivity (S/cm): " + "9" * 60])
def test_invalid_table_cell_is_skipped_without_discarding_valid_neighbour(bad_cell):
    from core_engine.report.text import numbers_supported

    text = bad_cell + " | Ionic conductivity (mS/cm): 3.4"
    assert numbers_supported("Conductivity is 3.4 mS/cm.", text)
    assert not numbers_supported("Conductivity is 1.2 mS/cm.", text)
    assert all(len(token) < 200 for token in numeric_tokens(text))


@pytest.mark.parametrize("malformed", ["1.2 × 10^{-3 S/cm", "1.2 × 10^-3} S/cm",
    "1.2 × 10^{-3}} S/cm", "1.2 × 10^-3.5 S/cm"])
def test_unbalanced_or_fractional_scientific_exponent_cannot_support_normalized_value(malformed):
    from core_engine.report.text import numbers_supported

    assert not numbers_supported("Conductivity is 0.0012 S/cm.", malformed)


@pytest.mark.parametrize("spelling", ["−1.2 × 10^{-3} S/cm", "-1.2×10⁻³ S/cm", "-1.2e-3 S/cm"])
def test_scientific_normalization_preserves_negative_values(spelling):
    from core_engine.report.text import numbers_supported

    assert numbers_supported("Conductivity is -0.0012 S/cm.", spelling)
    assert not numbers_supported("Conductivity is 0.0012 S/cm.", spelling)


def test_scientific_exponent_bounds_do_not_expand_oversized_strings():
    from core_engine.report.text import numbers_supported

    for text in ("1.2 × 10^{1000000} S/cm", "1.2×10⁻¹⁰⁰⁰ S/cm", "Ionic conductivity (S/cm): 10^1000000"):
        assert all(len(token) < 100 for token in numeric_tokens(text))
        assert not numbers_supported("Conductivity is 0.0012 S/cm.", text)
    assert numeric_tokens("A fraction of 1.2 × 10^{-3}.") == numeric_tokens("A fraction of 0.0012.")


def test_labelled_unit_binding_never_uses_a_different_field_or_row():
    from core_engine.report.text import numbers_supported

    source = "Ionic conductivity (mS/cm): 1.2\nIonic conductivity (S/m): 3.4"
    assert numbers_supported("Conductivity is 1.2 mS/cm.", source)
    assert numbers_supported("Conductivity is 3.4 S/m.", source)
    assert not numbers_supported("Conductivity is 3.4 mS/cm.", source)
    assert not numbers_supported("Conductivity is 1.2 S/m.", source)
    assert not numbers_supported("Conductivity is 3.4 mS/cm.", "Conductivity (mS/cm): | Other value: 3.4")
