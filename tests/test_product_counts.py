"""Regression: two patent measurements were rewritten as two different cells."""
import asyncio

from core_engine.config import get_settings
from core_engine.report.grounding import SentenceGrounder
from core_engine.report.text import numbers_supported, numeric_tokens


def test_chinese_product_counts_require_matching_source_counts_not_measurement_numbers():
    for chinese, arabic in (("两款电池", "2款电池"), ("二款电池", "2款电池"),
                            ("十一款产品", "11款产品"), ("二十三款器件", "23款器件")):
        assert numeric_tokens(chinese) == numeric_tokens(arabic)
    assert not numbers_supported("两款电池分别完成435圈和3C测试。", "本发明电池完成435圈及3C测试。")
    assert numbers_supported("两款电池完成435圈测试。", "共2款电池完成435圈测试。")
    assert not numbers_supported("两款电池完成测试。", "两款车型完成测试。")


def test_a_model_endorsement_cannot_override_an_unanchored_cell_count():
    class OverconfidentJudge:
        async def check_entailment(self, *_):
            raise AssertionError("Unanchored product counts must fail before model judgment")
    auditor = SentenceGrounder(OverconfidentJudge(), get_settings())
    assert not asyncio.run(auditor._entailed("另两款电池分别达到435圈和91.2%的容量保持率。",
        "本发明所提供的电池至少循环435圈，3C放电容量保持率为91.2%。"))


def test_reaudit_withdraws_a_previously_published_sentence_with_wrong_product_count():
    from core_engine.report.models import Claim, ReportSentence, Source, SourceKind

    source = Source("https://example.edu/patent", "example.edu", "专利", SourceKind.ACADEMIC,
        "本发明所提供的电池至少循环435圈，3C放电容量保持率为91.2%。")
    claim = Claim("c1", source.text, [source.url])
    sentence = ReportSentence("另两款电池分别达到435圈和91.2%的容量保持率。", claim_ids=["c1"],
                              citation_ids=["old"], status="supported", attribution="旧的归属")
    auditor = SentenceGrounder(object(), get_settings())
    result = asyncio.run(auditor.audit_sentence(sentence, {claim.id: claim}, {source.url: source}))
    assert result.status == "withheld" and result.citation_ids == []
    assert result.reason == "numeric_value_or_unit_not_supported" and result.attribution == ""
