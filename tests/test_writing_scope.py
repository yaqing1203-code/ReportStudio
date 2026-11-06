"""The real template article reversed lithium charging direction after a skipped review."""
import asyncio
import json

import pytest

from core_engine.app import writing

BAD = "充电时，锂离子从负极脱嵌，经过固体电解质抵达正极并嵌入其中。"
GOOD = "充电时，锂离子从正极向负极迁移；放电时由负极返回正极，电子通过外电路传输。"


def test_explicit_inverted_direction_and_mixed_component_roles_are_flagged():
    assert writing._scope_issues(BAD, "固态电池") == [BAD]
    assert writing._scope_issues("放电时锂离子从正极迁移到负极。", "固态电池")
    assert writing._scope_issues("充电期间Li⁺由负极经电解质到达正极。", "固态电池")
    assert writing._scope_issues("液态电解液承担离子传输与隔膜功能。", "固态电池")
    assert not writing._scope_issues(GOOD, "固态电池")
    assert not writing._scope_issues(BAD, "教学中的常见错误分析")
    assert not writing._scope_issues("不能说" + BAD, "固态电池")


class Editor:
    def __init__(self, replacement):
        self.replacement = replacement

    async def complete(self, system, user):
        payload = json.loads(user)
        assert payload["convention_references"] and "positive" in payload["conventions"]
        return json.dumps({"edits": [{"excerpt": BAD, "replacement": self.replacement}]})


def test_detected_direction_error_is_repaired_and_logged_without_truth_certification():
    text, audit = asyncio.run(writing.repair_scope(BAD, "固态电池", Editor(GOOD)))
    assert text == GOOD and audit["repaired"] == 1
    assert audit["is_factual_verification"] is False


@pytest.mark.parametrize("replacement", [BAD, GOOD + "能量密度为999 Wh/kg。"])
def test_unrepaired_or_numerically_fabricated_correction_cannot_be_published(replacement):
    with pytest.raises(writing.WritingError):
        asyncio.run(writing.repair_scope(BAD, "固态电池", Editor(replacement)))


def test_ordinary_article_needs_no_extra_model_request():
    text, audit = asyncio.run(writing.repair_scope("固态电池仍面临界面与制造挑战。", "固态电池", object()))
    assert text.endswith("。") and audit["detected"] == 0
