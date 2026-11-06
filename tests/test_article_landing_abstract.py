"""Regressions from the real OSTI solid-state battery template run."""
from core_engine.report.fulltext import html_article_document

ABSTRACT = ("Solid-state batteries require stable interfaces. This review discusses how "
            "electrolyte chemistry, contact and pressure affect interface reactions and resistance.")


def page(extra=""):
    return ('<html><head><meta name="citation_title" content="Solid-state battery interfaces">'
            f'<meta name="citation_abstract" content="{ABSTRACT}"></head><body>{extra}</body></html>')


def test_reference_table_cannot_replace_explicit_landing_page_abstract():
    references = "| A related paper | journal | January 2005 |\n" * 500
    doc = html_article_document(page(), "https://example.edu/paper", references)
    assert doc.text == ABSTRACT
    assert doc.document_format == "abstract"
    assert doc.metadata["source_role"] == "article_declared_abstract"
    assert not doc.metadata["full_text_accessed"]
    assert "article_abstract" not in doc.metadata


def test_full_article_keeps_body_and_does_not_shrink_to_abstract():
    body = "Experimental results and interface observations.\n" * 150
    doc = html_article_document(page("<h2>Introduction</h2><h2>Results</h2>"), "https://example.edu/paper", body)
    assert doc.text == body and doc.metadata["full_text_accessed"]
    assert doc.document_format == "html"


def test_abstract_recovers_empty_main_extraction_without_claiming_fulltext():
    doc = html_article_document(page(), "https://example.edu/paper", None)
    assert doc.text == ABSTRACT and doc.document_format == "abstract"


def test_generic_seo_description_is_not_a_research_abstract():
    html = '<meta name="description" content="' + ABSTRACT + '">'
    assert html_article_document(html, "https://example.edu", "Menu and search") is None


def test_short_or_unidentified_abstract_is_not_used_as_paper_content():
    html = page().replace(ABSTRACT, "See this paper")
    assert html_article_document(html, "https://example.edu/paper", "Landing text").text == "Landing text"
    html = page().replace('name="citation_title"', 'name="site_title"')
    assert html_article_document(html, "https://example.edu/paper", None) is None


def test_generic_manufacturing_terms_cannot_admit_wrong_openalex_subject():
    from core_engine.report.discovery import matches_subject
    topic = "固态电池技术研究报告：分析界面、制造、材料、温度和压力。"
    assert not matches_subject(topic, "石油化工机械设备维护检修技术分析",
        "设备材料的制造需要温度和压力控制。界面机械应力影响制造质量。")
    assert matches_subject(topic, "Solid-state battery interfaces", "Short institutional summary.")
    assert matches_subject("固态电池", "Interface mechanisms",
        "Solid-state batteries face interface reactions. Solid-state batteries also face contact loss.")


def test_long_report_instructions_do_not_become_cover_title():
    from core_engine.report.pipeline import _report_title
    assert _report_title("固态电池技术研究报告：分析技术路线，保留测试条件并注明缺口。", "zh") == "固态电池技术研究报告"


async def _collect_wrong_paper():
    from core_engine.config import get_settings
    from core_engine.report.models import SearchHit
    from core_engine.report.scholarly import FetchedDocument
    from core_engine.report.scrape import FakeFetcher, FakeSearchProvider, gather_sources
    topic = "固态电池技术研究报告：材料、制造、温度和压力"
    url = "https://example.edu/petrochemical"
    provider = FakeSearchProvider()
    provider.add(topic, [SearchHit(url, "石油化工设备维护", "", "example.edu")])
    document = FetchedDocument("设备材料与制造过程需要控制温度和压力。" * 30, "abstract", metadata={"title": "石油化工设备维护"})
    return await gather_sources(topic, provider, FakeFetcher({url: document}), queries=[topic],
        settings=get_settings().model_copy(update={"relevance_min_score": 0, "relevance_min_topic_hits": 0}))


def test_wrong_academic_subject_is_rejected_before_extraction():
    import asyncio
    sources, rejected = asyncio.run(_collect_wrong_paper())
    assert sources == []
    assert rejected[0].metadata["rejection_detail"] == "subject_identity_missing"
