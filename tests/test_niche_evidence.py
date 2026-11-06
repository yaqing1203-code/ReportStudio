from __future__ import annotations

import copy
import io
from pathlib import Path

import httpx
import pytest

from core_engine.config import get_settings
from core_engine.report.discovery import discovery_query, equivalent_terms
from core_engine.report.evidence_quality import assess_evidence, origin_groups, work_identifier
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import FakeLLM
from core_engine.report.models import Claim, PipelineStatus, SentenceCitation, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet, load_dataset, run_cases
from core_engine.report.router import BackendRegistry, QueryType, routed_search
from core_engine.report.scholarly import CrossrefSearchProvider, FetchedDocument, crossref_document
from core_engine.report.scrape import (
    FakeFetcher,
    FakeSearchProvider,
    HttpFetcher,
    SearchHit,
    gather_sources,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def configured(tmp_path, **updates):
    return get_settings().model_copy(update={"output_dir": tmp_path, "min_authoritative_sources": 2,
        "min_total_claims": 3, "enable_charts": False, "enable_deep_dive": False, "verify_strategy": "cross_reference",
        "verify_mode": "strict", "report_locale": "zh", "research_max_queries": 4, "research_max_rounds": 2,
        "enable_synthesis_sections": False, "active_verify_enabled": False, "research_deadline_s": 3,
        **updates})


def record(doi="10.1234/battery", abstract=None, **updates):
    return {"DOI": doi, "title": ["Solid-state batteries and polymer electrolyte interfaces"],
        "abstract": abstract or "<jats:p>Solid-state batteries use solid electrolytes. This study reports a polymer design for improving electrode interfaces under the stated laboratory conditions.</jats:p>",
        "type": "journal-article", "published": {"date-parts": [[2022, 12, 5]]}, "publisher": "Test Publisher", **updates}


def citation(claim, source, identifier="e1"):
    start = source.text.index(claim.text)
    return SentenceCitation(identifier, claim.id, source.url, claim.text, start, start + len(claim.text),
                            1, None, source.content_sha256, "verbatim")


@pytest.mark.parametrize(("topic", "wanted", "excluded"), [
    ("固态电池", "solid-state batteries", "semi-solid-state batteries"),
    ("全固态电池", "all-solid-state batteries", "semi-solid-state batteries"),
    ("半固态电池", "semi-solid-state batteries", "all-solid-state batteries"),
])
def test_discovery_preserves_solid_and_semisolid_distinctions(topic, wanted, excluded):
    assert wanted in equivalent_terms(topic) and excluded not in equivalent_terms(topic)
    strategies = [discovery_query(topic, "question_subject", i) for i in range(4)]
    assert len({query for _, query in strategies}) == 4
    assert strategies[0][0] == "scholarly" and "research paper" in strategies[0][1]


async def test_router_empty_hits_and_empty_cache_continue_to_another_backend(tmp_path, monkeypatch):
    from core_engine.report import router
    from core_engine.report.search_cache import SearchCache

    zero, good = FakeSearchProvider(), FakeSearchProvider()
    good.add("", [SearchHit("https://example.edu/paper", "Paper", "", "example.edu")])
    monkeypatch.setattr(router, "get_search_provider", lambda name: {"zero": zero, "good": good}[name])
    cache = SearchCache(tmp_path / "cache")
    cache.put("zero", "niche topic", [])
    registry = BackendRegistry({QueryType.GENERAL: ("zero", "good")})
    assert (await routed_search("niche topic", max_results=3, registry=registry, cache=cache))[0].title == "Paper"
    assert registry.routes["niche topic"] == "good"


async def test_all_successful_empty_backends_return_no_result_instead_of_outage(monkeypatch):
    from core_engine.report import router
    monkeypatch.setattr(router, "get_search_provider", lambda name: FakeSearchProvider())
    assert await routed_search("niche topic", max_results=3, registry=BackendRegistry({QueryType.GENERAL: ("empty",)})) == []


def test_abstract_is_identified_but_not_mislabeled_as_reviewed_full_text():
    document = crossref_document(record(), "10.1234/battery")
    assert document.document_format == "abstract" and document.published_at == "2022-12-05"
    assert not document.metadata["full_text_accessed"] and document.metadata["publication_status"] == "review_status_unknown"
    assert "<jats" not in document.text
    assert crossref_document(record(), "10.1234/different") is None
    no_abstract = record(); no_abstract.pop("abstract")
    assert crossref_document(no_abstract, "10.1234/battery") is None
    assert crossref_document(record(title=["Retraction: battery"]), "10.1234/battery") is None


def test_sparse_bibliographic_metadata_does_not_discard_an_accessible_abstract():
    document = crossref_document(record(author=None, license=None, title=None, published=None), "10.1234/battery")
    assert document and document.text and document.published_at is None
    assert document.metadata["authors"] == document.metadata["licenses"] == []


async def test_crossref_returns_fetchable_abstract_and_primary_leads_not_snippet_evidence(monkeypatch):
    from core_engine.security import network
    calls = []

    async def request(client, url, **kwargs):
        calls.append(url)
        row = record(resource={"primary": {"URL": "https://publisher.example/paper.pdf"}})
        return httpx.Response(200, json={"message": {"items": [row]}})

    monkeypatch.setattr(network, "read_public_text", request)
    hits = await CrossrefSearchProvider().search("solid-state batteries", max_results=3)
    assert len(hits) == 2 and "api.crossref.org/works/10.1234%2Fbattery" in hits[0].url
    assert hits[1].url.endswith("paper.pdf") and "type%3Ajournal-article" in calls[0]
    assert "This study" not in hits[0].snippet


async def test_short_topic_matched_abstract_is_not_dropped_by_article_length_rules(tmp_path):
    document = crossref_document(record(), "10.1234/battery")
    assert 100 < len(document.text) < 400
    hit = SearchHit("https://api.crossref.org/works/10.1234%2Fbattery", "Solid-state batteries", "", "api.crossref.org")
    search = FakeSearchProvider(); search.add("", [hit])

    class Fetch:
        async def fetch(self, url):
            return document

    sources, _ = await gather_sources("固态电池", search, Fetch(), queries=["solid-state batteries research paper"],
                                      settings=configured(tmp_path))
    assert len(sources) == 1 and sources[0].document_format == "abstract"
    assert sources[0].metadata["doi"] == "10.1234/battery"


async def test_unrelated_short_document_still_cannot_become_battery_evidence(tmp_path):
    hit = SearchHit("https://api.crossref.org/works/10.1234%2Fother", "Unrelated paper", "", "api.crossref.org")
    search = FakeSearchProvider(); search.add("", [hit])

    class Fetch:
        async def fetch(self, url):
            return FetchedDocument("Museum archives describe historical paintings and their pigments. " * 2, "abstract")

    sources, rejected = await gather_sources("固态电池", search, Fetch(), queries=["solid-state batteries"], settings=configured(tmp_path))
    assert not sources and rejected


async def test_online_pdf_is_parsed_with_page_boundaries(monkeypatch):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    from core_engine.report import scrape

    writer = PdfWriter()
    for text in ("Solid-state batteries use solid electrolytes.", "Reported results require stated test conditions."):
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject(); stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    data = io.BytesIO(); writer.write(data)

    async def read(client, url, **kwargs):
        return httpx.Response(200, content=data.getvalue(), headers={"content-type": "application/pdf"}, request=httpx.Request("GET", url))

    async def allowed(*args):
        return True

    monkeypatch.setattr(scrape, "read_public_text", read)
    monkeypatch.setattr(HttpFetcher, "_allowed", allowed)
    document = await HttpFetcher().fetch("https://example.edu/paper.pdf")
    assert isinstance(document, FetchedDocument) and document.document_format == "pdf" and "\f" in document.text
    assert "Solid-state batteries" in document.text and "test conditions" in document.text


def test_same_doi_versions_and_republications_do_not_add_corroboration():
    text = "Solid-state battery research reports electrode interface measurements under laboratory conditions."
    sources = [Source("https://api.crossref.org/works/10.1038%2Fs41467-025-64726-y", "api.crossref.org", "A", SourceKind.ACADEMIC, text),
               Source("https://www.nature.com/articles/s41467-025-64726-y.pdf", "nature.com", "A", SourceKind.ACADEMIC, text + " More methods."),
               Source("https://mirror.example/copy", "mirror.example", "Copy", SourceKind.GENERAL_WEB, text)]
    assert len(set(origin_groups(sources).values())) == 1
    preprint = Source("https://arxiv.org/pdf/2401.12345v2.pdf", "arxiv.org", "Preprint", SourceKind.ACADEMIC, "text")
    assert work_identifier(preprint) == "arxiv:2401.12345"


def test_near_identical_republication_is_conservatively_grouped():
    text = " ".join(f"token{i}" for i in range(100))
    sources = [Source("https://a.example/p", "a.example", "A", SourceKind.GENERAL_WEB, text),
               Source("https://b.example/p", "b.example", "B", SourceKind.GENERAL_WEB, "News release. " + text)]
    assert len(set(origin_groups(sources).values())) == 1


def test_evidence_quality_distinguishes_single_abstract_preprint_and_truth():
    source = Source("https://arxiv.org/abs/2401.12345", "arxiv.org", "Battery paper", SourceKind.ACADEMIC,
                    "Solid-state batteries use solid electrolytes.", document_format="abstract")
    claim = Claim("c1", source.text, candidate_source_urls=[source.url])
    profile = assess_evidence([claim], [source], [citation(claim, source)])
    assert profile["truth_probability"] is None
    assert claim.evidence_quality["truth_status"] == "not_established"
    assert {"single_origin_not_independently_confirmed", "abstract_only_methods_and_full_results_not_checked",
            "preprint_peer_review_not_established"}.issubset(claim.evidence_quality["caveats"])


async def test_single_real_passage_produces_limited_report_instead_of_false_consensus(tmp_path):
    source = Source("https://example.edu/battery", "example.edu", "Solid-state batteries", SourceKind.ACADEMIC,
        "Solid-state batteries use solid electrolytes.", document_format="abstract")
    llm = FakeLLM()
    llm.set_structured_claims(source.url, [{"text": source.text}])
    pipeline = ReportPipeline(llm=llm, search=FakeSearchProvider(), fetcher=FakeFetcher())
    pipeline._s = configured(tmp_path)
    result = await pipeline.run("固态电池", skip_search=True, extra_sources=[source], compile_to_pdf=False)
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert result.report.quality_status == "needs_review"
    assert result.report.source_quality["evidence_mode"] == "limited_evidence"
    packet = evidence_packet(result.report)
    assert not check_packet(packet)["integrity_errors"]
    forged = copy.deepcopy(packet); forged["source_quality"]["truth_probability"] = .99
    assert check_packet(forged)["integrity_errors"]
    assert "有限证据报告" in Path(result.tex_path).read_text(encoding="utf-8")


async def test_sparse_mode_does_not_invent_missing_profit_or_change_units(tmp_path):
    source = Source("https://example.edu/battery", "example.edu", "Solid-state battery lab", SourceKind.ACADEMIC,
                    "Solid-state batteries have a reported 20% improvement in a laboratory test in 2024.")
    pipeline = ReportPipeline(llm=FakeLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    pipeline._s = configured(tmp_path)
    result = await pipeline.run("Acme's 2025 net profit", skip_search=True, extra_sources=[source], compile_to_pdf=False)
    assert result.status is PipelineStatus.OUT_OF_SCOPE and not result.tex_path


async def test_document_metadata_survives_database_roundtrip():
    from core_engine.app.database import _dict_to_source, _source_to_dict
    doc = crossref_document(record(), "10.1234/battery")
    source = Source("https://api.crossref.org/works/10.1234%2Fbattery", "api.crossref.org", "Paper", SourceKind.ACADEMIC,
                    doc.text, document_format=doc.document_format, metadata=doc.metadata)
    restored = _dict_to_source(_source_to_dict(source))
    assert restored.metadata == source.metadata and restored.document_format == "abstract"


async def test_identical_republications_do_not_promote_a_general_web_claim_to_corroborated(tmp_path):
    from core_engine.report.models import ConfidenceTier
    from core_engine.report.verify import VerificationHarness

    text = "The experimental battery delivered 140 mAh per gram in 2022."
    sources = [Source(f"https://{host}/report", host, "Copy", SourceKind.GENERAL_WEB, text)
               for host in ("a.example", "b.example")]
    claim = Claim("c1", text, candidate_source_urls=[s.url for s in sources])
    harness = VerificationHarness(FakeLLM()); harness._s = configured(tmp_path)
    report = await harness.verify([claim], sources)
    assert report.kept_claims[0].confidence is ConfidenceTier.RUMOR
    assert len(report.kept_claims[0].supporting_sources()) == 2


async def test_real_battery_abstract_benchmark_preserves_limits_and_refuses_nonanswers(tmp_path):
    data = load_dataset(Path("evals/niche_quality/cases.json"))
    summary = await run_cases(data, "baseline", tmp_path / "battery-quality.json")
    assert summary["automatic_checks_passed"] == summary["cases_run"] == 6
    assert summary["live_model_evaluated"] is False


async def test_nonnumeric_research_observation_is_not_discarded_by_offline_extractor():
    text = "We created a polymer design for a solid-state battery."
    claims = await FakeLLM().extract_claims("https://example.edu/paper", text)
    assert [claim.text for claim in claims] == [text]


def test_commercialization_barriers_are_not_confused_with_proof_of_production():
    from core_engine.report.answerability import assess_coverage
    text = "Solid-state battery interfaces remain challenging under laboratory conditions."
    source = Source("https://example.edu/p", "example.edu", "Study", SourceKind.ACADEMIC, text)
    result = assess_coverage("固态电池的商业化有哪些技术挑战？", [Claim("c1", text)], [source], topic_aliases=equivalent_terms("固态电池"))
    assert "production_scale_evidence" not in result.missing


def test_other_company_production_cannot_answer_a_named_company_question():
    from core_engine.report.answerability import assess_coverage
    text = "Other's solid-state battery mass production began in 2026."
    source = Source("https://example.edu/p", "example.edu", "Update 2026", SourceKind.ACADEMIC, text)
    result = assess_coverage("Has Acme's solid-state battery mass production begun in 2026?", [Claim("c1", text)], [source])
    assert result.status == "insufficient" and "production_scale_evidence" in result.missing


def test_material_properties_are_not_routed_to_market_sizing_by_substring(tmp_path):
    pipeline = ReportPipeline(llm=FakeLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    pipeline._s = configured(tmp_path)
    facts = [Claim("c1", "We used the same polymer for solid-state batteries."),
             Claim("c2", "Dendrite growth depends on particle size under laboratory conditions."),
             Claim("c3", "TAM was 20 USD bn in 2024.")]
    buckets = pipeline._bucket_claims(facts)
    assert [c.id for c in buckets["market_size"]] == ["c3"]
    assert {c.id for c in buckets["industry_overview"]} == {"c1", "c2"}


@pytest.mark.parametrize(("claim", "source"), [("3.5 S/cm", "3.5 mS/cm"), ("140 Ah/g", "140 mAh/g"),
    ("300 Wh/L", "300 Wh/kg"), ("20 kPa", "20 MPa"), ("25 K", "25 °C"), ("80%", "80 mol%"),
    ("3.5 MS/cm", "3.5 mS/cm"), ("140 MAh/g", "140 mAh/g")])
def test_battery_units_cannot_silently_change_while_the_number_stays_the_same(claim, source):
    from core_engine.report.text import numbers_supported
    assert not numbers_supported(claim, source)


def test_battery_unit_notation_variants_preserve_scale_and_dimensions():
    from core_engine.report.text import numbers_supported
    assert numbers_supported("140 mAh/g", "140 mAh g−1")
    assert numbers_supported("3.5 mS/cm", "3.5 mS cm−1")
