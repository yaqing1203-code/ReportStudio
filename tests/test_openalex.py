"""Offline contracts for complementary discovery, provenance and failure isolation."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from core_engine.config import get_settings
from core_engine.report import openalex, router, scrape
from core_engine.report.discovery import equivalent_terms
from core_engine.report.evidence_quality import assess_evidence, origin_groups
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import FakeLLM
from core_engine.report.models import Claim, PipelineStatus, SearchHit, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet
from core_engine.report.scholarly import FetchedDocument
from core_engine.report.search_cache import SearchCache

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setenv("CE_SEARCH_API_KEY", "")
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "true")
    monkeypatch.setenv("CE_OPENALEX_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def inverted(text):
    index = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    return index


def record(**updates):
    return {"id": "https://openalex.org/W12345", "doi": "https://doi.org/10.1234/battery",
        "title": "Solid-state batteries: polymer electrolyte interfaces", "type": "article",
        "publication_date": "2022-12-05", "is_retracted": False,
        "abstract_inverted_index": inverted("Solid-state batteries use solid electrolytes. This laboratory study reports improved polymer electrolyte interfaces under explicitly stated testing conditions."),
        "authorships": [{"author": {"display_name": "Example Researcher"}}],
        "primary_location": {"landing_page_url": "https://example.edu/battery", "version": "publishedVersion"},
        "best_oa_location": {"pdf_url": "https://example.edu/battery.pdf", "version": "acceptedVersion",
                             "license": "cc-by", "is_oa": True}, **updates}


def hit(url):
    return SearchHit(url, "Example paper", "", urlsplit(url).hostname or "")


class Provider:
    def __init__(self, hits=(), error=False):
        self.hits, self.error, self.calls = list(hits), error, 0

    async def search(self, topic, *, max_results):
        self.calls += 1
        if self.error:
            raise scrape.SearchUnavailableError("test backend unavailable")
        return self.hits[:max_results]


def patch_backends(monkeypatch, primary, extra):
    def factory(name=None):
        return extra if name == "openalex" else primary
    monkeypatch.setattr(router, "get_search_provider", factory)


@pytest.mark.parametrize("index", [None, {}, {"a": [1]}, {"a": [0], "b": [0]},
    {"a": [True]}, {"a": [-1]}, {"a": [10000]}, {"a": "0"}, {"": [0]}])
def test_malformed_abstract_is_never_repaired_into_evidence(index):
    assert openalex.reconstruct_abstract(index) is None
    assert openalex.openalex_document(record(abstract_inverted_index=index), "W12345") is None


def test_abstract_reconstruction_preserves_order_numbers_units_and_provenance():
    text = "At 80 mol% the electrolyte measured 4.0 mS/cm at 25 °C."
    source = record(abstract_inverted_index=inverted(text))
    doc = openalex.openalex_document(source, "W12345")
    assert doc.text == text and doc.document_format == "abstract"
    assert doc.metadata["source_role"] == "aggregated_abstract"
    assert doc.metadata["publication_status"] == "review_status_unknown"
    assert doc.metadata["retraction_status"] == "not_flagged_by_openalex_not_independently_checked"
    assert not doc.metadata["full_text_accessed"]
    assert doc.metadata["licenses"] == []  # OA-copy licence is not the abstract's licence.
    assert doc.metadata["locations"][0]["version"] == "acceptedVersion"
    assert doc.metadata["doi"] == "10.1234/battery"
    assert doc.metadata["abstract_inverted_index"] == source["abstract_inverted_index"]


@pytest.mark.parametrize("updates", [{"id": "https://openalex.org/W999"},
    {"is_retracted": True}, {"title": "Retraction: Battery study"}, {"type": "paratext"},
    {"title": None}, {"type": []}])
def test_mismatched_retracted_or_nonresearch_records_are_not_study_evidence(updates):
    assert openalex.openalex_document(record(**updates), "W12345") is None


def test_preprint_and_sparse_metadata_are_explicit():
    doc = openalex.openalex_document(record(type="preprint", authorships=None,
        primary_location=None, best_oa_location=None, is_retracted=None), "W12345")
    assert doc.metadata["publication_status"] == "preprint"
    assert doc.metadata["authors"] == [] and doc.metadata["retraction_status"] == "not_determined"


def test_broad_battery_topic_matches_lithium_metal_wording_without_swapping_explicit_chemistry():
    assert "solid-state lithium metal batteries" in equivalent_terms("固态电池")
    assert "solid-state lithium-ion batteries" not in equivalent_terms("solid-state lithium metal batteries")
    specific = equivalent_terms("全固态锂离子电池")
    assert "all-solid-state lithium-ion batteries" in specific
    assert "all-solid-state lithium metal batteries" not in specific


def test_search_leads_require_a_readable_candidate_and_preserve_work_diversity():
    blank = record(id="https://openalex.org/W2", abstract_inverted_index=None,
                   primary_location=None, best_oa_location=None)
    other = record(id="https://openalex.org/W3", doi=None)
    hits = openalex.search_hits([record(), blank, other], 4)
    assert [h.url for h in hits[:2]] == ["https://api.openalex.org/works/W12345", "https://api.openalex.org/works/W3"]
    assert all("api_key" not in h.url for h in hits)
    assert hits[2].metadata["discovery_location"]["version"] == "acceptedVersion"
    assert openalex.search_hits([blank, record(is_retracted=True)], 6) == []
    assert openalex.search_query("全固态电池 2026 research paper") == "all-solid-state batteries 2026 research paper"


def test_readable_abstract_beats_resolver_only_and_landing_leads_including_old_cache():
    resolver_only = record(abstract_inverted_index=None, best_oa_location=None,
        primary_location={"landing_page_url": "https://doi.org/10.1234/battery"})
    assert openalex.search_hits([resolver_only], 6) == []
    no_abstract = record(abstract_inverted_index=None)
    readable = record(id="https://openalex.org/W456")
    hits = openalex.search_hits([no_abstract, readable], 6)
    assert hits[0].url == "https://api.openalex.org/works/W456"
    primary = [hit(f"https://example.edu/main{i}") for i in range(6)]
    old_cached = [hit("https://doi.org/10.1234/battery"), hit("https://api.openalex.org/works/W456")]
    merged = router._merge_complement(primary, old_cached, 6)
    assert merged[2].url == "https://api.openalex.org/works/W456"


def test_compact_evidence_context_represents_multiple_origins_and_english_papers():
    from core_engine.report.research import coverage_rows, related, requires_causal_support
    claims = [Claim(f"c{i}", "固态电池中文概述。" * 15, ["https://example.cn/a"]) for i in range(5)]
    claims.append(Claim("paper", "Solid-state lithium metal batteries require specified test conditions.", ["https://api.openalex.org/works/W456"]))
    rows = coverage_rows(claims, {}, 1200)
    assert any(row["id"] == "paper" for row in rows)
    assert related("固态电池", claims[-1].text)
    assert not requires_causal_support("固态电池的技术路线有哪些差异？")
    assert requires_causal_support("为什么固态电池界面阻抗影响性能？")


@pytest.mark.parametrize("hyphen", ["\u2010", "\u2011"])
async def test_typographic_hyphen_does_not_discard_relevant_english_abstract_for_chinese_topic(hyphen, tmp_path):
    from core_engine.report.research import related
    url = "https://api.openalex.org/works/W12345"
    text = (f"Solid{hyphen}state batteries are being studied using several electrolyte materials. "
            "This review reports unresolved production, safety and cost questions, not verified mass production.")
    class Fetch:
        async def fetch(self, url):
            return FetchedDocument(text, "abstract")
    provider = scrape.FakeSearchProvider()
    provider.add("", [SearchHit(url, f"A roadmap for solid{hyphen}state batteries", "", "api.openalex.org")])
    sources, rejected = await scrape.gather_sources("固态电池", provider, Fetch(), queries=["固态电池"],
        settings=get_settings().model_copy(update={"output_dir": tmp_path}))
    assert len(sources) == 1 and not rejected
    assert sources[0].text == text and related("固态电池", text)


@pytest.mark.parametrize("url", ["https://api.openalex.org.evil.test/works/W12345",
    "https://user:secret@api.openalex.org/works/W12345", "https://api.openalex.org/works/W12345?api_key=secret",
    "http://api.openalex.org/works/W12345", "https://api.openalex.org/works/W12345/extra"])
def test_credentials_only_apply_to_canonical_record_urls(url):
    assert openalex.api_work_id(url) is None


async def test_search_uses_bounded_public_reads_and_header_only_auth(monkeypatch):
    monkeypatch.setenv("CE_OPENALEX_API_KEY", "private-test-key")
    get_settings.cache_clear()
    requests = []

    async def read(client, url, *, max_bytes, timeout):
        requests.append((url, dict(client.headers)))
        assert not client.follow_redirects
        assert max_bytes == 3 * 1024 * 1024 and timeout <= 60
        return httpx.Response(200, json={"results": [record()]})

    monkeypatch.setattr(openalex, "read_public_text", read)
    hits = await openalex.OpenAlexSearchProvider().search("固态电池", max_results=100)
    url, headers = requests[0]
    params = parse_qs(urlsplit(url).query)
    assert params["search"] == ["solid-state batteries"]
    assert int(params["per_page"][0]) <= 8
    assert headers["authorization"] == "Bearer private-test-key"
    assert "private-test-key" not in json.dumps([h.metadata for h in hits]) + url


async def test_quota_retry_is_bounded_and_failure_does_not_expose_key(monkeypatch):
    monkeypatch.setenv("CE_OPENALEX_API_KEY", "sensitive-test-key")
    get_settings.cache_clear()
    requests = []

    async def read(client, url, **kwargs):
        requests.append(url)
        response = httpx.Response(429, request=httpx.Request("GET", url))
        response.raise_for_status()

    monkeypatch.setattr(openalex, "read_public_text", read)
    with pytest.raises(scrape.SearchUnavailableError, match="HTTP 429") as exc:
        await openalex.OpenAlexSearchProvider().search("battery paper", max_results=2)
    assert len(requests) == 2 and "sensitive-test-key" not in str(exc.value)


async def test_invalid_api_envelope_is_an_outage_not_no_results(monkeypatch):
    async def read(*args, **kwargs):
        return httpx.Response(200, json={"results": {"error": "invalid"}})
    monkeypatch.setattr(openalex, "read_public_text", read)
    with pytest.raises(scrape.SearchUnavailableError):
        await openalex.OpenAlexSearchProvider().search("battery", max_results=2)


async def test_default_router_merges_even_when_primary_is_full_and_uses_cache(monkeypatch, tmp_path):
    primary = Provider([hit(f"https://example.edu/{i}") for i in range(12)])
    extra = Provider([hit("https://api.openalex.org/works/W12345")])
    patch_backends(monkeypatch, primary, extra)
    cache = SearchCache(tmp_path)
    for _ in range(2):
        hits = await router.routed_search("battery paper", max_results=9, cache=cache)
        assert len(hits) == 9 and hits[2].url == "https://api.openalex.org/works/W12345"
        assert len({h.url for h in hits}) == len(hits)
    assert primary.calls == extra.calls == 1


@pytest.mark.parametrize(("query", "purpose", "prefer_record"), [
    ("battery impedance", "academic", True),
    ("battery impedance research paper", "gap", True),
    ("固态电池电导率论文", "gap", True),
    ("battery manufacturing", "gap", False),
    ("battery paper", "general", False),
])
async def test_only_targeted_academic_search_prioritizes_cached_record_at_limit_one(
    monkeypatch, tmp_path, query, purpose, prefer_record,
):
    primary = Provider([hit("https://example.edu/primary")])
    abstract = hit("https://api.openalex.org/works/W12345")
    landing = hit("https://example.edu/publisher-landing")
    extra = Provider(error=True)
    patch_backends(monkeypatch, primary, extra)
    cache = SearchCache(tmp_path)
    cache.put("openalex", query, [landing, abstract])

    hits = await router.routed_search(query, max_results=1, cache=cache, purpose=purpose)

    assert [h.url for h in hits] == [abstract.url if prefer_record else primary.hits[0].url]
    assert primary.calls == 1 and extra.calls == 0


async def test_academic_search_keeps_primary_if_no_direct_record_is_returned(monkeypatch):
    primary = Provider([hit("https://example.edu/primary")])
    extra = Provider([hit("https://example.edu/publisher-landing")])
    patch_backends(monkeypatch, primary, extra)

    hits = await router.routed_search("battery impedance", max_results=1, purpose="academic")

    assert [h.url for h in hits] == [primary.hits[0].url]
    assert extra.calls == 1


@pytest.mark.parametrize(("primary_error", "extra_error"), [(False, True), (True, False)])
async def test_either_channel_can_fail_without_discarding_other_results(monkeypatch, primary_error, extra_error):
    primary = Provider([hit("https://example.edu/primary")], error=primary_error)
    extra = Provider([hit("https://api.openalex.org/works/W12345")], error=extra_error)
    patch_backends(monkeypatch, primary, extra)
    hits = await router.routed_search("battery paper", max_results=4)
    assert len(hits) == 1
    assert ("openalex" in hits[0].url) == primary_error


async def test_all_channels_unavailable_remains_an_outage_and_empty_success_remains_empty(monkeypatch):
    patch_backends(monkeypatch, Provider(error=True), Provider(error=True))
    with pytest.raises(scrape.SearchUnavailableError, match="All search backends failed"):
        await router.routed_search("battery paper", max_results=4)
    patch_backends(monkeypatch, Provider(), Provider(error=True))
    assert await router.routed_search("battery paper", max_results=4) == []


async def test_complement_timeout_and_progress_callback_cannot_break_primary(monkeypatch):
    monkeypatch.setenv("CE_OPENALEX_TIMEOUT_S", "0.01")
    get_settings.cache_clear()
    class Slow:
        async def search(self, *args, **kwargs):
            await asyncio.sleep(10)
    primary = Provider([hit("https://example.edu/primary")])
    patch_backends(monkeypatch, primary, Slow())
    def bad_progress(message):
        raise RuntimeError("UI disconnected")
    assert len(await router.routed_search("battery paper", max_results=3, on_progress=bad_progress)) == 1


async def test_disable_explicit_registry_and_fake_runs_never_call_complement(monkeypatch):
    primary, extra = Provider([hit("https://example.edu/primary")]), Provider(error=True)
    patch_backends(monkeypatch, primary, extra)
    await router.routed_search("battery paper", max_results=3, registry=router.BackendRegistry())
    monkeypatch.setenv("CE_OPENALEX_SEARCH_ENABLED", "false")
    get_settings.cache_clear()
    await router.routed_search("battery paper", max_results=3)
    monkeypatch.setenv("CE_SEARCH_PROVIDER", "fake")
    get_settings.cache_clear()
    await router.routed_search("battery paper", max_results=3)
    assert extra.calls == 0


async def test_record_fetch_checks_returned_identity_and_does_not_send_key_to_publisher(monkeypatch):
    monkeypatch.setenv("CE_OPENALEX_API_KEY", "fetch-only-key")
    monkeypatch.setenv("CE_SCRAPE_RESPECT_ROBOTS", "false")
    get_settings.cache_clear()
    calls = []
    async def read(client, url, **kwargs):
        calls.append((url, client.headers.get("authorization")))
        return httpx.Response(200, json=record(), request=httpx.Request("GET", url))
    monkeypatch.setattr(scrape, "read_public_text", read)
    fetcher = scrape.HttpFetcher()
    assert (await fetcher.fetch("https://api.openalex.org/works/W12345")).document_format == "abstract"
    assert await fetcher.fetch("https://api.openalex.org/works/W999") is None
    assert await fetcher.fetch("https://example.edu/data.json") is None
    assert calls[0][1] == "Bearer fetch-only-key" and calls[-1][1] is None


async def test_gather_keeps_complement_before_page_cutoff_and_metadata_on_full_text(monkeypatch, tmp_path):
    from core_engine.app import runtime
    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    monkeypatch.setenv("CE_MAX_PAGES_PER_QUERY", "3")
    monkeypatch.setenv("CE_RELEVANCE_FILTER", "false")
    get_settings.cache_clear()
    main = [hit(f"https://example.edu/{i}") for i in range(12)]
    pdf = openalex.search_hits([record(type="preprint", abstract_inverted_index=None)], 1)[0]
    patch_backends(monkeypatch, Provider(main), Provider([pdf]))
    body = "Solid-state battery research conditions for electrolyte testing. " * 10
    fetcher = scrape.FakeFetcher({**{h.url: body for h in main},
        pdf.url: FetchedDocument(body, "pdf", metadata={"full_text_accessed": True})})
    sources, _ = await scrape.gather_sources("固态电池", None, fetcher, queries=["solid-state battery paper"])
    source = next(s for s in sources if s.url == pdf.url)
    assert len(sources) == 3 and source.document_format == "pdf"
    assert source.metadata["discovery_openalex_id"] == "W12345"
    assert source.metadata["full_text_accessed"] and source.metadata["publication_status"] == "preprint"
    assert source.published_at is None  # Registry primary date is not this PDF version's date.


async def test_academic_gap_schedules_abstracts_with_tight_round_budget(monkeypatch, tmp_path):
    from core_engine.app import runtime

    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    monkeypatch.setenv("CE_MAX_PAGES_PER_QUERY", "2")
    monkeypatch.setenv("CE_RELEVANCE_FILTER", "false")
    get_settings.cache_clear()
    # Match the observed q2/q3 gap round: each academic record previously sat
    # behind two web hits, and the round cap selected only one page per angle.
    queries = ["battery interface impedance research paper",
               "battery cycle life experimental paper", "battery manufacturing"]
    web_hits = {query: [hit(f"https://example.edu/{index}/{i}") for i in range(3)]
                for index, query in enumerate(queries)}
    abstracts = {query: [hit(f"https://api.openalex.org/works/W{12345 + index}")]
                 for index, query in enumerate(queries)}

    class PerQueryProvider:
        def __init__(self, hits):
            self.hits = hits

        async def search(self, topic, *, max_results):
            return self.hits[topic][:max_results]

    patch_backends(monkeypatch, PerQueryProvider(web_hits), PerQueryProvider(abstracts))
    fetched_urls = []
    body = "Solid-state battery experiment with detailed measurement conditions. " * 10

    class Fetcher:
        async def fetch(self, url):
            fetched_urls.append(url)
            return FetchedDocument(body + url, "abstract", metadata={"full_text_accessed": False})

    diagnostics = []
    sources, _ = await scrape.gather_sources(
        "solid-state battery", None, Fetcher(), queries=queries,
        max_pages=2, search_purpose="gap", diagnostics=diagnostics,
    )

    expected = [abstracts[q][0].url for q in queries[:2]]
    assert sorted(fetched_urls) == sorted(expected)
    assert [entry["scheduled_urls"] for entry in diagnostics] == [[expected[0]], [expected[1]], []]
    assert diagnostics[-1]["hit_urls"][0] == web_hits[queries[-1]][0].url
    assert {source.url for source in sources} == set(expected)
    assert all(s.document_format == "abstract" and not s.metadata["full_text_accessed"] for s in sources)


def test_crossref_openalex_and_pdf_of_same_work_are_one_origin_even_without_matching_text():
    crossref = Source("https://api.crossref.org/works/10.1234/battery", "api.crossref.org", "Paper", SourceKind.ACADEMIC, "Original abstract")
    doc = openalex.openalex_document(record(), "W12345")
    aggregated = Source("https://api.openalex.org/works/W12345", "api.openalex.org", "Paper", SourceKind.ACADEMIC, doc.text, metadata=doc.metadata)
    pdf = Source("https://example.edu/battery.pdf", "example.edu", "Paper", SourceKind.ACADEMIC, "Full paper with distinct methods and detailed results.", metadata={"discovered_via": "openalex", "discovery_openalex_id": "W12345", "discovery_doi": "10.1234/battery"})
    assert len(set(origin_groups([crossref, aggregated, pdf]).values())) == 1
    no_doi = copy.deepcopy(aggregated)
    no_doi.metadata.pop("doi")
    assert len(set(origin_groups([no_doi, pdf]).values())) == 1
    pdf.metadata["doi"] = "10.1234/different"
    assert len(set(origin_groups([crossref, aggregated, pdf]).values())) == 2


async def test_real_solid_state_excerpt_through_openalex_adapter_and_report_keeps_limits(tmp_path):
    pinned = json.loads(Path("evals/niche_quality/cases.json").read_text(encoding="utf-8"))["sources"][0]
    # Real CC BY excerpt in a simulated API envelope; no claim of live model evaluation.
    row = record(doi=pinned["metadata"]["doi"], title=pinned["title"], abstract_inverted_index=inverted(pinned["text"]))
    doc = openalex.openalex_document(row, "W12345")
    source = Source("https://api.openalex.org/works/W12345", "api.openalex.org", row["title"], SourceKind.ACADEMIC, doc.text, document_format="abstract", metadata=doc.metadata)
    pipeline = ReportPipeline(llm=FakeLLM(), search=scrape.FakeSearchProvider(), fetcher=scrape.FakeFetcher())
    pipeline._s = get_settings().model_copy(update={"output_dir": tmp_path, "enable_charts": False,
        "enable_deep_dive": False, "min_authoritative_sources": 2, "min_total_claims": 3,
        "active_verify_enabled": False, "verify_strategy": "cross_reference", "verify_mode": "strict"})
    result = await pipeline.run("固态电池", extra_sources=[source], skip_search=True, compile_to_pdf=False)
    assert result.status is PipelineStatus.COMPLETED
    assert result.report.source_quality["evidence_mode"] == "limited_evidence"
    packet = evidence_packet(result.report)
    assert not check_packet(packet)["integrity_errors"]
    assert all("aggregated_abstract_original_publication_not_directly_checked" in c.evidence_quality["caveats"] for c in result.report.claims)
    assessed = assess_evidence([Claim("unused", "Unsupported claim", [source.url])], [source], [])
    assert assessed["truth_probability"] is None


def test_openalex_gui_key_is_redacted_and_blank_preserves_it(monkeypatch, tmp_path):
    from core_engine.app import runtime
    from core_engine.app.server import SettingsUpdate
    monkeypatch.setattr(runtime, "user_data_dir", lambda: tmp_path)
    monkeypatch.setenv("CE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CE_OPENALEX_API_KEY", "")
    update = SettingsUpdate(openalex_api_key="gui-private-key", openalex_search_enabled=False)
    redacted = runtime.save_settings(update.model_dump(exclude_unset=True))
    assert redacted["openalex_api_key_set"] and "openalex_api_key" not in redacted
    assert not get_settings().openalex_search_enabled
    runtime.save_settings({"openalex_api_key": ""})
    assert runtime.load_settings()["openalex_api_key"] == "gui-private-key"
    runtime.save_settings({"openalex_api_key": None})
    assert not get_settings().openalex_api_key
