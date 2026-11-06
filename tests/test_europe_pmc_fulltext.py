"""Exact DOI OA lookup cannot substitute metadata or the wrong article for evidence."""
from __future__ import annotations

import httpx
import pytest

from core_engine.config import get_settings
from core_engine.report.fulltext import (
    europe_pmc_fulltext,
    europe_pmc_lookup,
    europe_pmc_lookup_doi,
    fulltext_candidates,
)
from core_engine.report.models import Source, SourceKind
from core_engine.report.scrape import HttpFetcher

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize(("identifier", "open_access", "doi", "accepted"), [
    ("PMC12345", "Y", "10.1234/battery", True),
    ("PMC12345", "N", "10.1234/battery", False),
    ("PMC12345", "Y", "10.1234/other", False),
    ("../private", "Y", "10.1234/battery", False),
    (None, "Y", "10.1234/battery", False),
])
def test_exact_open_access_identity_is_required(identifier, open_access, doi, accepted):
    payload = {"resultList": {"result": [{"pmcid": identifier, "isOpenAccess": open_access, "doi": doi}]}}
    target = europe_pmc_fulltext(payload, "10.1234/battery")
    assert bool(target) == accepted
    if target:
        assert target == "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12345/fullTextXML"


def test_lookup_url_is_fixed_and_doi_is_not_a_free_query():
    url = europe_pmc_lookup("10.1234/battery")
    assert europe_pmc_lookup_doi(url) == "10.1234/battery"
    assert europe_pmc_lookup_doi(url.replace("www.ebi.ac.uk", "evil.example")) is None
    assert europe_pmc_lookup_doi(url + "&pageSize=100") is None
    assert europe_pmc_lookup_doi(url + "#fragment") is None
    assert europe_pmc_lookup("not-a-doi") is None


@pytest.mark.parametrize("xml_doi", ["10.1234/battery", "10.1234/wrong"])
async def test_fetch_reads_actual_xml_and_checks_its_doi(monkeypatch, xml_doi):
    lookup = europe_pmc_lookup("10.1234/battery")
    requests = []
    fulltext = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12345/fullTextXML"
    xml = (f'<article><front><article-meta><article-id pub-id-type="doi">{xml_doi}</article-id>'
           '<title-group><article-title>Solid-state batteries</article-title></title-group></article-meta></front>'
           '<body><sec><title>Results</title><p>' + "Solid-state batteries use solid electrolytes. " * 30 + '</p></sec></body></article>')

    async def read(client, url, **kwargs):
        requests.append(url)
        if url == lookup:
            return httpx.Response(200, json={"resultList": {"result": [
                {"pmcid": "PMC12345", "doi": "10.1234/battery", "isOpenAccess": "Y"}]}}, request=httpx.Request("GET", url))
        assert url == fulltext
        return httpx.Response(200, text=xml, headers={"content-type": "application/xml"}, request=httpx.Request("GET", url))

    fetcher = HttpFetcher()
    fetcher._s = get_settings().model_copy(update={"scrape_robots_fail_open": True})

    async def allowed(client, url):
        return True

    monkeypatch.setattr(fetcher, "_allowed", allowed)
    monkeypatch.setattr("core_engine.report.scrape.read_public_text", read)
    result = await fetcher.fetch_fulltext(lookup)
    assert requests == [lookup, fulltext]
    if xml_doi != "10.1234/battery":
        assert result is None and fetcher.failures[lookup] == "fulltext_identity_mismatch"
    else:
        assert result.document_format == "jats" and result.metadata["full_text_accessed"] is True
        assert result.metadata["actual_url"] == fulltext
        assert result.metadata["fulltext_resolver"] == "europe_pmc"


def test_a_read_same_doi_fulltext_prevents_duplicate_copy_downloads():
    abstract = Source("https://api.openalex.org/works/W123", "api.openalex.org", "Paper", SourceKind.ACADEMIC,
                      "Abstract", document_format="abstract", metadata={"doi": "10.1234/battery"})
    copy = Source("https://repo.edu/paper", "repo.edu", "Paper", SourceKind.ACADEMIC, "Full text",
                  document_format="jats", metadata={"doi": "10.1234/battery", "full_text_accessed": True})
    assert fulltext_candidates([abstract], 1)[0].url == europe_pmc_lookup("10.1234/battery")
    assert fulltext_candidates([abstract, copy], 3) == []


async def test_crossref_fulltext_fallback_preserves_parent_and_preprint_status():
    from core_engine.report.scholarly import FetchedDocument
    from core_engine.report.scrape import FakeFetcher, FakeSearchProvider, gather_sources

    parent = Source("https://api.crossref.org/works/10.1234/battery", "api.crossref.org", "Solid-state batteries",
                    SourceKind.ACADEMIC, "Abstract", document_format="abstract",
                    metadata={"doi": "10.1234/battery", "publication_type": "preprint"})
    hits = fulltext_candidates([parent], 3)

    class Provider(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            return hits

    document = FetchedDocument("Solid-state batteries use solid electrolytes. " * 30, "jats",
                               metadata={"doi": "10.1234/battery", "full_text_accessed": True})
    sources, _ = await gather_sources("固态电池", Provider(), FakeFetcher({hits[0].url: document}), queries=["固态电池"])
    assert len(sources) == 1 and sources[0].metadata["fulltext_parent_url"] == parent.url
    assert sources[0].metadata["publication_status"] == "preprint"
