"""Failures observed during a real paired report run must remain auditable."""
from __future__ import annotations

import httpx
import pytest

from core_engine.config import get_settings
from core_engine.report.llm import FakeLLM
from core_engine.report.models import PipelineStatus, SearchHit
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.scrape import (
    DuckDuckGoSearchProvider,
    FakeFetcher,
    SearchUnavailableError,
    gather_sources,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class MixedSearch:
    async def search(self, topic, *, max_results):
        if topic == "offline":
            raise SearchUnavailableError("Test channel disconnected")
        if topic == "empty":
            return []
        return [SearchHit(f"https://example.edu/{topic}", "Battery study", "", "example.edu")]


def settings(tmp_path):
    return get_settings().model_copy(update={"output_dir": tmp_path, "relevance_filter": False,
        "research_max_rounds": 0, "research_max_queries": 0})


async def test_one_failed_angle_does_not_discard_successful_sources_and_audit_is_per_query(tmp_path):
    body = "Solid-state battery research reports laboratory testing conditions and interface limitations. " * 6
    fetcher = FakeFetcher({"https://example.edu/good": body, "https://example.edu/thin": "Too short"})
    diagnostics = []
    sources, rejected = await gather_sources("固态电池", MixedSearch(), fetcher,
        queries=["good", "thin", "offline"], settings=settings(tmp_path), diagnostics=diagnostics)
    assert [source.url for source in sources] == ["https://example.edu/good"]
    rows = {row["query"]: row for row in diagnostics}
    assert rows["good"]["retrieved_urls"] == ["https://example.edu/good"]
    assert rows["good"]["rejections"] == []
    assert rows["thin"]["retrieved_urls"] == []
    assert rows["thin"]["rejections"][0]["url"] == rejected[0].url
    assert rows["offline"]["status"] == "error" and rows["offline"]["hit_count"] == 0


async def test_all_angles_failing_is_outage_with_diagnostics(tmp_path):
    diagnostics = []
    with pytest.raises(SearchUnavailableError, match="All query angles failed"):
        await gather_sources("固态电池", MixedSearch(), FakeFetcher(), queries=["offline"],
                             settings=settings(tmp_path), diagnostics=diagnostics)
    assert diagnostics[0]["error_type"] == "SearchUnavailableError"


async def test_successful_empty_angle_is_not_relabelled_outage(tmp_path):
    sources, rejected = await gather_sources("固态电池", MixedSearch(), FakeFetcher(),
        queries=["offline", "empty"], settings=settings(tmp_path))
    assert sources == rejected == []


def test_unknown_publication_date_is_neutral_instead_of_freshly_published():
    from core_engine.report.scrape import _freshness_score
    assert _freshness_score(None) == _freshness_score("unknown") == 0.5
    assert _freshness_score("2000-01-01") < _freshness_score(None)


async def test_pipeline_preserves_research_audit_when_initial_retrieval_fails(tmp_path):
    class Offline(MixedSearch):
        async def search(self, *args, **kwargs):
            raise SearchUnavailableError("Test network disconnected")
    pipeline = ReportPipeline(llm=FakeLLM(), search=Offline(), fetcher=FakeFetcher())
    pipeline._s = settings(tmp_path)
    result = await pipeline.run("固态电池", compile_to_pdf=False)
    assert result.status is PipelineStatus.ERROR
    assert result.research_audit["stop_reason"] == "initial_retrieval_error"
    assert result.research_audit["initial_queries"][0]["status"] == "error"


@pytest.mark.parametrize("response", ["empty", "challenge", "network"])
async def test_duckduckgo_reports_observed_failures_without_inventing_http_status(monkeypatch, response):
    actual = httpx.AsyncClient
    def handler(request):
        if response == "network":
            raise httpx.ConnectError("Test disconnected", request=request)
        return httpx.Response(202 if response == "challenge" else 200, text="<html>No matches</html>")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: actual(transport=httpx.MockTransport(handler), **kwargs))
    async def no_library(*args):
        return []
    provider = DuckDuckGoSearchProvider()
    monkeypatch.setattr(provider, "_via_library", no_library)
    if response == "empty":
        assert await provider.search("battery", max_results=2) == []
    else:
        with pytest.raises(SearchUnavailableError) as caught:
            await provider.search("battery", max_results=2)
        assert ("202" in str(caught.value)) == (response == "challenge")
        assert ("ConnectError" in str(caught.value)) == (response == "network")
