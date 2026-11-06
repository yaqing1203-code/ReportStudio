from __future__ import annotations

from itertools import pairwise

import httpx
import pytest

from core_engine.config import get_settings
from core_engine.report.llm import FakeLLM, LLMResponseError, OpenAILLM, completion_text
from core_engine.report.models import PipelineStatus, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clear_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize(("text", "finish", "reason"), [
    ("", "stop", "empty_final_answer"), (None, "stop", "empty_final_answer"),
    ("<think>not a final answer</think>", "stop", "empty_final_answer"),
    ('[{"text":"truncated', "length", "output_limit"),
    ("", "max_tokens", "output_limit"), ("", "content_filter", "provider_refusal"),
])
def test_incomplete_or_empty_completion_is_not_successful_no_facts(text, finish, reason):
    with pytest.raises(LLMResponseError) as caught:
        completion_text(text, finish)
    assert caught.value.reason == reason
    assert completion_text("[]", "stop") == "[]"  # Valid empty extraction remains valid.


@pytest.mark.parametrize("host", ["api.minimaxi.com", "api.minimax.io", "compatible.example"])
async def test_minimax_specific_output_controls_do_not_leak_to_other_providers(monkeypatch, host):
    monkeypatch.setenv("CE_LLM_THINKING_MODE", "disabled")
    requests = []
    def handler(request):
        import json
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]})
    actual = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: actual(transport=httpx.MockTransport(handler), **kwargs))
    llm = OpenAILLM("test-key", "MiniMax-M3", 4096, "https://" + host + "/v1")
    assert await llm.complete("Extract facts", "Source") == "[]"
    payload = requests[0]
    if host == "compatible.example":
        assert payload["max_tokens"] == 4096 and "thinking" not in payload
    else:
        assert "max_tokens" not in payload and payload["max_completion_tokens"] == 4096
        assert payload["thinking"] == {"type": "disabled"} and payload["reasoning_split"] is True


async def test_unsupported_thinking_disable_fails_before_sending_request(monkeypatch):
    monkeypatch.setenv("CE_LLM_THINKING_MODE", "disabled")
    llm = OpenAILLM("test-key", "MiniMax-M3.1-Flash", 4096, "https://api.minimax.io/v1")
    with pytest.raises(LLMResponseError, match="thinking_disable_unsupported"):
        await llm.complete("Extract facts", "Source")


async def test_all_extraction_failures_are_pipeline_error_with_source_audit(tmp_path):
    class EmptyResponseLLM(FakeLLM):
        async def extract_claims(self, *args):
            raise LLMResponseError("output_limit")
    source = Source("https://example.edu/paper", "example.edu", "Solid-state batteries", SourceKind.ACADEMIC,
        "Solid-state batteries are being studied under specified laboratory conditions.")
    pipeline = ReportPipeline(llm=EmptyResponseLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    pipeline._s = get_settings().model_copy(update={"output_dir": tmp_path})
    result = await pipeline.run("固态电池", extra_sources=[source], skip_search=True, compile_to_pdf=False)
    assert result.status is PipelineStatus.ERROR
    assert "all_source_extractions_failed" in result.message
    audit = next(row for row in result.trace if row["stage"] == "extraction_sources")
    assert audit["sources"][0]["reason"] == "output_limit"


async def test_one_extraction_failure_keeps_other_source_claims_and_records_both():
    class PartialLLM(FakeLLM):
        async def extract_claims(self, url, text):
            if url.endswith("/bad"):
                raise LLMResponseError("empty_final_answer")
            return await super().extract_claims(url, text)
    sources = [Source(f"https://example.edu/{name}", "example.edu", "Paper", SourceKind.ACADEMIC,
        "Solid-state batteries are studied at 25 °C.") for name in ("bad", "good")]
    pipeline = ReportPipeline(llm=PartialLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    audit = []
    claims = await pipeline._extract_claims(sources, audit=audit)
    assert claims and {row["status"] for row in audit} == {"error", "completed"}


async def test_extraction_budgets_bound_claims_and_keep_later_corroborating_urls():
    from core_engine.report.llm import ExtractedClaim
    class VerboseLLM(FakeLLM):
        async def extract_claims(self, url, text):
            return [ExtractedClaim(f"Battery observation {i}.", [url]) for i in range(8)]
    sources = [Source(f"https://example.edu/{i}", "example.edu", "Paper", SourceKind.ACADEMIC, "Source") for i in range(2)]
    pipeline = ReportPipeline(llm=VerboseLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    pipeline._s = get_settings().model_copy(update={"extraction_max_claims_per_source": 3, "extraction_max_claims": 2})
    audit = []
    claims = await pipeline._extract_claims(sources, audit=audit)
    assert len(claims) == 2 and all(len(claim.candidate_source_urls) == 2 for claim in claims)
    assert all(row["budget_omitted"] == 5 for row in audit if "source_url" in row)
    assert next(row for row in audit if row.get("scope") == "total_claim_budget")["budget_omitted"] == 2


async def test_spacing_is_shared_across_parallel_model_requests(monkeypatch):
    import asyncio
    import time
    monkeypatch.setenv("CE_LLM_MIN_INTERVAL_S", "0.02")
    llm = OpenAILLM("test-key", "example-model", 4096, "https://compatible.example/v1")
    starts = []
    async def transport(*args):
        # Use a high-resolution clock on Windows Python 3.11.
        starts.append(time.perf_counter())
        return "[]"
    monkeypatch.setattr(llm, "_transport", transport)
    assert await asyncio.gather(*(llm.complete("s", "u") for _ in range(3))) == ["[]"] * 3
    assert all(b - a >= 0.018 for a, b in pairwise(starts))


async def test_analysis_cannot_borrow_invisible_claim_ids_from_gap_bookkeeping():
    import json

    from core_engine.report.reasoning import analyze
    class Inspect(FakeLLM):
        async def complete(self, system, user):
            if system.startswith("Evidence-based analyst"):
                payload = json.loads(user)
                assert payload["claims"][0]["id"] == "c1"
                assert "hidden_claim" not in json.dumps(payload["evidence_gaps"])
                assert "must be disjoint" in system
                return '{"findings":[]}'
            return await super().complete(system, user)
    plan = {"questions": [{"id": "q1", "question": "固态电池"}]}
    audit = {"questions": [{"id": "q1", "question": "固态电池", "status": "partial", "claim_ids": ["hidden_claim"]}]}
    evidence = {"claims": [{"id": "c1", "text": "Solid-state batteries are studied.", "verified": True}], "citations": []}
    findings, checked = await analyze(plan, audit, evidence, [], [], Inspect(), get_settings())
    assert not findings and checked["status"] == "completed"
