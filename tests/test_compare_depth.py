"""Benchmark accounting and isolation contracts; no live providers or paid calls."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from contextvars import copy_context
from dataclasses import asdict
from datetime import date

import httpx
import pytest

from core_engine.config import Settings, get_settings, settings_for_run
from core_engine.report import llm as llm_module
from core_engine.report.models import (
    PipelineResult,
    PipelineStatus,
    ReportData,
    ReportSection,
    ReportSentence,
    SentenceCitation,
    Source,
    SourceKind,
)
from core_engine.report.research import ResearchPlan, ResearchQuestion
from scripts import compare_depth as benchmark


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def plan():
    return ResearchPlan("固态电池", [ResearchQuestion(f"q{i}", "固态电池" if i == 1 else f"条件问题{i}",
                         required=i == 1) for i in range(1, 7)])


@pytest.fixture
def source():
    return Source("https://example.org/paper", "example.org", "Saved paper", SourceKind.ACADEMIC,
                  "The solid electrolyte was measured at 25 degrees Celsius.",
                  metadata={"full_text_accessed": True})


@pytest.fixture
def base(tmp_path):
    return Settings(llm_provider="openai", llm_model="fixture-model", llm_base_url="https://model.example/v1",
                    llm_api_key="never-serialize-this", output_dir=tmp_path / "unused", scrape_max_pages=25,
                    research_max_queries=8, research_max_rounds=3, research_enabled=True)


def test_frozen_plan_preserves_all_six_questions_and_requirements(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"research_plan": plan.to_dict()}), encoding="utf-8")
    assert benchmark.load_plan(path).to_dict() == plan.to_dict()
    changed = plan.to_dict()
    changed["questions"] = changed["questions"][:3]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="six-question"):
        benchmark.load_plan(path)


def test_coverage_never_drops_unanswered_questions_or_calls_failures_unsupported(plan):
    audit = {"questions": [
        {"id": "q1", "question": "固态电池", "status": "supported", "claim_ids": ["c1"]},
        {"id": "q2", "question": "条件问题2", "status": "partial"},
        {"id": "q3", "question": "条件问题3", "status": "missing"},
    ]}
    summary = benchmark.coverage_summary(plan, audit)
    assert summary["denominator"] == 6
    assert summary["complete"] == summary["partial"] == summary["unsupported"] == 1
    assert summary["not_evaluated"] == 3
    assert summary["complete_rate"] == 1 / 6
    assert benchmark.coverage_summary(plan, {})["not_evaluated"] == 6


def test_coverage_rejects_renamed_duplicate_or_unknown_questions(plan):
    for rows in ([{"id": "q1", "question": "rewritten", "status": "supported"}],
                 [{"id": "q1"}, {"id": "q1"}], [{"id": "q9"}]):
        with pytest.raises(ValueError):
            benchmark.coverage_summary(plan, {"questions": rows})


def test_unknown_dates_are_not_invented_and_future_dates_are_excluded():
    cutoff = date(2026, 11, 5)
    assert benchmark.before_cutoff(None, cutoff)
    assert benchmark.before_cutoff("unknown", cutoff)
    assert benchmark.before_cutoff("2026-11-05", cutoff)
    assert not benchmark.before_cutoff("2026-11-06", cutoff)
    assert not benchmark.before_cutoff("2027", cutoff)


def test_settings_redaction_removes_credentials_and_url_queries(base):
    settings = base.model_copy(update={"llm_base_url": "https://user:password@model.example/v1?api_key=secret"})
    public = benchmark.safe_settings(settings)
    encoded = json.dumps(public, default=str)
    assert "never-serialize-this" not in encoded and "password" not in encoded and "api_key" not in encoded
    assert public["llm_base_url"] == "https://model.example/v1"
    assert public["llm_max_tokens"] == base.llm_max_tokens
    assert benchmark.model_hostname(Settings(llm_provider="anthropic")) == "api.anthropic.com"


@pytest.mark.anyio
async def test_http_attempt_budget_counts_real_sends_and_blocks_next_before_transport(tmp_path):
    calls = []

    async def transport(request):
        calls.append(request.url)
        return httpx.Response(200, json={"ok": True})

    ledger = benchmark.Ledger(tmp_path, max_live_requests=1, mode="live", model_host="model.example")
    with ledger.http_accounting():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            await client.post("https://model.example/v1?api_key=never-log", headers={"Authorization": "never-log"})
            with pytest.raises(benchmark.RequestBudgetExceeded):
                await client.post("https://model.example/v1")
    assert len(calls) == ledger.counts["model_http_attempts"] == 1
    assert ledger.budget_exhausted
    assert "never-log" not in (tmp_path / "requests.json").read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_fixed_corpus_http_boundary_forbids_retrieval(tmp_path):
    ledger = benchmark.Ledger(tmp_path, max_live_requests=2, mode="replay", model_host="model.example")
    with ledger.http_accounting():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
            with pytest.raises(benchmark.ReplayNetworkDenied):
                await client.get("https://example.org/paper")
    assert ledger.counts["blocked_replay_http_attempts"] == 1
    assert ledger.counts["source_or_auxiliary_http_attempts"] == 0


@pytest.mark.anyio
async def test_exact_model_replays_are_separate_from_live_attempts_and_token_budgets(tmp_path, base):
    class Model:
        async def _complete(self, system, user):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
                await client.post("https://model.example/v1")
            return "recorded response"

        async def complete(self, system, user):
            return await self._complete(system, user)

    ledger = benchmark.Ledger(tmp_path, max_live_requests=3, mode="live", model_host="model.example")
    with settings_for_run(base), ledger.http_accounting():
        model = benchmark.cache_llm(Model(), tmp_path / "cache", ledger)
        assert await model.complete("system", "user") == "recorded response"
        assert await model.complete("system", "user") == "recorded response"
        token = llm_module._completion_budget.set(37)
        try:
            await model.complete("system", "user")
        finally:
            llm_module._completion_budget.reset(token)
    assert ledger.counts["model_http_attempts"] == 2
    assert ledger.counts["model_completion_operations"] == 2
    assert ledger.counts["model_cache_replays"] == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2
    assert ledger.snapshot()["monetary_cost"] is None


@pytest.mark.anyio
async def test_cache_only_fails_closed_without_hidden_live_fallback(tmp_path, base):
    class Model:
        async def complete(self, *_):
            pytest.fail("must never invoke live provider")

    ledger = benchmark.Ledger(tmp_path, max_live_requests=0, mode="replay", model_host="model.example")
    with settings_for_run(base):
        model = benchmark.cache_llm(Model(), tmp_path / "cache", ledger, cache_only=True)
        with pytest.raises(benchmark.ModelCacheMiss):
            await model.complete("system", "missing prompt")
    assert ledger.counts["model_http_attempts"] == ledger.counts["model_completion_operations"] == 0


@pytest.mark.anyio
async def test_concurrent_fetches_do_not_inflate_live_or_replay_counts(tmp_path):
    class Real:
        def __init__(self):
            self.failures = {}

        async def fetch(self, url):
            await asyncio.sleep(.005)
            return "A saved public page."

    ledger = benchmark.Ledger(tmp_path, max_live_requests=0, mode="live", model_host="model.example")
    fetcher = benchmark.RecordedFetcher(tmp_path / "sources", ledger)
    fetcher.real = Real()
    urls = ["https://example.org/a", "https://example.org/b"]
    await asyncio.gather(*(fetcher.fetch(url) for url in urls))
    await asyncio.gather(*(fetcher.fetch(url) for url in urls))
    assert ledger.counts["fetcher_live_operations"] == 2
    assert ledger.counts["fetcher_snapshot_replays"] == 2


@pytest.mark.anyio
async def test_manual_semantic_review_remains_null_until_someone_reviews(source):
    quote = source.text
    citation = SentenceCitation("e1", "c1", source.url, quote, 0, len(quote), 1, None,
                                hashlib.sha256(quote.encode()).hexdigest(), "verbatim")
    sentence = ReportSentence(quote, ["c1"], ["e1"], "supported", attribution="According to the paper: ")
    report = ReportData("topic", "title", "abstract", sources=[source], citations=[citation],
                        sections=[ReportSection("Heading", sentences=[sentence])])
    review = await benchmark.semantic_review(report, None, 0)
    assert review["semantic_citation_correctness"] is None
    assert review["status"] == "pending_manual_review"
    assert review["samples"][0]["citations"][0]["quote_matches_source"] is True
    assert review["samples"][0]["displayed_sentence"] == "According to the paper: " + quote


def test_comparison_refuses_different_denominators_and_marks_partial_pair(plan):
    row = {"status": "completed", "coverage": benchmark.coverage_summary(plan, {}),
           "plan_sha256": "same", "elapsed_wall_s": 1}
    result = benchmark.compare_arms({"standard": row, "deep": {**row, "status": "timeout"}})
    assert result["same_plan"] and not result["paired"]
    with pytest.raises(ValueError, match="denominators"):
        benchmark.compare_arms({"standard": row, "deep": {**row, "coverage": {"denominator": 3}}})


def test_execution_finished_is_not_a_usable_report_pair(plan):
    row = {"status": "completed", "coverage": benchmark.coverage_summary(plan, {}),
           "plan_sha256": "same", "elapsed_wall_s": 1, "report_available": False, "pipeline_status": "error"}
    assert not benchmark.compare_arms({"standard": row, "deep": row})["paired"]
    valid = {**row, "report_available": True, "pipeline_status": "completed"}
    assert benchmark.compare_arms({"standard": valid, "deep": valid})["paired"]
    assert not benchmark.compare_arms({"standard": valid, "deep": {**valid, "pipeline_status": "error"}})["paired"]


def test_matching_scientific_units_alone_do_not_make_observations_comparable():
    rows = [{"id": f"m{i}", "name": "ionic_conductivity", "entity": "electrolyte", "value": str(i),
             "unit": "S/cm"} for i in (1, 2)]
    pairs = benchmark.scientific_pairs(rows)
    assert pairs["scientific_observations"] == 2
    assert pairs["comparable_pairs"] == 0
    assert pairs["incomparable_candidates"][0]["reasons"]


def write_inputs(tmp_path, plan, source):
    plan_path, corpus = tmp_path / "plan.json", tmp_path / "corpus.json"
    plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    corpus.write_text(json.dumps({"report": {"sources": [asdict(source)]}}), encoding="utf-8")
    return plan_path, corpus


@pytest.mark.anyio
async def test_run_isolates_profiles_corpora_outputs_and_global_settings(tmp_path, plan, source, base, monkeypatch):
    plan_path, corpus = write_inputs(tmp_path, plan, source)
    original = copy.deepcopy(base.model_dump())
    original_global = get_settings()
    observed = []
    monkeypatch.setattr(benchmark, "code_state", lambda: {"sha256": "stable"})

    class Pipeline:
        def __init__(self, **_kwargs):
            self.profile = get_settings()

        async def run(self, topic, **kwargs):
            observed.append((self.profile.generation_depth, self.profile.scrape_max_pages,
                             str(self.profile.output_dir), kwargs["extra_sources"][0].text, self.profile.llm_thinking_mode))
            assert kwargs["skip_search"] and kwargs["research_plan"].to_dict() == plan.to_dict()
            kwargs["extra_sources"][0].text = "only this arm mutates its own copy"
            audit = {"questions": [{"id": "q1", "question": topic, "status": "supported", "claim_ids": []}]}
            return PipelineResult(PipelineStatus.COMPLETED, topic,
                                  report=ReportData(topic, "Fixture report", "Fixture only", research_audit=audit),
                                  research_plan=kwargs["research_plan"].to_dict(), research_audit=audit)

    args = benchmark.parse_args(["--mode", "replay", "--plan-file", str(plan_path), "--corpus-result", str(corpus),
                                 "--output", str(tmp_path / "runs"), "--locale", "zh", "--thinking-mode", "disabled"])
    result = await benchmark.run(args, base=base, pipeline_factory=Pipeline, llm_factory=llm_module.FakeLLM)
    assert result["comparison"]["paired"]
    assert observed[0][0:2] == ("standard", 12) and observed[1][0:2] == ("deep", 25)
    assert observed[0][2] != observed[1][2]
    assert observed[0][3] == observed[1][3] == source.text
    assert observed[0][4] == observed[1][4] == "disabled"
    assert base.model_dump() == original and get_settings() is original_global
    manifests = [json.loads((tmp_path / "runs" / arm / "manifest.json").read_text(encoding="utf-8")) for arm in ("standard", "deep")]
    assert manifests[0]["incoming_corpus"]["sha256"] == manifests[1]["incoming_corpus"]["sha256"]
    assert "never-serialize-this" not in json.dumps(manifests)
    run_manifest = json.loads((tmp_path / "runs/run-manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["task_overrides"]["thinking_mode"] == "disabled"
    assert run_manifest["task_overrides"]["saved_user_settings_modified"] is False


@pytest.mark.anyio
async def test_timeout_arm_keeps_partial_record_and_does_not_prevent_other_arm(tmp_path, plan, source, base, monkeypatch):
    plan_path, corpus = write_inputs(tmp_path, plan, source)
    monkeypatch.setattr(benchmark, "code_state", lambda: {"sha256": "stable"})

    class Pipeline:
        def __init__(self, **_):
            self.depth = get_settings().generation_depth

        async def run(self, topic, **kwargs):
            kwargs["on_progress"]("fixture_started", "Retain this partial record.")
            if self.depth == "standard":
                await asyncio.sleep(.2)
            return PipelineResult(PipelineStatus.OUT_OF_SCOPE, topic, research_plan=kwargs["research_plan"].to_dict())

    args = benchmark.parse_args(["--mode", "replay", "--plan-file", str(plan_path), "--corpus-result", str(corpus),
                                 "--output", str(tmp_path / "runs"), "--timeout", ".02"])
    result = await benchmark.run(args, base=base, pipeline_factory=Pipeline, llm_factory=llm_module.FakeLLM)
    assert result["arms"]["standard"]["status"] == "timeout"
    assert result["arms"]["standard"]["coverage"]["not_evaluated"] == 6
    assert result["arms"]["deep"]["status"] == "completed"
    assert not result["comparison"]["paired"]
    assert (tmp_path / "runs/standard/partial-manifest.json").exists()
    assert json.loads((tmp_path / "runs/standard/progress.json").read_text())[0]["stage"] == "fixture_started"


@pytest.mark.anyio
async def test_live_collection_requires_explicit_opt_in(tmp_path, base):
    args = benchmark.parse_args(["--mode", "live", "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="allow-live"):
        await benchmark.run(args, base=base)


@pytest.mark.anyio
async def test_hard_budget_arm_keeps_cache_and_next_arm_has_independent_budget(tmp_path, plan, source, base, monkeypatch):
    plan_path, corpus = write_inputs(tmp_path, plan, source)
    monkeypatch.setattr(benchmark, "code_state", lambda: {"sha256": "stable"})

    class Model:
        async def complete(self, *_):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
                await client.post("https://model.example/v1")
            return "ok"

    class Pipeline:
        def __init__(self, *, llm, **_):
            self.llm, self.depth = llm, get_settings().generation_depth

        async def run(self, topic, **kwargs):
            await self.llm.complete("first", "request")
            if self.depth == "standard":
                await self.llm.complete("second", "request")
            return PipelineResult(PipelineStatus.OUT_OF_SCOPE, topic, research_plan=kwargs["research_plan"].to_dict())

    args = benchmark.parse_args(["--mode", "replay", "--plan-file", str(plan_path), "--corpus-result", str(corpus),
                                 "--output", str(tmp_path / "runs"), "--max-live-requests", "1"])
    result = await benchmark.run(args, base=base, pipeline_factory=Pipeline, llm_factory=Model)
    assert result["arms"]["standard"]["status"] == "budget_exhausted"
    assert result["arms"]["deep"]["status"] == "completed"
    assert all(arm["accounting"]["counts"]["model_http_attempts"] == 1 for arm in result["arms"].values())
    assert len(list((tmp_path / "runs/standard/cache/llm").glob("*.json"))) == 1


@pytest.mark.anyio
async def test_model_failure_records_reason_and_effective_budget_without_prompt_text(tmp_path, base):
    class Model:
        async def complete(self, *_):
            raise llm_module.LLMResponseError("output_limit")

    ledger = benchmark.Ledger(tmp_path, max_live_requests=2, mode="live", model_host="model.example")
    with settings_for_run(base):
        model = benchmark.cache_llm(Model(), tmp_path / "cache", ledger)
        with pytest.raises(llm_module.LLMResponseError):
            await model.complete("private-system", "private-user")
    saved = json.loads((tmp_path / "requests.json").read_text(encoding="utf-8"))
    event = next(row for row in saved if row["kind"] == "model_completion_failure")
    assert event["reason"] == "output_limit" and event["effective_max_tokens"] == base.llm_max_tokens
    assert "private-system" not in json.dumps(saved) and "private-user" not in json.dumps(saved)


@pytest.mark.anyio
async def test_provider_usage_is_observed_not_inferred_or_called_a_bill(tmp_path):
    bodies = [
        {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
         "choices": [{"finish_reason": "length", "message": {"content": "private-response", "reasoning_content": "private-thought"}}]},
        {"choices": [{"finish_reason": ["unexpected"]}]},
        {"usage": {"prompt_tokens": -1, "completion_tokens": 1.5, "total_tokens": True}},
    ]

    def response(_):
        return httpx.Response(200, json=bodies.pop(0))

    ledger = benchmark.Ledger(tmp_path, max_live_requests=3, mode="live", model_host="model.example")
    with ledger.http_accounting():
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            for _ in range(3):
                await client.post("https://model.example/v1")
    result = ledger.snapshot()
    assert result["provider_reported_usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert result["counts"]["provider_usage_response_count"] == 1
    assert result["billed_tokens"] is result["monetary_cost"] is None
    assert "private-response" not in (tmp_path / "requests.json").read_text(encoding="utf-8")
    assert "private-thought" not in (tmp_path / "requests.json").read_text(encoding="utf-8")
    assert benchmark.Ledger(tmp_path / "empty", max_live_requests=1, mode="live", model_host="model.example").snapshot()["provider_reported_usage"] is None


@pytest.mark.anyio
async def test_budget_stop_drains_owned_requests_and_never_borrows_next_arm_budget(tmp_path, plan, source, base, monkeypatch):
    plan_path, corpus = write_inputs(tmp_path, plan, source)
    monkeypatch.setattr(benchmark, "code_state", lambda: {"sha256": "stable"})
    started, cancelled, unrelated_sent = asyncio.Event(), asyncio.Event(), asyncio.Event()
    stale_contexts = []
    loop = asyncio.get_running_loop()
    initial_factory = loop.get_task_factory()

    async def unrelated():
        await started.wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
            await client.post("https://model.example/unrelated")
        unrelated_sent.set()
        await asyncio.Future()

    outsider = asyncio.create_task(unrelated())

    class Model:
        async def complete(self, system, _user):
            async def respond(_):
                if system == "hold":
                    started.set()
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                return httpx.Response(200)

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                await client.post("https://model.example/owned")
            return "ok"

    class Pipeline:
        def __init__(self, *, llm, **_):
            self.llm, self.depth = llm, get_settings().generation_depth

        async def run(self, topic, **kwargs):
            if self.depth == "standard":
                stale_contexts.append(copy_context())
                asyncio.create_task(self.llm.complete("hold", "request"))
                await unrelated_sent.wait()
                await self.llm.complete("second", "request")  # Budget stop while first still in flight.
            else:
                assert cancelled.is_set() and not outsider.done()

                async def stale_http():
                    def unexpected(_):
                        pytest.fail("A closed arm sent a request through the new arm")
                    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
                        await client.post("https://model.example/stale")

                with pytest.raises(benchmark.ArmScopeClosed):
                    await asyncio.create_task(stale_http(), context=stale_contexts[0])
                await self.llm.complete("deep", "request")
            return PipelineResult(PipelineStatus.OUT_OF_SCOPE, topic, research_plan=kwargs["research_plan"].to_dict())

    args = benchmark.parse_args(["--mode", "replay", "--plan-file", str(plan_path), "--corpus-result", str(corpus),
                                 "--output", str(tmp_path / "runs"), "--max-live-requests", "1"])
    try:
        result = await benchmark.run(args, base=base, pipeline_factory=Pipeline, llm_factory=Model)
        assert not outsider.done()
    finally:
        outsider.cancel()
        await asyncio.gather(outsider, return_exceptions=True)
    assert loop.get_task_factory() is initial_factory
    standard, deep = (result["arms"][arm] for arm in ("standard", "deep"))
    assert standard["status"] == "budget_exhausted" and deep["status"] == "completed"
    assert standard["accounting"]["counts"]["background_tasks_cancelled"] == 1
    assert standard["accounting"]["counts"]["background_cleanup_pending"] == 0
    assert standard["accounting"]["counts"]["model_http_attempts"] == deep["accounting"]["counts"]["model_http_attempts"] == 1
