"""Output scope changes budgets, never evidence rules or another job's settings."""
from __future__ import annotations

import asyncio

import pytest

from core_engine.config import Settings, get_settings, settings_for_run
from core_engine.report.generation import profile_audit, report_profile

pytestmark = pytest.mark.anyio

# This is the permitted scope of the product control, not the implementation's
# cap mapping: any new change to evidence/verification settings must fail here.
SCOPE_FIELDS = frozenset({
    "scrape_max_pages", "scrape_max_results", "extraction_max_claims",
    "extraction_max_chunks_per_source", "research_max_rounds", "research_max_queries",
    "research_deadline_s", "research_fulltext_max_documents", "research_local_max_chunks",
    "reasoning_max_findings", "max_deep_dives", "deep_dive_max_claims",
    "deep_dive_min_paragraphs", "deep_dive_target_words", "section_min_paragraphs",
    "section_target_words",
})


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clear_base_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def configured(**changes):
    # Supply every field explicitly so real operator environment/secrets cannot
    # change the profile contract or appear in an assertion's diagnostic output.
    values = Settings.model_construct().model_dump()
    values.update(changes)
    return Settings(_env_file=None, **values)


def test_standard_and_deep_have_materially_different_research_and_output_budgets():
    base = configured()
    before = base.model_dump()
    standard, deep = report_profile(base, "standard"), report_profile(base, "deep")
    assert standard.generation_depth == "standard" and deep.generation_depth == "deep"
    assert standard.scrape_max_pages < deep.scrape_max_pages
    assert standard.extraction_max_claims < deep.extraction_max_claims
    assert standard.research_max_queries < deep.research_max_queries
    assert standard.research_deadline_s < deep.research_deadline_s
    assert standard.reasoning_max_findings < deep.reasoning_max_findings
    assert standard.section_target_words < deep.section_target_words
    assert standard.max_deep_dives < deep.max_deep_dives
    assert {key: value for key, value in deep.model_dump().items() if key != "generation_depth"} == {
        key: value for key, value in before.items() if key != "generation_depth"}
    assert base.model_dump() == before and base.generation_depth is None
    assert standard is not base and deep is not base and standard is not deep
    standard.scrape_max_pages = 1
    assert base.scrape_max_pages == deep.scrape_max_pages == before["scrape_max_pages"]


@pytest.mark.parametrize("depth", ["standard", "deep"])
def test_profiles_respect_lower_operator_caps_and_disabled_budgets(depth):
    low = {
        "scrape_max_pages": 2, "scrape_max_results": 2, "extraction_max_claims": 1,
        "extraction_max_chunks_per_source": 1, "research_max_rounds": 0,
        "research_max_queries": 0, "research_deadline_s": 0.25,
        "research_fulltext_max_documents": 0, "research_local_max_chunks": 0,
        "reasoning_max_findings": 1, "max_deep_dives": 0, "deep_dive_max_claims": 0,
        "deep_dive_min_paragraphs": 1, "deep_dive_target_words": 50,
        "section_min_paragraphs": 1, "section_target_words": 50,
    }
    base = configured(**low)
    run = report_profile(base, depth)
    assert {key: getattr(run, key) for key in low} == low
    assert all(getattr(run, key) <= getattr(base, key) for key in SCOPE_FIELDS)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("depth", ["standard", "deep"])
def test_depth_never_changes_evidence_publication_fulltext_flags_or_question_denominator(depth, enabled):
    base = configured(
        verify_mode="strict" if enabled else "traceable",
        verify_strategy="cross_reference" if enabled else "aligned",
        research_max_questions=9, research_enabled=enabled,
        research_fulltext_enabled=enabled, active_verify_enabled=enabled,
        enable_synthesis_sections=enabled, scholarly_search_enabled=enabled,
        openalex_search_enabled=enabled, allow_general_web=enabled,
        sparse_evidence_enabled=enabled, scrape_respect_robots=enabled,
    )
    run = report_profile(base, depth)
    protected = set(base.model_dump()) - SCOPE_FIELDS - {"generation_depth"}
    assert {key: getattr(run, key) for key in protected} == {
        key: getattr(base, key) for key in protected}
    assert run.research_max_questions == 9
    assert run.research_fulltext_enabled is enabled


@pytest.mark.parametrize("depth", ["standard", "deep"])
def test_profile_audit_reports_actual_budgets_without_mutating_settings(depth):
    run = report_profile(configured(), depth)
    audit = profile_audit(run)
    assert audit["depth"] == depth
    assert set(audit["budgets"]) == SCOPE_FIELDS
    assert all(value == getattr(run, key) for key, value in audit["budgets"].items())
    audit["budgets"]["research_max_queries"] = 999
    assert run.research_max_queries < 999


@pytest.mark.parametrize("depth", ["full", "DEEP", "", "standard ", None])
def test_invalid_profile_depth_is_rejected_without_editing_base(depth):
    base = configured()
    before = base.model_dump()
    with pytest.raises(ValueError):
        report_profile(base, depth)
    assert base.model_dump() == before


def test_nested_contexts_and_exceptions_restore_the_previous_settings():
    original = get_settings()
    standard, deep = report_profile(configured(), "standard"), report_profile(configured(), "deep")
    with settings_for_run(standard) as yielded:
        assert yielded is standard and get_settings() is standard
        with pytest.raises(RuntimeError, match="inner failure"), settings_for_run(deep):
            assert get_settings() is deep
            raise RuntimeError("inner failure")
        assert get_settings() is standard
    assert get_settings() is original
    with pytest.raises(RuntimeError, match="outer failure"), settings_for_run(deep):
        raise RuntimeError("outer failure")
    assert get_settings() is original


def test_cache_clear_reloads_base_but_keeps_active_job_snapshot(monkeypatch):
    original = get_settings()
    snapshot = report_profile(configured(llm_model="snapshot-model"), "standard")
    with settings_for_run(snapshot):
        monkeypatch.setenv("CE_LLM_MODEL", "replacement-base-model")
        get_settings.cache_clear()
        assert get_settings() is snapshot and get_settings().llm_model == "snapshot-model"
    reloaded = get_settings()
    assert reloaded is not original and reloaded.llm_model == "replacement-base-model"


async def test_overlapping_jobs_children_and_threads_keep_their_own_profiles():
    original = get_settings()
    standard, deep = report_profile(configured(), "standard"), report_profile(configured(), "deep")
    both_entered, release = asyncio.Event(), asyncio.Event()
    entered = []

    async def child(expected):
        assert get_settings() is expected
        await asyncio.sleep(0)
        assert await asyncio.to_thread(get_settings) is expected
        return get_settings().generation_depth

    async def worker(snapshot):
        with settings_for_run(snapshot):
            entered.append(snapshot.generation_depth)
            if len(entered) == 2:
                both_entered.set()
            await release.wait()
            assert get_settings() is snapshot
            result = await asyncio.create_task(child(snapshot))
            assert get_settings() is snapshot
        assert get_settings() is original
        return result

    tasks = [asyncio.create_task(worker(standard)), asyncio.create_task(worker(deep))]
    await asyncio.wait_for(both_entered.wait(), timeout=2)
    assert get_settings() is original
    release.set()
    assert await asyncio.gather(*tasks) == ["standard", "deep"]
    assert get_settings() is original
    assert await asyncio.to_thread(get_settings) is original


async def test_child_task_retains_its_snapshot_after_parent_context_exits():
    original = get_settings()
    snapshot = report_profile(configured(), "deep")
    release = asyncio.Event()

    async def child():
        await release.wait()
        assert get_settings() is snapshot
        return await asyncio.to_thread(get_settings)

    with settings_for_run(snapshot):
        task = asyncio.create_task(child())
    assert get_settings() is original
    release.set()
    assert await task is snapshot
    assert get_settings() is original


async def test_settings_save_in_other_task_does_not_replace_running_snapshot(monkeypatch):
    original = get_settings()
    snapshot = report_profile(configured(llm_model="running-model"), "standard")
    started, release = asyncio.Event(), asyncio.Event()

    async def running_job():
        with settings_for_run(snapshot):
            started.set()
            await release.wait()
            assert get_settings() is snapshot
            assert await asyncio.to_thread(get_settings) is snapshot
        return get_settings()

    task = asyncio.create_task(running_job())
    await asyncio.wait_for(started.wait(), timeout=2)
    monkeypatch.setenv("CE_LLM_MODEL", "new-operator-model")
    get_settings.cache_clear()
    refreshed = get_settings()
    assert refreshed is not original and refreshed.llm_model == "new-operator-model"
    release.set()
    assert await task is refreshed
    assert get_settings() is refreshed


async def test_async_cancellation_restores_settings_outside_the_context():
    original = get_settings()
    snapshot = report_profile(configured(), "deep")
    started = asyncio.Event()
    restored = []

    async def worker():
        try:
            with settings_for_run(snapshot):
                started.set()
                await asyncio.Event().wait()
        finally:
            restored.append(get_settings())

    task = asyncio.create_task(worker())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert restored == [original]
    assert get_settings() is original
