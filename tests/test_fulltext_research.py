"""Regression for real failure modes: abstract starvation and unsearched gaps."""
from __future__ import annotations

import json

import pytest

from core_engine.config import get_settings
from core_engine.report.discovery import discovery_query
from core_engine.report.fulltext import article_metadata, fulltext_candidates, jats_document
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import FakeLLM
from core_engine.report.models import Claim, Source, SourceKind
from core_engine.report.passages import extraction_windows
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet
from core_engine.report.research import ResearchPlan, ResearchQuestion, ResearchSession
from core_engine.report.scholarly import FetchedDocument
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider, gather_sources

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def paper(text="Solid-state batteries use solid electrolytes.", **kwargs):
    return Source("https://api.openalex.org/works/W1234", "api.openalex.org", "Solid-state batteries",
                  SourceKind.ACADEMIC, text, **kwargs)


def test_gap_queries_retain_question_specific_requirements():
    questions = ["固态电池硫化物与氧化物路线的离子电导率和制造工艺？", "固态电池压力与枝晶的相互影响？",
                 "固态电池安全性和循环寿命的可比实验？"]
    queries = [discovery_query(q, "experimental conditions", 0)[1] for q in questions]
    assert len(set(queries)) == 3
    assert "conductivity" in queries[0] and "pressure" in queries[1] and "safety" in queries[2]


def test_title_only_topic_does_not_replace_explicit_user_evidence_requirements():
    from core_engine.report.research import evidence_requirement

    question = ResearchQuestion("q1", "固态电池", required=True, evidence_needed="需要完整电芯的温度、压力和测试方法原文。")
    plan = ResearchPlan("固态电池", [question])
    assert evidence_requirement(plan, question, "zh") == question.evidence_needed


def test_gap_scheduler_reaches_all_six_questions_before_repeating():
    plan = ResearchPlan("固态电池", [ResearchQuestion(f"q{i}", f"固态电池问题{i}") for i in range(1, 7)])
    session = ResearchSession(plan, FakeLLM(), get_settings())
    for row in session.audit["questions"]:
        row["missing"] = [f"specific missing requirement {row['id']}", f"second missing condition {row['id']}"]
    batches = [session.next_queries(2) for _ in range(3)]
    assert [q["question_id"] for batch in batches for q in batch] == [f"q{i}" for i in range(1, 7)]


async def test_per_question_context_and_missing_requirements_drive_search():
    class Assessor(FakeLLM):
        def __init__(self):
            super().__init__()
            self.inputs = []

        async def complete(self, system, user):
            payload = json.loads(user)
            self.inputs.append(payload)
            q = payload["questions"][0]
            return json.dumps({"assessments": [{"id": q["id"], "status": "partial", "claim_ids": ["c1"],
                "missing": ["stack pressure under matched full-cell cycling"],
                "search_queries": ["solid-state battery stack pressure full-cell cycling protocol"]}]})

    llm = Assessor()
    src = paper()
    plan = ResearchPlan("固态电池", [ResearchQuestion("q1", "固态电池压力管理的测试条件？", evidence_needed="原始压力条件")])
    session = ResearchSession(plan, llm, get_settings())
    await session.evaluate([Claim("c1", src.text, candidate_source_urls=[src.url], verified=True)], [src])
    row = session.audit["questions"][0]
    assert row["missing"] == ["stack pressure under matched full-cell cycling"]
    assert llm.inputs[0]["questions"][0]["evidence_needed"] == "原始压力条件"
    query = session.next_queries(1)[0]
    assert query["strategy"] == "evidence_gap" and "stack pressure" in query["query"]


async def test_abstract_cannot_close_explicit_full_text_requirement():
    src = paper(document_format="abstract", metadata={"full_text_accessed": False})
    llm = FakeLLM()
    llm.set_complete("Research coverage assessor", '{"assessments":[{"id":"q1","status":"supported","claim_ids":["c1"],"missing":[]}]}')
    session = ResearchSession(ResearchPlan("固态电池", [ResearchQuestion("q1", "固态电池的测试？", evidence_needed="需要全文中的完整电芯数据")]), llm, get_settings())
    await session.evaluate([Claim("c1", src.text, candidate_source_urls=[src.url], verified=True)], [src])
    assert session.audit["questions"][0]["status"] == "partial"
    assert "full_text_methods_or_full_cell_data" in session.audit["questions"][0]["missing"]


def test_fulltext_candidates_use_open_copies_preserve_versions_and_reject_secret_urls():
    src = paper(document_format="abstract", metadata={"openalex_id": "W1234", "doi": "10.1234/battery",
        "locations": [{"is_oa": True, "pdf_url": "https://repo.edu/paper.pdf", "version": "acceptedVersion", "license": "cc-by"},
                      {"is_oa": False, "pdf_url": "https://publisher.com/paywall.pdf"},
                      {"is_oa": True, "pdf_url": "https://repo.edu/private.pdf?token=secret"}]})
    leads = fulltext_candidates([src], 3)
    assert len(leads) == 2 and leads[0].metadata["discovery_doi"] == "10.1234/battery"
    assert leads[0].metadata["discovery_location"]["version"] == "acceptedVersion"
    assert leads[1].metadata["fulltext_resolver"] == "europe_pmc"
    assert all("secret" not in lead.url and "paywall" not in lead.url for lead in leads)


@pytest.mark.parametrize("bad", ["<article><abstract>Only abstract</abstract></article>",
    '<!DOCTYPE article [<!ENTITY x "bomb">]><article><body><p>&x;</p></body></article>', "<html>not XML article</html>"])
def test_incomplete_or_unsafe_xml_is_not_fulltext(bad):
    assert jats_document(bad) is None


def test_jats_methods_and_table_body_are_accessed_without_bibliographic_results():
    raw = '<article><front><article-meta><article-id pub-id-type="doi">10.1234/battery</article-id></article-meta></front><body><sec><title>Methods</title><p>' + "Solid-state battery stack pressure was measured. " * 20 + '</p></sec></body><back><ref-list><p>Unrelated cited result</p></ref-list></back></article>'
    doc = jats_document(raw)
    assert doc and doc.document_format == "jats" and doc.metadata["full_text_accessed"]
    assert "Methods" in doc.text and "Unrelated cited result" not in doc.text


def test_long_document_windows_reach_later_methods_and_limitations():
    text = "Introduction. Background about solid-state batteries.\n" * 300 + "\nMethods\nThe stack pressure and cycling protocol constrain the energy density.\n" * 30
    windows = extraction_windows(text, ["固态电池的压力、能量密度与循环寿命测试条件"], 3000, 3)
    assert len(windows) == 3 and any("stack pressure" in text[a:b] for a, b in windows)
    assert all(0 <= a < b <= len(text) and b - a <= 3000 for a, b in windows)


def test_bibliography_does_not_supply_experimental_results():
    text = "Battery background and introduction.\n" * 100 + "\nReferences\n" + "Stack pressure cycling energy density results in cited titles.\n" * 100
    windows = extraction_windows(text, ["固态电池压力循环能量密度"], 1000, 3)
    assert all("cited titles" not in text[a:b] for a, b in windows)


def test_one_readable_fulltext_does_not_trigger_duplicate_copy_downloads():
    src = paper(document_format="abstract", metadata={"locations": [
        {"is_oa": True, "pdf_url": "https://repo.edu/paper.pdf"},
        {"is_oa": True, "pdf_url": "https://other.edu/copy.pdf"}]})
    copy = Source("https://repo.edu/paper.pdf", "repo.edu", src.title, SourceKind.ACADEMIC, "body",
        document_format="pdf", metadata={"full_text_accessed": True, "fulltext_parent_url": src.url})
    assert fulltext_candidates([src, copy], 3) == []


async def test_open_landing_page_follows_declared_pdf_and_preserves_actual_url():
    src = paper(document_format="abstract", metadata={"openalex_id": "W1234", "locations": [
        {"is_oa": True, "landing_page_url": "https://repo.edu/record/123", "version": "acceptedVersion"}]})
    leads = fulltext_candidates([src], 1)

    class Provider(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            return leads

    pdf_url = "https://repo.edu/bitstream/paper.pdf"
    fetcher = FakeFetcher({leads[0].url: FetchedDocument("Only the landing page", "html", metadata={"article_pdf_url": pdf_url}),
        pdf_url: FetchedDocument("Solid-state batteries use solid electrolytes. " * 30, "pdf", metadata={"full_text_accessed": True})})
    sources, _ = await gather_sources("固态电池", Provider(), fetcher, queries=["固态电池"], settings=get_settings())
    assert len(sources) == 1 and sources[0].url == pdf_url
    assert sources[0].metadata["retrieval_entry_url"] == leads[0].url
    assert sources[0].metadata["fulltext_parent_url"] == src.url


async def test_fulltext_with_conflicting_doi_is_rejected():
    src = paper(document_format="abstract", metadata={"doi": "10.1234/battery", "locations": [
        {"is_oa": True, "pdf_url": "https://repo.edu/paper.pdf"}]})
    leads = fulltext_candidates([src], 1)

    class Provider(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            return leads

    fetcher = FakeFetcher({leads[0].url: FetchedDocument("Solid-state batteries use solid electrolytes. " * 30,
        "pdf", metadata={"doi": "10.1234/other", "full_text_accessed": True})})
    sources, rejected = await gather_sources("固态电池", Provider(), fetcher, queries=["固态电池"], settings=get_settings())
    assert not sources and rejected[0].metadata["rejection_reason"] == "fulltext_identity_mismatch"


async def test_fulltext_outer_timeout_exports_actual_budget_and_failure_reason(monkeypatch, caplog):
    import asyncio

    from core_engine.report.models import SearchHit

    hit = SearchHit("https://nature.com/paper.pdf", "固态电池全文", "", "nature.com",
                    metadata={"require_fulltext": True})

    class Provider(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            return [hit]

    actual_wait = asyncio.wait_for
    budgets = []

    async def timeout_read(awaitable, timeout):
        if getattr(getattr(awaitable, "cr_code", None), "co_name", "") == "read_candidate":
            budgets.append(timeout)
            awaitable.close()
            raise TimeoutError("simulated outer deadline")
        return await actual_wait(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", timeout_read)
    settings = get_settings().model_copy(update={"research_fulltext_timeout_s": 17, "scrape_per_url_timeout_s": 4})
    sources, rejected = await gather_sources("固态电池", Provider(), FakeFetcher({}), queries=["固态电池"], settings=settings)
    assert not sources and rejected[0].metadata["rejection_reason"] == "timeout"
    assert budgets == [22] and "exceeded 22s" in caplog.text


@pytest.mark.parametrize("body,retrieved", [("Solid-state batteries use solid electrolytes. ", True),
                                           ("Unrelated neuroscience introduction. ", False)])
async def test_long_same_doi_fulltext_requires_body_subject_match_not_keyword_density(body, retrieved):
    src = paper(document_format="abstract", metadata={"doi": "10.1234/battery", "locations": [
        {"is_oa": True, "pdf_url": "https://unlisted.example/paper.pdf"}]})
    leads = fulltext_candidates([src], 1)

    class Provider(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            return leads

    fetcher = FakeFetcher({leads[0].url: FetchedDocument(body + "Details of specialized measurements and procedures. " * 2000,
        "pdf", metadata={"doi": "10.1234/battery", "full_text_accessed": True})})
    sources, _ = await gather_sources("固态电池", Provider(), fetcher, queries=["固态电池"], settings=get_settings())
    assert bool(sources) is retrieved


@pytest.mark.parametrize("related_body", [True, False])
async def test_direct_fulltext_does_not_need_an_openalex_parent_to_survive_density_filter(related_body):
    from core_engine.report.models import SearchHit

    url = "https://frontiersin.org/paper"

    class Provider(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            return [SearchHit(url, "Scientific paper", "", "frontiersin.org")]

    body = ("Solid-state batteries use sulfide electrolytes. " if related_body else "Neuroscience imaging methods. ")
    body += "Detailed measurement methods and procedures with calibration. " * 500
    document = FetchedDocument(body, "html", metadata={"doi": "10.3389/example", "full_text_accessed": True})
    sources, rejected = await gather_sources("固态电池", Provider(), FakeFetcher({url: document}), queries=["固态电池"],
        settings=get_settings().model_copy(update={"relevance_min_score": .5, "relevance_min_topic_hits": 10}))
    assert bool(sources) == related_body
    if not related_body:
        assert rejected[0].metadata["rejection_reason"] == "topic_mismatch"


def test_html_pdf_link_metadata_is_only_a_discovery_lead():
    data = article_metadata('<meta name="citation_doi" content="10.1234/battery"><meta name="citation_pdf_url" content="/paper.pdf">', "https://repo.edu/record")
    assert data["article_pdf_url"] == "https://repo.edu/paper.pdf"
    assert "full_text_accessed" not in data
    repository = article_metadata('<meta name="citation_pdf_url" content="/assets/logo.png"><a href="/bitstreams/123/download">Paper.pdf</a>', "https://repo.edu/record")
    assert repository["article_pdf_url"] == "https://repo.edu/bitstreams/123/download"


async def test_document_only_run_never_downloads_open_fulltext(tmp_path):
    src = paper("Solid-state batteries are electrochemical cells using solid electrolytes.",
                document_format="abstract", metadata={"locations": [{"is_oa": True, "pdf_url": "https://repo.edu/paper.pdf"}]})

    class NoNetwork(FakeFetcher):
        async def fetch(self, url):
            pytest.fail("Document-only mode fetched a full-text link")

    pipeline = ReportPipeline(llm=FakeLLM(), search=FakeSearchProvider(), fetcher=NoNetwork())
    pipeline._s = get_settings().model_copy(update={"min_total_claims": 1, "min_authoritative_sources": 1,
        "output_dir": tmp_path, "enable_charts": False, "enable_deep_dive": False,
        "verify_strategy": "cross_reference", "verify_mode": "strict"})
    result = await pipeline.run("固态电池", skip_search=True, extra_sources=[src], compile_to_pdf=False)
    assert result.report and not result.research_audit.get("fulltext_attempts")
    assert not check_packet(evidence_packet(result.report))["integrity_errors"]


@pytest.mark.parametrize("failure", [None, "timeout", "http_403"])
async def test_research_follows_open_copy_and_closes_a_source_bound_gap(tmp_path, failure):
    src = paper("Solid-state batteries are electrochemical cells using solid electrolytes.", document_format="abstract",
        metadata={"openalex_id": "W1234", "locations": [{"is_oa": True, "pdf_url": "https://repo.edu/paper.pdf"}]})
    body = "Solid-state batteries are tested with a stack pressure of 5 MPa at 25 °C."

    class Analyst(FakeLLM):
        async def complete(self, system, user):
            if "Research coverage assessor" in system:
                payload = json.loads(user)
                witnesses = [c["id"] for c in payload["claims"] if "stack pressure" in c["text"]]
                return json.dumps({"assessments": [{"id": "q1", "status": "supported" if witnesses else "partial",
                    "claim_ids": witnesses or ["c1"], "missing": [] if witnesses else ["full text pressure conditions"]}]})
            return await super().complete(system, user)

    class RecoveringFetcher(FakeFetcher):
        def __init__(self):
            super().__init__({"https://repo.edu/paper.pdf": FetchedDocument(body * 10, "pdf", metadata={"full_text_accessed": True})})
            self.failures = {}
            self.attempts = 0

        async def fetch_fulltext(self, url):
            self.attempts += 1
            if failure and self.attempts == 1:
                self.failures[url] = failure
                return None
            self.failures.pop(url, None)
            return await self.fetch(url)

    fetcher = RecoveringFetcher()
    pipeline = ReportPipeline(llm=Analyst(), search=FakeSearchProvider(), fetcher=fetcher)
    pipeline._s = get_settings().model_copy(update={"min_authoritative_sources": 1, "min_total_claims": 1,
        "verify_strategy": "cross_reference", "verify_mode": "strict", "output_dir": tmp_path,
        "enable_charts": False, "enable_deep_dive": False, "active_verify_enabled": False,
        "relevance_min_body_chars": 1, "relevance_min_score": 0, "relevance_min_topic_hits": 0})
    plan = ResearchPlan("固态电池", [ResearchQuestion("q1", "固态电池", required=True, evidence_needed="全文压力测试条件")])
    result = await pipeline.run("固态电池", extra_sources=[src], compile_to_pdf=False, research_plan=plan)
    if failure == "http_403":
        assert result.report and result.research_audit["questions"][0]["status"] != "supported"
        assert fetcher.attempts == 1
        return
    assert result.report and result.research_audit["questions"][0]["status"] == "supported"
    assert result.research_audit["fulltext_attempts"][-1]["retrieved_url"] == "https://repo.edu/paper.pdf"
    assert fetcher.attempts == (2 if failure else 1)
    assert len(set(result.research_audit["evidence_quality"]["groups"].values())) == 1
    assert not check_packet(evidence_packet(result.report))["integrity_errors"]


async def test_fetched_originals_survive_a_later_verification_deadline(tmp_path):
    from core_engine.report.models import PipelineResult, PipelineStatus, SearchHit
    from core_engine.report.verify import VerificationReport

    src = paper("Solid-state batteries use solid electrolytes.")
    new_url = "https://nature.com/new-paper"

    class Search(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            return [SearchHit(new_url, "Solid-state battery methods", "", "nature.com")]

    class Pipeline(ReportPipeline):
        async def _extract_and_verify(self, *args, **kwargs):
            raise TimeoutError("simulated verification deadline after successful download")

    pipeline = Pipeline(llm=FakeLLM(), search=Search(), fetcher=FakeFetcher({
        new_url: "Solid-state batteries use ceramic electrolytes. " * 30}))
    pipeline._s = get_settings().model_copy(update={"research_fulltext_enabled": False,
        "research_max_rounds": 1, "research_max_queries": 1, "relevance_filter": False})
    plan = ResearchPlan("固态电池", [ResearchQuestion("q1", "固态电池", required=True, evidence_needed="需要压力测试方法。")])
    session = ResearchSession(plan, FakeLLM(), pipeline._s)
    verification = VerificationReport(kept_claims=[Claim("c1", src.text, candidate_source_urls=[src.url])])
    sources, returned = await pipeline._fill_evidence_gaps(session, [src], verification,
        PipelineResult(PipelineStatus.ERROR, "固态电池"), lambda *args: None, skip_search=False)
    assert {s.url for s in sources} == {src.url, new_url}
    assert returned is verification and session.audit["stop_reason"] == "deadline"
    assert {c.id for c in session.claims} == {"c1"}  # downloaded is not verified


async def test_resume_research_skips_broad_queries_but_searches_evidence_gaps(tmp_path):
    class Search(FakeSearchProvider):
        def __init__(self):
            super().__init__()
            self.queries = []

        async def search(self, topic, *, max_results):
            self.queries.append(topic)
            return []

    src = paper("Solid-state batteries use solid electrolytes.")
    search = Search()
    pipeline = ReportPipeline(llm=FakeLLM(), search=search, fetcher=FakeFetcher())
    pipeline._s = get_settings().model_copy(update={"output_dir": tmp_path, "min_authoritative_sources": 1,
        "min_total_claims": 1, "research_max_queries": 1, "research_max_rounds": 1,
        "research_fulltext_enabled": False, "research_local_max_chunks": 0,
        "verify_strategy": "cross_reference", "verify_mode": "strict", "enable_charts": False,
        "enable_deep_dive": False, "reasoning_enabled": False})
    plan = ResearchPlan("固态电池", [ResearchQuestion("q1", "固态电池", required=True, evidence_needed="压力实验原文")])
    result = await pipeline.run("固态电池", extra_sources=[src], skip_initial_search=True,
        seed_claims=[Claim("c1", src.text, candidate_source_urls=[src.url])], research_plan=plan, compile_to_pdf=False,
        retained_leads=[{"text": "固态电池已达到9999 Wh/kg。", "candidate_source_urls": [src.url]}])
    assert result.report and result.research_audit["search_policy"] == "web_allowed"
    assert result.research_audit["initial_queries"] == []
    assert search.queries and all(query != "固态电池" for query in search.queries)
    assert result.report.leads[0]["text"] == "固态电池已达到9999 Wh/kg。"
    assert not check_packet(evidence_packet(result.report))["integrity_errors"]


async def test_new_fulltext_is_not_starved_by_global_extraction_budget():
    from core_engine.report.llm import ExtractedClaim

    class Extractor(FakeLLM):
        async def extract_claims(self, url, text):
            return [ExtractedClaim(f"Observation {i} in {url}", [url]) for i in range(5)]

    sources = [Source(f"https://example.edu/{i}", "example.edu", "Paper", SourceKind.ACADEMIC, "Body") for i in range(4)]
    pipeline = ReportPipeline(llm=Extractor(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    pipeline._s = get_settings().model_copy(update={"extraction_max_claims": 4})
    claims = await pipeline._extract_claims(sources)
    assert len(claims) == 4 and {c.candidate_source_urls[0] for c in claims} == {s.url for s in sources}


async def test_coverage_receives_exact_source_quotes_and_source_format():
    class Quoted(FakeLLM):
        async def complete(self, system, user):
            row = json.loads(user)["claims"][0]
            assert row["source_quotes"][0]["quote"] == src.text
            assert row["source_quotes"][0]["format"] == "abstract"
            assert not row["source_quotes"][0]["full_text_accessed"]
            return '{"assessments":[]}'

    src = paper(document_format="abstract")
    session = ResearchSession(ResearchPlan("固态电池", [ResearchQuestion("q1", "固态电池", required=True)]), Quoted(), get_settings())
    await session.evaluate([Claim("c1", src.text, candidate_source_urls=[src.url], verified=True)], [src])
    assert "assessment_error" not in session.audit["questions"][0]


async def test_title_overview_does_not_invent_a_numeric_requirement():
    from core_engine.report.research import build_plan

    settings = get_settings().model_copy(update={"report_locale": "zh", "research_max_questions": 1})
    plan = await build_plan("固态电池", FakeLLM(), settings)
    assert "定义" in plan.questions[0].evidence_needed and "观测值" not in plan.questions[0].evidence_needed
    metric_plan = await build_plan("Acme's net profit in 2024", FakeLLM(), settings)
    assert "观测值" in metric_plan.questions[0].evidence_needed


def test_specific_relevant_paper_passages_survive_many_irrelevant_origins():
    from core_engine.report.research import coverage_rows

    claims = [Claim(f"c{i}", "Sulfide oxide conductivity pressure sintering interface manufacturing challenges.",
                    candidate_source_urls=["paper"]) for i in range(3)]
    claims += [Claim(f"c{i}", "Global battery market news.", candidate_source_urls=[str(i)]) for i in range(3, 20)]
    selected = coverage_rows(claims, {}, 1300, focus="固态电池硫化物氧化物电导率压力烧结制造挑战")
    assert [c["id"] for c in selected[:3]] == ["c0", "c1", "c2"]


async def test_real_extractor_treats_research_requirements_as_goals_not_source():
    from core_engine.report.llm import OpenAILLM

    llm = OpenAILLM("test-key", "example-model", 4096, "https://example.org/v1")
    calls = []

    async def complete(system, user):
        calls.append((system, json.loads(user)))
        return "[]"

    llm._complete = complete
    assert await llm.extract_research_claims("https://repo.edu/paper.pdf", "Original article passage.", ["Pressure conditions?"]) == []
    assert calls[0][1]["SOURCE"] == "Original article passage."
    assert calls[0][1]["research_questions"] == ["Pressure conditions?"]
    assert "NOT evidence" in calls[0][0]


async def test_coverage_has_its_own_timeout_instead_of_the_short_planner_budget():
    import asyncio

    class SlowAssessor(FakeLLM):
        async def complete(self, system, user):
            await asyncio.sleep(.03)
            return '{"assessments":[{"id":"q1","status":"partial","claim_ids":["c1"],"missing":["experimental conditions"]}]}'

    settings = get_settings().model_copy(update={"research_plan_timeout_s": .001, "research_coverage_timeout_s": .5})
    src = paper()
    session = ResearchSession(ResearchPlan("固态电池", [ResearchQuestion("q1", "固态电池有哪些实验条件？")]), SlowAssessor(), settings)
    await session.evaluate([Claim("c1", src.text, candidate_source_urls=[src.url], verified=True)], [src])
    assert session.audit["questions"][0]["method"] == "model_coverage_assessment"
    assert "assessment_error" not in session.audit["questions"][0]


def test_fulltext_budget_keeps_a_late_distinct_limitation_instead_of_repeated_background():
    from core_engine.report.llm import ExtractedClaim
    from core_engine.report.passages import select_research_claims

    claims = [ExtractedClaim(f"Solid electrolyte conductivity observation {i}.", ["paper"]) for i in range(20)]
    limitation = ExtractedClaim("Oxide electrolytes require energy-intensive sintering, increasing manufacturing cost.", ["paper"])
    claims.append(limitation)
    selected = select_research_claims(claims, ["固态电池电导率与氧化物烧结制造挑战"], 8)
    assert len(selected) == 8 and limitation in selected


async def test_translated_late_fact_uses_original_window_for_verification_and_citation():
    from core_engine.report.grounding import SentenceGrounder
    from core_engine.report.llm import EntailmentJudgement
    from core_engine.report.verify import VerificationHarness

    quote = "Oxide solid-state battery electrolytes require energy-intensive sintering, increasing manufacturing cost."
    prefix = "General introductory background without this experimental observation.\n\n" * 400
    src = paper(prefix + quote + "\n\nEnd of the discussion.", document_format="pdf", metadata={"full_text_accessed": True})
    claim = Claim("c1", "固态电池的氧化物电解质需要高能耗烧结，从而提高制造成本。", candidate_source_urls=[src.url],
                  source_windows={src.url: [[len(prefix), len(prefix) + len(quote)]]})

    class Oracle(FakeLLM):
        async def check_entailment(self, text, source_text):
            return EntailmentJudgement(quote in source_text, quote if quote in source_text else "")

    settings = get_settings().model_copy(update={"verify_mode": "strict", "llm_context_chars": 1000, "verify_passage_max_chars": 200})
    harness = VerificationHarness(Oracle())
    harness._s = settings
    assert not (await harness._check_source(claim.text, src, 0))[0]
    assert (await harness._check_source(claim.text, src, 0, claim=claim))[0]
    anchored, citations = await SentenceGrounder(Oracle(), settings).anchor_claims([claim], [src])
    assert anchored and citations[0].start_char == len(prefix)
    assert src.text[citations[0].start_char:citations[0].end_char] == quote
    assert citations[0].source_sha256 == src.content_sha256
    # A location hint is not evidence for a number absent from the quotation.
    forged = Claim("c2", claim.text + "成本提高99%。", candidate_source_urls=[src.url], source_windows=claim.source_windows)
    assert not (await SentenceGrounder(Oracle(), settings).anchor_claims([forged], [src]))[0]


async def test_extraction_records_actual_window_and_narrows_a_valid_quote():
    from core_engine.report.llm import ExtractedClaim

    quote = "Oxide electrolytes need energy-intensive sintering during manufacturing."
    text = "General introductory background.\n" * 400 + quote + "\nDiscussion."

    class Extractor(FakeLLM):
        async def extract_claims(self, url, excerpt):
            return [ExtractedClaim("氧化物电解质的烧结制造消耗较多能源。", [url], source_quote=quote)] if quote in excerpt else []

    src = paper(text, document_format="pdf", metadata={"full_text_accessed": True})
    pipeline = ReportPipeline(llm=Extractor(), search=FakeSearchProvider(), fetcher=FakeFetcher())
    pipeline._s = get_settings().model_copy(update={"llm_context_chars": 1000})
    pipeline._research_questions = ["固态电池氧化物电解质烧结制造挑战"]
    claims = await pipeline._extract_claims([src])
    start, end = claims[0].source_windows[src.url][0]
    assert start > 0 and quote in text[start:end] and end - start <= 1000


async def test_nonverbatim_model_quote_is_ignored_and_window_hints_remain_internal():
    from core_engine.report.llm import OpenAILLM

    llm = OpenAILLM("test-key", "example-model", 4096, "https://example.org/v1")

    async def complete(system, user):
        return '[{"text":"Observation.","quote":"Invented source quotation","source_windows":{"fake":[[0,10]]}}]'

    llm._complete = complete
    claims = await llm.extract_research_claims("https://repo.edu/paper.pdf", "Actual original passage.", ["Source conditions?"])
    assert claims[0].source_quote is None and not claims[0].source_windows


async def test_small_gather_budget_retains_a_candidate_from_each_query():
    from core_engine.report.models import SearchHit

    class Provider(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            return [SearchHit(f"https://example.edu/{topic}{i}", "Solid-state batteries", "", "example.edu") for i in range(2)]

    fetcher = FakeFetcher({f"https://example.edu/{q}{i}": "Solid-state batteries are tested under stated laboratory conditions. " * 10
                           for q in ("a", "b") for i in range(2)})
    sources, _ = await gather_sources("固态电池", Provider(), fetcher, queries=["a", "b"], max_pages=2,
        settings=get_settings().model_copy(update={"max_pages_per_query": 2}))
    assert {s.url for s in sources} == {"https://example.edu/a0", "https://example.edu/b0"}


async def test_source_budget_reserves_first_search_opportunities_for_all_questions(tmp_path):
    import hashlib

    from core_engine.report.models import SearchHit

    class Provider(FakeSearchProvider):
        async def search(self, topic, *, max_results):
            if topic == "固态电池":
                return []
            key = hashlib.sha256(topic.encode()).hexdigest()[:8]
            return [SearchHit(f"https://example.edu/{key}/{i}", "Solid-state batteries", "", "example.edu") for i in range(4)]

    class Fetcher(FakeFetcher):
        async def fetch(self, url):
            return "Solid-state batteries are tested at 25 °C."

    questions = [ResearchQuestion("q1", "固态电池", required=True)] + [
        ResearchQuestion(f"q{i}", f"固态电池的{term}？", evidence_needed=term)
        for i, term in enumerate(("压力", "离子电导率", "制造", "湿度", "烧结"), 2)]
    pipeline = ReportPipeline(llm=FakeLLM(), search=Provider(), fetcher=Fetcher())
    pipeline._s = get_settings().model_copy(update={"output_dir": tmp_path, "research_max_rounds": 3,
        "research_max_queries": 6, "scrape_max_pages": 7, "max_pages_per_query": 4,
        "min_authoritative_sources": 1, "min_total_claims": 1, "verify_strategy": "cross_reference",
        "verify_mode": "strict", "enable_charts": False, "enable_deep_dive": False,
        "reasoning_enabled": False, "research_fulltext_enabled": False,
        "relevance_min_body_chars": 1, "relevance_min_score": 0, "relevance_min_topic_hits": 0})
    src = paper("Solid-state batteries are tested at 25 °C.")
    result = await pipeline.run("固态电池", compile_to_pdf=False, extra_sources=[src],
                                research_plan=ResearchPlan("固态电池", questions))
    assert result.report and len(result.report.sources) == 7
    assert {r["question_id"] for r in result.research_audit["searches"]} == {q.id for q in questions}
    assert result.research_audit["queries_used"] == 6


@pytest.mark.parametrize("invent_number", [False, True])
@pytest.mark.parametrize("document_only", [False, True])
async def test_local_gap_read_recovers_late_atomic_evidence_without_web_or_false_numbers(tmp_path, invent_number, document_only):
    from core_engine.report.llm import EntailmentJudgement, ExtractedClaim

    definition = "Solid-state batteries are electrochemical cells using solid electrolytes."
    quote = "Oxide solid-state battery electrolytes require energy-intensive sintering, increasing manufacturing cost."
    prefix = definition + "\n\n" + "General introduction about the topic.\n" * 300
    src = paper(prefix + quote, document_format="pdf", metadata={"full_text_accessed": True})

    class Analyst(FakeLLM):
        async def extract_research_claims(self, url, excerpt, goals, *, max_claims=None):
            text = "氧化物固态电池电解质的烧结消耗较多能源，增加制造成本。" + ("成本增加99%。" if invent_number else "")
            return [ExtractedClaim(text, [url], source_quote=quote)] if quote in excerpt else []

        async def check_entailment(self, text, source_text):
            if "烧结" in text:
                return EntailmentJudgement(quote in source_text, quote if quote in source_text else "")
            return await super().check_entailment(text, source_text)

        async def complete(self, system, user):
            if "Research coverage assessor" in system:
                payload = json.loads(user)
                qid = payload["questions"][0]["id"]
                witnesses = [c["id"] for c in payload["claims"] if "烧结" in c["text"]]
                return json.dumps({"assessments": [{"id": qid, "status": "supported" if qid == "q1" or witnesses else "partial",
                    "claim_ids": ["c1"] if qid == "q1" else witnesses or ["c1"],
                    "missing": [] if qid == "q1" or witnesses else ["oxide electrolyte sintering manufacturing cost"]}]})
            return await super().complete(system, user)

    class NoWeb(FakeSearchProvider):
        async def search(self, *args, **kwargs):
            pytest.fail("Local deficit reading performed an external search")

    pipeline = ReportPipeline(llm=Analyst(), search=NoWeb(), fetcher=FakeFetcher())
    pipeline._s = get_settings().model_copy(update={"output_dir": tmp_path, "llm_context_chars": 1000,
        "min_authoritative_sources": 1, "min_total_claims": 1, "verify_strategy": "cross_reference",
        "verify_mode": "strict", "research_local_max_chunks": 2, "research_max_rounds": 2,
        "scrape_max_pages": 1,
        "enable_charts": False, "enable_deep_dive": False, "reasoning_enabled": False})
    plan = ResearchPlan("固态电池", [ResearchQuestion("q1", "固态电池", required=True),
        ResearchQuestion("q2", "固态电池氧化物电解质的烧结成本？", evidence_needed="原文中的烧结与制造成本关系")])
    seed = Claim("c1", definition, candidate_source_urls=[src.url])
    result = await pipeline.run("固态电池", compile_to_pdf=False, skip_search=document_only, extra_sources=[src],
                                seed_claims=[seed], research_plan=plan)
    expected_stop = "document_only" if document_only else "source_budget" if invent_number else "questions_covered"
    assert result.report and result.research_audit["stop_reason"] == expected_stop
    assert result.research_audit["local_reads"] and not result.research_audit["searches"]
    question = next(q for q in result.research_audit["questions"] if q["id"] == "q2")
    assert (question["status"] == "supported") is not invent_number
    assert len(result.research_audit["local_reads"]) <= 2
    assert not check_packet(evidence_packet(result.report))["integrity_errors"]


def test_long_pdf_quote_wrapping_preserves_characters_and_escapes_source_commands():
    from core_engine.report.latex import tex_escape, tex_quote

    source = "ConcatenatedPDFWord" * 8 + r"\input{private}" + " 25 mS/cm"
    rendered = tex_quote(source)
    assert rendered.replace(r"\allowbreak{}", "") == tex_escape(source)
    assert r"\input{" not in rendered
