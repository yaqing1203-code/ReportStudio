"""Tests for the report-generation pipeline.

These exercise the GATES with fakes — no network, no API key, no TeX toolchain
required. The point is to prove the pipeline HALTS correctly (source filter,
scope gate, verification gate) and only produces a report when every gate passes.

Run: pytest -q tests/test_report_pipeline.py
"""
from __future__ import annotations

import pytest

from core_engine.report.latex import render_tex, tex_escape
from core_engine.report.models import (
    OUT_OF_SCOPE_MESSAGE,
    Claim,
    PipelineStatus,
    ReportData,
    ReportSection,
    SearchHit,
    Source,
    SourceKind,
)
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.scrape import FakeFetcher, FakeSearchProvider
from core_engine.report.sources import classify, filter_hits
from core_engine.report.verify import VerificationHarness

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def cross_ref(monkeypatch):
    """Opt a test into the legacy 'cross_reference' verification strategy.

    The default is now 'conflict_only' (accept-all + conflict resolution). Tests that
    assert the broad-collection tiering/entailment behavior request this fixture so they
    exercise the legacy path explicitly."""
    from core_engine.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("CE_VERIFY_STRATEGY", "cross_reference")
    yield
    get_settings.cache_clear()


# --------------------------------------------------------------------------
# Source filter — the strict allowlist gate.
# --------------------------------------------------------------------------
def test_classify_tiers_curated_high_trust_and_general_web():
    from core_engine.report.models import SourceKind as SK
    # Academic databases classify as ACADEMIC (checked before the .gov suffix).
    assert classify("https://www.nature.com/articles/x") is SK.ACADEMIC
    assert classify("https://pubmed.ncbi.nlm.nih.gov/123") is SK.ACADEMIC
    assert classify("https://www.cdc.gov/flu/index.html") is SK.ACADEMIC  # in academic list
    # A plain government/IGO domain not in the academic list -> GOVERNMENT.
    assert classify("https://ec.europa.eu/info") is SK.GOVERNMENT
    assert classify("https://www.reuters.com/world/x") is SK.AUTHORITATIVE_MEDIA
    # Denylisted hosts (social / UGC platforms) are ALWAYS rejected.
    assert classify("https://someone.medium.com/post") is SK.REJECTED
    assert classify("https://www.reddit.com/r/news") is SK.REJECTED
    # Tiered trust: a non-denylisted, non-curated site is now GENERAL_WEB.
    assert classify("https://randomblog.example.com/") is SK.GENERAL_WEB
    assert classify("https://notagov.com/") is SK.GENERAL_WEB
    # Curated tiers (incl. academic) are high-trust; general web is not.
    assert classify("https://www.nature.com/x").high_trust is True
    assert classify("https://randomblog.example.com/").high_trust is False


def test_filter_hits_partitions_curated_and_general_but_denylists():
    hits = [
        SearchHit("https://cdc.gov/a", "A", "", "cdc.gov"),
        SearchHit("https://cdc.gov/a", "A dup", "", "cdc.gov"),  # exact dup
        SearchHit("https://someblog.net/x", "X", "", "someblog.net"),  # general web -> kept
        SearchHit("https://reddit.com/r/y", "Y", "", "reddit.com"),   # denylisted -> rejected
    ]
    kept, rejected = filter_hits(hits)
    assert [h.url for h in kept] == ["https://cdc.gov/a", "https://someblog.net/x"]
    assert [h.url for h in rejected] == ["https://reddit.com/r/y"]


def test_relevance_score_separates_deep_dive_from_superficial():
    from core_engine.report.scrape import relevance_score

    topic = "solid-state battery supply chain"
    deep = ("The solid-state battery supply chain spans upstream lithium and "
            "electrolyte suppliers, midstream cell manufacturing, and downstream "
            "battery pack integration. Solid-state battery makers depend on this "
            "supply chain for scale. ") * 5
    superficial = ("This article is mostly about unrelated consumer gadgets and "
                   "lifestyle trends. " * 40) + "It mentions a solid-state battery once."

    d_deep, hits_deep = relevance_score(topic, deep)
    d_sup, _hits_sup = relevance_score(topic, superficial)
    # The deep-dive article has much higher topic-term density than the name-drop.
    assert d_deep > d_sup
    assert hits_deep >= 3       # covers several distinct topic terms
    assert d_deep >= 0.04       # clears the default keep threshold


# --------------------------------------------------------------------------
# Verification harness — triple-check + cross-reference.
# --------------------------------------------------------------------------
async def test_harness_requires_cross_referenced_domains(cross_ref):
    from core_engine.report.llm import FakeLLM

    # One claim supported verbatim by TWO distinct gov domains -> verified.
    shared = "The program covered 4 million people in 2023."
    sources = [
        Source("https://cdc.gov/p", "cdc.gov", "CDC", SourceKind.GOVERNMENT,
               text=f"Background. {shared} More text."),
        Source("https://europa.eu/p", "europa.eu", "EU", SourceKind.GOVERNMENT,
               text=f"Intro. {shared} Additional detail."),
    ]
    claim = Claim(id="c1", text=shared,
                  candidate_source_urls=[s.url for s in sources])

    harness = VerificationHarness(FakeLLM())
    report = await harness.verify([claim], sources)
    assert report.verified_count == 1
    assert len(report.distinct_supporting_domains()) == 2


async def test_harness_tiers_single_source_general_web_as_rumor(cross_ref):
    """Broad-collection model: a single GENERAL_WEB source is KEPT as a RUMOR (not
    discarded); a single HIGH-TRUST source is verified as HIGH confidence."""
    from core_engine.report.llm import FakeLLM
    from core_engine.report.models import ConfidenceTier

    shared = "The tariff rate was set at 12 percent in 2024."
    # One general-web source only -> uncorroborated -> RUMOR, kept in the pool.
    gw = Source("https://someblog.net/x", "someblog.net", "Blog",
                SourceKind.GENERAL_WEB, text=f"Notice. {shared} End.")
    claim = Claim(id="c1", text=shared, candidate_source_urls=[gw.url])
    report = await VerificationHarness(FakeLLM()).verify([claim], [gw])
    assert report.verified_count == 0                 # not a verified fact
    assert len(report.rumor_claims) == 1              # but kept as a rumor
    assert report.rumor_claims[0].confidence is ConfidenceTier.RUMOR
    assert report.rejected_claims == []               # NOT discarded

    # A single HIGH-TRUST (gov) source is enough for HIGH confidence.
    gov = Source("https://cdc.gov/only", "cdc.gov", "CDC", SourceKind.GOVERNMENT,
                 text=f"Notice. {shared} End.")
    claim2 = Claim(id="c2", text=shared, candidate_source_urls=[gov.url])
    report2 = await VerificationHarness(FakeLLM()).verify([claim2], [gov])
    assert report2.verified_count == 1
    assert report2.verified_claims[0].confidence is ConfidenceTier.HIGH


async def test_harness_cross_references_against_whole_pool(cross_ref):
    """Whole-pool cross-referencing: a claim whose OWN source is only GENERAL_WEB (so the
    authority fast-path does NOT fire) is still corroborated by ANOTHER high-trust source
    in the pool that also states it — even though the claim only listed its own source.
    This is the fix for '0 verified out of N', and it must survive the fast-path being on
    because the fast-path only skips cross-referencing once a HIGH-TRUST source endorses."""
    from core_engine.report.llm import FakeLLM
    from core_engine.report.models import ConfidenceTier

    shared = "The program reached 4 million participants in 2023 across the country."
    # The claim's own source is general web (not high-trust) -> fast-path can't short it,
    # so the pool source in a distinct high-trust domain must be found by cross-reference.
    src_a = Source("https://someblog.net/a", "someblog.net", "Blog",
                   SourceKind.GENERAL_WEB, text=f"Report. {shared} More context here.")
    src_b = Source("https://iea.org/b", "iea.org", "IEA", SourceKind.INDUSTRY_INSTITUTION,
                   text=f"Analysis. {shared} Additional discussion.")
    claim = Claim(id="c1", text=shared, candidate_source_urls=["https://someblog.net/a"])
    report = await VerificationHarness(FakeLLM()).verify([claim], [src_a, src_b])
    assert report.verified_count == 1
    # Corroborated across 2 distinct domains via cross-reference -> HIGH.
    assert len(report.verified_claims[0].supporting_sources()) == 2
    assert report.verified_claims[0].confidence is ConfidenceTier.HIGH


async def test_authority_fast_path_short_circuits_cross_referencing(cross_ref, monkeypatch):
    """Requirement #3: a claim endorsed by its own HIGH-TRUST source is classified HIGH
    WITHOUT scanning the rest of the pool. We prove the short-circuit by counting LLM
    support checks: with the fast-path on, only the claim's own (high-trust) source is
    checked; the second pool source is never touched."""
    from core_engine.config import get_settings
    from core_engine.report.llm import TraceJudgement
    from core_engine.report.models import ConfidenceTier

    shared = "The subsidy budget was 8 billion dollars in the 2024 fiscal year period."

    # Default verify mode is 'traceable', so only check_traceability is exercised.
    class CountingLLM:
        def __init__(self):
            self.checks = 0
        async def check_traceability(self, claim, source_text):
            self.checks += 1
            return TraceJudgement(supported=True, hallucinated=False,
                                  evidence_span="q", confidence=1.0)

    get_settings.cache_clear()
    monkeypatch.setenv("CE_VERIFY_FAST_PATH_HIGH_TRUST", "true")
    try:
        gov = Source("https://cdc.gov/a", "cdc.gov", "CDC", SourceKind.GOVERNMENT,
                     text=f"Report. {shared} More.")
        pool = Source("https://iea.org/b", "iea.org", "IEA",
                      SourceKind.INDUSTRY_INSTITUTION, text=f"Also. {shared} End.")
        claim = Claim(id="c1", text=shared, candidate_source_urls=["https://cdc.gov/a"])
        llm = CountingLLM()
        report = await VerificationHarness(llm).verify([claim], [gov, pool])
    finally:
        get_settings.cache_clear()

    assert report.verified_count == 1
    assert report.verified_claims[0].confidence is ConfidenceTier.HIGH
    # Only the claim's own high-trust source was checked; the pool source was skipped.
    assert llm.checks == 1, f"fast-path did not short-circuit (checked {llm.checks})"


async def test_single_credible_source_is_likely_not_rumor(cross_ref):
    """A single non-high-trust but credible (media) source -> LIKELY (reportable),
    not RUMOR — the relaxed threshold."""
    from core_engine.report.llm import FakeLLM
    from core_engine.report.models import ConfidenceTier

    shared = "The sector added 12 thousand jobs during the last fiscal year period."
    media = Source("https://reuters.com/x", "reuters.com", "R",
                   SourceKind.AUTHORITATIVE_MEDIA, text=f"News. {shared} End.")
    claim = Claim(id="c1", text=shared, candidate_source_urls=[media.url])
    report = await VerificationHarness(FakeLLM()).verify([claim], [media])
    # media is high_trust -> single source is HIGH; verify it's reportable either way.
    assert report.verified_count == 1
    assert report.verified_claims[0].confidence in (
        ConfidenceTier.HIGH, ConfidenceTier.LIKELY)


async def test_harness_drops_only_hallucinations(cross_ref, monkeypatch):
    """A claim a source flags as hallucinated is DROPPED; an uncorroborated one is not."""
    from core_engine.config import get_settings
    from core_engine.report.llm import TraceJudgement

    get_settings.cache_clear()
    monkeypatch.setenv("CE_VERIFY_MODE", "traceable")

    class StubLLM:
        async def check_traceability(self, claim, source_text):
            if "fabricated" in claim:
                return TraceJudgement(supported=False, hallucinated=True)
            return TraceJudgement(supported=True, hallucinated=False,
                                  evidence_span="ok", confidence=0.9)

    good = Claim(id="c1", text="a real finding",
                 candidate_source_urls=["https://someblog.net/a"])
    bad = Claim(id="c2", text="a fabricated finding",
                candidate_source_urls=["https://someblog.net/b"])
    sources = [
        Source("https://someblog.net/a", "someblog.net", "A", SourceKind.GENERAL_WEB, text="x"),
        Source("https://someblog.net/b", "someblog.net", "B", SourceKind.GENERAL_WEB, text="x"),
    ]
    try:
        report = await VerificationHarness(StubLLM()).verify([good, bad], sources)
    finally:
        get_settings.cache_clear()
    kept_ids = {c.id for c in report.kept_claims}
    assert "c1" in kept_ids                # uncorroborated but kept (rumor)
    assert "c2" not in kept_ids            # hallucinated -> dropped
    assert "c2" in report.reasons


async def test_harness_rejects_unsupported_claim(cross_ref):
    from core_engine.report.llm import FakeLLM

    # Claim text appears in NO source -> no verbatim quote -> rejected.
    sources = [
        Source("https://cdc.gov/p", "cdc.gov", "CDC", SourceKind.GOVERNMENT,
               text="This document is about something else entirely."),
        Source("https://europa.eu/p", "europa.eu", "EU", SourceKind.GOVERNMENT,
               text="Unrelated content here."),
    ]
    claim = Claim(id="c1", text="The unemployment rate fell to 3 percent in 2022.",
                  candidate_source_urls=[s.url for s in sources])
    harness = VerificationHarness(FakeLLM())
    report = await harness.verify([claim], sources)
    assert report.verified_count == 0


# --------------------------------------------------------------------------
# Conflict-only strategy — accept all claims, resolve only direct contradictions
# between same-subject claims (no per-claim source verification). The DEFAULT is
# now "aligned", so these tests pin "conflict_only" explicitly.
# --------------------------------------------------------------------------
@pytest.fixture
def conflict_only(monkeypatch):
    """Opt a test into the fast 'conflict_only' verification strategy (the aligned
    strategy is the default now; this keeps the legacy fast path under test)."""
    from core_engine.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("CE_VERIFY_STRATEGY", "conflict_only")
    yield
    get_settings.cache_clear()


async def test_conflict_only_accepts_all_without_source_checks(conflict_only):
    """conflict_only accepts every extracted claim as fact and never calls the
    per-claim entailment/traceability oracle — proving the expensive step is bypassed."""
    from core_engine.report.models import ConfidenceTier

    class NoVerifyLLM:
        """Fails loudly if any per-claim verification is attempted; contradiction
        checks are allowed (and here, never fire because subjects differ)."""
        def __init__(self):
            self.contradiction_checks = 0
        async def check_traceability(self, claim, source_text):
            raise AssertionError("check_traceability must NOT be called in conflict_only")
        async def check_entailment(self, claim, source_text):
            raise AssertionError("check_entailment must NOT be called in conflict_only")
        async def check_contradiction(self, a, b):
            self.contradiction_checks += 1
            from core_engine.report.llm import ContradictionJudgement
            return ContradictionJudgement(contradict=False)

    sources = [
        Source("https://cdc.gov/p", "cdc.gov", "CDC", SourceKind.GOVERNMENT, text="x"),
        Source("https://iea.org/q", "iea.org", "IEA", SourceKind.INDUSTRY_INSTITUTION, text="y"),
    ]
    claims = [
        Claim(id="c1", text="The tariff rate was 12 percent in 2024.",
              candidate_source_urls=["https://cdc.gov/p"]),
        Claim(id="c2", text="Steel output reached 8 million tonnes in 2023.",
              candidate_source_urls=["https://iea.org/q"]),
    ]
    llm = NoVerifyLLM()
    report = await VerificationHarness(llm).verify(claims, sources)
    # Both accepted as fact; nothing rejected; both reportable.
    assert report.kept_count == 2
    assert report.rejected_claims == []
    assert report.verified_count == 2                       # HIGH/LIKELY are reportable
    # Different subjects -> no candidate pair -> not even a contradiction call.
    assert llm.contradiction_checks == 0
    # High-trust provenance -> HIGH tier, and citations still resolve to real URLs.
    assert {c.confidence for c in report.kept_claims} == {ConfidenceTier.HIGH}
    assert report.kept_claims[0].supporting_sources() == ["https://cdc.gov/p"]


async def test_conflict_only_drops_lower_authority_of_conflicting_pair(conflict_only):
    """Two same-subject claims with diverging numbers contradict; the one backed by the
    lower-authority source is dropped, the higher-authority one kept."""
    shared_subject = "The national unemployment rate for 2024"
    gov = Source("https://bls.gov/a", "bls.gov", "BLS", SourceKind.GOVERNMENT, text="x")
    blog = Source("https://someblog.net/b", "someblog.net", "Blog",
                  SourceKind.GENERAL_WEB, text="y")
    hi = Claim(id="gov1", text=f"{shared_subject} was 4 percent.",
               candidate_source_urls=["https://bls.gov/a"])
    lo = Claim(id="blog1", text=f"{shared_subject} was 9 percent.",
               candidate_source_urls=["https://someblog.net/b"])
    # Order the low-authority claim first to prove authority (not order) decides.
    from core_engine.report.llm import FakeLLM
    report = await VerificationHarness(FakeLLM()).verify([lo, hi], [gov, blog])
    kept_ids = {c.id for c in report.kept_claims}
    assert kept_ids == {"gov1"}
    assert "blog1" in report.reasons
    assert "contradicted" in report.reasons["blog1"]


async def test_conflict_only_keeps_nonconflicting_claim_untouched(conflict_only):
    """A claim with no contradicting counterpart is accepted with no verification
    overhead even when a contradiction oracle exists."""
    from core_engine.report.llm import FakeLLM as _F
    lone = Claim(id="c1", text="Adoption grew 30 percent in 2025.",
                 candidate_source_urls=["https://reuters.com/x"])
    src = Source("https://reuters.com/x", "reuters.com", "R",
                 SourceKind.AUTHORITATIVE_MEDIA, text="z")
    report = await VerificationHarness(_F()).verify([lone], [src])
    assert [c.id for c in report.kept_claims] == ["c1"]
    assert report.rejected_claims == []


def test_report_renders_rumors_section_separately():
    """Rumor-tier claims render in a labeled 'Unverified Signals & Rumors' section,
    separate from the verified body — and are escaped like all untrusted text."""
    from core_engine.report.latex import render_tex
    from core_engine.report.models import ConfidenceTier

    report = ReportData(
        topic="widgets",
        title="Widgets: Report",
        abstract="abstract",
        sections=[ReportSection(heading="Industry Overview", section_id="industry_overview",
                                paragraphs=["Verified fact about widgets."], claim_ids=["c1"])],
        claims=[Claim(id="c1", text="Verified fact about widgets.", verified=True,
                      confidence=ConfidenceTier.HIGH,
                      candidate_source_urls=["https://cdc.gov/a"])],
        sources=[Source("https://cdc.gov/a", "cdc.gov", "A", SourceKind.GOVERNMENT, text="...")],
        rumors=[Claim(id="r1", text="Rumor: a new widget plant may open in 2026.",
                      confidence=ConfidenceTier.RUMOR,
                      evidence=[], candidate_source_urls=["https://someblog.net/x"])],
    )
    tex = render_tex(report)
    assert "Unverified Signals" in tex
    assert "new widget plant may open" in tex
    assert "unverified" in tex.lower()


# --------------------------------------------------------------------------
# LaTeX escaping — untrusted scraped text cannot inject commands.
# --------------------------------------------------------------------------
def test_tex_escape_neutralizes_special_chars():
    dangerous = r"Cost rose 50% & \input{/etc/passwd} #$_{}"
    escaped = tex_escape(dangerous)
    assert "\\input{" not in escaped        # command neutralized
    assert r"\%" in escaped
    assert r"\&" in escaped
    assert r"\#" in escaped


def test_render_tex_escapes_report_content():
    report = ReportData(
        topic="tariffs & trade",
        title="Tariffs & Trade: 100% Verified",
        abstract="A report about 50% growth & $budgets.",
        sections=[ReportSection(heading="Overview & Scope",
                                paragraphs=["Spending rose 10% in Q1."], claim_ids=["c1"])],
        claims=[Claim(id="c1", text="x", verified=True,
                      candidate_source_urls=["https://cdc.gov/a"])],
        sources=[Source("https://cdc.gov/a", "cdc.gov", "Title & More",
                        SourceKind.GOVERNMENT, text="...")],
    )
    tex = render_tex(report)
    assert r"100\% Verified" in tex
    assert r"Overview \& Scope" in tex
    # raw unescaped '&' should not appear inside the rendered title
    assert "100% Verified" not in tex


# --------------------------------------------------------------------------
# End-to-end pipeline: out-of-scope halt vs. successful generation.
# --------------------------------------------------------------------------
def _authoritative_fixture():
    """Build a search provider + fetcher where 3 gov domains all state the same
    5 verifiable facts, so claims clear triple-check AND cross-reference."""
    facts = (
        "The initiative launched in 2021."
        " It allocated 5 billion dollars in funding."
        " Participation increased by 30 percent in 2022."
        " The program operated in 12 regions."
        " Officials reported a 95 percent satisfaction rate."
    )
    body = f"Official statement. {facts} This concludes the summary of the record."
    urls = {
        "https://cdc.gov/report": body,
        "https://europa.eu/report": body,
        "https://who.int/report": body,
    }
    search = FakeSearchProvider()
    search.add("clean energy", [
        SearchHit(u, "Official Report", "", u.split("/")[2]) for u in urls
    ] + [SearchHit("https://randomblog.net/x", "Blog", "", "randomblog.net")])
    fetcher = FakeFetcher(urls)
    return search, fetcher


async def test_pipeline_out_of_scope_when_no_authoritative_sources():
    from core_engine.report.llm import FakeLLM

    search = FakeSearchProvider()  # no fixtures -> no hits for any topic
    fetcher = FakeFetcher({})
    pipe = ReportPipeline(llm=FakeLLM(), search=search, fetcher=fetcher)
    result = await pipe.run("obscure niche topic", compile_to_pdf=False)
    assert result.status is PipelineStatus.OUT_OF_SCOPE
    assert result.message == OUT_OF_SCOPE_MESSAGE
    assert result.pdf_path is None


async def test_gather_survives_a_hanging_url(monkeypatch):
    """A single hanging/slow URL must NOT stall the whole gather step. The gather-level
    defensive timeout bounds each fetch, keeps the good sources, and emits progress."""
    import asyncio
    import time as _time

    from core_engine.config import get_settings
    from core_engine.report.scrape import gather_sources

    class HangingFetcher:
        async def fetch(self, url: str):
            if "hang" in url:
                await asyncio.sleep(10_000)          # would hang forever
            await asyncio.sleep(0.01)
            return "Solid state battery supply chain analysis. " * 40

    class Prov:
        async def search(self, topic, *, max_results):
            return [
                SearchHit("https://nih.gov/a", "A", "", "nih.gov"),
                SearchHit("https://europa.eu/b", "B", "", "europa.eu"),
                SearchHit("https://who.int/hang", "H", "", "who.int"),   # hangs
                SearchHit("https://oecd.org/c", "C", "", "oecd.org"),
            ]

    get_settings.cache_clear()
    monkeypatch.setenv("CE_SCRAPE_PER_URL_TIMEOUT_S", "1")
    monkeypatch.setenv("CE_RELEVANCE_FILTER", "false")
    monkeypatch.setenv("CE_MULTI_QUERY_RESEARCH", "false")
    monkeypatch.setenv("CE_SCRAPE_CONCURRENCY", "8")

    events = []
    try:
        t0 = _time.monotonic()
        sources, _rejected = await gather_sources(
            "solid state battery supply chain", Prov(), HangingFetcher(),
            on_progress=lambda d, t, detail: events.append((d, t)))
        dt = _time.monotonic() - t0
    finally:
        get_settings.cache_clear()

    assert dt < 8, "a hanging URL must not stall the whole gather step"
    assert len(sources) == 3, "the 3 good URLs should survive the one that hangs"
    assert {s.url for s in sources} == {
        "https://nih.gov/a", "https://europa.eu/b", "https://oecd.org/c"}
    assert len(events) >= 4, "progress must be emitted per page (feeds the UI watchdog)"


async def test_llm_retries_429_and_caps_concurrency(monkeypatch):
    """The LLM chokepoint must (a) retry a 429 with backoff instead of crashing the
    pipeline, and (b) never exceed the global concurrency cap across parallel calls."""
    import asyncio

    from core_engine.config import get_settings
    from core_engine.report.llm import AnthropicLLM, LLMRateLimitError

    get_settings.cache_clear()
    monkeypatch.setenv("CE_LLM_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("CE_LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("CE_LLM_RETRY_BASE_S", "0.01")
    monkeypatch.setenv("CE_LLM_RETRY_MAX_S", "0.05")

    class _Resp:
        def __init__(self, status): self.status_code = status; self.headers = {}

    class _HTTPStatusError(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.response = _Resp(status)

    # (a) two 429s then success — must recover, not raise
    class Flaky(AnthropicLLM):
        def __init__(self): super().__init__("k", "m", 100); self.calls = 0
        async def _transport(self, system, user):
            self.calls += 1
            if self.calls <= 2:
                raise _HTTPStatusError(429)
            return "ok"

    try:
        flaky = Flaky()
        assert await flaky._complete("s", "u") == "ok"
        assert flaky.calls == 3

        # (b) global cap: 20 concurrent calls, peak in-flight must stay <= 3
        class Conc(AnthropicLLM):
            def __init__(self): super().__init__("k", "m", 100); self.cur = 0; self.peak = 0
            async def _transport(self, system, user):
                self.cur += 1; self.peak = max(self.peak, self.cur)
                await asyncio.sleep(0.02)
                self.cur -= 1
                return "x"

        conc = Conc()
        await asyncio.gather(*[conc._complete("s", "u") for _ in range(20)])
        assert conc.peak <= 3, f"concurrency cap breached: peak={conc.peak}"

        # (c) auth 4xx fails fast (no retry); exhausted 429 budget -> LLMRateLimitError
        class Auth(AnthropicLLM):
            def __init__(self): super().__init__("k", "m", 100); self.calls = 0
            async def _transport(self, system, user):
                self.calls += 1
                raise _HTTPStatusError(401)

        auth = Auth()
        try:
            await auth._complete("s", "u")
            assert False, "401 should propagate"
        except _HTTPStatusError:
            assert auth.calls == 1, "must not retry a 401"

        class Always(AnthropicLLM):
            async def _transport(self, system, user):
                raise _HTTPStatusError(429)

        try:
            await Always("k", "m", 100)._complete("s", "u")
            assert False, "exhausted budget should raise"
        except LLMRateLimitError as e:
            assert e.status == 429
    finally:
        get_settings.cache_clear()


async def test_assembly_runs_concurrently_under_deadline(monkeypatch):
    """Assembly (KG + section drafts + deep-dives) must run CONCURRENTLY, not serialize
    into a timeout. With a slow LLM (~0.3s/call) and enough verified claims to trigger
    every section + deep-dive, a sequential run would take many seconds; the concurrent
    fan-out (bounded by the global LLM semaphore) finishes far faster and still fills
    every section."""
    import asyncio
    import time as _t

    from core_engine.config import get_settings
    from core_engine.report.llm import FakeLLM

    class SlowLLM(FakeLLM):
        async def draft_section(self, heading, claims):
            await asyncio.sleep(0.3)
            return await super().draft_section(heading, claims)
        async def draft_deep_dive(self, title, kind, claims):
            await asyncio.sleep(0.3)
            return ["Analytical paragraph one.", "Analytical paragraph two."]
        async def extract_kg(self, topic, verified_claims):
            await asyncio.sleep(0.3)
            return await super().extract_kg(topic, verified_claims)
        async def identify_insights(self, topic, verified_claims):
            await asyncio.sleep(0.3)
            return []

    get_settings.cache_clear()
    monkeypatch.setenv("CE_LLM_MAX_CONCURRENCY", "8")
    monkeypatch.setenv("CE_ASSEMBLE_DEADLINE_S", "30")

    # Enough verified claims (spread across sections) to draft every section + fallbacks.
    verified = []
    kw = ["overview sector", "policy regulation subsidy", "upstream supply component",
          "market size CAGR revenue", "competitor share leader"]
    for i in range(15):
        c = Claim(id=f"c{i}", text=f"The {kw[i % 5]} figure was {i} percent in 2024.",
                  candidate_source_urls=[f"https://cdc.gov/{i}"])
        c.verified = True
        from core_engine.report.models import ConfidenceTier
        c.confidence = ConfidenceTier.HIGH
        verified.append(c)
    sources = [Source(f"https://cdc.gov/{i}", "cdc.gov", "CDC", SourceKind.GOVERNMENT,
                      text="gov text") for i in range(15)]

    from core_engine.report.verify import VerificationReport
    report = VerificationReport()
    report.kept_claims = verified

    pipe = ReportPipeline(llm=SlowLLM(), search=FakeSearchProvider(), fetcher=FakeFetcher({}))
    try:
        t0 = _t.monotonic()
        data = await pipe._assemble("battery supply chain", report, sources)
        dt = _t.monotonic() - t0
    finally:
        get_settings.cache_clear()

    # Sequential would be ~ (1 KG + 1 insights + 5 sections + up-to-5 deep-dives) * 0.3s
    # ≈ 3.6s+. Concurrent (cap 8) collapses that to roughly one wave.
    assert dt < 2.5, f"assembly did not run concurrently (took {dt:.1f}s)"
    # 5 mandated sections + the deterministic 'limitations' closing section
    # (no clusters in this fixture, so no 'source_comparison' section).
    assert len(data.sections) == 6


async def test_pipeline_generates_tex_when_gates_pass(monkeypatch):
    from core_engine.config import get_settings
    from core_engine.report.llm import FakeLLM

    # This test exercises the PIPELINE logic (verify -> assemble -> render), not the
    # relevance filter, whose thresholds the compact fixture bodies don't meet. Turn
    # the relevance filter off and pin a 2-source bar so the assertions below hold.
    get_settings.cache_clear()
    monkeypatch.setenv("CE_RELEVANCE_FILTER", "false")
    monkeypatch.setenv("CE_MIN_SOURCES_PER_CLAIM", "2")
    monkeypatch.setenv("CE_MIN_AUTHORITATIVE_SOURCES", "3")
    monkeypatch.setenv("CE_MIN_VERIFIED_CLAIMS", "5")

    search, fetcher = _authoritative_fixture()
    pipe = ReportPipeline(llm=FakeLLM(), search=search, fetcher=fetcher)
    try:
        result = await pipe.run("clean energy", compile_to_pdf=False)
    finally:
        get_settings.cache_clear()
    assert result.status is PipelineStatus.COMPLETED, result.message
    assert result.tex_path is not None
    assert result.report is not None
    # every claim in the report is verified and cross-referenced
    assert all(c.verified for c in result.report.claims)
    assert len(result.report.claims) >= 5
    for c in result.report.claims:
        assert len({u.split("/")[2] for u in c.supporting_sources()}) >= 2


async def test_pipeline_rejects_blog_only_topic():
    """Even with plenty of content, if it's all non-authoritative -> out of scope."""
    search = FakeSearchProvider()
    search.add("gossip", [
        SearchHit("https://blog.medium.com/a", "A", "", "blog.medium.com"),
        SearchHit("https://forum.reddit.com/b", "B", "", "forum.reddit.com"),
    ])
    fetcher = FakeFetcher({
        "https://blog.medium.com/a": "lots of text " * 50,
        "https://forum.reddit.com/b": "more text " * 50,
    })
    from core_engine.report.llm import FakeLLM

    pipe = ReportPipeline(llm=FakeLLM(), search=search, fetcher=fetcher)
    result = await pipe.run("gossip", compile_to_pdf=False)
    assert result.status is PipelineStatus.OUT_OF_SCOPE


# --------------------------------------------------------------------------
# Expanded source allowlist — IBs, industry institutions, non-profits pass;
# tabloids/gossip (小道媒体) are denied.
# --------------------------------------------------------------------------
def test_expanded_allowlist_classifies_new_categories():
    assert classify("https://www.mckinsey.com/insights") is SourceKind.INVESTMENT_BANK
    assert classify("https://www.spglobal.com/x") is SourceKind.INVESTMENT_BANK
    assert classify("https://www.iea.org/reports/x") is SourceKind.INDUSTRY_INSTITUTION
    assert classify("https://www.oecd.org/x") is SourceKind.INDUSTRY_INSTITUTION
    assert classify("https://www.pewresearch.org/x") is SourceKind.NONPROFIT
    assert classify("https://www.reuters.com/x") is SourceKind.AUTHORITATIVE_MEDIA


def test_tabloid_denylist_rejects_gossip_media():
    # tabloids / gossip are rejected even though they are "media" outlets
    assert classify("https://www.dailymail.co.uk/news") is SourceKind.REJECTED
    assert classify("https://www.tmz.com/x") is SourceKind.REJECTED
    assert classify("https://www.infowars.com/x") is SourceKind.REJECTED


# --------------------------------------------------------------------------
# Traceable verification mode — paraphrase passes; fabricated fact is a hard fail.
# --------------------------------------------------------------------------
async def test_traceable_mode_accepts_paraphrase_rejects_hallucination(cross_ref, monkeypatch):
    from core_engine.config import get_settings
    from core_engine.report.llm import TraceJudgement

    get_settings.cache_clear()
    monkeypatch.setenv("CE_VERIFY_MODE", "traceable")
    monkeypatch.setenv("CE_MIN_SOURCES_PER_CLAIM", "2")
    monkeypatch.setenv("CE_VERIFY_ROUNDS", "3")

    # A stub LLM: paraphrase-supported by two domains, plus one claim a source flags
    # as hallucinated (which must poison that claim regardless of other support).
    class StubLLM:
        async def check_traceability(self, claim, source_text):
            if "fabricated" in claim:
                if "flagme" in source_text:
                    return TraceJudgement(supported=False, hallucinated=True)
                return TraceJudgement(supported=True, hallucinated=False,
                                      evidence_span="ok", confidence=0.9)
            return TraceJudgement(supported=True, hallucinated=False,
                                  evidence_span="ok", confidence=0.9)

    from core_engine.report.verify import VerificationHarness

    good = Claim(id="c1", text="paraphrased finding",
                 candidate_source_urls=["https://cdc.gov/a", "https://iea.org/b"])
    bad = Claim(id="c2", text="a fabricated finding",
                candidate_source_urls=["https://cdc.gov/a", "https://iea.org/flagme"])
    sources = [
        Source("https://cdc.gov/a", "cdc.gov", "A", SourceKind.GOVERNMENT, text="..."),
        Source("https://iea.org/b", "iea.org", "B", SourceKind.INDUSTRY_INSTITUTION, text="..."),
        Source("https://iea.org/flagme", "iea.org", "F", SourceKind.INDUSTRY_INSTITUTION,
               text="flagme"),
    ]
    try:
        harness = VerificationHarness(StubLLM())
        report = await harness.verify([good, bad], sources)
    finally:
        get_settings.cache_clear()

    verified_ids = {c.id for c in report.verified_claims}
    assert "c1" in verified_ids                       # semantically supported -> verified
    assert "c2" not in verified_ids                   # directly contradicted -> dropped
    assert "contradicted" in report.reasons["c2"]     # only contradiction is a hard drop


# --------------------------------------------------------------------------
# KG extraction + chart generation from verified claims.
# --------------------------------------------------------------------------
async def test_kg_build_and_charts_from_verified_claims():
    from core_engine.report.charts import build_all
    from core_engine.report.kg import ChainTier, build_report_kg
    from core_engine.report.llm import FakeLLM

    verified = [
        Claim(id="k1", text="UPSTREAM: Polysilicon", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
        Claim(id="k2", text="MIDSTREAM: Cell Manufacturing", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
        Claim(id="k3", text="DOWNSTREAM: Installers", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
        Claim(id="k4", text="Polysilicon supplies Cell Manufacturing", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
        Claim(id="k5", text="TAM is 200 USD bn", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
        Claim(id="k6", text="In 2021 the market was 90 USD bn", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
        Claim(id="k7", text="In 2022 the market was 120 USD bn", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
        Claim(id="k8", text="COMPETITOR: FirstCo share 30% advantage scale", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
        Claim(id="k9", text="COMPETITOR: SecondCo share 20% advantage cost", verified=True,
              candidate_source_urls=["https://iea.org/a"]),
    ]
    kg = await build_report_kg("solar", verified, FakeLLM())

    assert kg.has_chain()
    assert [n.name for n in kg.tier(ChainTier.UPSTREAM)] == ["Polysilicon"]
    assert kg.market.tam == 200
    assert kg.market.has_chartable_series()
    assert len(kg.competitors) == 2

    blocks = build_all(kg)
    assert "tikzpicture" in blocks["chain_tikz"]
    assert "axis" in blocks["market_plot"]
    assert "tabular" in blocks["competitive_table"]
    assert "xbar" in blocks["competitive_bar"]


def test_kg_drops_elements_without_verified_provenance():
    """A KGExtraction element whose claim_ids aren't verified must be dropped."""
    from core_engine.report.kg import build_report_kg
    from core_engine.report.llm import KGExtraction

    class StubLLM:
        async def extract_kg(self, topic, verified_claims):
            # references claim 'ghost' which is NOT in the verified set
            return KGExtraction(
                chain=[{"name": "Ghost", "tier": "upstream", "claim_ids": ["ghost"]}],
                competitors=[{"name": "GhostCo", "market_share_pct": 50,
                              "claim_ids": ["ghost"]}],
            )

    async def _run():
        verified = [Claim(id="real", text="x", verified=True,
                          candidate_source_urls=["https://cdc.gov/a"])]
        return await build_report_kg("t", verified, StubLLM())

    import anyio
    kg = anyio.run(_run)
    assert not kg.has_chain()          # Ghost node dropped (unverified provenance)
    assert not kg.has_competitors()    # GhostCo dropped


# --------------------------------------------------------------------------
# The 5 mandated sections are always present, in order.
# --------------------------------------------------------------------------
async def test_pipeline_emits_five_mandated_sections_in_order(monkeypatch):
    from core_engine.config import get_settings
    from core_engine.report.llm import FakeLLM

    # Pipeline-structure test; the compact fixture bodies are below the relevance
    # filter's thresholds, so disable it here (relevance has its own dedicated test).
    get_settings.cache_clear()
    monkeypatch.setenv("CE_RELEVANCE_FILTER", "false")
    monkeypatch.setenv("CE_MIN_AUTHORITATIVE_SOURCES", "3")

    search, fetcher = _industry_fixture()
    pipe = ReportPipeline(llm=FakeLLM(), search=search, fetcher=fetcher)
    try:
        result = await pipe.run("solar power", compile_to_pdf=False)
    finally:
        get_settings.cache_clear()
    assert result.status is PipelineStatus.COMPLETED, result.message
    ids = [s.section_id for s in result.report.sections]
    # The 5 mandated sections come first, in order; the synthesis layer may append
    # 'source_comparison' (only when alignment clusters exist) and always appends
    # the deterministic 'limitations' closing section.
    assert ids[:5] == ["industry_overview", "policy_analysis", "industry_chain",
                       "market_size", "competitive_landscape"]
    assert set(ids[5:]) <= {"source_comparison", "limitations"}


def _industry_fixture():
    """3 authoritative domains (gov + IB + institution) stating the same industry
    facts, including chain/market/competitor markers the FakeLLM KG extractor reads."""
    facts = (
        "The solar sector background shows strong adoption."
        " Government policy introduced a subsidy in 2021."
        " UPSTREAM: Polysilicon is a key input."
        " MIDSTREAM: Cell Manufacturing integrates components."
        " DOWNSTREAM: Installers serve end markets."
        " Polysilicon supplies Cell Manufacturing."
        " TAM is 200 USD bn for the market."
        " In 2021 the market was 90 USD bn."
        " In 2022 the market was 120 USD bn."
        " CAGR is 15% for the sector."
        " COMPETITOR: FirstCo share 30% advantage scale."
        " COMPETITOR: SecondCo share 20% advantage cost leadership."
    )
    body = f"Official industry report. {facts} End of record."
    urls = {
        "https://energy.gov/report": body,
        "https://iea.org/report": body,
        "https://mckinsey.com/report": body,
    }
    search = FakeSearchProvider()
    search.add("solar power", [
        SearchHit(u, "Industry Report", "", u.split("/")[2]) for u in urls
    ])
    fetcher = FakeFetcher(urls)
    return search, fetcher
