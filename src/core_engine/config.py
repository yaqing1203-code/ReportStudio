"""Central configuration. All values are env-overridable (12-factor).

Domain adaptation NEVER happens here — this is invariant-core config only.
Vertical-specific settings live in the adapter layer (ontology.yaml, profiles).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

# Localized display headings for every known section id (the 5 mandated sections plus
# the synthesis-layer sections). resolved_sections() falls back to the configured
# heading for ids not listed here.
_SECTION_HEADINGS: dict[str, dict[str, str]] = {
    "industry_overview": {"en": "Industry Overview", "zh": "行业概览"},
    "policy_analysis": {"en": "Policy Analysis", "zh": "政策分析"},
    "industry_chain": {"en": "Industry Chain Map", "zh": "产业链图谱"},
    "market_size": {"en": "Market Size", "zh": "市场规模"},
    "competitive_landscape": {"en": "Competitive Landscape", "zh": "竞争格局"},
    "source_comparison": {"en": "Source Comparison & Divergence",
                          "zh": "来源对比与分歧分析"},
    "limitations": {"en": "Information Limitations", "zh": "信息局限性声明"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CE_", env_file=".env", extra="ignore")

    # --- Postgres (single instance hosts AGE graph + pgvector + relational) ---
    pg_dsn: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/core_engine",
        description="Libpq DSN. The same DB holds relational tables, the AGE graph, and pgvector.",
    )
    pg_pool_min: int = 1
    pg_pool_max: int = 10

    # --- Knowledge Graph (Apache AGE) ---
    age_graph_name: str = "core_kg"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-m3"  # dense + sparse from one model
    embedding_dim: int = 1024

    # --- Retrieval defaults (overridable per query) ---
    retrieval_top_k: int = 50        # candidates pulled before re-rank
    rerank_top_n: int = 8            # returned after cross-encoder
    max_hops: int = 3                # multi-hop traversal depth ceiling
    rrf_k: int = 60                  # Reciprocal Rank Fusion constant

    # --- Ontology (adapter layer) ---
    ontology_path: Path = REPO_ROOT / "adapters" / "default" / "ontology.yaml"

    # --- Safety ---
    min_retrieval_score: float = 0.15  # below this -> abstain rather than answer

    # ======================================================================
    # Report-generation pipeline (the current project objective).
    # The KG/RAG settings above are retained for the dormant data-agent modules
    # but are NOT used by the report pipeline.
    # ======================================================================

    # --- LLM (used for query planning, claim extraction, drafting) ---
    # "anthropic" : native Anthropic Messages API (anthropic_api_key).
    # "openai"    : any OpenAI-COMPATIBLE endpoint (OpenAI, Azure, local vLLM, Ollama,
    #               Together, Groq, ...) via llm_base_url + llm_api_key. The GUI
    #               Settings panel writes these three fields.
    # "fake"      : deterministic offline stand-in (default, zero setup).
    llm_provider: str = "fake"               # "anthropic" | "openai" | "fake"
    llm_model: str = "claude-opus-4-8"
    llm_max_tokens: int = 4096
    llm_base_url: str = ""                    # CE_LLM_BASE_URL (openai-compatible)
    llm_api_key: str = ""                     # CE_LLM_API_KEY (generic; used by "openai")
    anthropic_api_key: str = ""              # CE_ANTHROPIC_API_KEY
    # Hard per-call timeout. No single LLM request may hang the pipeline; on timeout
    # the caller treats that (claim, source) check as 'unsupported' and moves on.
    llm_timeout_s: float = 120.0
    # --- LLM rate limiting (applied at the ONE _complete chokepoint all phases share) ---
    # GLOBAL cap on concurrent LLM requests across the WHOLE pipeline (extract + verify +
    # KG + drafting), independent of any per-phase semaphore. This is the primary guard
    # against tripping the provider's rate limit (429). Keep conservative for shared/free
    # tiers (e.g. MiniMax); raise it for high-quota keys.
    llm_max_concurrency: int = 4
    # 429/5xx are retried with exponential backoff + full jitter, honouring Retry-After.
    # attempts total (incl. the first try); auth/4xx are NOT retried (won't self-heal).
    llm_max_retries: int = 5
    llm_retry_base_s: float = 1.0            # backoff = base * 2**(attempt-1), + jitter
    llm_retry_max_s: float = 30.0            # cap on any single backoff sleep
    # Max source characters fed into a single LLM call. The context window is fixed by
    # the model/provider (we cannot "expand" it client-side); this bounds how much of a
    # source we send per call. Deep-dive/full-text phases raise this via llm_deepdive_
    # context_chars so verified sources are read in full, not just a snippet.
    llm_context_chars: int = 12000
    # Larger input budget for the FULL-TEXT deep-dive phase: once a claim is verified we
    # feed the whole verified source (up to this many chars) so the analysis captures
    # tables/methodology a snippet misses. ~48k chars ≈ ~16k tokens of source context.
    llm_deepdive_context_chars: int = 48000
    # Chat context guard: when the running conversation exceeds this many words,
    # older turns are summarized into one paragraph before the next call.
    chat_summarize_word_limit: int = 100_000

    # --- Web search / scraping ---
    # "duckduckgo" : keyless, zero-setup default — works out-of-the-box in the .exe.
    # "tavily"     : higher-quality search (needs CE_SEARCH_API_KEY).
    # "fake"       : deterministic offline fixtures (tests).
    search_provider: str = "duckduckgo"      # "duckduckgo" | "tavily" | "fake"
    search_api_key: str = ""                 # CE_SEARCH_API_KEY
    scrape_max_results: int = 25             # search hits to consider before filtering
    scrape_max_pages: int = 24               # authoritative pages to actually fetch
    scrape_timeout_s: float = 20.0
    scrape_user_agent: str = "CoreEngineReportBot/0.1 (+contact@example.com)"
    scrape_respect_robots: bool = True
    # When robots.txt can't be fetched (timeout/error): True permits the page anyway
    # (fail-open, still rate-limited); False skips it (fail-closed).
    scrape_robots_fail_open: bool = True
    # --- Scrape performance / stability ---
    # Pages are fetched CONCURRENTLY (bounded) instead of one-by-one, so 24 pages don't
    # serialise into minutes. Each fetch also has a HARD per-URL wall-clock so one slow
    # or hanging server can't block the whole gather step.
    scrape_fetch_concurrency: int = 8
    scrape_per_url_timeout_s: float = 25.0   # hard ceiling per URL (robots + page combined)
    scrape_concurrency: int = 6              # max pages fetched in parallel (bounded, polite)
    # Search API calls retry on transient failure with exponential backoff.
    search_max_retries: int = 3
    search_retry_base_s: float = 1.0         # backoff: base * 2**attempt (+ jitter)
    search_timeout_s: float = 30.0           # strict per-attempt timeout for a search call
    # --- Search routing + extra backends (Search Router) ---
    # Optional higher-quality backends the router can pick per query type. Both are
    # instantiated by name via scrape.get_search_provider("firecrawl" | "jina").
    firecrawl_api_key: str = ""              # CE_FIRECRAWL_API_KEY (structured/table queries)
    jina_api_key: str = ""                   # CE_JINA_API_KEY (optional; keyless works, tighter limits)
    # When True and no single provider was explicitly injected, gather_sources routes
    # each query through report/router.py (query-type classification + per-type backend
    # fallback chain). Set False to pin the legacy single-provider behavior.
    search_router_enabled: bool = True
    # On-disk search-result cache (user_data_dir()/search_cache), keyed by
    # sha256(backend + query). 0 disables caching.
    search_cache_ttl_s: float = 86400.0
    # --- Credibility-weighted source ranking (L1-L4) ---
    # After the relevance filter, kept sources are ordered by
    #   relevance * rank_w_relevance + credibility.weight * rank_w_source
    #   + freshness * rank_w_freshness
    # so a high-relevance but weak (L4) source cannot outrank L1/L2 anchors.
    rank_w_relevance: float = 0.5
    rank_w_source: float = 0.35
    rank_w_freshness: float = 0.15
    # robots.txt check has its own short timeout so it can't double per-page latency.
    robots_timeout_s: float = 5.0
    # Multi-query research: instead of one search on the raw topic, the gatherer
    # issues several SECTION-TARGETED queries (overview, policy, chain, market,
    # competition) and merges the authoritative survivors. This is what gives each
    # mandated section its own supporting material instead of starving on a single
    # generic search. Set to False to fall back to a single-query gather.
    multi_query_research: bool = True
    max_pages_per_query: int = 6             # cap fetches per sub-query (breadth vs. cost)

    # --- Source filtering (allowlist gate — EXPANDED for industry reports) ---
    # Domain SUFFIXES always treated as authoritative government sources.
    gov_domain_suffixes: tuple[str, ...] = (
        ".gov", ".gov.uk", ".gov.au", ".gov.cn", ".gc.ca",
        ".europa.eu", ".un.org", ".who.int", ".mil", ".edu",
    )
    # Exact authoritative-media hosts (mainstream outlets with editorial standards).
    media_allowlist: tuple[str, ...] = (
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "nytimes.com", "wsj.com", "washingtonpost.com", "economist.com",
        "ft.com", "bloomberg.com", "npr.org", "pbs.org", "nature.com",
        "science.org", "scientificamerican.com",
    )
    # Investment banks / research houses / market-data providers. These publish the
    # TAM/SAM/SOM, market-share, and industry-chain data the report's charts need.
    investment_bank_allowlist: tuple[str, ...] = (
        "goldmansachs.com", "morganstanley.com", "jpmorgan.com", "gs.com",
        "citigroup.com", "credit-suisse.com", "ubs.com", "barclays.com",
        "mckinsey.com", "bcg.com", "bain.com", "deloitte.com", "pwc.com",
        "kpmg.com", "ey.com", "spglobal.com", "moodys.com", "fitchratings.com",
        "gartner.com", "forrester.com", "idc.com", "statista.com",
        "nielsen.com", "mckinsey.com.cn",
    )
    # Authoritative industry institutions / trade associations / standards bodies.
    industry_institution_allowlist: tuple[str, ...] = (
        "iea.org", "oecd.org", "imf.org", "worldbank.org", "wto.org",
        "iso.org", "ieee.org", "itu.int", "weforum.org", "bis.org",
        "semi.org", "iata.org", "opec.org", "irena.org",
    )
    # Academic databases / peer-reviewed publishers / scientific bodies. High-trust:
    # targeted for scientific and factual claims (requirement: Nature, Science, PubMed…).
    academic_allowlist: tuple[str, ...] = (
        "nature.com", "science.org", "sciencedirect.com", "springer.com",
        "link.springer.com", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
        "nih.gov", "arxiv.org", "ssrn.com", "jstor.org", "wiley.com",
        "onlinelibrary.wiley.com", "tandfonline.com", "cell.com", "pnas.org",
        "thelancet.com", "nejm.org", "bmj.com", "acs.org", "aps.org",
        "scholar.google.com", "semanticscholar.org", "researchgate.net",
        "who.int", "un.org", "cdc.gov", "nasa.gov",
    )
    # Non-profit / NGO research organisations with editorial + methodological rigor.
    nonprofit_allowlist: tuple[str, ...] = (
        "pewresearch.org", "brookings.edu", "rand.org", "gatesfoundation.org",
        "wri.org", "iisd.org", "resourcesforthefuture.org",
    )
    # HARD DENY: gossip / tabloid / 小道媒体. Rejected even if some heuristic matches.
    tabloid_denylist: tuple[str, ...] = (
        "dailymail.co.uk", "thesun.co.uk", "mirror.co.uk", "nypost.com",
        "buzzfeed.com", "tmz.com", "breitbart.com", "infowars.com",
        "dailystar.co.uk", "express.co.uk", "rt.com", "sputniknews.com",
    )

    # --- Verification harness (the gatekeeper) ---
    # "strict"    : every claim needs a REAL verbatim span present in the source
    #               (the original zero-hallucination contract).
    # "traceable" : claims need strict TRACEABILITY + factual accuracy — the LLM must
    #               judge the claim fully supported by an authoritative source and
    #               flag any unsupported/hallucinated fact — but a verbatim quote is
    #               no longer mandatory. This is the relaxed contract requested for
    #               synthesized industry-report claims.
    verify_mode: str = "traceable"           # "traceable" | "strict"
    # --- Verification STRATEGY (speed vs. strictness) ---------------------------------
    #   "aligned" (DEFAULT) — alignment/fusion layer. Like conflict_only it accepts every
    #       claim up-front, but claims carrying structured slots (entity/attribute/value)
    #       are first alias-normalized and clustered by (entity, attribute); inside a
    #       cluster pairs are classified as corroborate / complement / conflict (conflicts
    #       drop the lower-authority claim). Single-source L3/L4 claims are marked
    #       ISOLATED (孤证) and get consistency / deviation / source-note review plus an
    #       optional one-round active-verification re-search. Claims without structured
    #       slots fall back to the lexical conflict path.
    #   "conflict_only" — fast path (previous default). Accept all claims; spend LLM calls
    #       ONLY resolving DIRECT CONTRADICTIONS between same-subject claims (lexical
    #       prefilter + contradiction oracle; lower authority loses).
    #   "cross_reference" — legacy broad-collection model: every claim is traced/entailed
    #       against its own sources and the whole pool, then tiered. Stricter, slower.
    verify_strategy: str = "aligned"         # "aligned" | "conflict_only" | "cross_reference"
    # --- Aligned strategy knobs ---
    # Deviation score (0-1, LLM domain-common-sense judgement) above which an ISOLATED
    # claim is considered an outlier vs. its field's baseline and downgraded one
    # credibility level.
    alignment_deviation_threshold: float = 0.6
    # Active verification for core isolated claims: at most this many proxy-metric
    # queries are generated and re-searched (one round only, no recursion).
    active_verify_max_queries: int = 3
    active_verify_enabled: bool = True
    # --- Human-in-the-loop review (workflow F, default OFF) ---
    # When True, the pipeline PAUSES after verification (before assembly) whenever at
    # least one CORE isolated claim survived — a claim backed by a single source whose
    # credibility is L3 or stronger (numeric level <= 3). The run emits an
    # 'awaiting_confirmation' progress event carrying a summary of those claims and
    # waits for a human decision via a ConfirmationGate: "continue" proceeds to
    # assembly, "abort" halts the run. A gate timeout defaults to "continue".
    hitl_on_isolated_core_claim: bool = False
    # conflict_only: max distinct subject terms two claims must share (as a fraction of the
    # smaller claim's terms) before we spend an LLM contradiction check on the pair. Higher
    # = fewer, more-precise pair checks. The gate is purely lexical, so it costs nothing.
    conflict_min_term_overlap: float = 0.5
    # conflict_only: cap on the number of LLM contradiction checks per run, a hard ceiling
    # so a pathological set of near-duplicate claims can't explode into O(n^2) calls.
    conflict_max_pair_checks: int = 200
    verify_rounds: int = 3                   # legacy; single cross-ref pass is used now
    min_sources_per_claim: int = 1           # corroboration; 1 keeps coverage (grounded)
    min_authoritative_sources: int = 2       # distinct domains overall (softened)
    # Max distinct sources to cross-reference a single claim against (bounds cost of
    # checking every claim against the whole pool). The claim's own source(s) go first.
    max_cross_ref_sources: int = 8
    # --- Verification performance ---
    # Claims are classified CONCURRENTLY (bounded) rather than one-by-one, so 100+
    # claims finish within the deadline. The cap also respects LLM API rate limits.
    verify_max_concurrency: int = 8
    # Before sending a source to the LLM, we retrieve only the most claim-relevant
    # passages instead of the full page (was: up to 12k chars/call). This is the big
    # latency win — the model reads ~2k focused chars, not a whole article.
    verify_passage_max_chars: int = 2000
    # Cheap lexical PREFILTER for CROSS-REFERENCED (non-own) sources: if a source shares
    # fewer than this many distinct content terms with the claim, skip the LLM call —
    # it almost certainly can't corroborate it. The claim's OWN source(s) are always
    # checked regardless, so grounding is never weakened. 0 disables the prefilter.
    verify_prefilter_min_hits: int = 1
    # FAST-PATH: once a HIGH-TRUST (gov/academic/IB/institution/media) source supports a
    # claim, it is already at the top confidence tier (HIGH) — checking further sources
    # cannot raise it, so we stop immediately and move to the next claim. This is the big
    # per-claim win: proven claims don't get over-verified. Set False to always cross-
    # reference up to max_cross_ref_sources even after a high-trust endorsement.
    verify_fast_path_high_trust: bool = True
    # Fail-safe wall-clock for the WHOLE verification stage. When exceeded, the harness
    # stops issuing new checks and returns whatever it has classified so far — the run
    # proceeds with partial results instead of hanging forever. DOUBLED from 180s to 360s
    # to prevent premature timeout on large workloads.
    verify_deadline_s: float = 360.0
    min_verified_claims: int = 3             # below this -> topic is out-of-scope
    max_reverify_attempts: int = 2           # re-scrape/re-verify loops before failing

    # --- Confidence tiering (broad collection + cross-verification) ---
    # The harness KEEPS every traceable, non-hallucinated claim and labels it by how
    # well it is corroborated, instead of discarding uncorroborated ones. This lets
    # the report separate verified facts from early signals / rumors.
    #   HIGH  : >= high_confidence_min_domains distinct domains, OR any high-trust
    #           (curated-allowlist) source endorses it.
    #   CORROBORATED : supported by >1 distinct domain but below the HIGH bar.
    #   RUMOR : single source, or general-web only.
    high_confidence_min_domains: int = 2     # distinct domains for automatic HIGH tier
    include_rumors_in_report: bool = True    # keep rumors (labeled) vs. drop them
    # A report is out-of-scope only if fewer than this many claims survive at ANY tier
    # (rumors included) — so broad collection can still produce a report to analyze.
    min_total_claims: int = 3

    # --- Tiered source trust ---
    # When True, sites that match no curated allowlist but survive the denylist are
    # admitted as GENERAL_WEB (lower trust weight) so valid topics aren't starved of
    # sources. Grounding is preserved: every claim still traces to a fetched source.
    # Set False to revert to the strict curated-allowlist-only behavior.
    allow_general_web: bool = True

    # --- Full-body relevance filtering (requirement #2) ---
    # After fetching, each page's FULL body is scored for topical relevance. Sources
    # that only mention the topic superficially are discarded; deep-dive articles with
    # substantial on-topic content are kept.
    relevance_filter: bool = True
    # Loosened per requirement #2 ("broaden relevance / semantic matching"). Combined with
    # the stemming + synonym-expansion + prefix matching in relevance_score(), these lower
    # thresholds let approximately-related pages through instead of demanding dense,
    # exact-keyword coverage. Raise them again if too much marginal material gets in.
    relevance_min_score: float = 0.015       # min topic-term density to keep a source
    relevance_min_body_chars: int = 400      # a real article body, not a stub/nav page
    relevance_min_topic_hits: int = 1        # at least one (stem/synonym) topic hit

    # --- Report structure (the 5 MANDATORY sections, in order) ---
    # Keys are stable ids used by the synthesizer; values are the display headings.
    # The KG drives 'industry_chain'; pgfplots drives 'market_size'; booktabs drives
    # 'competitive_landscape'.
    report_sections: tuple[tuple[str, str], ...] = (
        ("industry_overview", "Industry Overview"),
        ("policy_analysis", "Policy Analysis"),
        ("industry_chain", "Industry Chain Map"),
        ("market_size", "Market Size"),
        ("competitive_landscape", "Competitive Landscape"),
    )
    # Report language: "en" (default) or "zh". Drives localized section headings
    # (resolved_sections), the report title, the out-of-scope message, and the
    # LaTeX preamble (ctex for zh).
    report_locale: str = "en"                # "en" | "zh"
    # Synthesis-layer sections appended after the mandated five: 'source_comparison'
    # (only when alignment clusters exist) and 'limitations' (always, when enabled).
    enable_synthesis_sections: bool = True

    def resolved_sections(self) -> list[tuple[str, str]]:
        """The full ordered section plan with locale-appropriate headings.

        The 5 mandated sections keep their configured headings under "en" and switch
        to the Chinese titles under "zh"; the synthesis-layer sections
        (source_comparison, limitations) are appended when enabled."""
        locale = (self.report_locale or "en").lower()
        out: list[tuple[str, str]] = []
        for sid, heading in self.report_sections:
            localized = _SECTION_HEADINGS.get(sid, {}).get(locale) if locale != "en" else None
            out.append((sid, localized or heading))
        if self.enable_synthesis_sections:
            for sid in ("source_comparison", "limitations"):
                titles = _SECTION_HEADINGS[sid]
                out.append((sid, titles.get(locale, titles["en"])))
        return out

    # --- Charts / KG ---
    enable_charts: bool = True               # emit TikZ / pgfplots / booktabs
    kg_project_to_age: bool = False          # also persist the report KG into AGE
    kg_max_chain_nodes_per_tier: int = 6     # keep the chain diagram legible

    # --- Assembly (synthesis) performance ---
    # The assemble phase issues many LLM generations (KG + deep-dives + section prose).
    # Running them SEQUENTIALLY behind the rate limiter was the "stuck on synthesizing"
    # hang. They are now run CONCURRENTLY (bounded) and under their own wall-clock
    # deadline so synthesis degrades to what's finished instead of hanging to the ceiling.
    assemble_max_concurrency: int = 6        # max parallel synthesis LLM calls
    assemble_deadline_s: float = 300.0       # fail-safe ceiling for the whole assemble phase (doubled)
    # DEEP-DIVE SCALE: only the top-N most-corroborated verified claims feed the deep-dive
    # phase, so 156 claims don't spawn dozens of generations. The rest still appear in the
    # (cheap, non-LLM) section buckets and rumor list — they're not discarded.
    deep_dive_max_claims: int = 30           # top-N verified claims considered for deep dives
    # FULL-TEXT DEEP DIVE: after a claim is verified, feed the most relevant passages of
    # its supporting SOURCE text (not just the claim sentence) into the analysis, so the
    # deep dive captures data/tables/nuance a one-line claim misses. 0 disables (claim
    # text only). This is chars of extracted source passage per claim, capped for cost.
    deep_dive_source_context_chars: int = 1500

    # --- Deep-dive analysis (report length / depth expansion) ---
    # When enabled, the harness's highest-value verified findings trigger multi-
    # paragraph analytical insights (bottlenecks, policy impacts, competitor edges)
    # rather than brief summaries.
    enable_deep_dive: bool = True
    max_deep_dives: int = 8                   # cap total deep-dives per report
    deep_dive_min_paragraphs: int = 2         # each deep-dive is multi-paragraph
    deep_dive_target_words: int = 320         # target length per deep-dive (guidance)
    # Guarantee analytical depth per section: any mandated section holding at least
    # this many verified claims but no insight-driven deep-dive gets one synthesized
    # from its own claims, so no section is left with only a thin summary.
    deep_dive_section_min_claims: int = 3

    # --- Section drafting depth ---
    # Guidance passed to draft_section so every mandated section gets a substantive,
    # analytically-structured treatment (not a one-line claim concatenation) — while
    # still using ONLY verified claims. These are targets, not hard limits; the model
    # is told never to pad or invent to hit them.
    section_min_paragraphs: int = 2           # ask for multi-paragraph section prose
    section_target_words: int = 260           # target length per section (guidance)

    # --- LaTeX ---
    # "auto" resolves through compile.available_engine(): tectonic (bundled/self-
    # contained) first, then a system xelatex/pdflatex/latexmk. Pin a concrete engine
    # to override. The packaged .exe relies on "auto" finding the bundled Tectonic.
    latex_engine: str = "auto"               # "auto" | "tectonic" | "xelatex" | "pdflatex" | "latexmk"
    latex_template_dir: Path = REPO_ROOT / "src" / "core_engine" / "report" / "templates"
    output_dir: Path = REPO_ROOT / "output"


@lru_cache
def get_settings() -> Settings:
    return Settings()
