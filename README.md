# ReportStudio
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

A desktop application that transforms a research topic into a **verified, professionally typeset PDF report** with knowledge-graph-driven analysis and deep insights. Built with a correctness-first philosophy: the pipeline either produces a fully verified report or refuses to generate one.

## 🎯 Overview

ReportStudio combines intelligent web scraping, strict source verification, knowledge graph synthesis, and professional LaTeX typesetting into a single desktop application. It operates in two modes:

- **Chat Mode** — Instant conversational responses for questions, guidance, and quick queries
- **Report Mode** — Full research pipeline with multi-source verification and PDF generation

```text
topic → search router (query-type backend chain + disk cache)
      → STRICT filter → fetch → extract claims (structured slots)
      → VERIFY (aligned: entity clustering + isolated-claim review)
      → [optional HITL confirmation gate] → [scope gate]
      → synthesize KG + deep-dive analysis → assemble (5 + 2 sections)
      → LaTeX (TikZ/pgfplots/booktabs/comparison matrix) → compile PDF
```

## ✨ Key Features

### Verification Harness
Every claim passes through a configurable verification gate:
- **`aligned`** (default) — Entity alignment + fusion: structured claims are clustered by (entity, attribute), intra-cluster pairs are classified as corroborate/complement/conflict, and isolated single-source claims get a dedicated review
- **`conflict_only`** — Fast path: accept all claims, resolve only direct contradictions
- **`cross_reference`** — Legacy strict model: every claim traced against its sources and the whole pool
- **`traceable` / `strict` verification modes** — Whether paraphrase support suffices or verbatim spans are mandatory
- **Out-of-scope fallback** — Pipeline halts rather than guessing when insufficient sources found

### Search Router
No single search backend serves every query shape well, so each query is classified and routed:
- **Six query types** — `fact`, `deep_research`, `chinese`, `structured`, `academic`, `general`
- **Rule short-circuit + LLM tiebreak** — deterministic rules classify first; the LLM only breaks ties and any failure degrades gracefully to `general`
- **Backend fallback chains** — e.g. `structured` → Firecrawl → Tavily → DuckDuckGo; an unavailable backend falls through to the next entry
- **On-disk search cache** — TTL'd JSON cache (default 24 h) keyed by backend + query, atomic writes

See [Search Router](#-search-router) below for the full routing table.

### Source Credibility Grading (L1–L4)
Beyond the allowlist gate, every kept source is graded L1–L4 and that grade drives ranking, conflict resolution, and isolated-claim review:

| Level | Weight | Definition | Typical sources | Handling |
|-------|--------|------------|-----------------|----------|
| **L1** | 1.0 | Authoritative anchor | Government, regulators, standards bodies, top academic venues | Ranking anchors; never outranked by weak sources |
| **L2** | 0.8 | Professional endorsement | Leading media, investment banks / research houses, think tanks, user-provided documents | High-trust; wins conflicts vs. L3/L4 |
| **L3** | 0.5 | Industry consensus | Trade media, expert interviews; default for unrecognized-but-allowed sites | Single-source L3 claims are marked isolated and reviewed |
| **L4** | 0.2 | Weak signal | Personal blogs, anonymous forums, unattributed material | Never ranked ahead of L1/L2; dagger (†) marker in the comparison matrix |

Kept sources are ordered by `relevance × 0.5 + credibility × 0.35 + freshness × 0.15` (weights tunable via `CE_RANK_W_*`).

### Report Structure (5 Mandated Sections + Synthesis Layer)
Every report follows a standardized structure backed by knowledge graphs:
1. **Industry Overview** — Comprehensive sector analysis
2. **Policy Analysis** — Regulatory landscape and implications
3. **Industry Chain Map** — TikZ flow diagram of supply relationships
4. **Market Size** — pgfplots charts with historical data + TAM/SAM/SOM tables
5. **Competitive Landscape** — booktabs tables + market share visualizations

When `CE_ENABLE_SYNTHESIS_SECTIONS=true` (default) two more sections are appended:
6. **Source Comparison & Divergence** — LLM-drafted analysis of where sources agree/disagree, backed by a **booktabs comparison matrix** (one row per (entity, attribute), one column per source; L4-backed cells carry a † marker). Only emitted when alignment clusters exist.
7. **Information Limitations** — Deterministic closing section stating what the available sources could and could not establish.

`CE_REPORT_LOCALE=zh` localizes all headings, the title, and the out-of-scope message, and switches the LaTeX preamble to `ctex`.

### Isolated-Claim (孤证) Handling
A claim backed by a *single* source at credibility L3/L4 is flagged **isolated** and gets four layers of scrutiny (all fail-open):
1. **Internal consistency** — self-contradiction check within the claim
2. **Baseline deviation** — an LLM-scored deviation from domain common sense; above `CE_ALIGNMENT_DEVIATION_THRESHOLD` (default 0.6) the claim is downgraded one credibility level
3. **Provenance / timeliness note** — a source note recording where the claim comes from and any timeliness or motivation caveats
4. **Active verification** (optional, `CE_ACTIVE_VERIFY_ENABLED`, default on) — up to `CE_ACTIVE_VERIFY_MAX_QUERIES` (default 3) proxy-metric queries are generated and re-searched through the router in ONE bounded round; corroborating hits can lift the isolated flag

Isolated claims that survive are rendered with hedged language: claim lines marked `[UNVERIFIED - single source]` force explicit attribution ("According to a single source, …") in the drafted prose.

### Human-in-the-Loop Review (optional)
Set `CE_HITL_ON_ISOLATED_CORE_CLAIM=true` to pause the pipeline after verification whenever at least one **core isolated claim** survived (single source, credibility L3 or stronger):

- The run emits an `awaiting_confirmation` SSE event carrying a JSON summary of the claims, and the UI shows a confirmation card.
- A human decides via `POST /api/jobs/{job_id}/confirm` with `{"decision": "continue" | "abort"}`.
- The gate times out after 600 s and defaults to **continue**; **abort** halts the run with status `BLOCKED`.
- Default is **off** — runs proceed unattended.

### Desktop Application
- **Native UI** — pywebview-based chat interface
- **Zero-setup PDF** — Bundled Tectonic LaTeX compiler (no system TeX required)
- **Keyless search** — DuckDuckGo integration (no API key needed)
- **Loopback session token** — per-launch random token guards `/api/*` against other local processes
- **Persistent settings** — API configurations saved locally
- **Real-time progress** — SSE streaming of pipeline stages

## 🚀 Quick Start

### Running from Source

```bash
# Install dependencies
pip install -e ".[app]"

# Launch the desktop app
python -m core_engine.app.shell
```

### Configuration

Set API keys via the Settings panel in the UI, or use environment variables:

```bash
# LLM Provider (Anthropic or OpenAI-compatible)
export CE_LLM_PROVIDER=anthropic
export CE_ANTHROPIC_API_KEY=sk-...

# Search Provider (optional - defaults to keyless DuckDuckGo)
export CE_SEARCH_PROVIDER=tavily
export CE_SEARCH_API_KEY=tvly-...

# Verification mode
export CE_VERIFY_MODE=traceable  # 'traceable' (default) or 'strict'
```

See [.env.example](.env.example) for the full, commented variable list.

## 📦 Building from Source

### Prerequisites

```bash
# Create Python 3.11 environment
conda create -n py311 python=3.11 -y
conda activate py311

# Install dependencies (run from the repository root — it IS the project root;
# the package lives under src/core_engine/)
pip install -e ".[app]"
pip install pyinstaller
```

### Optional: Bundle Tectonic for Zero-Setup PDFs

1. Download Tectonic from https://github.com/tectonic-typesetting/tectonic/releases/latest
2. Place as `packaging/bin/tectonic.exe` (~48 MB)
3. The build script will automatically bundle it

### Build the Executable

```bash
# Automated build with cleanup
./build.sh

# Or manual build
python -m PyInstaller packaging/ReportStudio.spec --clean --noconfirm
```

**Output:** `dist/ReportStudio/ReportStudio.exe` (134 MB with Tectonic)

## 🏗️ Architecture

### Project Structure

```
src/core_engine/
├── report/                      # Report generation pipeline (active)
│   ├── models.py               # Data models for all stages (incl. CredibilityLevel L1-L4)
│   ├── sources.py              # Strict source allowlist + credibility_for() grading
│   ├── scrape.py               # Search & fetch (DuckDuckGo/Tavily/Firecrawl/Jina/fake)
│   ├── router.py               # Search Router: query-type classification + backend chains
│   ├── search_cache.py         # On-disk TTL cache for search results (atomic writes)
│   ├── llm.py                  # LLM operations (Anthropic/OpenAI) + hedging rules
│   ├── verify.py               # Verification harness (aligned / conflict_only / cross_reference)
│   ├── kg.py                   # Knowledge graph synthesis
│   ├── charts.py               # LaTeX chart generation (incl. comparison matrix)
│   ├── pipeline.py             # Orchestration, gates, HITL ConfirmationGate
│   ├── latex.py                # LaTeX rendering with Jinja2 (collision-proof filenames)
│   ├── compile.py              # PDF compilation (Tectonic/xelatex)
│   ├── cli.py                  # CLI entry: `python -m core_engine.report.cli`
│   ├── documents.py            # Uploaded-document parsing (PDF/DOCX/XLSX)
│   └── templates/report.tex.j2 # LaTeX template
├── app/                         # Desktop application (pywebview + FastAPI)
│   ├── shell.py                # Entry point + pywebview window (issues session token)
│   ├── server.py               # FastAPI backend (loopback token middleware, HITL confirm API)
│   ├── runtime.py              # Settings persistence
│   ├── intent.py               # Chat/report routing
│   ├── conversation.py         # Chat-mode conversation state
│   ├── database.py             # JSON-file session store
│   └── history.py              # Report run history (atomic writes)
├── agents/                      # Agent abstractions (base classes)
├── gateway/                     # Tool router (LLM tool-calling layer)
├── ontology/                    # YAML schema loader
│                              # (dormant data-agent layer — see below)
├── retrieval/                   # Reranking + retrieval helpers
├── security/                    # Context guards for untrusted source text
├── stores/                      # Vector / graph / Postgres backends
│   ├── graph.py                #   Apache AGE knowledge graph
│   ├── vector.py               #   pgvector embeddings
│   └── postgres.py             #   SQLAlchemy + psycopg
├── config.py                    # pydantic-settings configuration (single source of truth)
└── embeddings.py                # fastembed (BGE-M3) wrapper

adapters/default/                # Domain config (ontology + agent YAMLs)
├── ontology.yaml
└── agents/
    ├── compliance_checker.yaml
    ├── critic.yaml
    ├── data_query.yaml
    └── report_gen.yaml

db/init/                         # Postgres init SQL (runs on first docker-compose up)
├── 001_extensions.sql           #   pgvector + Apache AGE
├── 002_chunks.sql               #   source-chunks table
└── 003_rls.sql                  #   row-level security policies

web/                             # Frontend UI (vanilla, no build step)
├── index.html                  # Chat interface
├── app.js                      # Client logic (incl. HITL confirmation card)
└── styles.css                  # Light theme styling

packaging/                       # Build system
├── ReportStudio.spec           # PyInstaller spec
├── build.py                    # Build wrapper (optional --fetch-tectonic)
└── make_icon.py                # Generates reportstudio.ico

scripts/                         # Developer utilities
└── network_diag.py             # Connectivity diagnostics (search backends, LLM endpoints)

tests/                           # Test suite (pytest + anyio, fully offline by default)
├── test_report_pipeline.py     # Gates, LaTeX escaping, end-to-end happy path
├── test_search_router.py       # Query classification + backend fallback chains
├── test_search_cache.py        # Disk cache TTL / corruption / atomicity
├── test_credibility.py         # L1-L4 grading + weighted ranking
├── test_alignment.py           # Aligned strategy: clustering, tri-classification, 孤证 review
├── test_comparison_report.py   # Comparison matrix + synthesis sections
├── test_hitl.py                # ConfirmationGate + confirm endpoint
├── test_report_session_wiring.py
├── test_database_mode.py
├── test_database_store.py
├── test_history_store.py
└── test_wiring.py
```

> **Dormant layer:** `agents/`, `gateway/`, `ontology/`, `retrieval/`, `security/`, `stores/`,
> and `embeddings.py` are leftovers from the original data-agent project (KG + RAG over
> Postgres). They are kept on disk for reference but are not invoked by the report
> pipeline. Install the `data-agent` extra (`pip install -e ".[data-agent]"`) only if you
> want to revive them.

### Dual-Mode Operation

**Intent Classification:** LLM-based routing determines user intent

```
User Message
    ↓
Intent Classifier
    ↓
    ├─→ GENERAL_CHAT: "Hello", "What can you do?"
    │   → Direct LLM response (instant, $0.001)
    │
    └─→ REPORT_GENERATION: "Generate a report on X"
        → Full pipeline (30-120s, $0.15-0.30)
```

## 🔎 Search Router

When `CE_SEARCH_ROUTER_ENABLED=true` (default) and no provider was explicitly injected, `gather_sources` routes each query through `report/router.py`:

**Classification** — rules short-circuit first (cheap, deterministic); when no rule fires an LLM classifier breaks the tie, and any failure degrades to `general`:

| QueryType | Triggered by | Backend chain |
|-----------|--------------|---------------|
| `fact` | `site:.gov` / `site:.edu` / official-source markers | configured provider → tavily → duckduckgo |
| `deep_research` | "对比", "深度", "in-depth", "analysis", … | configured provider → tavily → duckduckgo |
| `chinese` | >30 % CJK characters in the query | **jina** → tavily → duckduckgo |
| `structured` | "表格", "营收", "市场规模", "table", "revenue", … | **firecrawl** → tavily → duckduckgo |
| `academic` | "paper", "study", "arxiv", "论文", … | configured provider → tavily → duckduckgo |
| `general` | fallback when nothing matched | configured provider → tavily → duckduckgo |

**Fallback semantics** — a backend that can't be constructed (missing API key) is skipped; a backend raising `SearchUnavailableError` (auth failure, blocked, timeout after retries) falls through to the next chain entry. If every entry fails, `SearchUnavailableError` is raised naming all attempts. The registry records which backend actually served each query (`routes` diagnostics).

**Caching** — `search_cache.py` stores hit lists as JSON files under `user_data_dir()/search_cache/`, keyed by `sha256(backend + query)[:16]`, TTL `CE_SEARCH_CACHE_TTL_S` (default 24 h, `0` disables). Writes are atomic (tmp file + `os.replace`); corrupt or expired entries read as a miss. Cache failures never break a search.

**Backends** — instantiated by name via `scrape.get_search_provider(name)`:

| Name | Key | Notes |
|------|-----|-------|
| `duckduckgo` | none | Keyless, zero-setup default |
| `tavily` | `CE_SEARCH_API_KEY` | Higher-quality search API |
| `firecrawl` | `CE_FIRECRAWL_API_KEY` | Extraction-oriented; preferred for `structured` queries |
| `jina` | `CE_JINA_API_KEY` (optional) | Works keyless with tighter rate limits; preferred for `chinese` queries |
| `fake` | none | Deterministic offline fixtures (tests) |

Set `CE_SEARCH_ROUTER_ENABLED=false` to pin the legacy single-provider behavior.

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CE_VERIFY_STRATEGY` | `aligned` | Verification strategy (`aligned`, `conflict_only`, `cross_reference`) |
| `CE_VERIFY_MODE` | `traceable` | Verification mode (`traceable` or `strict`) |
| `CE_VERIFY_ROUNDS` | `3` | Triple-check rounds per claim (legacy) |
| `CE_MIN_SOURCES_PER_CLAIM` | `1` | Distinct authoritative domains per claim |
| `CE_MIN_AUTHORITATIVE_SOURCES` | `2` | Minimum total authoritative sources |
| `CE_MIN_VERIFIED_CLAIMS` | `3` | Threshold for out-of-scope fallback |
| `CE_SEARCH_ROUTER_ENABLED` | `true` | Route queries through the Search Router |
| `CE_SEARCH_CACHE_TTL_S` | `86400` | On-disk search cache TTL (`0` disables) |
| `CE_FIRECRAWL_API_KEY` | — | Firecrawl backend key |
| `CE_JINA_API_KEY` | — | Jina Reader key (optional; keyless works) |
| `CE_SCRAPE_ROBOTS_FAIL_OPEN` | `true` | When robots.txt can't be fetched: allow (true) or skip (false) the page |
| `CE_RANK_W_RELEVANCE` | `0.5` | Source ranking weight: relevance |
| `CE_RANK_W_SOURCE` | `0.35` | Source ranking weight: credibility (L1-L4) |
| `CE_RANK_W_FRESHNESS` | `0.15` | Source ranking weight: freshness |
| `CE_ALIGNMENT_DEVIATION_THRESHOLD` | `0.6` | Baseline-deviation score that downgrades an isolated claim |
| `CE_ACTIVE_VERIFY_ENABLED` | `true` | One bounded active-verification round for core isolated claims |
| `CE_ACTIVE_VERIFY_MAX_QUERIES` | `3` | Max proxy-metric re-search queries per run |
| `CE_HITL_ON_ISOLATED_CORE_CLAIM` | `false` | Pause for human confirmation on core isolated claims |
| `CE_ENABLE_SYNTHESIS_SECTIONS` | `true` | Append source_comparison + limitations sections |
| `CE_REPORT_LOCALE` | `en` | Report language (`en` or `zh`; `zh` uses ctex) |
| `CE_LLM_TIMEOUT_S` | `120` | Per-LLM call timeout |
| `CE_LLM_MAX_CONCURRENCY` | `4` | Global cap on concurrent LLM requests |
| `CE_VERIFY_DEADLINE_S` | `360` | Verification stage deadline |
| `CE_ASSEMBLE_DEADLINE_S` | `300` | Assembly stage deadline |
| `CE_LATEX_ENGINE` | `auto` | LaTeX compiler engine (`auto`/`tectonic`/`xelatex`/`pdflatex`/`latexmk`) |
| `CE_MULTI_QUERY_RESEARCH` | `true` | Section-targeted searches |
| `CE_MAX_DEEP_DIVES` | `8` | Maximum analytical deep-dive sections |
| `CE_APP_PORT` | random | Fixed port for the local backend (debugging) |
| `CE_APP_SESSION_TOKEN` | per-launch | Loopback session token (set by the shell; not a user setting) |

### Verification Strategies

**`aligned` (Default):**
- Accepts all claims up-front (like `conflict_only`), then adds an alignment/fusion layer
- Structured claims (entity/attribute/value/qualifier/time_scope slots, extracted when possible) are alias-normalized and clustered by (entity, attribute)
- Intra-cluster pairs are tri-classified: **corroborate** / **complement** / **conflict** (conflicts drop the lower-authority claim)
- Single-source L3/L4 claims are marked isolated and get the [isolated-claim review](#isolated-claim-孤证-handling)
- Cluster data feeds the source-comparison matrix in the report

**`conflict_only` (Fast Mode):**
- Accept claims by default
- Only detect and resolve contradictions (lexical prefilter + contradiction oracle)
- 50-100x faster than cross-reference verification
- Best for trusted source bases

**`cross_reference` (Legacy Strict):**
- Every claim is traced/entailed against its own sources and the whole pool, then tiered
- Stricter and slower; kept for backwards compatibility

**Verification modes** (orthogonal to strategy):
- **`traceable`** (default) — Claims must be judged fully supported by an authoritative source; paraphrase and synthesis allowed
- **`strict`** — Every claim requires verbatim spans from sources; no paraphrase allowed

## 📊 Report Output

### Generated Files

- **`output/<slug>-<hash8>.pdf`** — Final typeset report
- **`output/<slug>-<hash8>.tex`** — LaTeX source (for manual editing)

`slug` is derived from the topic and `hash8` is a short SHA-1 prefix of the raw topic, so two topics whose slugs collide (punctuation/casing differences, truncation) never overwrite each other.

### Report Sections

1. **Industry Overview** — Comprehensive sector introduction
2. **Policy Analysis** — Regulatory environment and implications
3. **Industry Chain Map** — TikZ visualization of supply relationships
4. **Market Size** — Historical data charts + TAM/SAM/SOM analysis
5. **Competitive Landscape** — Market share tables and visualizations
6. **Source Comparison & Divergence** — Comparison matrix + divergence analysis (when alignment clusters exist)
7. **Information Limitations** — Deterministic statement of what the sources could not establish

### LaTeX Features

- **TikZ flow diagrams** — Industry chain visualization
- **pgfplots charts** — Market size time series
- **booktabs tables** — Professional competitive analysis + source comparison matrix
- **XeLaTeX compilation** — Unicode support for international text (`ctex` when `CE_REPORT_LOCALE=zh`)
- **Automatic escaping** — Safe rendering of untrusted source text

## 🛠️ Build History & Fixes

### Major Updates

#### Python 3.11 Migration (v0.1)
- **Issue:** Python 3.13 DLL loading failures on Windows
- **Solution:** Rebuilt with Python 3.11.15 for stable PyInstaller support
- **Result:** Reliable executable with Visual C++ runtime bundled

#### Tectonic Integration (v0.3)
- **Feature:** Bundled LaTeX compiler for zero-setup PDF generation
- **Size:** 48 MB binary auto-included in build
- **Benefit:** No system TeX distribution required

#### Light Theme UI (v0.4)
- **Update:** Switched from dark theme to light theme
- **Colors:** Light gray background (#f5f5f5), dark text (#1a1a1a)
- **Result:** Improved readability and professional appearance

#### Dual-Mode Intelligence (v0.6)
- **Feature:** Intelligent chat/report routing with intent classification
- **Chat Mode:** Instant responses for greetings, questions, guidance
- **Report Mode:** Full pipeline for comprehensive research
- **Cost Optimization:** 50-100x cheaper for simple queries

#### Runtime Fixes (v0.7)
- **Fix:** stdout/stderr NoneType crash in frozen executables
- **Solution:** Log file redirection to `%LOCALAPPDATA%\reportstudio\sessions\`
- **Fallback:** Temp directory if primary location fails

#### Settings Persistence (v0.8)
- **Fix:** Settings not persisting after save
- **Solution:** Proper HTTP error checking and field retention
- **Enhancement:** Better error messages and console logging

#### Verification Optimization (v0.10)
- **Update:** Doubled all timeout values (120s/360s/300s)
- **Strategy:** Adopted `conflict_only` for faster processing (superseded in 0.2.0 — the default is now `aligned`)
- **Result:** 2x time budget, adaptive pair generation, fast-fail logic

#### Research-Quality Overhaul (v0.2.0)
- **Search Router** — Query-type classification + per-type backend fallback chains + disk cache
- **New backends** — Firecrawl and Jina join DuckDuckGo/Tavily
- **Credibility grading** — L1-L4 source levels drive ranking and conflict resolution
- **Aligned verification** — Entity clustering, isolated-claim (孤证) review, active re-verification
- **Synthesis layer** — Source-comparison matrix, limitations section, zh locale
- **HITL** — Optional human confirmation gate on core isolated claims
- **Hardening** — Loopback session token, atomic history writes, collision-proof output filenames, CI workflow

### Known Issues (Resolved)

| Issue | Status | Resolution |
|-------|--------|------------|
| Python 3.13 DLL loading | ✅ Fixed | Python 3.11 rebuild |
| Qt bindings conflict | ✅ Fixed | Excluded in spec |
| Tectonic --synctex error | ✅ Fixed | Removed incompatible option |
| stdout NoneType crash | ✅ Fixed | Log file redirection |
| Dark theme readability | ✅ Fixed | Light theme CSS |
| Build artifacts confusion | ✅ Fixed | Auto-cleanup script |
| Settings not persisting | ✅ Fixed | Error handling + field retention |
| LLM client import error | ✅ Fixed | Correct module path |
| Intent classification failure | ✅ Fixed | Proper LLM interface |
| Concurrent report overwrites | ✅ Fixed | Topic-hash filename suffix |
| Local API open to other processes | ✅ Fixed | Loopback session token middleware |

## 🧪 Testing

### Run the Test Suite

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run the full test suite (offline, no API keys needed for the default fake providers)
pytest -q tests/
```

### Manual Testing

**Chat Mode:**
```
"Hello" → Friendly greeting (instant)
"What can you do?" → Feature explanation
"Tell me about AI" → Concise overview
```

**Report Mode:**
```
"Generate a report on quantum computing" → Full pipeline
"Analyze cryptocurrency trends" → Status updates + PDF
"Research renewable energy" → Verified report download
```

### Debug Logs

**Location:** `%LOCALAPPDATA%\reportstudio\sessions\app.log`

**Check logs:**
```bash
# Windows PowerShell
Get-Content $env:LOCALAPPDATA\reportstudio\sessions\app.log -Tail 50

# Git Bash / WSL
tail -f "$LOCALAPPDATA/reportstudio/sessions/app.log"
```

**Network diagnostics** (connectivity to search backends / LLM endpoints):

```bash
python scripts/network_diag.py
```

## 📈 Performance & Cost

### Chat Mode
- **Latency:** <2 seconds
- **Tokens:** ~400-900 per message
- **Cost:** $0.001-0.003 per message

### Report Mode
- **Latency:** 30-120 seconds (with 2x timeout budget)
- **Tokens:** ~50,000-100,000 per report
- **Cost:** $0.15-0.30 per report (Anthropic Claude)

### Verification Strategies
- **cross_reference:** 3-5 minutes for 50 claims (strict)
- **conflict_only:** ~30 seconds for 50 claims (fast)
- **aligned:** ~1 minute for 50 claims (default; adds clustering + isolated-claim review on top of conflict_only)
- **traceable:** 1-2 minutes for 50 claims (balanced)

## 🔒 Security Notes

### LaTeX Rendering
- All source text is **untrusted** and fully escaped
- Custom Jinja delimiters (`<< >>`, `<% %>`) prevent injection
- LaTeX specials (`\ & % $ # _ { } ~ ^`) automatically escaped
- Compilation in temp directory with `-halt-on-error`

### API Keys
- Stored locally in `%LOCALAPPDATA%\reportstudio\settings.json`
- File permissions set to user-only
- Never logged or transmitted to unauthorized endpoints
- Password fields cleared after save in UI

### Local API
- The desktop shell generates a random **session token** per launch (`CE_APP_SESSION_TOKEN`)
- When set, every `/api/*` request must present it (`?token=` or `X-Session-Token` header); `/api/health` is exempt so the shell can probe readiness
- Static assets are exempt (`<script>`/`<link>` tags can't carry tokens); the protected surface is the state-mutating API
- When unset (CLI runs, tests) all requests pass — preserving the no-auth desktop default

### Source Filtering
- Strict allowlist prevents malicious content sources
- robots.txt compliance for ethical scraping (`CE_SCRAPE_ROBOTS_FAIL_OPEN` controls behavior when robots.txt itself can't be fetched)
- User-Agent identification for transparency

## 🎓 Advanced Usage

### Command-Line Interface

```bash
# Run pipeline from CLI (for automation/testing)
python -m core_engine.report.cli "your topic here"

# With custom settings
CE_VERIFY_MODE=strict \
CE_MIN_SOURCES_PER_CLAIM=3 \
python -m core_engine.report.cli "quantum computing"
```

### Programmatic API

```python
from core_engine.report.pipeline import ReportPipeline

async def generate_report(topic: str):
    pipeline = ReportPipeline()
    result = await pipeline.run(
        topic,
        on_progress=lambda msg: print(f"[{msg.stage}] {msg.status}")
    )
    
    if result.status == "completed":
        print(f"PDF: {result.pdf_path}")
        print(f"LaTeX: {result.latex_path}")
    else:
        print(f"Failed: {result.error}")
```

## 🤝 Contributing

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[app,dev]"

# Run tests
pytest tests/

# Run linting
ruff format src/
ruff check src/
mypy src/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide (ground rules, test conventions, style).

### Project Goals

- **Correctness over coverage** — Better to refuse than fabricate
- **Transparency** — Every claim traceable to authoritative sources
- **Professional output** — Publication-quality LaTeX typesetting
- **User experience** — Natural conversational interaction

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for the full text.

In short: you can use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the software, as long as the copyright notice and permission notice are kept.
The software is provided "as is", without warranty of any kind.

## 🆘 Troubleshooting

### Issue: Executable won't launch
**Check:** Windows SmartScreen (click "More info" → "Run anyway")  
**Check:** Antivirus false positives (whitelist ReportStudio.exe)

### Issue: PDF generation fails
**Check:** Log file for API key status: `%LOCALAPPDATA%\reportstudio\sessions\app.log`  
**Check:** Settings panel has valid API keys configured  
**Look for:** `[DEBUG] Has Anthropic Key: True` in logs

### Issue: Settings don't persist
**Check:** Browser console (F12) for JavaScript errors  
**Check:** File permissions on `%LOCALAPPDATA%\reportstudio\`  
**Try:** Run as administrator (temporary test)

### Issue: Verification timeout
**Solution:** Increase timeout: `CE_VERIFY_DEADLINE_S=600`  
**Or:** Switch to faster strategy: `CE_VERIFY_STRATEGY=conflict_only`

### Issue: Out of scope errors
**Cause:** Insufficient authoritative sources for topic  
**Solution:** Try a more mainstream topic or lower `CE_MIN_AUTHORITATIVE_SOURCES`

## 📞 Support

For issues and questions:
1. **Open an issue:** https://github.com/yaqing1203-code/ReportStudio/issues
   *(bug reports, feature requests, security — please don't email for these)*
2. Check the log file: `%LOCALAPPDATA%\reportstudio\sessions\app.log`
3. Look for `[DEBUG]` and `[ERROR]` messages
4. Review the troubleshooting section above
5. Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a PR

---

**Version:** 0.2.0  
**Build:** ReportStudio.exe (134 MB)  
**Last Updated:** 2026-07-17  
**Status:** Production Ready ✅
