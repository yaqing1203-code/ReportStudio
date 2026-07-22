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
topic → search → STRICT filter → fetch → extract claims
      → VERIFY (triple-check + cross-reference) → [scope gate]
      → synthesize KG + deep-dive analysis → assemble (5 sections)
      → LaTeX (TikZ/pgfplots/booktabs) → compile PDF
```

## ✨ Key Features

### Verification Harness
Every claim must pass a strict verification gate with configurable strategies:
- **`traceable`** (default) — Claims fully supported by authoritative sources with cross-domain corroboration
- **`strict`** — Every claim requires verbatim spans from authoritative sources
- **Triple-check** — Multiple independent verification rounds per claim
- **Cross-referencing** — Minimum distinct authoritative domains required
- **Out-of-scope fallback** — Pipeline halts rather than guessing when insufficient sources found

### Report Structure (5 Mandated Sections)
Every report follows a standardized structure backed by knowledge graphs:
1. **Industry Overview** — Comprehensive sector analysis
2. **Policy Analysis** — Regulatory landscape and implications
3. **Industry Chain Map** — TikZ flow diagram of supply relationships
4. **Market Size** — pgfplots charts with historical data + TAM/SAM/SOM tables
5. **Competitive Landscape** — booktabs tables + market share visualizations

### Source Quality Control
Strict allowlist-based filtering:
- ✅ Government agencies & IGOs
- ✅ Investment banks & research houses
- ✅ Academic institutions (.edu)
- ✅ Industry institutions & non-profit research orgs
- ✅ Mainstream media outlets
- ❌ Blogs, forums, social media, tabloids (rejected before download)

### Desktop Application
- **Native UI** — pywebview-based chat interface
- **Zero-setup PDF** — Bundled Tectonic LaTeX compiler (no system TeX required)
- **Keyless search** — DuckDuckGo integration (no API key needed)
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

## 📦 Building from Source

### Prerequisites

```bash
# Create Python 3.11 environment
conda create -n py311 python=3.11 -y
conda activate py311

# Install dependencies
cd core-engine
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
│   ├── models.py               # Data models for all stages
│   ├── sources.py              # Strict source allowlist
│   ├── scrape.py               # Search & fetch (DuckDuckGo/Tavily)
│   ├── llm.py                  # LLM operations (Anthropic/OpenAI)
│   ├── verify.py               # Verification harness (traceable / strict)
│   ├── kg.py                   # Knowledge graph synthesis
│   ├── charts.py               # LaTeX chart generation
│   ├── pipeline.py             # Orchestration & gates
│   ├── latex.py                # LaTeX rendering with Jinja2
│   ├── compile.py              # PDF compilation (Tectonic/xelatex)
│   ├── cli.py                  # CLI entry: `python -m core_engine.report.cli`
│   ├── documents.py            # Uploaded-document parsing (PDF/DOCX/XLSX)
│   └── templates/report.tex.j2 # LaTeX template
├── app/                         # Desktop application (pywebview + FastAPI)
│   ├── shell.py                # Entry point + pywebview window
│   ├── server.py               # FastAPI backend
│   ├── runtime.py              # Settings persistence
│   ├── intent.py               # Chat/report routing
│   ├── conversation.py         # Chat-mode conversation state
│   ├── database.py             # SQLite-backed session store
│   └── history.py              # Report run history
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
├── config.py                    # pydantic-settings configuration
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
├── app.js                      # Client logic
└── styles.css                  # Light theme styling

packaging/                       # Build system
├── ReportStudio.spec           # PyInstaller spec
├── build.py                    # Build wrapper (optional --fetch-tectonic)
└── make_icon.py                # Generates reportstudio.ico

tests/                           # Test suite (pytest + anyio)
├── test_report_pipeline.py
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

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CE_VERIFY_MODE` | `traceable` | Verification strategy (`traceable` or `strict`) |
| `CE_VERIFY_ROUNDS` | `3` | Triple-check rounds per claim |
| `CE_MIN_SOURCES_PER_CLAIM` | `2` | Distinct authoritative domains per claim |
| `CE_MIN_AUTHORITATIVE_SOURCES` | `3` | Minimum total authoritative sources |
| `CE_MIN_VERIFIED_CLAIMS` | `5` | Threshold for out-of-scope fallback |
| `CE_LLM_TIMEOUT_S` | `120` | Per-LLM call timeout |
| `CE_VERIFY_DEADLINE_S` | `360` | Verification stage deadline |
| `CE_ASSEMBLE_DEADLINE_S` | `300` | Assembly stage deadline |
| `CE_LATEX_ENGINE` | `xelatex` | LaTeX compiler engine |
| `CE_MULTI_QUERY_RESEARCH` | `true` | Section-targeted searches |
| `CE_MAX_DEEP_DIVES` | `5` | Maximum analytical deep-dive sections |

### Verification Strategies

**`traceable` (Recommended Default):**
- Claims must be fully supported by authoritative sources
- Cross-domain corroboration required
- Paraphrase and synthesis allowed (not verbatim quotes)
- Faster processing, suitable for most use cases

**`strict` (Maximum Rigor):**
- Every claim requires verbatim spans from sources
- No paraphrase allowed
- Slower but highest verification standard

**`conflict_only` (Fast Mode):**
- Accept claims by default
- Only detect and resolve contradictions
- 50-100x faster than cross-reference verification
- Best for trusted source bases

## 📊 Report Output

### Generated Files

- **`output/<topic>_report.pdf`** — Final typeset report
- **`output/<topic>_report.tex`** — LaTeX source (for manual editing)

### Report Sections

1. **Industry Overview** — Comprehensive sector introduction
2. **Policy Analysis** — Regulatory environment and implications
3. **Industry Chain Map** — TikZ visualization of supply relationships
4. **Market Size** — Historical data charts + TAM/SAM/SOM analysis
5. **Competitive Landscape** — Market share tables and visualizations

### LaTeX Features

- **TikZ flow diagrams** — Industry chain visualization
- **pgfplots charts** — Market size time series
- **booktabs tables** — Professional competitive analysis
- **XeLaTeX compilation** — Unicode support for international text
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
- **Strategy:** Default to `conflict_only` for faster processing
- **Result:** 2x time budget, adaptive pair generation, fast-fail logic

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
- **conflict_only:** ~30 seconds for 50 claims (fast, default)
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

### Source Filtering
- Strict allowlist prevents malicious content sources
- robots.txt compliance for ethical scraping
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
black src/
mypy src/
```

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

**Version:** v0.10  
**Build:** ReportStudio.exe (134 MB)  
**Last Updated:** 2026-07-17  
**Status:** Production Ready ✅
