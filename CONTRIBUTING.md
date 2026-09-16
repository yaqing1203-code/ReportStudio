# Contributing to ReportStudio

Thanks for your interest in making ReportStudio better! 🎉
This guide covers everything you need to send a clean PR.

> **TL;DR** — Fork → branch → `pip install -e ".[app,dev]"` → make changes → add tests →
> `pytest -q` → `ruff check` → open a PR against `main`.

---

## 🧭 Ground Rules

1. **Correctness over features.** The pipeline's whole point is "verified or refuse".
   Don't loosen the verification harness to make a claim pass — fix the source instead.
2. **Be transparent.** Every change should be motivated in the PR description:
   *what* you changed, *why*, and *how* you tested it.
3. **Stay in scope.** One PR = one logical change. Don't bundle refactors with bug fixes.
4. **No drive-by formatting.** A PR that only reformats files is hard to review. Keep
   formatting changes inside the PR that legitimately needs them.

---

## 🛠️ Development Setup

### Prerequisites

- **Python 3.11+** (3.11.15 recommended — matches the build environment)
- **Git**
- **Optional:** Docker (only if you want to test the data-agent layer with Postgres)

### Install

```bash
# Clone your fork
git clone https://github.com/yaqing1203-code/ReportStudio.git
cd ReportStudio

# Create a clean environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install with the app + dev extras
pip install -e ".[app,dev]"
```

> **Heads up:** The old README sometimes lists `.[dev,report,app]`. There is **no
> `report` extra** in `pyproject.toml` — the report pipeline ships in the base install.
> Use `.[app,dev]`. If you also build executables, add `,build`.

### Optional: bundle Tectonic for local PDF testing

The default install relies on whatever LaTeX engine is on your `PATH` (`xelatex`).
To use the zero-setup bundled Tectonic:

```bash
python packaging/build.py --fetch-tectonic
# Tectonic lands in packaging/bin/tectonic(.exe) and is auto-detected
```

---

## 📁 Project Layout

```
ReportStudio/
├── src/core_engine/         # All Python source
│   ├── report/              # Active report pipeline (CLI + library)
│   ├── app/                 # Desktop shell (pywebview + FastAPI)
│   ├── agents/              # Agent abstractions
│   ├── gateway/             # Tool router
│   ├── ontology/            # YAML schema loader
│   ├── retrieval/           # Reranking + retrieval helpers
│   ├── security/            # Context guards
│   ├── stores/              # Vector / graph / Postgres backends (dormant)
│   ├── config.py            # pydantic-settings configuration
│   └── embeddings.py        # fastembed wrapper (dormant)
├── web/                     # Frontend (vanilla HTML/CSS/JS, no build step)
├── adapters/default/        # Domain config (ontology + agent YAMLs)
├── db/init/                 # Postgres init SQL (run on first docker-compose up)
├── packaging/               # Build system (PyInstaller spec + helpers)
├── tests/                   # pytest suite (anyio for async)
├── docker-compose.yml       # Optional Postgres with AGE + pgvector
├── pyproject.toml
└── README.md
```

> The `agents/`, `gateway/`, `ontology/`, `retrieval/`, `security/`, `stores/` modules
> are the dormant data-agent layer. **Don't break them when refactoring the report
> pipeline** — they're kept around in case someone wants to revive the KG/RAG features.

---

## 🌿 Branching & Commits

- Branch off `main`. Use a descriptive prefix:
  - `feat/...` — new feature
  - `fix/...` — bug fix
  - `docs/...` — docs only
  - `refactor/...` — no behavior change
  - `test/...` — tests only
  - `chore/...` — tooling, CI, deps

- Commit messages: imperative mood, ≤ 72 chars on the subject line.

  ```
  fix: cap retry budget in scrape fallback
  docs: clarify CE_VERIFY_MODE default in .env.example
  ```

  A short body is welcome for non-trivial changes ("why" not "what").

---

## ✅ Before You Open a PR

Run these from the repo root and make sure they all pass:

```bash
# Tests (offline, no API keys needed for the default suite)
pytest -q

# Lint
ruff check src tests

# Format (auto-fix)
ruff format src tests

# Type-check (advisory — warnings won't block but please address)
mypy src
```

If your change touches LaTeX templates or the verification pipeline, also run the
**smoke test** in `tests/test_report_pipeline.py` with the fake LLM/search provider
to confirm a full happy-path run still completes:

```bash
CE_LLM_PROVIDER=fake CE_SEARCH_PROVIDER=fake pytest -q tests/test_report_pipeline.py
```

---

## 🧪 Writing Tests

- Tests live in `tests/` and are plain `pytest` files. Async tests use
  `pytestmark = pytest.mark.anyio` (NOT `pytest-asyncio`).
- Prefer **offline** tests with `CE_LLM_PROVIDER=fake` and `CE_SEARCH_PROVIDER=fake`
  so CI doesn't burn API credits.
- For unit tests, target the smallest interesting function or class.
- For integration tests that need Postgres, gate them behind a marker so they're
  easy to skip locally:

  ```python
  import pytest

  pytestmark = pytest.mark.skipif(
      not os.getenv("DATABASE_URL"),
      reason="needs DATABASE_URL (docker-compose up postgres first)",
  )
  ```

- **Name tests after behavior**, not implementation:
  - ✅ `test_verification_rejects_unsupported_claim`
  - ❌ `test_verify_function_3rd_assertion`

---

## 🎨 Style

- **Formatter:** `ruff format` (line-length 100, see `pyproject.toml`).
- **Linter:** `ruff check` — please don't disable rules without a comment explaining why.
- **Type hints:** encouraged on public functions. `mypy` is advisory, not blocking.
- **Imports:** prefer `from __future__ import annotations` at the top of new modules.
- **Naming:** snake_case modules & functions, PascalCase classes, UPPER_CASE constants.

---

## 🐛 Reporting Bugs

Open an issue with:

1. **What you ran** (the exact command or chat message).
2. **What you expected.**
3. **What happened.** Include the full traceback if there is one.
4. **Environment:** OS, Python version, commit SHA, `pip freeze` excerpt.

The log file at `%LOCALAPPDATA%\reportstudio\sessions\app.log` (or
`$LOCALAPPDATA/reportstudio/sessions/app.log` on Git Bash) is the first place to look.

---

## 💡 Suggesting Features

Open an issue tagged `enhancement`. Tell us:

- The **user story** ("As a researcher, I want … so that …").
- The **acceptance criteria** — how do we know it works?
- Any **trade-offs** you've thought about.

Features that loosen verification defaults need a stronger bar than usual — see
*Ground Rules #1*.

---

## 🔐 Security Issues

**Do not open a public issue for security bugs.** Email the maintainer directly (see
README) and we'll coordinate a fix and disclosure.

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the
**MIT License** — the same license as the rest of the project. See `LICENSE` for
the full text.

---

Welcome aboard, and thanks for helping make ReportStudio better 🙏
