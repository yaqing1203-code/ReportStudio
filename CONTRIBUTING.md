# Contributing to ReportStudio

## Set up a development environment

Fork the repository, clone your fork, and run the commands from the repository root. Use Git and Python 3.11 or later.

```sh
python -m venv .venv
```

Activate the environment in PowerShell.

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS, activate it in a compatible shell.

```sh
source .venv/bin/activate
```

Install the desktop application and development dependencies.

```sh
python -m pip install -e ".[app,dev]"
```

Run `python -m core_engine.app.shell` to open the desktop application for a manual check. The frontend in `web/` loads directly and requires no asset build.

## Run the checks

```sh
python -m pytest tests/ -q
python -m ruff check src tests
python -m mypy src/core_engine --ignore-missing-imports
```

CI treats the type check as advisory. For frontend changes, install Node.js and run the browser-independent regression tests.

```sh
node --test tests/frontend_render.test.cjs tests/frontend_theme.test.cjs
```

For changes to retrieval, evidence, or report generation, run the pinned evaluations used by CI.

```sh
python -m core_engine.report.quality_eval --mode check
python -m core_engine.report.quality_eval --mode baseline --output output/quality-ci.json
python -m core_engine.report.research_eval --output output/research-quality-ci.json
python -m core_engine.report.quality_eval --mode baseline --dataset evals/niche_quality/cases.json --output output/quality-niche-ci.json
```

Put regression tests in `tests/` and use the AnyIO pytest marker for async cases. Use fixtures or mocked providers for automated tests. For evidence changes, keep source attribution, units, experimental conditions, and explicit gaps in the expected results.

## Build the Windows application

Run the packaging commands on Windows with the virtual environment active.

```powershell
python -m pip install -e ".[app,build]"
python packaging/build.py
```

The executable is `dist/ReportStudio/ReportStudio.exe`. Keep the entire `dist/ReportStudio/` folder together when sharing a build.

You can request a Tectonic download as part of the build.

```powershell
python packaging/build.py --fetch-tectonic
```

## Submit a change

1. Create a branch with a descriptive name and keep the pull request focused on one change.
2. Describe the problem and the resulting behavior. Add a short reproduction or screenshots when they help a reviewer verify the change.
3. Report the checks you ran and any remaining limitations. Include platform and Python versions for platform-specific issues.
4. Open a pull request against this repository's default branch. Keep credentials, private source material, local logs, and generated reports out of the diff.

For a bug report, include the steps to reproduce it, the expected result, the actual result, and the affected commit. Share a redacted traceback when available. Contact a maintainer privately for security vulnerabilities.

Contributions are covered by the project's MIT license. Preserve source and license notices when adding third-party fixtures. See [LICENSE](LICENSE).
