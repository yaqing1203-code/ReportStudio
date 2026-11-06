# ReportStudio

ReportStudio is a Windows desktop workspace for chat, research reports, and long-form writing. It can research a topic, assemble a report from cited evidence, and export the result as a PDF. The desktop interface is currently in Chinese. Reports can use English or Chinese headings.

## Choose a mode

| Mode | Use it for |
| --- | --- |
| Chat | Ask questions or discuss ideas in a conversation. |
| Research | Generate a source-cited report. Attach local documents to create a report from those materials. |
| Writing | Draft or revise a complete article from a brief, outline, or notes. Optional web research supplies background. |

Automatic routing chooses a mode from the prompt. Research and Writing also offer Standard and Deep settings. Deep allows more research or drafting work within the configured limits. The workspace supports light and dark themes and keeps a local history.

Research attachments can be PDF, Word, Excel, CSV, plain text, or Markdown files.

Research and Writing can export to an uploaded `.docx` template. Start with the example template available in the app. Put `{{content}}` or `{{section:Section name}}` in its own paragraph. Template-based PDF export uses LibreOffice or Microsoft Word on Windows. The default research PDF uses LaTeX.

## Get started

### Run from source

Install Git and Python 3.11 or later. Run these commands in PowerShell.

```powershell
git clone https://github.com/taishanglaoxu/ReportStudio.git
Set-Location ReportStudio
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[app]"
.\.venv\Scripts\python.exe -m core_engine.app.shell
```

The desktop window uses Microsoft Edge WebView2 on Windows. Install a supported LaTeX engine such as [Tectonic](https://tectonic-typesetting.github.io/) or XeLaTeX for default research PDF export.

### Use a Windows portable package

When a Windows x64 ZIP is available on [Releases](https://github.com/taishanglaoxu/ReportStudio/releases), extract the entire archive and run `ReportStudio.exe` from the extracted `ReportStudio` folder. The package requires WebView2. Bundled Tectonic may download LaTeX resources on the first compilation.

The portable build uses pypdf for PDF text extraction. The PyMuPDF fallback for complex layouts and native PDF table extraction are available in source installations. HTML and JATS table extraction are available in both builds.

## Configure a provider

Open the settings panel to choose a model and enter its credentials. Select Anthropic or an OpenAI-compatible endpoint and use a model name supported by that service. DuckDuckGo is the default search provider. Tavily requires a search key. OpenAlex can supplement scholarly and evidence-gap queries.

For the command-line interface, copy [.env.example](.env.example) to `.env` and fill in the settings for your providers. Desktop settings are stored separately and applied at launch.

| Setting | Purpose |
| --- | --- |
| `CE_LLM_PROVIDER` and `CE_LLM_MODEL` | Choose the model provider and model. |
| `CE_ANTHROPIC_API_KEY` | Authenticate to Anthropic. |
| `CE_LLM_BASE_URL` and `CE_LLM_API_KEY` | Configure an OpenAI-compatible endpoint. |
| `CE_SEARCH_PROVIDER` and `CE_SEARCH_API_KEY` | Select a search backend and provide a key when it requires one. |
| `CE_OPENALEX_SEARCH_ENABLED` and `CE_OPENALEX_API_KEY` | Configure complementary scholarly search. |
| `CE_REPORT_LOCALE` | Choose English (`en`) or Chinese (`zh`) report headings and typesetting. |
| `CE_LATEX_ENGINE` | Select a LaTeX engine or let the app choose one. |

Keep `.env`, local settings, and generated research outputs out of public commits.

## Research output and evidence

The report pipeline plans research questions, gathers sources, and searches for missing evidence within its budgets. It checks draft claims against saved source passages. Reports distinguish attributed statements, source disagreements, and unresolved gaps.

Evidence packages record source locations, numeric observations, and calculation inputs. ReportStudio can link conductivity measurements in structured JATS research articles to applicable Methods passages. Comparisons use explicit units, measurement scope, and experimental conditions. Review the cited sources and stated limitations before using findings. Writing mode checks for internal inconsistencies and treats supplied material as writing premises.

To generate a report from the repository root after configuring providers, run the following command.

```powershell
.\.venv\Scripts\python.exe -m core_engine.report.cli "Solid-state battery electrolytes"
```

Add `--no-pdf` to render LaTeX without compiling a PDF. The CLI writes to `output/` by default. The desktop app stores settings, history, and reports under `%LOCALAPPDATA%\CoreEngineReports` on Windows. The packaged app's startup log is under `sessions/app.log` in that directory.

## Build the Windows package

Use a Windows x64 Python environment. Activate it so the build helper can find PyInstaller.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[app,build]"
python packaging/build.py --fetch-tectonic
```

Run `dist/ReportStudio/ReportStudio.exe` to verify the build. A portable ZIP should contain the complete `dist/ReportStudio/` folder and the applicable license notices.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, test commands, and pull request checklist.

This fork builds on [yaqing1203-code/ReportStudio](https://github.com/yaqing1203-code/ReportStudio). The source is distributed under the [MIT License](LICENSE).
