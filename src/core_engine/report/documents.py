"""Document parsing for Database Mode — user-uploaded files -> plain text.

A user can inject their own articles/documents into a report's database. Those
files arrive as raw bytes with a filename; this module turns them into the plain
text the pipeline already understands (claim extraction reads `Source.text`).

Design rules:
  - Pure functions over (filename, bytes) — no disk I/O, no network. The caller
    (server) decides where anything is stored.
  - Parsers are imported LAZILY inside each branch so a missing optional dependency
    only affects the one format that needs it, never the whole app.
  - Untrusted input: text is length-capped and control chars are stripped. The LaTeX
    renderer additionally escapes everything, so parsed text can never inject commands.
  - Failure is explicit: an unreadable/empty/oversized file raises DocumentError with
    a user-facing reason rather than silently yielding empty text.
"""
from __future__ import annotations

import csv
import io
import zipfile
from itertools import islice
from pathlib import Path

# Formats we accept. Legacy binary .doc is intentionally NOT here — there is no
# reliable pure-Python parser bundled; we ask the user to convert to .docx.
SUPPORTED_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md", ".markdown", ".text",
})

# Hard ceiling on an uploaded file (bytes). Guards against a pathological upload
# exhausting memory during parse. 25 MB is generous for text-bearing documents.
MAX_FILE_BYTES = 25 * 1024 * 1024

# Cap on extracted text length (characters). The LLM only reads the first
# llm_context_chars anyway; this bounds storage + keeps the DB session small.
MAX_TEXT_CHARS = 200_000
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 2000
MAX_PDF_PAGES = 500
MAX_SHEET_ROWS = 20_000
MAX_SHEET_COLUMNS = 256


class DocumentError(Exception):
    """A user-facing parse failure (unsupported type, corrupt file, empty result)."""


def _clean(text: str) -> str:
    """Normalize whitespace and drop control chars that confuse downstream stages."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip other C0 control chars except tab/newline.
    text = "".join(ch for ch in text if ch in "\n\t\f" or ord(ch) >= 32)
    # Collapse runs of blank lines.
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    blanks = 0
    for ln in lines:
        if ln:
            out.append(ln)
            blanks = 0
        else:
            blanks += 1
            if blanks <= 1:
                out.append("")
    cleaned = "\n".join(out).strip()
    return cleaned[:MAX_TEXT_CHARS]


def parse_document(filename: str, raw: bytes) -> tuple[str, str]:
    """Parse an uploaded document to (title, text).

    `title` is derived from the filename (stem). `text` is the extracted plain-text
    body. Raises DocumentError with an actionable message on any failure.
    """
    if not raw:
        raise DocumentError("The file is empty.")
    if len(raw) > MAX_FILE_BYTES:
        mb = MAX_FILE_BYTES // (1024 * 1024)
        raise DocumentError(f"File is too large (limit {mb} MB).")

    ext = Path(filename).suffix.lower()
    title = Path(filename).stem.strip() or "Untitled document"
    if ext in {".docx", ".xlsx"}:
        _check_archive(raw)

    if ext == ".pdf":
        text = _parse_pdf(raw)
    elif ext == ".docx":
        text = _parse_docx(raw)
    elif ext in (".xlsx", ".xls"):
        text = _parse_xlsx(raw, ext)
    elif ext == ".csv":
        text = _parse_csv(raw)
    elif ext in (".txt", ".md", ".markdown", ".text"):
        text = _parse_text(raw)
    elif ext == ".doc":
        raise DocumentError(
            "Legacy .doc files are not supported. Please save the document as .docx "
            "(or export to PDF) and upload again."
        )
    else:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentError(
            f"Unsupported file type '{ext or '(none)'}'. Supported types: {allowed}."
        )

    text = _clean(text)
    if not text:
        raise DocumentError(
            "No readable text could be extracted from this file. If it is a scanned "
            "PDF (images only), it needs OCR before it can be used."
        )
    return title, text


def _check_archive(raw: bytes) -> None:
    """Office formats are ZIP containers; cap expansion before loading their XML."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) > MAX_ARCHIVE_ENTRIES
                    or sum(entry.file_size for entry in entries) > MAX_ARCHIVE_BYTES):
                raise DocumentError("Document archive expands beyond the safe parsing limit.")
    except (zipfile.BadZipFile, OSError) as exc:
        raise DocumentError("The document is not a valid Office archive.") from exc


def _join_bounded(parts, separator: str = "\n") -> str:
    result = []
    used = 0
    for part in parts:
        remaining = MAX_TEXT_CHARS - used
        if remaining <= 0:
            break
        result.append(part[:remaining])
        used += len(result[-1]) + 1
        if used >= MAX_TEXT_CHARS:
            break
    return separator.join(result)[:MAX_TEXT_CHARS]


# --------------------------------------------------------------------------
# Per-format parsers. Each imports its dependency lazily and raises DocumentError
# (never a bare ImportError) so the failure is actionable.
# --------------------------------------------------------------------------
def _parse_pdf(raw: bytes) -> str:
    """Extract text from a PDF. Prefer pypdf; fall back to PyMuPDF (fitz) which
    handles some files pypdf trips on."""
    # Attempt 1: pypdf (pure Python, always bundled).
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentError("PDF exceeds the 500-page parsing limit.")
        text = _join_bounded((page.extract_text() or "" for page in reader.pages), "\f").strip()
        if text:
            return text
    except DocumentError:
        raise
    except Exception:
        text = ""

    # Attempt 2: PyMuPDF, if present — better on complex layouts.
    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=raw, filetype="pdf") as doc:
            if len(doc) > MAX_PDF_PAGES:
                raise DocumentError("PDF exceeds the 500-page parsing limit.")
            text = _join_bounded((page.get_text() for page in doc), "\f").strip()
        if text:
            return text
    except DocumentError:
        raise
    except Exception:
        pass

    if not text:
        raise DocumentError(
            "Could not extract text from the PDF. It may be a scanned/image-only "
            "document (which needs OCR) or corrupt."
        )
    return text


def _parse_docx(raw: bytes) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        raise DocumentError(
            "Reading .docx files requires the 'python-docx' package, which is not "
            "available in this build."
        )
    try:
        document = docx.Document(io.BytesIO(raw))
    except Exception as e:
        raise DocumentError(f"Could not open the .docx file: {e}")

    def parts():
        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                yield paragraph.text
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    yield " | ".join(cells)
    return _join_bounded(parts())


def _parse_xlsx(raw: bytes, ext: str) -> str:
    try:
        import openpyxl
    except ImportError:
        raise DocumentError(
            "Reading spreadsheet files requires the 'openpyxl' package, which is not "
            "available in this build. Export the sheet to CSV and upload that instead."
        )
    if ext == ".xls":
        raise DocumentError(
            "Legacy .xls files are not supported. Save the workbook as .xlsx or export "
            "to CSV and upload again."
        )
    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    except Exception as e:
        raise DocumentError(f"Could not open the spreadsheet: {e}")

    def rows():
        for ws in wb.worksheets:
            yield f"# Sheet: {ws.title}"
            # Ignore untrusted declared dimensions and stop at actual XML row data.
            ws.reset_dimensions()
            for row in islice(ws.iter_rows(max_col=MAX_SHEET_COLUMNS, values_only=True),
                              MAX_SHEET_ROWS):
                cells = [str(c) for c in row if c is not None and str(c).strip()]
                if cells:
                    yield " | ".join(cells)
    try:
        return _join_bounded(rows())
    finally:
        wb.close()


def _parse_csv(raw: bytes) -> str:
    text = _decode(raw)
    reader = csv.reader(io.StringIO(text))
    try:
        return _join_bounded(" | ".join(c.strip() for c in row if c.strip()) for row in reader)
    except csv.Error as exc:
        raise DocumentError(f"Could not parse CSV: {exc}") from exc


def _parse_text(raw: bytes) -> str:
    return _decode(raw)


def _decode(raw: bytes) -> str:
    """Decode bytes to text, trying the common encodings before giving up."""
    for enc in ("utf-8-sig", "utf-8", "utf-16", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # latin-1 above never fails, but keep a defensive fallback.
    return raw.decode("utf-8", errors="replace")
