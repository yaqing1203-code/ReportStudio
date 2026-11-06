"""Bounded scientific-table representations, retaining headers and local context.

Table text is a transparent linearization of fetched cells, not an inferred
measurement or a guarantee that the original experiment is comparable. Offsets
refer to the final extracted document text, never to the HTML/XML byte stream.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

MAX_TABLES = 12
MAX_ROWS = 60
MAX_COLUMNS = 20
MAX_CELL_CHARS = 1600
MAX_CONTEXT_CHARS = 4000
MAX_RAW_CHARS = 2_000_000
MAX_NODES = 60_000
MAX_ROW_CHARS = 12_000
MAX_TABLE_CHARS = 50_000
MAX_PDF_TABLE_PAGES = 12
_NUMBER = re.compile(r"(?<!\w)[+−-]?\d+(?:[.,]\d+)?")
_BLOCKS = {"p", "div", "br", "break", "li", "list-item", "title", "label"}


def _clean(value: str) -> str:
    return " ".join("".join(c for c in value if ord(c) >= 32 or c in "\t\n\r").split())


@dataclass(eq=False)
class _Node:
    tag: str
    attrs: dict
    start: int = 0
    end: int | None = None
    parent: _Node | None = None
    children: list = field(default_factory=list)


class _LimitReached(Exception):
    pass


class _Tree(HTMLParser):
    def __init__(self, raw: str):
        super().__init__(convert_charrefs=True)
        self.raw = raw
        self.line_starts = [0] + [match.end() for match in re.finditer("\n", raw)]
        self.root = _Node("root", {})
        self.stack = [self.root]
        self.nodes: list[_Node] = []

    def _position(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if len(self.nodes) >= MAX_NODES:
            raise _LimitReached
        tag = tag.split(":")[-1]
        position = self._position()
        # HTML permits omitted closing tags for cells and rows.
        close = {"td", "th"} if tag in {"td", "th"} else {"tr"} if tag == "tr" else set()
        if close:
            for i in range(len(self.stack) - 1, 0, -1):
                if self.stack[i].tag in close:
                    while len(self.stack) > i:
                        self.stack.pop().end = position
                    break
                if self.stack[i].tag == "table":
                    break
        node = _Node(tag, dict(attrs), position, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        self.nodes.append(node)
        if tag in {"br", "hr", "img", "input", "meta", "link", "source", "wbr"}:
            node.end = position + len(self.get_starttag_text() or "")
        else:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        tag = tag.split(":")[-1]
        if self.stack[-1].tag == tag:
            self.stack.pop().end = self._position() + len(self.get_starttag_text() or "")

    def handle_endtag(self, tag):
        tag = tag.split(":")[-1]
        end = self.raw.find(">", self._position()) + 1
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                while len(self.stack) > i:
                    self.stack.pop().end = end
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _descendants(node: _Node):
    for child in node.children:
        if isinstance(child, _Node):
            yield child
            yield from _descendants(child)


def _text(node: _Node) -> str:
    if node.tag in {"script", "style", "noscript"}:
        return ""
    parts = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        else:
            if child.tag in _BLOCKS:
                parts.append(" ")
            value = _text(child)
            if child.tag == "sup" and re.fullmatch(r"\s*[+−-]?\d+\s*", value):
                referenced = any(node.tag in {"a", "xref"} for node in _descendants(child))
                marker = bool(re.search(r"ref|note|fn", str(child.attrs.get("class", "")), re.IGNORECASE))
                # A visible superscript exponent needs an explicit plaintext
                # boundary; otherwise 10⁻⁵ is indistinguishable from a range or
                # a corrupted decimal. Linked citation markers are not powers.
                value = " [" + value.strip() + "]" if referenced or marker else "^" + value.strip()
            parts.append(value)
            if child.tag in _BLOCKS:
                parts.append(" ")
    return "".join(parts)


def _within(node: _Node, ancestor: _Node, tag: str) -> bool:
    parent = node.parent
    while parent is not None and parent is not ancestor:
        if parent.tag == tag:
            return True
        parent = parent.parent
    return False


def _outer_texts(node: _Node, predicate) -> list[str]:
    result = []
    for child in node.children:
        if not isinstance(child, _Node):
            continue
        if predicate(child):
            text = _clean(_text(child))
            if text:
                result.append(text)
        else:
            result.extend(_outer_texts(child, predicate))
    return result


def _table_context(table: _Node) -> _Node:
    context = table
    parent = table.parent
    for _ in range(5):
        if parent is None or parent.tag in {"body", "article", "root"}:
            break
        if sum(node.tag == "table" for node in _descendants(parent)) > 1:
            break  # An outer group must not mix notes from neighboring tables.
        if parent.tag in {"table-wrap", "figure"}:
            return parent
        if "table" in str(parent.attrs.get("class", "")).lower():
            context = parent
        parent = parent.parent
    return context


def _context(table: _Node, tree: _Tree) -> tuple[str, list[str]]:
    context = _table_context(table)
    captions = _outer_texts(context, lambda node: node.tag in {"caption", "figcaption"}
                            or bool(re.search(r"caption|table[-_ ]?title",
                                              str(node.attrs.get("class", "")), re.IGNORECASE)))
    labels = [_clean(_text(child)) for child in context.children
              if isinstance(child, _Node) and child.tag == "label"]
    notes = _outer_texts(context, lambda node: node.tag in {"tfoot", "table-wrap-foot", "fn"}
                        or bool(re.search(r"footnote|table[-_ ]?note", str(node.attrs.get("class", "")), re.IGNORECASE)))
    # Resolve local footnote references only. This never follows a URL or downloads.
    by_id = {node.attrs["id"]: node for node in tree.nodes if node.attrs.get("id")}
    for node in _descendants(table):
        target = node.attrs.get("rid") or str(node.attrs.get("href", "")).removeprefix("#")
        if target in by_id and (node.attrs.get("ref-type") == "table-fn" or str(node.attrs.get("href", "")).startswith("#")):
            referenced = by_id[target]
            if referenced.tag == "fn" or re.search(r"footnote|table[-_ ]?note", str(referenced.attrs.get("class", "")), re.IGNORECASE):
                notes.append(_clean(_text(referenced)))
    return _clean(" ".join(dict.fromkeys(labels + captions))), list(dict.fromkeys(notes))


def _span(cell: _Node, name: str, limit: int) -> int | None:
    value = str(cell.attrs.get(name) or "1")
    if not value.isascii() or not value.isdigit() or len(value) > 3:
        return None
    amount = int(value)
    return amount if 1 <= amount <= limit else None


def _html_table(table: _Node, tree: _Tree, index: int, format: str) -> dict | None:
    if table.end is None or any(node.tag == "table" for node in _descendants(table)):
        return None
    rows = [node for node in _descendants(table) if node.tag == "tr" and not _within(node, table, "tfoot")]
    raw_rows = [[cell for cell in row.children if isinstance(cell, _Node) and cell.tag in {"td", "th"}]
                for row in rows]
    pairs = [(row, cells) for row, cells in zip(rows, raw_rows, strict=True) if cells]
    if len(pairs) < 2:
        return None
    header_indices = {i for i, (row, _) in enumerate(pairs) if _within(row, table, "thead")}
    inferred = False
    if not header_indices:
        for i, (_, cells) in enumerate(pairs):
            if all(cell.tag == "th" for cell in cells):
                header_indices.add(i)
            else:
                break
    if not header_indices and all(not _NUMBER.search(_text(cell)) for cell in pairs[0][1]):
        header_indices = {0}
        inferred = True
    caption, footnotes = _context(table, tree)
    if len(caption) + sum(map(len, footnotes)) > MAX_CONTEXT_CHARS:
        return None  # Never silently cut a footnote that may qualify the numbers.
    expanded = []
    pending: dict[int, tuple[str, int]] = {}
    truncated = len(pairs) > MAX_ROWS + len(header_indices)
    for ri, (_, cells) in enumerate(pairs[:MAX_ROWS + min(len(header_indices), 5)]):
        values = {column: text for column, (text, until) in pending.items() if until >= ri}
        inherited = sorted(values)
        col = 0
        ambiguous = False
        for cell in cells:
            while col in values:
                col += 1
            width, height = _span(cell, "colspan", MAX_COLUMNS), _span(cell, "rowspan", MAX_ROWS)
            value = _clean(_text(cell))
            if width is None or height is None or col + width > MAX_COLUMNS or len(value) > MAX_CELL_CHARS:
                return None
            if any(column in values for column in range(col, col + width)):
                return None
            if width > 1 and ri not in header_indices:
                ambiguous = True
            for offset in range(width):
                # Header spans describe all child columns. Data spans do not
                # license copying one numeric value into multiple measurements.
                resolved = value if ri in header_indices or offset == 0 else ""
                values[col + offset] = resolved
                if height > 1:
                    pending[col + offset] = (resolved, ri + height - 1)
            col += width
        expanded.append((values, inherited, ambiguous))
    width = max((max(values, default=-1) + 1 for values, _, _ in expanded), default=0)
    if width < 2:
        return None
    body_subheader = False
    first_body = next((i for i in range(len(expanded)) if i not in header_indices), None)
    grouped_header = any((_span(cell, "colspan", MAX_COLUMNS) or 0) > 1
                         for ri in header_indices for cell in pairs[ri][1])
    if first_body is not None and grouped_header:
        values, inherited, _ = expanded[first_body]
        labels = [value for column, value in sorted(values.items()) if column > 0 and value]
        # Some publishers put grouped-column sublabels (e.g. XRD 4a/4d sites)
        # in tbody, leaving the row-header cell empty. Preserve those labels;
        # a scalar/scientific number must never be promoted to a header.
        if ((not values.get(0) or 0 in inherited) and len(labels) >= 2
                and all(len(value) <= 60 and any(c.isalpha() for c in value)
                        and not re.fullmatch(r"[+−-]?\d+(?:[.,]\d+)?(?:[eE][+−-]?\d+)?", value)
                        for value in labels)):
            header_indices.add(first_body)
            body_subheader = True
    headers = []
    for col in range(width):
        path = [expanded[ri][0].get(col, "") for ri in sorted(header_indices) if ri < len(expanded)]
        headers.append(" / ".join(dict.fromkeys(value for value in path if value)))
    data = []
    row_meta = []
    for ri, (values, inherited, ambiguous) in enumerate(expanded):
        if ri in header_indices:
            continue
        row = [values.get(col, "") for col in range(width)]
        if not any(row):
            continue
        data.append(row)
        row_meta.append({"inherited_columns": inherited,
                         "structure_status": "ambiguous_data_span" if ambiguous else "parsed"})
    if not data or not any(_NUMBER.search(cell) for row in data for cell in row):
        return None
    context = _table_context(table)
    container_id = context.attrs.get("id")
    element_id = table.attrs.get("id") or container_id
    cited_table = format in {"jats", "xml"} and any(
        node.tag == "xref" and node.attrs.get("ref-type") == "bibr" for node in _descendants(context))
    return {"id": str(element_id or f"table-{index}")[:120], "caption": caption,
            "headers": headers, "rows": data, "footnotes": footnotes,
            "header_status": ("explicit_with_body_subheaders" if body_subheader else "inferred_first_row"
                              if inferred else "explicit" if header_indices else "unknown"),
            "conditions_status": "unknown", "truncated": truncated,
            **({"has_bibliography_links": True} if cited_table else {}),
            "locator": {"format": format, "table_index": index, "element_id": element_id,
                        **({"container_id": container_id} if container_id and container_id != element_id else {})},
            "_raw_span": [table.start, table.end], "_row_meta": row_meta}


def extract_tables(raw: str | bytes, format: str) -> list[dict]:
    """Extract bounded tables from received content; no file/network access.

    Missing headers, conditions or PDF footnotes are never guessed. HTML entity
    decoding and whitespace normalization match ordinary extracted source text.
    """
    if format == "pdf":
        return _pdf_tables(raw) if isinstance(raw, bytes) else []
    if format not in {"html", "jats", "xml"}:
        return []
    if isinstance(raw, bytes):
        raw = raw[:MAX_RAW_CHARS * 4].decode("utf-8", errors="replace")
    if re.search(r"<!ENTITY", raw, re.IGNORECASE):
        return []
    if format in {"jats", "xml"}:
        import xml.etree.ElementTree as ET

        safe_xml = re.sub(r'<!DOCTYPE\s+article\s+(?:PUBLIC\s+"[^"]*"\s+"[^"]*"|SYSTEM\s+"[^"]*")\s*>',
                          "", raw, flags=re.IGNORECASE)
        if len(safe_xml) > MAX_RAW_CHARS or re.search(r"<!DOCTYPE", safe_xml, re.IGNORECASE):
            return []
        try:
            ET.fromstring(safe_xml)
        except ET.ParseError:
            return []
    tree = _Tree(raw[:MAX_RAW_CHARS])
    try:
        tree.feed(tree.raw)
    except (_LimitReached, RecursionError):
        pass
    tables: list[dict] = []
    seen: dict[str, dict] = {}
    table_index = 0
    for node in tree.nodes:
        if node.tag != "table":
            continue
        table_index += 1
        try:
            table = _html_table(node, tree, table_index, format)
        except (RecursionError, ValueError, TypeError):
            continue
        if table is None:
            continue
        fingerprint = repr((table["caption"], table["headers"], table["rows"], table["footnotes"]))
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()
        if digest in seen:
            seen[digest].setdefault("_duplicate_raw_spans", []).append(table["_raw_span"])
            seen[digest].setdefault("_duplicate_table_indices", []).append(table_index)
            continue
        seen[digest] = table
        tables.append(table)
        if len(tables) == MAX_TABLES:
            break
    return tables


def strip_extracted_tables(raw: str, tables: list[dict]) -> str:
    """Remove accepted HTML tables only, avoiding a second flattened cell copy."""
    spans = [span for table in tables for span in [table.get("_raw_span"), *table.get("_duplicate_raw_spans", [])]]
    for span in sorted((span for span in spans if isinstance(span, list) and len(span) == 2), reverse=True):
        start, end = span
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(raw):
            raw = raw[:start] + " " + raw[end:]
    return raw


def append_tables(text: str, tables: list[dict], *, max_chars: int = 200_000) -> tuple[str, list[dict]]:
    """Attach complete contextual rows, returning offsets into the exact result.

    Existing prose is never truncated. Oversized rows are omitted in full rather
    than slicing off a unit, exponent, caption or qualifying footnote.
    """
    attached = []
    for original in tables[:MAX_TABLES]:
        table = {key: value for key, value in original.items() if not key.startswith("_")}
        records = []
        used = 0
        for index, cells in enumerate(table.get("rows", [])[:MAX_ROWS]):
            labels = table.get("headers", [])
            content = " | ".join(f"{labels[col]}: {value}" if col < len(labels) and labels[col] else value
                                 for col, value in enumerate(cells))
            parts = ([table["caption"]] if table.get("caption") else []) + [content] + table.get("footnotes", [])
            row_text = "\n".join(parts)
            if not row_text or len(row_text) > MAX_ROW_CHARS or used + len(row_text) > MAX_TABLE_CHARS:
                table["truncated"] = True
                continue
            existing = text.find(row_text)
            if existing < 0:
                separator = "\n\n" if text else ""
                if len(text) + len(separator) + len(row_text) > max_chars:
                    table["truncated"] = True
                    break
                start = len(text) + len(separator)
                text += separator + row_text
            else:
                start = existing
            details = original.get("_row_meta") or [
                {key: value for key, value in row.items() if key in {"structure_status", "inherited_columns"}}
                for row in original.get("row_records", [])]
            records.append({"row_index": index + 1, "cells": cells, "text": row_text,
                            "start": start, "end": start + len(row_text),
                            **(details[index] if index < len(details) else {"structure_status": "parsed", "inherited_columns": []})})
            used += len(row_text)
        if records:
            table.update(start=min(row["start"] for row in records), end=max(row["end"] for row in records),
                         rows=[row["cells"] for row in records], row_records=records,
                         representation="table_linearization_v1")
            attached.append(table)
    return text, attached


def _pdf_tables(raw: bytes) -> list[dict]:
    """Only ruled, captioned PDF tables with readable headers; no OCR inference."""
    if len(raw) > 25 * 1024 * 1024:
        return []
    try:
        import pymupdf
    except ImportError:
        return []
    result: list[dict] = []
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            if len(document) > 500:
                return []
            for page_index in range(min(len(document), MAX_PDF_TABLE_PAGES)):
                page = document[page_index]
                try:
                    # New PyMuPDF releases can apply a learned layout grid by
                    # default. Retain only explicit vector-line geometry here.
                    found = page.find_tables(strategy="lines_strict", use_layout=False)
                except TypeError:
                    found = page.find_tables(strategy="lines_strict")
                for candidate in found.tables[:MAX_TABLES - len(result)]:
                    matrix = candidate.extract()
                    if not matrix or len(matrix) < 2 or not 2 <= len(matrix[0]) <= MAX_COLUMNS:
                        continue
                    matrix = [[_clean(str(value or "")) for value in row] for row in matrix[:MAX_ROWS + 1]]
                    headers = matrix[0]
                    if (not all(headers) or all(_NUMBER.fullmatch(cell) for cell in headers)
                            or any(len(row) != len(headers) for row in matrix)
                            or any(len(cell) > MAX_CELL_CHARS for row in matrix for cell in row)):
                        continue
                    exact_cells = True
                    for row, geometry in zip(matrix, candidate.rows, strict=False):
                        for cell, box in zip(row, geometry.cells, strict=False):
                            if box and cell != _clean(page.get_text("text", clip=pymupdf.Rect(box))):
                                exact_cells = False
                    if not exact_cells:
                        continue
                    x0, y0, x1, y1 = candidate.bbox
                    before = page.get_text("text", clip=pymupdf.Rect(0, max(0, y0 - 90), page.rect.width, y0))
                    match = re.search(r"(?im)^\s*(?:table|表)\s*[\dIVX]+\b[^\n]*(?:\n[^\n]*)?", before)
                    if not match:
                        continue
                    caption = _clean(match.group())
                    after = page.get_text("text", clip=pymupdf.Rect(x0, y1, x1, min(page.rect.height, y1 + 70)))
                    footnotes = [_clean(line) for line in after.splitlines() if re.match(r"^\s*(?:[*†‡]|[a-z][):.]|Note\b|注[:：])", line)]
                    rows = matrix[1:]
                    if not any(_NUMBER.search(cell) for row in rows for cell in row):
                        continue
                    result.append({"id": f"pdf-table-{len(result) + 1}", "caption": caption,
                        "headers": headers, "rows": rows, "footnotes": footnotes,
                        "header_status": "pdf_grid_first_row", "conditions_status": "unknown",
                        "truncated": candidate.row_count > MAX_ROWS + 1,
                        "physical_page_count": len(document),
                        "locator": {"format": "pdf", "page": page_index + 1,
                                    "page_count": len(document), "bbox": list(candidate.bbox)},
                        "_row_meta": [{"structure_status": "parsed", "inherited_columns": []} for _ in rows]})
                if len(result) >= MAX_TABLES:
                    break
    except Exception:
        # Ordinary PDF text remains available if geometry extraction is unsupported.
        return result
    return result
