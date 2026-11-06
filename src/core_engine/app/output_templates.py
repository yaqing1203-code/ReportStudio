"""DOCX templates are layout inputs, never research evidence or executable code.

Patch the copied OOXML package, preserving all untouched parts verbatim. Supports
explicit text/block slots, ordinary heading-based templates and a styles-only mode.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import uuid
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from lxml import etree as ET

from core_engine.app import runtime
from core_engine.security.files import atomic_write_text, storage_path

MAX_TEMPLATE_BYTES = 10 * 1024 * 1024
_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_W = "{" + _NS["w"] + "}"
_SLOT = re.compile(r"\{\{\s*([^{}\n]{1,100}?)\s*\}\}")
_ALIASES = {"标题": "title", "正文": "content", "日期": "date", "摘要": "abstract", "参考文献": "references"}
_CORE = {"title", "content", "date", "abstract", "references"}


class TemplateError(ValueError):
    pass


def _root() -> Path:
    root = runtime.user_data_dir() / "output_templates"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _xml(raw: bytes):
    if re.search(br"<!DOCTYPE|<!ENTITY", raw, re.IGNORECASE):
        raise TemplateError("模板包含不支持的 XML 声明。")
    return ET.fromstring(raw, ET.XMLParser(resolve_entities=False, no_network=True))


def _package(raw: bytes) -> dict[str, bytes]:
    if not raw or len(raw) > MAX_TEMPLATE_BYTES:
        raise TemplateError("模板不能为空，大小上限为 10 MB。")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if (len(entries) > 2000 or len(set(names)) != len(names)
                    or sum(e.file_size for e in entries) > 100 * 1024 * 1024
                    or any(e.flag_bits & 1 for e in entries)):
                raise TemplateError("模板压缩包过大、加密或包含重复文件。")
            for name in names:
                if (name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/")
                        or re.search(r"vbaProject|activeX|embeddings/", name, re.IGNORECASE)):
                    raise TemplateError("请使用不含宏、嵌入对象或活动控件的普通 .docx 模板。")
            parts = {e.filename: archive.read(e) for e in entries}
        if "word/document.xml" not in parts or "[Content_Types].xml" not in parts:
            raise TemplateError("文件不是有效的 Word .docx 模板。")
        if b"macroenabled" in parts["[Content_Types].xml"].lower():
            raise TemplateError("请先另存为不含宏的普通 .docx 模板。")
        for name, data in parts.items():
            if name.endswith((".xml", ".rels")):
                root = _xml(data)
                if name.startswith("word/"):
                    instructions = " ".join(root.xpath("//w:fldSimple/@w:instr", namespaces=_NS)) + " " + "".join(
                        root.xpath("//w:instrText/text()", namespaces=_NS))
                    if (re.search(r"\b(?:DDEAUTO|DDE|INCLUDETEXT|INCLUDEPICTURE|LINK|DATABASE|MACROBUTTON|AUTOTEXT)\b",
                                  instructions, re.IGNORECASE) or root.find(".//" + _W + "altChunk") is not None
                            or root.find(".//" + _W + "object") is not None):
                        raise TemplateError("模板包含动态外部内容或活动域，请转换为普通文字或内嵌图片后再上传。")
                if name.endswith(".rels"):
                    for rel in root:
                        if re.search(r"/(?:vbaProject|oleObject|control|aFChunk)$", str(rel.get("Type", "")), re.IGNORECASE):
                            raise TemplateError("模板包含不支持的活动对象。")
                        if (rel.get("TargetMode") == "External" and
                                (not str(rel.get("Type", "")).endswith("/hyperlink")
                                 or not re.match(r"^(?:https?://|mailto:)", rel.get("Target", ""), re.IGNORECASE))):
                            raise TemplateError("模板含外部图片、附加模板或其他外部资源，请先嵌入到文件中。")
        return parts
    except TemplateError:
        raise
    except (zipfile.BadZipFile, ET.XMLSyntaxError, OSError, RuntimeError, ValueError) as exc:
        raise TemplateError("无法读取模板，请用 Word 另存为普通 .docx 后重试。") from exc


def _text(p) -> str:
    return "".join(p.itertext()) if p.tag == _W + "t" else "".join(p.xpath(".//w:t/text()", namespaces=_NS))


def _key(key: str) -> str:
    key = key.strip()
    if key.startswith(("章节:", "章节：", "section:")):
        return "section:" + re.split(r"[:：]", key, maxsplit=1)[1].strip()
    return _ALIASES.get(key, key)


def _heading_styles(parts: dict[str, bytes]) -> set[str]:
    if "word/styles.xml" not in parts:
        return {"Heading1", "Heading2", "1", "2"}
    root = _xml(parts["word/styles.xml"])
    return {s.get(_W + "styleId") for s in root.findall(_W + "style")
            if s.find(".//" + _W + "outlineLvl") is not None
            or re.match(r"(?:heading|标题)\s*[1-6]$", (s.find(_W + "name").get(_W + "val", "")
                if s.find(_W + "name") is not None else ""), re.IGNORECASE)}


def _is_heading(p, styles: set[str]) -> bool:
    style = p.find("./" + _W + "pPr/" + _W + "pStyle")
    if p.find("./" + _W + "pPr/" + _W + "outlineLvl") is not None or (style is not None and style.get(_W + "val") in styles):
        return True
    text = _text(p).strip()
    return bool(len(text) <= 80 and re.match(r"^(?:第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十\d]+[、.．])", text)
                and (p.find(".//" + _W + "b") is not None or any(int(s.get(_W + "val", "0")) >= 28
                     for s in p.findall(".//" + _W + "sz") if str(s.get(_W + "val", "")).isdigit())))


def inspect_template(raw: bytes) -> dict:
    parts = _package(raw)
    doc = _xml(parts["word/document.xml"])
    body = doc.find(_W + "body")
    if body is None:
        raise TemplateError("模板缺少正文。")
    slots: list[str] = []
    for name, data in parts.items():
        if not re.fullmatch(r"word/(?:document|header\d+|footer\d+)\.xml", name):
            continue
        for p in _xml(data).iter(_W + "p"):
            text = _text(p)
            for match in _SLOT.finditer(text):
                key = _key(match[1])
                if not key or key == "section:":
                    raise TemplateError("章节占位符需要章节名称。")
                block = key in {"content", "references"} or key.startswith("section:")
                if block:
                    if name != "word/document.xml" or text.strip() != match[0]:
                        raise TemplateError("正文、参考文献和章节占位符须单独占正文中的一个段落。")
                    if p.xpath("ancestor::w:txbxContent|ancestor::w:sdt|.//w:drawing|.//w:fldChar", namespaces=_NS):
                        raise TemplateError("长正文填充位置请放在普通段落或可自动增高的表格中。")
                    if p.xpath("ancestor::w:tr/w:trPr/w:trHeight[@w:hRule='exact']", namespaces=_NS):
                        raise TemplateError("正文填充位置所在表格行使用了固定高度，请先改为自动高度。")
                slots.append(key)
    sections = [s[8:] for s in slots if s.startswith("section:")]
    if len(sections) != len(set(sections)) or slots.count("content") > 1:
        raise TemplateError("同一章节或整篇正文只能设置一个填充位置。")
    if sections and "content" in slots:
        raise TemplateError("请在整篇正文占位符与分章节占位符中选择一种，避免正文重复。")
    styles = _heading_styles(parts)
    headings = [_text(p).strip() for p in body if p.tag == _W + "p" and _is_heading(p, styles) and _text(p).strip()]
    mode = "slots" if "content" in slots or sections else "outline" if headings else "styles"
    if mode == "outline":
        sections = headings
    if len(sections) > 12 or len(sections) != len(set(sections)):
        raise TemplateError("请使用至多 12 个不重复的主要章节标题。")
    if mode == "styles" and len(doc.findall(".//" + _W + "sectPr")) > 1:
        raise TemplateError("多节版式模板请用 {{content}} 或 {{section:章节名}} 标明正文位置。")
    fields = sorted({key for key in slots if key not in _CORE and not key.startswith("section:")})
    return {"mode": mode, "sections": sections, "fields": fields, "slots": sorted(set(slots)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "hint": {"slots": "保留固定文字，在占位符位置填入内容。", "outline": "按模板章节顺序填入正文；章节下的示例段落会被替换，固定表格保留。",
                     "styles": "未识别到正文位置：仅套用页面、页眉页脚和样式，原正文将替换。保留封面固定文字请使用占位符。"}[mode]}


def add_template(filename: str, raw: bytes) -> dict:
    if Path(filename).suffix.lower() != ".docx":
        raise TemplateError("目前支持 Word .docx 模板；.doc/.dotx 请先另存为 .docx。")
    info = inspect_template(raw)
    identifier = uuid.uuid4().hex
    info.update(id=identifier, name=filename.replace("\\", "/").rsplit("/", 1)[-1][:180],
                created_at=datetime.now(UTC).isoformat())
    root = _root()
    _write_bytes(storage_path(root, identifier, ".docx"), raw)
    atomic_write_text(storage_path(root, identifier, ".json"), json.dumps(info, ensure_ascii=False))
    return info


def get_template(identifier: str) -> tuple[dict, bytes]:
    try:
        root = _root()
        info = json.loads(storage_path(root, identifier, ".json").read_text(encoding="utf-8"))
        raw = storage_path(root, identifier, ".docx").read_bytes()
        if hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise TemplateError("模板文件发生变化，请重新上传。")
        return info, raw
    except (OSError, KeyError, ValueError) as exc:
        raise TemplateError("模板不存在或已不可用，请重新选择。") from exc


def list_templates() -> list[dict]:
    items = []
    for path in _root().glob("*.json"):
        try:
            items.append(get_template(path.stem)[0])
        except TemplateError:
            continue
    return sorted(items, key=lambda i: i["created_at"], reverse=True)


def delete_template(identifier: str) -> None:
    get_template(identifier)
    for suffix in (".docx", ".json"):
        storage_path(_root(), identifier, suffix).unlink(missing_ok=True)


def selection(identifier: str | None, fields: dict[str, str]) -> tuple[dict, bytes] | None:
    if not identifier:
        return None
    info, raw = get_template(identifier)
    if (len(fields) > 30 or sum(len(k) + len(v) for k, v in fields.items()) > 15000
            or set(fields) - set(info["fields"])):
        raise TemplateError("模板补充字段不匹配或过长，请重新选择模板。")
    return info, raw


def _write_bytes(path: Path, raw: bytes) -> None:
    fd, filename = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
        os.replace(filename, path)
    finally:
        Path(filename).unlink(missing_ok=True)


def _inline(p, values: dict[str, str]) -> None:
    nodes = list(p.iter(_W + "t"))
    text = "".join(n.text or "" for n in nodes)
    for match in reversed(list(_SLOT.finditer(text))):
        key = _key(match[1])
        if key not in values:
            continue
        start, end = match.span()
        pos = 0
        for node in nodes:
            value = node.text or ""
            a, b = max(start - pos, 0), min(end - pos, len(value))
            if a < b:
                node.text = value[:a] + (values[key] if pos <= start < pos + len(value) else "") + value[b:]
                node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            pos += len(value)


def _paragraph(prototype, text: str, heading: bool = False, *, first=False, last=False, heading_style="Heading1", links=None):
    p = deepcopy(prototype)
    for attr in list(p.attrib):
        if attr.endswith(("}paraId", "}textId")):
            p.set(attr, uuid.uuid4().hex[:8].upper())
    rpr = prototype.find(".//" + _W + "rPr")
    for child in list(p):
        if child.tag != _W + "pPr":
            p.remove(child)
    props = p.find(_W + "pPr")
    if props is not None:
        for prop in list(props):
            if (prop.tag == _W + "sectPr" and not last) or (prop.tag == _W + "pageBreakBefore" and not first):
                props.remove(prop)
    if heading:
        if props is None:
            props = ET.SubElement(p, _W + "pPr")
        style = props.find(_W + "pStyle")
        if style is None:
            style = ET.SubElement(props, _W + "pStyle")
        style.set(_W + "val", heading_style)
        ET.SubElement(props, _W + "keepNext")
    links = links or {}
    reference = next((link for link in links.values() if text.startswith(link["prefix"])), None)
    if reference:
        ET.SubElement(p, _W + "bookmarkStart", {_W + "id": reference["id"], _W + "name": reference["name"]})
    def append(parent, value):
        run = ET.SubElement(parent, _W + "r")
        if rpr is not None and not heading:
            run.append(deepcopy(rpr))
        t = ET.SubElement(run, _W + "t")
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = value
    cursor = 0
    for match in re.finditer(r"\[([A-Za-z0-9][A-Za-z0-9_-]{0,127})\]", text):
        link = links.get(match[1])
        if link and not reference:
            append(p, text[cursor:match.start()])
            hyperlink = ET.SubElement(p, _W + "hyperlink", {_W + "anchor": link["name"], _W + "history": "1"})
            append(hyperlink, match[0])
            cursor = match.end()
    append(p, text[cursor:])
    if reference:
        ET.SubElement(p, _W + "bookmarkEnd", {_W + "id": reference["id"]})
    return p


def _replace_block(p, lines: list[tuple[str, bool]], *, heading_style="Heading1", links=None):
    lines = lines or [("", False)]
    parent = p.getparent()
    offset = parent.index(p)
    paragraphs = [_paragraph(p, text, heading, first=i == 0, last=i == len(lines) - 1, heading_style=heading_style, links=links)
                  for i, (text, heading) in enumerate(lines)]
    for marker in p.iter():
        if marker.tag in {_W + "bookmarkStart", _W + "commentRangeStart"}:
            paragraphs[0].insert(int(paragraphs[0].find(_W + "pPr") is not None), deepcopy(marker))
        elif marker.tag in {_W + "bookmarkEnd", _W + "commentRangeEnd"}:
            paragraphs[-1].append(deepcopy(marker))
    for i, paragraph in enumerate(paragraphs):
        parent.insert(offset + i, paragraph)
    parent.remove(p)


def render_template(raw: bytes, info: dict, payload: dict, fields: dict[str, str]) -> bytes:
    parts = _package(raw)
    sections: dict[str, list[str]] = payload["sections"]
    values = {**{key: fields.get(key, "").replace("\n", " ") for key in info["fields"]},
              "title": payload["title"], "date": payload.get("date", datetime.now(UTC).strftime("%Y-%m-%d")),
              "abstract": payload.get("abstract", "")}
    references = payload.get("references", [])
    styles = _heading_styles(parts)
    heading_style = "Heading1" if "Heading1" in styles else next(iter(sorted(styles)), "Heading1")
    original_doc = _xml(parts["word/document.xml"])
    bookmark_ids = [int(v) for v in original_doc.xpath("//w:bookmarkStart/@w:id", namespaces=_NS) if str(v).isdigit()]
    start_id = max(bookmark_ids, default=0) + 1
    links = {key: {"id": str(start_id + i), "name": "rs_ev_" + hashlib.sha256(key.encode()).hexdigest()[:20],
                   "prefix": f"[{key}] {url}"}
             for i, (key, url) in enumerate(payload.get("citation_sources", {}).items())}
    content: list[tuple[str, bool]] = []
    for heading, paragraphs in sections.items():
        content.append((heading, True))
        content.extend((para, False) for para in paragraphs)
    if references and ("references" not in info["slots"] or info["mode"] == "styles"):
        content += [("参考资料", True), *[(r, False) for r in references]]
    changed = {}
    for name, data in parts.items():
        if not re.fullmatch(r"word/(?:document|header\d+|footer\d+)\.xml", name):
            continue
        root = _xml(data)
        before = ET.tostring(root)
        if name == "word/document.xml":
            body = root.find(_W + "body")
            if info["mode"] == "styles":
                prototype = next((p for p in body if p.tag == _W + "p" and not _is_heading(p, _heading_styles(parts))), ET.Element(_W + "p"))
                title_prototype = next((p for p in body if p.find("./" + _W + "pPr/" + _W + "pStyle[@" + _W + "val='Title']") is not None), None)
                for child in list(body):
                    if child.tag != _W + "sectPr":
                        body.remove(child)
                title = _paragraph(title_prototype if title_prototype is not None else ET.Element(_W + "p"), payload["title"])
                if title_prototype is None:
                    props = ET.Element(_W + "pPr")
                    style = ET.SubElement(props, _W + "pStyle")
                    style.set(_W + "val", "Title")
                    title.insert(0, props)
                body.insert(0, title)
                anchor = _paragraph(prototype, "")
                body.insert(1, anchor)
                _replace_block(anchor, content, heading_style=heading_style, links=links)
            elif info["mode"] == "outline":
                headings = [p for p in body if p.tag == _W + "p" and _text(p).strip() in info["sections"]]
                for p in reversed(headings):
                    heading = _text(p).strip()
                    following = p.getnext()
                    prototype = following if (following is not None and following.tag == _W + "p"
                        and not _is_heading(following, styles) and following.find(".//" + _W + "sectPr") is None) else ET.Element(_W + "p")
                    while following is not None and following not in headings and following.tag != _W + "sectPr":
                        nxt = following.getnext()
                        if following.tag == _W + "p" and following.find(".//" + _W + "sectPr") is None:
                            body.remove(following)
                        following = nxt
                    anchor = deepcopy(prototype)
                    # Preserve heading format, but do not use a heading style for its body.
                    if _is_heading(prototype, styles):
                        props = anchor.find(_W + "pPr")
                        if props is not None:
                            for tag in ("pStyle", "outlineLvl", "keepNext"):
                                for child in list(props.findall(_W + tag)):
                                    props.remove(child)
                    body.insert(body.index(p) + 1, anchor)
                    _replace_block(anchor, [(para, False) for para in sections.get(heading, ["本节暂无可用内容。请查看证据缺口说明。 "])], links=links)
                for p in body:
                    style = p.find("./" + _W + "pPr/" + _W + "pStyle")
                    if style is not None and style.get(_W + "val", "").casefold() == "title":
                        _replace_block(p, [(payload["title"], False)])
                        break
                if references:
                    anchor = ET.Element(_W + "p")
                    body.insert(len(body) - int(body[-1].tag == _W + "sectPr"), anchor)
                    _replace_block(anchor, [("参考资料", True), *[(r, False) for r in references]], links=links)
        for p in list(root.iter(_W + "p")):
            text = _text(p).strip()
            match = _SLOT.fullmatch(text)
            if match and name == "word/document.xml":
                key = _key(match[1])
                if key == "content":
                    _replace_block(p, content, heading_style=heading_style, links=links)
                    continue
                if key == "references":
                    _replace_block(p, [(r, False) for r in references], links=links)
                    continue
                if key.startswith("section:"):
                    _replace_block(p, [(para, False) for para in sections.get(key[8:], ["本节暂无可用内容。请查看证据缺口说明。 "])], links=links)
                    continue
            _inline(p, values)
        if name == "word/document.xml" and payload.get("appendix"):
            body = root.find(_W + "body")
            anchor = ET.Element(_W + "p")
            body.insert(len(body) - int(body[-1].tag == _W + "sectPr"), anchor)
            lines = [("附录 已审核的补充内容", True)]
            for heading, paragraphs in payload["appendix"]:
                lines += [(heading, True), *[(para, False) for para in paragraphs]]
            _replace_block(anchor, lines, links=links)
        if (name == "word/document.xml" and info["mode"] == "slots" and "content" not in info["slots"]
                and "references" not in info["slots"] and references):
            body = root.find(_W + "body")
            anchor = ET.Element(_W + "p")
            body.insert(len(body) - int(body[-1].tag == _W + "sectPr"), anchor)
            _replace_block(anchor, [("参考资料与证据", True), *[(r, False) for r in references]], links=links)
        if ET.tostring(root) != before:
            changed[name] = ET.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)
    output = io.BytesIO()
    field_instructions = []
    for name, data in parts.items():
        if re.fullmatch(r"word/(?:document|header\d+|footer\d+)\.xml", name):
            root = _xml(data)
            field_instructions += root.xpath("//w:fldSimple/@w:instr|//w:instrText/text()", namespaces=_NS)
    if "word/settings.xml" in parts and re.search(r"\b(?:PAGE|NUMPAGES|TOC|REF|PAGEREF)\b", " ".join(field_instructions)):
        settings = _xml(parts["word/settings.xml"])
        field = settings.find(_W + "updateFields")
        if field is None:
            field = ET.SubElement(settings, _W + "updateFields")
        field.set(_W + "val", "true")
        changed["word/settings.xml"] = ET.tostring(settings, encoding="UTF-8", xml_declaration=True, standalone=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, changed.get(name, data))
    return output.getvalue()


def artifact_path(identifier: str, extension: str) -> Path:
    if extension not in {"docx", "pdf"}:
        raise TemplateError("不支持的导出格式。")
    root = runtime.user_data_dir() / "generated_documents"
    root.mkdir(parents=True, exist_ok=True)
    return storage_path(root, identifier, "." + extension)


def export_document(identifier: str, template: tuple[dict, bytes], payload: dict, fields: dict) -> dict:
    from core_engine.app.word_pdf import convert_docx_pdf

    info, raw = template
    docx = artifact_path(identifier, "docx")
    _write_bytes(docx, render_template(raw, info, payload, fields))
    result = {"template_id": info["id"], "template_name": info["name"], "has_docx": True,
              "has_pdf": False, "warnings": list(payload.get("warnings", []))}
    try:
        convert_docx_pdf(docx, artifact_path(identifier, "pdf"))
        result["has_pdf"] = True
    except Exception:
        artifact_path(identifier, "pdf").unlink(missing_ok=True)
        result["warnings"].append("模板 Word 已生成；PDF 转换未完成，可下载 Word 后另存为 PDF。需要本机 Word 或 LibreOffice。")
    return result


def article_payload(article: dict) -> dict:
    sections = {}
    for part in re.split(r"(?m)^## ", article["content"]):
        if not part.strip():
            continue
        heading, _, body = part.partition("\n")
        sections[heading.strip()] = [p.strip() for p in body.strip().split("\n\n") if p.strip()]
    abstract = next(iter(sections.values()), [""])[0]
    if article.get("notice"):
        sections = {"写作说明": [article["notice"]], **sections}
        if article.get("review_status") in {"partial", "unavailable"}:
            sections["写作说明"].append("自动行文复核有部分未完成，请结合人工检查使用。")
    references = [f"{s['title']}：{s['url']}（背景资料，未逐条核验）" for s in article.get("sources", [])]
    seen_urls = {s["url"] for s in article.get("sources", [])}
    for source in article.get("manual_review", {}).get("references", []):
        if source["url"] not in seen_urls:
            references.append(f"{source['title']}：{source['url']}（内容复核资料）")
            seen_urls.add(source["url"])
    for heading in article.get("template_sections", []):
        if heading.casefold() in {"参考文献", "参考资料", "references", "bibliography"}:
            sections[heading] = references or ["本次未采用联网参考资料。"]
            references = []
    return {"title": article["title"], "date": article["created_at"][:10], "sections": sections,
            "abstract": abstract, "references": references}


def report_payload(report, template_info: dict) -> dict:
    notice = ("本报告为待复核研究稿，仍有证据缺口，请结合下文的研究问题覆盖说明使用。"
              if report.quality_status != "source_supported" else "本报告正文已完成原文定位检查。")
    notice += "原文支持与多来源支持描述的是证据关系，不保证事实真实、独立复现或适用于量产电芯。"
    sections = {"摘要": [notice, report.abstract]}
    questions = report.research_audit.get("questions", [])
    if questions:
        complete = sum(q.get("status") == "supported" for q in questions)
        status = {"supported": "完整覆盖", "partial": "部分覆盖", "missing": "缺少证据", "undetermined": "尚未确定"}
        sections["研究问题覆盖与证据缺口"] = [f"研究问题完整覆盖 {complete}/{len(questions)}。覆盖判断由模型与规则共同完成，仍需专业复核。"]
        for row in questions:
            missing = row.get("missing") or []
            if isinstance(missing, str):
                missing = [missing]
            missing = [re.sub(r"(?<![A-Za-z0-9_])c\d+(?![A-Za-z0-9_])", "现有证据", str(item)).strip().rstrip("。；; ") for item in missing]
            sections["研究问题覆盖与证据缺口"].append(str(row.get("question", "")) + "——"
                + status.get(row.get("status"), "尚未确定")
                + ("；缺口：" + "；".join(missing) + "。" if missing else ""))
    used_citations = {eid for section in report.sections for container in [section, *section.deep_dives]
                      for sentence in container.sentences if sentence.status in {"supported", "editorial"}
                      for eid in sentence.citation_ids}
    for section in report.sections:
        paragraphs = [s.attribution + s.text + " " + " ".join(f"[{c}]" for c in s.citation_ids)
                      for s in section.sentences if s.status in {"supported", "editorial"}]
        for dive in section.deep_dives:
            paragraphs += [dive.title, *[s.attribution + s.text + " " + " ".join(f"[{c}]" for c in s.citation_ids)
                           for s in dive.sentences if s.status in {"supported", "editorial"}]]
        sections[section.heading] = paragraphs or ["本节暂无通过逐句检查的正文，详见证据文件。"]
    if report.metrics:
        labels = {"market_size": "市场规模", "market_share": "市场份额", "revenue": "营收", "net_profit": "净利润",
                  "growth_rate": "同比增长", "cagr": "年复合增长", "share": "占比", "unit_conversion": "单位换算",
                  "ionic_conductivity": "离子电导率", "electronic_conductivity": "电子电导率",
                  "gravimetric_energy_density": "质量能量密度", "volumetric_energy_density": "体积能量密度",
                  "specific_capacity": "比容量", "areal_capacity": "面容量", "current_density": "电流密度",
                  "cycle_life": "循环寿命", "capacity_retention": "容量保持率",
                  "coulombic_efficiency": "库仑效率", "interfacial_impedance": "界面阻抗"}
        sections["指标与计算"] = [" ".join(str(labels.get(str(row.get(k)), row.get(k)) or "") for k in ("entity", "name", "period"))
            + f"：{row.get('value', '')} {row.get('unit', '')} " + " ".join(f"[{eid}]" for eid in row.get("evidence_ids", [])) for row in report.metrics]
        used_citations.update(eid for row in report.metrics for eid in row.get("evidence_ids", []))
        sections["指标与计算"] += [f"{labels.get(row.get('operation'), '计算结果')}（{row.get('period', '')}）："
            f"{row.get('value', '')} {row.get('unit', '')}；计算过程详见证据文件。" for row in report.calculations]
        scientific = report.research_audit.get("scientific_comparisons", {})
        if scientific.get("observations"):
            sections["实验条件与比较缺口"] = [("以下为来源报告的观测。性能排名需要可比较的条件。缺口仅针对当前绑定的原文。"
                                              "带 M 编号的条件可在本节查阅同篇论文的方法原文。")]
            for row in scientific["observations"]:
                refs = row.get("method_ids", {})
                conditions = "、".join(f"{key} {value}" + (f" [{refs[key]}]" if key in refs else "")
                                     for key, value in row.get("conditions", {}).items() if value)
                missing = "; ".join(row.get("missing_labels", [])) or "所列条件齐全"
                sections["实验条件与比较缺口"].append(
                    f"{row.get('entity') or '样品未说明'} / {labels.get(row.get('name'), row.get('name'))}："
                    f"{row['value']} {row['unit']}；边界：{row.get('boundary') or '未报告'}；"
                    f"条件：{conditions or '未报告'}；比较所缺条件：{missing} "
                    + " ".join(f"[{eid}]" for eid in row.get("evidence_ids", [])))
            for witness in scientific.get("methods", []):
                section_title = witness["locator"].get("section_title", "")
                sections["实验条件与比较缺口"].append(
                    f"[{witness['id']}] {section_title} 字符范围 [{witness['start']}, {witness['end']})。"
                    f" {witness['source_url']}\n{witness['quote']}")
    if report.analysis:
        sections["分析与限制"] = ["以下为基于证据的分析推断，需人工复核，不是原文已证实的结论。"]
        for row in report.analysis:
            sections["分析与限制"].append(str(row.get("text") or row.get("conclusion") or row.get("finding") or "")
                + " " + " ".join(f"[{eid}]" for eid in row.get("citation_ids", [])))
            if row.get("assumptions"):
                sections["分析与限制"].append("成立条件：" + "；".join(item.strip().rstrip("。；; ") for item in row["assumptions"]))
            if row.get("alternatives"):
                sections["分析与限制"].append("其他解释：" + "；".join(item.strip().rstrip("。；; ") for item in row["alternatives"]))
            if row.get("uncertainty"):
                sections["分析与限制"].append("不确定性：" + row["uncertainty"])
        used_citations.update(eid for row in report.analysis for eid in row.get("citation_ids", []))
    # Analytical premises / metrics use research citation IDs, which need not be
    # the final drafted-prose IDs. Preserve their independently located spans too.
    from types import SimpleNamespace

    by_id = {c.id: c for c in report.citations}
    sources = {s.url: s for s in report.sources}
    for row in report.research_evidence.get("citations", []):
        if row.get("id") not in used_citations or row.get("id") in by_id:
            continue
        source = sources.get(row.get("source_url"))
        start, end = row.get("start_char"), row.get("end_char")
        if (source and type(start) is int and type(end) is int and 0 <= start < end <= len(source.text)
                and source.text[start:end] == row.get("quote") and row.get("quote")
                and source.content_sha256 == row.get("source_sha256")):
            by_id[row["id"]] = SimpleNamespace(**row)
    premise_ids = {eid for row in report.analysis for eid in row.get("citation_ids", [])}
    if premise_ids - by_id.keys():
        raise TemplateError("分析引用缺少可定位原文，无法导出，请先重新核验证据。")
    citations = [c for identifier, c in by_id.items() if identifier in used_citations]
    urls = {c.source_url for c in citations}
    references = [f"[{s.url}] {s.title}" for s in report.bibliography() if s.url in urls]
    references += [f"[{c.id}] {c.source_url}，原文位置 {c.start_char}–{c.end_char}：{c.quote}" for c in citations]
    remaining = []
    if template_info["sections"]:
        mapped = {}
        aliases = {"industry_overview": ("行业概况", "行业概述", "行业现状", "背景"), "policy_analysis": ("政策分析", "政策"),
                   "industry_chain": ("产业链", "产业链分析"), "market_size": ("市场规模",),
                   "competitive_landscape": ("竞争格局", "竞争分析")}
        for heading in template_info["sections"]:
            if heading.casefold() in {"参考文献", "参考资料", "references", "bibliography"}:
                mapped[heading] = references
                references = []
                continue
            found = sections.get(heading)
            if found is None:
                match = next((s for s in report.sections if heading.casefold() == s.section_id.casefold()
                    or any(a == heading for a in aliases.get(s.section_id, ()))), None)
                found = sections.get(match.heading) if match else None
            mapped[heading] = found or ["现有已核验内容未覆盖本章节，不能据模板要求补造结论。详见证据文件。"]
        # Do not discard verified content simply because a template lacks a matching slot.
        remaining = [(h, ps) for h, ps in sections.items() if ps not in mapped.values()]
        sections = mapped
    return {"title": report.title, "abstract": report.abstract, "date": report.generated_at[:10],
            "sections": sections, "references": references, "appendix": remaining,
            "citation_sources": {c.id: c.source_url for c in citations},
            "warnings": ["模板 Word 中的结构化指标以数据文字呈现，原始图表与计算审计保留在证据文件中。"]
                if any(s.latex_blocks for s in report.sections) else []}


def example_template() -> bytes:
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Cm(2.5)
    section.left_margin = section.right_margin = Cm(2.8)
    for name in ("Normal", "Title", "Heading 1"):
        style = doc.styles[name]
        style.font.name = "宋体"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
        for border in style.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    doc.styles["Normal"].font.size = Pt(11)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.5
    doc.add_paragraph("{{title}}", style="Title")
    doc.add_paragraph("作者：{{作者}}    课程：{{课程}}")
    doc.add_paragraph("日期：{{date}}")
    doc.add_paragraph("{{content}}")
    section.header.paragraphs[0].text = "课程报告"
    footer = section.footer.paragraphs[0]
    footer.text = "第 "
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    footer.add_run(" 页")
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()
