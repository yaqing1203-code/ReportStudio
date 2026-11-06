"""Link explicit measurement protocols to compatible rows in primary JATS articles."""
from __future__ import annotations

import re

from core_engine.report.metrics import _experiment_key
from core_engine.report.models import Source

MAX_METHOD_SPANS = 64
MAX_METHOD_CHARS = 4000

_HEADINGS = {"methods", "materials and methods", "material and methods", "experimental",
             "experimental section", "experimental methods", "methodology"}
_KIND = r"(ionic|electronic)\s+conductivit(?:y|ies)"
_MENTION = re.compile(r"\b" + _KIND + r"\b", re.IGNORECASE)
_JOINT = re.compile(r"\b(?:ionic|electronic)\s*(?:and|/|&)\s*(?:ionic|electronic)\s+conductivit(?:y|ies)\b",
                    re.IGNORECASE)
_UNNAMED = re.compile(r"\b(?<!ionic )(?<!electronic )conductivit(?:y|ies)\s+"
                      r"(?:was|were)\s+(?:measured|determined)\b")
_INTRO = re.compile(
    r"\b(?:the\s+)?" + _KIND + r"(?:\s+of\s+all\s+(?:the\s+)?samples)?\s+"
    r"(?:was|were)\s+(?:measured|determined)\b|\bfor\s+(?:the\s+)?" + _KIND + r"\s+measurements\b",
    re.IGNORECASE,
)
_MEASURE = re.compile(r"(?:^|[.!?,;]\s*|\band\s+)(?:the\s+)?measurements\s+"
                      r"(?:were|was)\s+(?:performed|conducted|taken|made)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_SETUP = re.compile(r"with\s+(?:an?|the)\s+[^,;]{1,100}\s+(?:cell|setup),", re.IGNORECASE)
_TABLE_SCOPE = re.compile(
    r"for\s+(?:all\s+)?(?:the\s+)?(?:samples|specimens|measurements)\s+(?:in|listed in|reported in)\s+"
    r"(table\s+[A-Z]?\d+)\s*,", re.IGNORECASE,
)
_QUALIFIED = re.compile(
    r"\b(?:only|except|subset|selected\s+samples?|respectively|depending\s+on|whereas|alternatively|or)\b"
    r"|\bfor\s+(?:(?:the|this|that|different|other|some|these|each|such)\s+)?"
    r"(?:samples?|specimens?|pellets?|batch|series)\b"
    r"|\b(?:not|without)\s+(?:using|by)\b",
    re.IGNORECASE,
)
_PREP = re.compile(r"[;]|\b(?:cold[-‐– ]?press(?:ed|ing)?|press(?:ed|ing)|dry(?:ing|ied)|"
                   r"anneal(?:ed|ing)?|sinter(?:ed|ing)?|fabricat(?:ed|ion)|prepar(?:ed|ation)|compact(?:ed|ion))\b",
                   re.IGNORECASE)
_METHOD = re.compile(r"\b(?:using|by)\s+(?:an?\s+)?"
                     r"(electrochemical impedance spectroscopy|EIS|direct[- ]current polarization|DC polarization)\b",
                     re.IGNORECASE)
_QUANTITY = (r"([+\-−]?\d+(?:\.\d+)?)\s*(°\s*C|K|GPa|MPa|kPa|Pa|bar)\b"
             r"(?!\s*(?:[~±/–−-]|to\b|or\b))")
_AT = re.compile(r"(?:at|under)\s+(?:(?:a\s+)?(?:pressure|temperature)\s+of\s+)?"
                 + _QUANTITY + r"(?:\s+and\s+" + _QUANTITY + r")?")
_AT_CONTINUATION = re.compile(r"\s*(?:and\s+)?")
_METHOD_SETUP = re.compile(r"\s*(?:\([^)]{1,120}\)\s*)?")
_NONSCALAR = re.compile(r"\d\s*(?:[~±–−-]|\bto\b)\s*\d+(?:\.\d+)?\s*(?:°\s*C|K|GPa|MPa|kPa|Pa|bar)\b")
_LITERATURE = re.compile(r"\bliterature\b|\b(?:previously|elsewhere)\s+(?:published|reported)\b|"
                         r"\b(?:other|prior)\s+(?:studies|work)\b|文献", re.IGNORECASE)
_LITERATURE_HEADER = re.compile(r"^\s*(?:references?|refs?\.?|source|doi)\b", re.IGNORECASE)
_OBSERVED_CAPTION = re.compile(r"\b(?:synthesi[sz]ed|prepared|measured)\b", re.IGNORECASE)


def jats_methods_heading(heading: str, sec_type: str) -> bool:
    heading = re.sub(r"\s+", " ", heading.casefold().replace("&", "and")).strip()
    return heading in _HEADINGS or sec_type.casefold() in {"methods", "materials|methods", "methodology"}


def _within_methods_heading(text: str, start: int, end: int, locator: dict) -> bool:
    heading = locator.get("methods_heading")
    if not isinstance(heading, dict):
        return False
    left, right, section_end = (heading.get(key) for key in ("start", "end", "section_end"))
    title = heading.get("text")
    return (type(left) is int and type(right) is int and type(section_end) is int
            and 0 <= left < right <= start < end <= section_end <= len(text)
            and isinstance(title, str) and jats_methods_heading(title, "")
            and text[left:right] == title
            and (left == 0 or text[max(0, left - 2):left] == "\n\n")
            and text[right:right + 2] == "\n\n")


def _conditions(protocol: str, introduction: re.Match[str]) -> dict[str, str]:
    values: dict[str, set[str]] = {}
    markers = list(_MEASURE.finditer(protocol))
    if not introduction[0].casefold().endswith("measurements"):
        markers.insert(0, introduction)
    for marker in markers:
        stop = _SENTENCE_END.search(protocol, marker.end())
        tail = protocol[marker.end():stop.start() if stop else len(protocol)]
        # The closest measurement verb governs its conditions. Preparation
        # clauses elsewhere in the same paragraph cannot donate a pressure.
        tail = _PREP.split(tail, maxsplit=1)[0].lstrip()
        if _NONSCALAR.search(tail):
            return {}
        # A condition must follow the measurement verb or its named method
        # directly. Equipment ratings elsewhere in the clause do not count.
        leading_method = _METHOD.match(tail)
        direct = _METHOD_SETUP.sub("", tail[leading_method.end():], count=1) if leading_method else tail
        while match := _AT.match(direct):
            for number, unit in ((match[1], match[2]), (match[3], match[4])):
                if number is None or unit is None:
                    continue
                unit = "".join(unit.split())
                key = "temperature" if unit in {"°C", "K"} else "pressure"
                values.setdefault(key, set()).add(f"{number} {unit}")
            direct = _AT_CONTINUATION.sub("", direct[match.end():], count=1)
        for method_match in _METHOD.finditer(tail):
            method = method_match[1]
            normalized_method = _experiment_key("test_method", method)
            values.setdefault("test_method", set()).add("EIS" if normalized_method == "eis" else method)
    # Two different values inside one nominal protocol indicate a variant, not
    # permission to choose the one matching the desired row.
    if any(len({_experiment_key(key, value) for value in observed}) != 1
           for key, observed in values.items()):
        return {}
    return {key: min(observed) for key, observed in values.items()}


def method_condition_evidence(
    source: Source, table: dict, row: dict, metric_name: str, row_conditions: dict[str, str],
) -> dict[str, dict]:
    """Return independently anchored conditions absent from the observed row."""
    table_locator = table.get("locator")
    if (metric_name not in {"ionic_conductivity", "electronic_conductivity"}
            or source.document_format != "jats"
            or source.metadata.get("article_type_declared") != "research-article"
            or source.metadata.get("full_text_accessed") is not True
            or source.metadata.get("methods_spans_complete") is not True
            or not isinstance(table_locator, dict) or table_locator.get("format") != "jats"):
        return {}
    caption = table.get("caption", "")
    if (not isinstance(caption, str) or _LITERATURE.search(caption) or table.get("has_bibliography_links")
            or any(_LITERATURE_HEADER.match(header) for header in table["headers"])):
        return {}
    spans = source.metadata.get("methods_spans", [])
    if not isinstance(spans, list) or not 0 < len(spans) <= MAX_METHOD_SPANS:
        return {}

    matched = []
    kind = metric_name.split("_", 1)[0]
    for span in spans:
        if not isinstance(span, dict):
            return {}
        text, start, end = span.get("text"), span.get("start"), span.get("end")
        locator = span.get("locator")
        if (not isinstance(text, str) or not 0 < len(text) <= MAX_METHOD_CHARS
                or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source.text)
                or source.text[start:end] != text or not isinstance(locator, dict)
                or locator.get("format") != "jats" or _JOINT.search(text)
                or not _within_methods_heading(source.text, start, end, locator)
                or _UNNAMED.search(" ".join(text.casefold().split()))):
            return {}
        mentions = list(_MENTION.finditer(text))
        if not any(mention[1].casefold() == kind for mention in mentions):
            continue
        boundaries = [match.end() for match in _SENTENCE_END.finditer(text)]
        starts = [max((point for point in boundaries if point <= mention.start()), default=0)
                  for mention in mentions]
        for index, mention in enumerate(mentions):
            if mention[1].casefold() != kind:
                continue
            local_start = starts[index]
            local_end = starts[index + 1] if index + 1 < len(starts) else len(text)
            quote = text[local_start:local_end].rstrip()
            intro = _INTRO.search(quote)
            if (intro is None or not intro.start() <= mention.start() - local_start < intro.end()
                    or _QUALIFIED.search(quote[intro.start():])):
                return {}
            prefix = quote[:intro.start()].strip()
            explicit_scope = _TABLE_SCOPE.fullmatch(prefix)
            applicability = "unique_article_measurement_protocol"
            if explicit_scope:
                ref_label = " ".join(explicit_scope[1].casefold().split())
                caption_label = " ".join(caption.casefold().split())
                refs = span.get("table_refs")
                if not isinstance(refs, list) or not re.match(re.escape(ref_label) + r"(?!\w)", caption_label):
                    return {}
                table_ids = {ref["id"] for ref in refs
                             if isinstance(ref, dict) and isinstance(ref.get("text"), str)
                             and isinstance(ref.get("id"), str) and ref_label in {
                                 " ".join(ref["text"].casefold().split()),
                                 "table " + " ".join(ref["text"].casefold().split())}}
                if len(table_ids) != 1 or not table_ids.issubset(
                        {table_locator.get("element_id"), table_locator.get("container_id")}):
                    return {}
                applicability = "explicit_table_reference"
            elif prefix and not _SETUP.fullmatch(prefix):
                return {}
            observed = _conditions(quote, intro)
            if not observed:
                return {}
            matched.append((observed, quote, start + local_start, locator, applicability))
    if len(matched) != 1:
        return {}
    observed, quote, start, locator, applicability = matched[0]
    end = start + len(quote)
    if start < row["end"] and row["start"] < end:
        return {}
    if any(_experiment_key(key, row_conditions[key]) != _experiment_key(key, value)
           for key, value in observed.items() if key in row_conditions):
        return {}
    if applicability == "unique_article_measurement_protocol" and (
            "temperature" not in row_conditions or "temperature" not in observed
            or not _OBSERVED_CAPTION.search(caption)):
        return {}
    version = source.content_sha256
    return {key: {"source_url": source.url, "source_version": version, "quote": quote, "start": start, "end": end,
                  "locator": locator, "applicability": applicability, "table_id": table["id"],
                  "row_index": row["row_index"], "value": value}
            for key, value in observed.items() if key not in row_conditions}
