"""Shared Unicode text and numeric handling for retrieval, claims and audits.

Normalization deliberately preserves signs, decimal points, units and negation.
It is an identity key, not a semantic-equivalence or unit-conversion engine.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

_CJK = re.compile(r"[\u3400-\u9fff]+")
_WORDS = re.compile(r"[a-z][a-z0-9_]*|\d+(?:\.\d+)?", re.IGNORECASE)
_SCIENTIFIC_UNIT = (
    r"(?:(?:mAh|Ah|mA\s+h)\s*(?:/\s*(?:kg|g|cm\^?2|m\^?2)|[· ]\s*(?:kg|g)\s*\^?-1|"
    r"[· ]\s*(?:cm|m)\s*\^?-2)|"
    r"(?:mWh|kWh|Wh|W\s+h)\s*(?:/\s*(?:kg|g|L|l|cm\^?3|m\^?3)|[· ]\s*(?:kg|g|L)\s*\^?-1|"
    r"[· ]\s*(?:cm|m)\s*\^?-3)|"
    r"(?:mS|μS|uS|S)\s*(?:/\s*(?:cm|m)(?:\^?1)?|[· ]\s*(?:cm|m)\s*\^?-1)|"
    r"(?:mA|μA|uA|A|mg|g)\s*(?:/\s*(?:cm|m)\^?2|[· ]\s*(?:cm|m)\s*\^?-2)|"
    r"(?:mΩ|Ω|mohm|ohms?)\s*(?:[· ]\s*(?:cm|m)\s*\^?2)?|"
    r"wt\s*\.?%|mol\s*%|MPa|kPa|GPa|Pa|°C|K|μm|um|nm|mm|cm|m\^?3|cm\^?3|m|"
    r"cycles?|次循环|循环|圈|次)(?![a-z\d/^·])"
)
_NUMBER = re.compile(
    r"(?<![\d.])(?P<currency>US\$|USD\s*|CNY\s*|RMB\s*|[$¥￥])?"
    r"(?P<value>[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?)"
    r"\s*(?P<unit>" + _SCIENTIFIC_UNIT + r"|"
    r"percentage\s+points?|basis\s+points?|percent|pct|bps?|"
    r"USD\s*bn|USD\s*million|USD\s*billion|billion|million|trillion|"
    r"亿美元|万美元|亿元|万元|人民币|美元|个百分点|百分点|个基点|基点|%|‰|亿|万|千|元|"
    r"gigawatts?|megawatts?|GWh|MWh|kWh|GW|MW|年)?", re.IGNORECASE)
_PRODUCT_COUNT = re.compile(
    r"(?<![\d零一二两三四五六七八九十百千万])(?P<count>\d{1,6}|[一二两三四五六七八九十]{1,3})"
    r"\s*款\s*(?:新型|全固态|半固态|固态|锂离子|锂金属|动力|储能)?(?P<entity>电池|电芯|车型|产品|器件)")


def _count_integer(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {ch: index for index, ch in enumerate("零一二三四五六七八九")}
    digits["两"] = 2
    if len(value) == 1 and value in digits:
        return digits[value]
    if value.count("十") == 1:
        tens, ones = value.split("十")
        if (not tens or len(tens) == 1 and tens in digits) and (not ones or len(ones) == 1 and ones in digits):
            return (digits[tens] if tens else 1) * 10 + (digits[ones] if ones else 0)
    return None


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).replace("\u2212", "-")
    # Publication typography must not hide an otherwise identical scientific term.
    # Raw source snapshots/offsets stay unchanged; ranges and signs remain present.
    normalized = normalized.replace("\u2010", "-").replace("\u2011", "-").replace("\u00ad", "")
    return " ".join(normalized.casefold().split())


def lexical_tokens(text: str) -> list[str]:
    """English words/numbers and overlapping CJK bigrams, retaining token frequency."""
    normalized = normalize_text(text)
    tokens = _WORDS.findall(normalized)
    for match in _CJK.finditer(normalized):
        run = match.group()
        tokens.extend([run] if len(run) == 1 else (run[i:i + 2] for i in range(len(run) - 1)))
    return tokens


def content_terms(text: str, stopwords=frozenset()) -> set[str]:
    return {token for token in lexical_tokens(text)
            if token not in stopwords and (len(token) > 1 or token.isdigit())}


def numeric_tokens(text: str) -> set[str]:
    """Decimal-safe signatures. 3.50% == 3.5 percent; 3.5%, 35%, -3.5 and 3.5 pp differ.

    Scale/currency conversions intentionally require the later calculation layer.
    """
    units = {
        "percent": "%", "pct": "%", "percentage point": "pp", "percentage points": "pp",
        "个百分点": "pp", "百分点": "pp", "basis point": "bp", "basis points": "bp",
        "bps": "bp", "基点": "bp", "个基点": "bp",
        "billion": "bn", "trillion": "tn", "million": "mn", "usd billion": "usd bn",
        "usd million": "usd mn", "gigawatt": "gw", "gigawatts": "gw",
        "megawatt": "mw", "megawatts": "mw", "年": "",
    }
    result = set()
    table_text = text
    # Preserve a true superscript exponent before NFKC erases its position.
    text = re.sub(r"(?<![\w.])10([⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
                  lambda match: "10^" + unicodedata.normalize("NFKC", match[1]), text)
    text = " ".join(unicodedata.normalize("NFKC", text).replace("\u2212", "-").split())
    exponent = r"(\{[+-]?\d+\}|[+-]?\d+)(?![\d}]|\.\d)"
    text = re.sub(r"(?<![\w.])([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[×*]\s*10\s*\^?\s*" + exponent,
                  lambda match: match[1] + "e" + match[2].strip("{}"), text)
    text = re.sub(r"(?<![\w.])10\s*\^\s*" + exponent,
                  lambda match: "1e" + match[1].strip("{}"), text)
    text = re.sub(r"百分之\s*([+-]?\d+(?:\.\d+)?)", r"\1%", text)
    text = re.sub(r"(?<![\d.])\d{1,3}(?:[ \u202f]\d{3})+(?!\d)",
                  lambda match: re.sub(r"[ \u202f]", "", match.group()), text)
    for match in _NUMBER.finditer(text):
        try:
            value = Decimal(match["value"].replace(",", ""))
            # Avoid expanding untrusted scientific notation into unbounded strings.
            if abs(value.adjusted()) > 100:
                result.add(match.group().strip())
                continue
            number = format(value, "f")
            if "." in number:
                number = number.rstrip("0").rstrip(".")
            if value == 0:
                number = "0"
        except InvalidOperation:
            continue
        raw_unit = " ".join((match["unit"] or "").split())
        unit = raw_unit.casefold()
        if raw_unit and re.fullmatch(_SCIENTIFIC_UNIT, raw_unit, re.IGNORECASE):
            from core_engine.report.metrics import MetricError, parse_unit

            try:
                unit = parse_unit(raw_unit).symbol
            except MetricError:
                # Unsupported or wrong-case scientific spelling stays distinct;
                # never collapse an apparent physical unit into a bare number.
                unit = "scientific:" + re.sub(r"\s+", "", raw_unit)
        unit = units.get(unit, unit)
        currency = (match["currency"] or "").strip().casefold()
        currency = {"$": "usd", "us$": "usd", "¥": "cny", "rmb": "cny"}.get(currency, currency)
        result.add(f"{currency}:{number}:{unit}")
    # A source table states units in its header, not after every numeric cell.
    # Reconstruct signatures only from explicitly labelled scalar/unit pairs;
    # scientific catalog checks still independently bind metric and test scope.
    if ":" in text or "：" in text:
        from core_engine.report.metrics import labelled_observations

        for value, unit in labelled_observations(table_text):
            result.update(numeric_tokens(f"{value} {unit}"))
    # Product counts are factual quantities too. Two measurements from one cell
    # must not silently become "two cells" merely because Chinese 二/两 was ignored.
    for match in _PRODUCT_COUNT.finditer(text):
        count = _count_integer(match["count"])
        if count is not None:
            result.update({f":{count}:", f"count:{count}:{match['entity']}"})
    return result


def numbers_supported(text: str, evidence: str) -> bool:
    return numeric_tokens(text).issubset(numeric_tokens(evidence))


def split_sentences(text: str) -> list[str]:
    """Split CJK/English prose without splitting decimals, initials or common abbreviations."""
    text = str(text).strip()
    if not text:
        return []
    protected = re.sub(r"\b(?:e\.g\.|i\.e\.|U\.S\.|U\.K\.|Mr\.|Mrs\.|Dr\.|vs\.)",
                       lambda m: m.group().replace(".", "\u0001"), text, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+|[\r\n\f]+", protected)
    return [part.replace("\u0001", ".").strip() for part in parts if part.strip()]
