"""Source-backed metric observations, explicit units and comparison boundaries.

Unknown scope stays unknown. Currency conversion and implicit actual/forecast,
nominal/real, country or population assumptions are never supplied by this module.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation, localcontext
from typing import Any

from core_engine.report.models import Claim, SentenceCitation, Source
from core_engine.report.text import normalize_text, split_sentences


class MetricError(ValueError):
    pass


_SUPERSCRIPT_DIGITS = str.maketrans("⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "+-0123456789")
_MANTISSA = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NUMBER = (r"(?:[+-]?10\s*\^\s*[+-]?\d+|" + _MANTISSA
           + r"(?:[eE][+-]?\d+|\s*[×x*·]\s*10\s*\^?\s*[+-]?\d+)?)")


def _notation(text: str) -> str:
    # Preserve the exponent marker before NFKC turns superscripts into plain
    # digits. In particular, 10⁻³ is not the range 10-3 or the integer 103.
    text = re.sub(r"(?<=10)[⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+",
                  lambda m: "^" + m[0].translate(_SUPERSCRIPT_DIGITS), str(text))
    text = unicodedata.normalize("NFKC", text).replace("\u2212", "-").replace("⋅", "·")
    return re.sub(r"(?<=\^)\{([+-]?\d+)\}", r"\1", text)


def _decimal_notation(value: Any) -> str:
    text = _notation(str(value)).strip()
    power = re.fullmatch(r"([+-]?)10\s*\^\s*([+-]?\d+)", text)
    if power:
        return f"{power[1]}1e{power[2]}"
    product = re.fullmatch("(" + _MANTISSA + r")\s*[×x*·]\s*10\s*\^?\s*([+-]?\d+)", text)
    if product:
        return f"{product[1]}e{product[2]}"
    return text


def decimal_value(value: Any) -> Decimal:
    if isinstance(value, (bool, float)):
        raise MetricError("Values must be decimal strings, integers or Decimal, not binary floats")
    text = _decimal_notation(value)
    if "," in text and not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        raise MetricError("Ambiguous decimal/grouping separator")
    try:
        result = Decimal(text.replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise MetricError("Invalid decimal value") from exc
    if not result.is_finite() or abs(result.adjusted()) > 50 or len(result.as_tuple().digits) > 50:
        raise MetricError("Non-finite or oversized metric value")
    return result


def decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(frozen=True)
class Unit:
    symbol: str
    dimension: str
    currency: str | None
    factor: Decimal
    offset: Decimal = Decimal(0)


_UNITS: dict[str, Unit] = {}
_SI_UNITS: dict[str, Unit] = {}


def _register(symbol: str, dimension: str, factor: str, aliases: tuple[str, ...], currency=None):
    unit = Unit(symbol, dimension, currency, Decimal(factor))
    for name in (symbol, *aliases):
        _UNITS[normalize_text(name)] = unit


_register("CNY", "money", "1", ("元", "人民币元", "RMB", "人民币"), "CNY")
_register("CNY 10k", "money", "10000", ("万元",), "CNY")
_register("CNY 100m", "money", "100000000", ("亿元", "亿人民币"), "CNY")
_register("CNY bn", "money", "1000000000", ("billion CNY", "RMB bn"), "CNY")
_register("CNY tn", "money", "1000000000000", ("万亿元", "trillion CNY"), "CNY")
_register("USD", "money", "1", ("美元", "$", "US$"), "USD")
_register("USD mn", "money", "1000000", ("million USD", "USD million"), "USD")
_register("USD 10k", "money", "10000", ("万美元",), "USD")
_register("USD bn", "money", "1000000000", ("billion USD", "USD billion"), "USD")
_register("USD 100m", "money", "100000000", ("亿美元",), "USD")
_register("EUR", "money", "1", ("欧元", "€"), "EUR")
_register("EUR mn", "money", "1000000", ("million EUR", "EUR million"), "EUR")
_register("EUR bn", "money", "1000000000", ("billion EUR", "EUR billion"), "EUR")
_register("%", "ratio", "0.01", ("percent", "pct", "百分比"))
_register("ratio", "ratio", "1", ("fraction",))
_register("pp", "rate_difference", "0.01", ("percentage points", "percentage point", "百分点", "个百分点"))
_register("bp", "rate_difference", "0.0001", ("basis points", "basis point", "bps", "基点", "个基点"))
_register("GW", "power", "1000000000", ("gigawatts", "gigawatt"))
_register("MW", "power", "1000000", ("megawatts", "megawatt"))
_register("kW", "power", "1000", ("kilowatts", "kilowatt"))
_register("W", "power", "1", ("watts", "watt"))
_register("GWh", "energy", "1000000000", ())
_register("MWh", "energy", "1000000", ())
_register("kWh", "energy", "1000", ())
_register("vehicles", "vehicles", "1", ("辆", "vehicle"))
_register("10k vehicles", "vehicles", "10000", ("万辆",))
_register("people", "people", "1", ("人",))
_register("10k people", "people", "10000", ("万人",))
_register("cycles", "cycles", "1", ("cycle", "次循环", "循环", "圈", "次"))


def _unit_key(value: str) -> str:
    return " ".join(_notation(value).strip().split())


def _scientific_unit(symbol: str, dimension: str, factor: str, aliases=(), *, offset="0"):
    unit = Unit(symbol, dimension, None, Decimal(factor), Decimal(offset))
    for alias in (symbol, *aliases):
        _SI_UNITS[_unit_key(alias)] = unit


def _inverse_units(symbol: str, stem: str, length: str, power: int) -> tuple[str, ...]:
    suffix = str(power)
    return (f"{stem} {length}-{suffix}", f"{stem} {length}^-{suffix}",
            f"{stem}·{length}-{suffix}", f"{stem}·{length}^-{suffix}",
            f"{stem}/{length}^{suffix}", symbol)


for stem, factor in (("S", "100"), ("mS", "0.1"), ("μS", "0.0001"), ("uS", "0.0001")):
    canonical = "μS/cm" if stem == "uS" else f"{stem}/cm"
    _scientific_unit(canonical, "conductivity", factor,
                    (f"{stem}/cm", *_inverse_units(canonical, stem, "cm", 1)))
_scientific_unit("S/m", "conductivity", "1", _inverse_units("S/m", "S", "m", 1))
_scientific_unit("mS/m", "conductivity", "0.001", _inverse_units("mS/m", "mS", "m", 1))
_scientific_unit("Wh/kg", "specific_energy", "1", ("W h/kg", "Wh kg-1", "Wh kg^-1", "Wh·kg-1"))
_scientific_unit("Wh/g", "specific_energy", "1000", ("Wh g-1", "Wh g^-1", "Wh·g-1"))
_scientific_unit("kWh/kg", "specific_energy", "1000", ("kWh kg-1", "kWh kg^-1"))
_scientific_unit("Wh/L", "volumetric_energy", "1000", ("Wh/l", "Wh L-1", "Wh L^-1", "Wh·L-1"))
_scientific_unit("Wh/m3", "volumetric_energy", "1", ("Wh/m^3", "Wh m-3", "Wh m^-3"))
_scientific_unit("mWh/cm3", "volumetric_energy", "1000", ("mWh/cm^3", "mWh cm-3", "mWh cm^-3"))
_scientific_unit("mAh/g", "specific_capacity", "1",
                ("mAh g-1", "mAh g^-1", "mAh·g-1", "mA h g-1", "mA h g^-1", "mA h/g"))
_scientific_unit("Ah/kg", "specific_capacity", "1", ("Ah kg-1", "Ah kg^-1"))
_scientific_unit("mAh/cm2", "areal_capacity", "10", _inverse_units("mAh/cm2", "mAh", "cm", 2))
_scientific_unit("Ah/m2", "areal_capacity", "1", _inverse_units("Ah/m2", "Ah", "m", 2))
for stem, factor in (("mA", "10"), ("A", "10000"), ("μA", "0.01"), ("uA", "0.01")):
    canonical = "μA/cm2" if stem == "uA" else f"{stem}/cm2"
    _scientific_unit(canonical, "current_density", factor,
                    (f"{stem}/cm2", *_inverse_units(canonical, stem, "cm", 2)))
_scientific_unit("A/m2", "current_density", "1", _inverse_units("A/m2", "A", "m", 2))
_scientific_unit("Ω", "resistance", "1", ("ohm", "Ohm", "ohms"))
_scientific_unit("mΩ", "resistance", "0.001", ("mohm",))
_scientific_unit("Ω cm2", "resistance_area", "0.0001",
                ("Ω cm^2", "Ω·cm2", "Ω·cm^2", "ohm cm2", "ohm cm^2", "ohm·cm2", "ohm cm²"))
_scientific_unit("Ω m2", "resistance_area", "1", ("Ω m^2", "Ω·m2", "ohm m2", "ohm m^2"))
_scientific_unit("mΩ cm2", "resistance_area", "0.0000001", ("mΩ cm^2", "mΩ·cm2"))
# Explicit experimental-condition units also use case-sensitive SI symbols.
_scientific_unit("°C", "temperature", "1", ("℃",), offset="273.15")
_scientific_unit("K", "temperature", "1")
_scientific_unit("Pa", "pressure", "1")
_scientific_unit("kPa", "pressure", "1000")
_scientific_unit("MPa", "pressure", "1000000")
_scientific_unit("GPa", "pressure", "1000000000")
_scientific_unit("m", "length", "1")
_scientific_unit("cm", "length", "0.01")
_scientific_unit("mm", "length", "0.001")
_scientific_unit("μm", "length", "0.000001", ("um",))
_scientific_unit("nm", "length", "0.000000001")
_scientific_unit("m3", "volume", "1", ("m^3",))
_scientific_unit("cm3", "volume", "0.000001", ("cm^3",))
_scientific_unit("wt%", "mass_fraction", "0.01", ("wt %", "wt.%"))


def parse_unit(value: str) -> Unit:
    raw = _unit_key(value)
    if raw in _SI_UNITS:
        return _SI_UNITS[raw]
    if raw.casefold() in {alias.casefold() for alias in _SI_UNITS}:
        raise MetricError("Use explicit case-sensitive scientific units; SI prefixes cannot be interchanged")
    if raw.casefold() in {"gw", "mw", "kw", "w", "gwh", "mwh", "kwh"} and raw not in {"GW", "MW", "kW", "W", "GWh", "MWh", "kWh"}:
        raise MetricError("Use explicit case-sensitive SI units; milli and mega cannot be interchanged")
    try:
        return _UNITS[normalize_text(value)]
    except KeyError as exc:
        raise MetricError(f"Unknown or ambiguous unit: {value!r}") from exc


def convert_value(value: Decimal, source: str, target: str) -> Decimal:
    left, right = parse_unit(source), parse_unit(target)
    if (left.dimension, left.currency) != (right.dimension, right.currency):
        raise MetricError("Incompatible dimensions or currencies; an explicit FX calculation is required")
    with localcontext(Context(prec=64, rounding=ROUND_HALF_EVEN)):
        return decimal_value((value + left.offset) * left.factor / right.factor - right.offset)


def decimal_sum(values: Iterable[Decimal]) -> Decimal:
    with localcontext(Context(prec=64, rounding=ROUND_HALF_EVEN)):
        return sum(values, Decimal(0))


_ALIASES = sorted(set(_UNITS) | set(_SI_UNITS), key=len, reverse=True)
_QUANTITY = re.compile(
    r"(?<![a-z\d.,^{])(" + _NUMBER + r")\s*("
    + "|".join(re.escape(alias) for alias in _ALIASES) + r")(?![a-z\d/^·])", re.IGNORECASE)


def quantities(text: str) -> list[tuple[Decimal, str]]:
    normalized = " ".join(_notation(text).split())
    normalized = re.sub(r"(?<![\d.])\d{1,3}(?: \d{3})+(?!\d)",
                        lambda match: match.group().replace(" ", ""), normalized)
    # Normalize explicit prefix currencies without assuming what an unqualified
    # dollar or an isolated 'billion' means.
    normalized = re.sub(
        r"(?<![A-Za-z])(?:US\$|\$|USD)\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*(billion|million|bn|mn)?\b",
        lambda m: f"{m[1]} USD" + ({"billion": " bn", "bn": " bn", "million": " mn", "mn": " mn"}.get((m[2] or "").lower(), "")),
        normalized, flags=re.IGNORECASE)
    matches = list(_QUANTITY.finditer(normalized))
    bounded: set[int] = set()
    for index, match in enumerate(matches):
        before, after = normalized[:match.start()], normalized[match.end():]
        if (re.search(r"(?:[<>≤≥≈≃~]|\d[\d.,]*\s*(?:[-–—~至到±]|\+/-|\bto\b))\s*$", before)
                or re.match(r"\s*(?:±|\+/-)", after)):
            bounded.add(index)
        if index and re.fullmatch(r"\s*(?:[-–—~至到±]|\+/-|to)\s*",
                                  normalized[matches[index - 1].end():match.start()], re.IGNORECASE):
            bounded.update((index - 1, index))
    return [(decimal_value(match[1]), parse_unit(match[2]).symbol)
            for index, match in enumerate(matches) if index not in bounded]


_HEADER_UNIT = re.compile(r"(?<![a-z])(" + "|".join(re.escape(alias) for alias in _ALIASES)
                          + r")(?![a-z\d/^·])", re.IGNORECASE)


def header_observation(header: str, cell: str) -> tuple[Decimal, str] | None:
    """Read an exact scalar cell with an explicitly unit-bearing header.

    Ranges, inequalities and uncertainty cells are deliberately not collapsed to
    a point estimate. The helper returns no metric name or experimental defaults.
    """
    content = " ".join(_notation(cell).strip().split())
    full_quantity = _QUANTITY.fullmatch(content)
    if full_quantity:
        return decimal_value(full_quantity[1]), parse_unit(full_quantity[2]).symbol
    if not re.fullmatch(_NUMBER, content):
        return None
    candidates: dict[str, Unit] = {}
    for match in _HEADER_UNIT.finditer(_notation(header)):
        unit = parse_unit(match[1])
        candidates[unit.symbol] = unit
    dimensions = set().union(*(_SCIENTIFIC_DIMENSIONS[name] for name in SCIENTIFIC_METRICS
                              if mentions_metric(header, name)))
    if dimensions:
        candidates = {symbol: unit for symbol, unit in candidates.items() if unit.dimension in dimensions}
    if len(candidates) != 1:
        return None
    return decimal_value(content), next(iter(candidates))


def labelled_observations(text: str) -> list[tuple[Decimal, str]]:
    """Explicit scientific/condition header:value pairs, each bound locally.

    This helper does not supply a metric identity, experimental defaults, or a
    point value for a range. Monetary/financial header inference is excluded.
    It is safe for the numeric text layer to import lazily after module loading.
    """
    allowed = set().union(*_SCIENTIFIC_DIMENSIONS.values()) | {
        "temperature", "pressure", "length", "volume", "mass_fraction"}
    result = []
    for field_text in re.split(_ROW_FIELDS, str(text)):
        match = re.fullmatch(r"\s*(.*?)\s*[:=：]\s*(.*?)\s*", field_text)
        if not match:
            continue
        header, cell = match[1], match[2].rstrip("。.;；")
        try:
            observed = header_observation(header, cell)
        except MetricError:
            # One malformed or oversized fetched cell must not discard other
            # explicitly labelled observations or abort sentence verification.
            continue
        if observed is None or parse_unit(observed[1]).dimension not in allowed:
            continue
        # Percentages remain semantically labelled: market-share columns cannot
        # acquire a scientific retention/efficiency meaning merely through %.
        if parse_unit(observed[1]).dimension in {"ratio", "cycles"} and not (
                any(mentions_metric(header, name) for name in SCIENTIFIC_METRICS)
                or any(normalize_text(alias) in normalize_text(header)
                       for name in ("cycle_count", "retention_threshold") for alias in _CONDITION_ALIASES[name])):
            continue
        result.append(observed)
    return result


METRIC_ALIASES = {
    "net_profit": ("net profit", "net income", "净利润", "净收益"),
    "revenue": ("revenue", "sales revenue", "营收", "营业收入"),
    "market_size": ("market size", "market was", "市场规模", "市场销售额"),
    "market_share": ("market share", "share", "市场份额", "市占率"),
    "gdp": ("gdp", "gross domestic product", "国内生产总值"),
    "growth_rate": ("growth", "grew", "增长", "增速"),
    "cpi": ("cpi", "consumer price index", "inflation", "通胀", "消费者价格指数"),
    "installed_capacity": ("installed capacity", "power capacity", "装机容量", "装机规模"),
    "capacity_addition": ("capacity addition", "新增装机"),
    "tam": ("tam", "total addressable market"),
    "sam": ("sam", "serviceable addressable market"),
    "som": ("som", "serviceable obtainable market"),
    "cagr": ("cagr", "compound annual growth", "复合增长率"),
    "ionic_conductivity": ("ionic conductivity", "ion conductivity", "lithium-ion conductivity",
                           "Li-ion conductivity", "Li+ conductivity", "σLi", "σLi+", "σ_Li", "离子电导率"),
    "electronic_conductivity": ("electronic conductivity", "electron conductivity", "电子电导率"),
    "conductivity": ("conductivity", "σ", "电导率"),
    "gravimetric_energy_density": ("gravimetric energy density", "specific energy", "energy density",
                                   "质量能量密度", "比能量", "能量密度"),
    "volumetric_energy_density": ("volumetric energy density", "volumetric specific energy", "energy density",
                                  "体积能量密度", "能量密度"),
    "specific_capacity": ("specific capacity", "gravimetric capacity", "capacity", "比容量", "质量比容量", "容量"),
    "areal_capacity": ("areal capacity", "area capacity", "capacity", "面容量", "面积比容量", "容量"),
    "critical_current_density": ("critical current density", "临界电流密度"),
    "current_density": ("current density", "电流密度"),
    "cycle_life": ("cycle life", "cycling life", "循环寿命", "循环次数"),
    "capacity_retention": ("capacity retention", "retention", "容量保持率", "容量保留率"),
    "coulombic_efficiency": ("coulombic efficiency", "库仑效率"),
    "interfacial_impedance": ("interfacial impedance", "interfacial resistance", "interface resistance", "界面阻抗", "界面电阻"),
    "impedance": ("impedance", "阻抗"),
}

SCIENTIFIC_METRICS = frozenset({
    "ionic_conductivity", "electronic_conductivity", "conductivity", "gravimetric_energy_density",
    "volumetric_energy_density", "specific_capacity", "areal_capacity", "current_density",
    "critical_current_density", "cycle_life", "capacity_retention", "coulombic_efficiency",
    "interfacial_impedance", "impedance",
})
_SCIENTIFIC_DIMENSIONS = {
    "ionic_conductivity": {"conductivity"}, "electronic_conductivity": {"conductivity"},
    "conductivity": {"conductivity"}, "gravimetric_energy_density": {"specific_energy"},
    "volumetric_energy_density": {"volumetric_energy"}, "specific_capacity": {"specific_capacity"},
    "areal_capacity": {"areal_capacity"}, "current_density": {"current_density"},
    "critical_current_density": {"current_density"}, "cycle_life": {"cycles"},
    "capacity_retention": {"ratio"}, "coulombic_efficiency": {"ratio"},
    "interfacial_impedance": {"resistance", "resistance_area"}, "impedance": {"resistance", "resistance_area"},
}
_CONDITION_ALIASES = {
    "temperature": ("temperature", "temp.", "温度"), "pressure": ("stack pressure", "pressure", "堆叠压力", "压力"),
    "test_method": ("test method", "measurement method", "method", "测试方法", "测量方法"),
    "sample_form": ("sample form", "sample geometry", "样品形态", "样品形式"),
    "cell_type": ("cell type", "cell configuration", "cell format", "电池类型", "电池构型"),
    "current_density": ("current density", "电流密度"), "c_rate": ("c-rate", "c rate", "倍率"),
    "voltage_window": ("voltage window", "voltage range", "电压窗口", "电压范围"),
    "cycle_count": ("cycle count", "cycles", "循环次数"),
    "retention_threshold": ("retention threshold", "寿命阈值", "容量阈值"),
    "mass_basis": ("mass basis", "质量口径", "质量基准"),
    "volume_basis": ("volume basis", "体积口径", "体积基准"),
    "electrolyte_thickness": ("electrolyte thickness", "电解质厚度"),
    "electrode_loading": ("electrode loading", "mass loading", "电极载量", "负载量"),
    "drying_temperature": ("drying temperature", "drying temp.", "干燥温度"),
    "sintering_temperature": ("sintering temperature", "烧结温度"),
    "fabrication_pressure": ("fabrication pressure", "pellet pressing pressure", "pressing pressure", "制备压力", "压片压力"),
    "solvent_amount": ("amount of solvent", "solvent amount", "溶剂用量"),
    "solid_loading": ("solid loading", "固体载量", "固体负载"),
    "ball_size": ("ball size", "球尺寸", "球径"),
    "lithium_salt": ("lithium salt", "li salt", "锂盐"),
}
_CONDITION_UNITS = {"temperature": "K", "drying_temperature": "K", "sintering_temperature": "K",
                    "pressure": "Pa", "fabrication_pressure": "Pa", "current_density": "A/m2", "ball_size": "m",
                    "electrolyte_thickness": "m", "solvent_amount": "m3", "solid_loading": "wt%"}
_PREPARATION_TEMPERATURE = re.compile(r"drying|anneal|sinter|烧结|退火|干燥", re.IGNORECASE)
_PREPARATION_PRESSURE = re.compile(r"fabrication|pellet pressing|pressing pressure|制备|压片", re.IGNORECASE)
_COMMON_EXPERIMENT = {"temperature", "pressure", "test_method"}
_SCIENCE_REQUIRED = {
    **{name: _COMMON_EXPERIMENT | {"sample_form"} for name in
       ("conductivity", "ionic_conductivity", "electronic_conductivity")},
    "gravimetric_energy_density": _COMMON_EXPERIMENT | {"cell_type", "mass_basis"},
    "volumetric_energy_density": _COMMON_EXPERIMENT | {"cell_type", "volume_basis"},
    "specific_capacity": _COMMON_EXPERIMENT | {"cell_type", "mass_basis"},
    "areal_capacity": _COMMON_EXPERIMENT | {"cell_type"},
    "current_density": _COMMON_EXPERIMENT | {"cell_type"},
    "critical_current_density": _COMMON_EXPERIMENT | {"cell_type"},
    "cycle_life": _COMMON_EXPERIMENT | {"cell_type", "voltage_window", "retention_threshold"},
    "capacity_retention": _COMMON_EXPERIMENT | {"cell_type", "voltage_window", "cycle_count"},
    "coulombic_efficiency": _COMMON_EXPERIMENT | {"cell_type", "voltage_window", "cycle_count"},
    "interfacial_impedance": _COMMON_EXPERIMENT | {"cell_type"},
    "impedance": _COMMON_EXPERIMENT | {"cell_type"},
}
_RATE_REQUIRED = {"gravimetric_energy_density", "volumetric_energy_density", "specific_capacity",
                  "areal_capacity", "cycle_life", "capacity_retention", "coulombic_efficiency"}
_UNKNOWN = {"", "unknown", "unspecified", "not reported", "n/a", "na", "未知", "未说明", "未报告"}
_ROW_FIELDS = r"[;；\n]|\s+\|\s+(?=[^|:：]{1,120}[:：])"


def mentions_metric(text: str, name: str) -> bool:
    text = normalize_text(text)
    return any(re.search(r"(?<![a-z])" + re.escape(normalize_text(alias)) + r"(?![a-z])", text)
               for alias in METRIC_ALIASES.get(name, (name,)))


def explicit_metadata(text: str) -> dict[str, str]:
    """Parse explicitly stated key=value metadata (also useful for uploaded data notes)."""
    keys = "entity|geography|scope|price_basis|rate_basis|value_kind|period|part_of|boundary|" + "|".join(_CONDITION_ALIASES)
    return {key: value.strip() for key, value in re.findall(
        r"\b(" + keys + r")\s*=\s*([^;\]\n]+)", text)}


@dataclass
class Metric:
    id: str
    name: str
    entity: str | None
    value: Decimal
    unit: str
    period: str | None = None
    geography: str | None = None
    scope: str | None = None
    price_basis: str | None = None
    rate_basis: str | None = None
    value_kind: str | None = None
    part_of: str | None = None
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    source_versions: list[str] = field(default_factory=list)
    origin: str = "source"
    boundary: str | None = None
    conditions: dict[str, str | None] = field(default_factory=dict)
    condition_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        self.value = decimal_value(self.value)
        self.unit = parse_unit(self.unit).symbol
        dimension = parse_unit(self.unit).dimension
        if self.name in {"revenue", "net_profit", "gdp", "market_size", "tam", "sam", "som"} and dimension != "money":
            raise MetricError("Level monetary metrics require an explicit currency unit")
        if self.name in {"installed_capacity", "capacity_addition"} and dimension != "power":
            raise MetricError("Installed power capacity requires power units, not energy or percentages")
        if self.name in {"market_share", "growth_rate", "cagr"} and parse_unit(self.unit).dimension != "ratio":
            raise MetricError("Percentage metrics require a ratio unit")
        if self.name == "market_share" and not Decimal(0) <= convert_value(self.value, self.unit, "%") <= 100:
            raise MetricError("Market share must be between 0% and 100%")
        if self.name in {"market_size", "tam", "sam", "som", "installed_capacity", "capacity_addition"} and self.value < 0:
            raise MetricError("Size/capacity cannot be negative")
        if self.name in SCIENTIFIC_METRICS:
            if dimension not in _SCIENTIFIC_DIMENSIONS[self.name]:
                raise MetricError("Scientific metric requires its explicit compatible experimental unit")
            if self.value < 0:
                raise MetricError("Scientific magnitude cannot be negative")
        if self.boundary is not None and (not isinstance(self.boundary, str) or not self.boundary.strip()
                                           or len(self.boundary) > 500):
            raise MetricError("Invalid scientific boundary metadata")
        if not isinstance(self.conditions, dict) or len(self.conditions) > 32:
            raise MetricError("Invalid experimental conditions")
        for key, condition in self.conditions.items():
            if (not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,48}", key)
                    or condition is not None and (not isinstance(condition, str)
                                                  or not condition.strip() or len(condition) > 500)):
                raise MetricError("Experimental conditions must be explicit short text or unknown")
        if not isinstance(self.condition_evidence, dict) or len(self.condition_evidence) > len(self.conditions):
            raise MetricError("Invalid experimental condition evidence")
        for key, witness in self.condition_evidence.items():
            if (key not in self.conditions or not isinstance(witness, dict)
                    or self.conditions[key] is None or witness.get("value") != self.conditions[key]):
                raise MetricError("Experimental condition evidence must name the retained condition and value")

    def to_dict(self) -> dict:
        return {**asdict(self), "value": decimal_text(self.value)}


def _comparison_key(name: str, value: str) -> str:
    key = normalize_text(value)
    equivalences = {
        "price_basis": {"nominal": "current", "current prices": "current", "现价": "current",
                        "real": "constant", "constant prices": "constant", "不变价": "constant", "不变价格": "constant"},
        "rate_basis": {"year-on-year": "yoy", "year over year": "yoy", "同比": "yoy",
                       "month-on-month": "mom", "环比": "mom", "annual rate": "annualized", "年化": "annualized"},
    }
    return equivalences.get(name, {}).get(key, key)


def _known(value: str | None) -> bool:
    return value is not None and normalize_text(value) not in _UNKNOWN


def _experiment_key(name: str, value: str | None) -> str | None:
    if not _known(value):
        return None
    assert value is not None
    normalized = normalize_text(value).replace("_", " ").replace("-", " ")
    categories = {
        "boundary": {"electrolyte": "electrolyte", "电解质": "electrolyte", "electrode": "electrode",
                     "电极": "electrode", "cell": "cell", "电芯": "cell", "电池": "cell",
                     "pack": "pack", "电池包": "pack", "full cell": "full_cell", "全电池": "full_cell",
                     "half cell": "half_cell", "半电池": "half_cell"},
        "cell_type": {"full cell": "full_cell", "全电池": "full_cell", "half cell": "half_cell",
                      "半电池": "half_cell", "symmetric cell": "symmetric_cell", "对称电池": "symmetric_cell"},
        "test_method": {"eis": "eis", "electrochemical impedance spectroscopy": "eis", "电化学阻抗谱": "eis"},
        "mass_basis": {"active material": "active_material", "活性材料": "active_material",
                       "active material mass": "active_material", "活性物质": "active_material",
                       "total cell mass": "total_cell", "cell mass": "total_cell", "电芯总质量": "total_cell"},
    }
    if name in categories:
        return categories[name].get(normalized, normalized)
    if name in _CONDITION_UNITS:
        match = _QUANTITY.fullmatch(" ".join(_notation(value).split()))
        if not match:
            return None
        try:
            amount = convert_value(decimal_value(match[1]), match[2], _CONDITION_UNITS[name])
        except MetricError:
            return None
        return decimal_text(amount) + " " + _CONDITION_UNITS[name]
    if name == "cycle_count":
        try:
            text = re.sub(r"\s*(?:cycles?|次循环|循环|圈|次)\s*$", "", value.strip(), flags=re.IGNORECASE)
            return decimal_text(decimal_value(text))
        except MetricError:
            return None
    if name == "c_rate":
        text = "".join(_notation(value).split())
        try:
            if re.fullmatch(r"C/[1-9]\d*(?:\.\d+)?", text):
                with localcontext(Context(prec=64, rounding=ROUND_HALF_EVEN)):
                    return decimal_text(Decimal(1) / decimal_value(text[2:]))
            return decimal_text(decimal_value(text.removesuffix("C")))
        except MetricError:
            return None
    # Keep unrecognized scientific condition text explicit and case-sensitive.
    # Casefolding an arbitrary physical unit could silently equate milli/mega.
    return re.sub(r"\s+", "", _notation(value)).replace("–", "-").replace("—", "-")


def _scientific_comparison_errors(left: Metric, right: Metric, *, allow_entity: bool, same_period: bool) -> list[str]:
    errors = []
    for name in ("boundary", *(() if allow_entity else ("entity",))):
        a, b = getattr(left, name), getattr(right, name)
        if not _known(a) or not _known(b):
            errors.append(f"missing_{name}")
        elif _experiment_key(name, a) != _experiment_key(name, b):
            errors.append(f"different_{name}")
    for name in ("scope", "value_kind"):
        a, b = getattr(left, name), getattr(right, name)
        if _known(a) or _known(b):
            if not _known(a) or not _known(b):
                errors.append(f"missing_{name}")
            elif _comparison_key(name, a) != _comparison_key(name, b):
                errors.append(f"different_{name}")
    required = _SCIENCE_REQUIRED.get(left.name, set()) | _SCIENCE_REQUIRED.get(right.name, set())
    for key in sorted(required | set(left.conditions) | set(right.conditions)):
        a = _experiment_key(key, left.conditions.get(key))
        b = _experiment_key(key, right.conditions.get(key))
        if a is None or b is None:
            errors.append(f"missing_condition_{key}")
        elif a != b:
            errors.append(f"different_condition_{key}")
    if ((left.name in _RATE_REQUIRED or right.name in _RATE_REQUIRED)
            and not any(_experiment_key(key, left.conditions.get(key)) is not None
                        and _experiment_key(key, right.conditions.get(key)) is not None
                        for key in ("current_density", "c_rate"))):
        errors.append("missing_condition_rate")
    if left.name != right.name:
        errors.append("different_metric")
    if same_period and (left.period or right.period):
        if not left.period or not right.period:
            errors.append("missing_period")
        elif left.period != right.period:
            errors.append("different_period")
    try:
        convert_value(right.value, right.unit, left.unit)
    except MetricError:
        errors.append("incompatible_unit_or_currency")
    return list(dict.fromkeys(errors))


def comparison_errors(left: Metric, right: Metric, *, allow_entity=False, same_period=False) -> list[str]:
    if left.name in SCIENTIFIC_METRICS or right.name in SCIENTIFIC_METRICS:
        return _scientific_comparison_errors(left, right, allow_entity=allow_entity, same_period=same_period)
    errors = []
    required = ["geography", "scope", "value_kind"]
    if not allow_entity:
        required.append("entity")
    if parse_unit(left.unit).dimension == "money":
        required.append("price_basis")
    if parse_unit(left.unit).dimension == "ratio":
        required.append("rate_basis")
    for name in required:
        a, b = getattr(left, name), getattr(right, name)
        if not a or not b:
            errors.append(f"missing_{name}")
        elif _comparison_key(name, a) != _comparison_key(name, b):
            errors.append(f"different_{name}")
    if left.name != right.name:
        errors.append("different_metric")
    if not left.period or not right.period:
        errors.append("missing_period")
    elif same_period and left.period != right.period:
        errors.append("different_period")
    if _comparison_key("rate_basis", left.rate_basis or "") != _comparison_key("rate_basis", right.rate_basis or ""):
        errors.append("different_rate_basis")
    try:
        convert_value(right.value, right.unit, left.unit)
    except MetricError:
        errors.append("incompatible_unit_or_currency")
    return list(dict.fromkeys(errors))


def _field_supported(name: str, value: str, text: str) -> bool:
    normalized = normalize_text(text)
    if normalize_text(explicit_metadata(text).get(name, "")) == normalize_text(value):
        return True
    if name in _CONDITION_ALIASES:
        expected = _experiment_key(name, value)
        if expected is not None:
            labelled_values: set[str | None] = set()
            physical_fields = []
            for field_text in re.split(_ROW_FIELDS, text):
                labelled = re.match(r"\s*(.*?)\s*[:=：]\s*(.*?)\s*$", field_text)
                if not labelled:
                    physical_fields.append(field_text)
                    continue
                label, cell = labelled[1], labelled[2].rstrip("。.;；")
                if name == "temperature" and _PREPARATION_TEMPERATURE.search(label):
                    continue
                if name == "pressure" and _PREPARATION_PRESSURE.search(label):
                    continue
                if name == "temperature" and any(mentions_metric(label, metric_name) for metric_name in SCIENTIFIC_METRICS):
                    labelled_values.update(_experiment_key(name, f"{decimal_text(number)} {unit}")
                        for number, unit in quantities(label) if parse_unit(unit).dimension == "temperature")
                if name in _CONDITION_UNITS:
                    header_dimensions = {parse_unit(match[1]).dimension
                                         for match in _HEADER_UNIT.finditer(_notation(label))}
                    if header_dimensions and header_dimensions != {parse_unit(_CONDITION_UNITS[name]).dimension}:
                        # E.g. a Capacity(mAh/g, test temperature) compound cell
                        # is not the conductivity's temperature column. Explicit
                        # "conductivity at25°C" header context was read above.
                        continue
                physical_fields.append(field_text)
                if not any(normalize_text(alias) in normalize_text(label) for alias in _CONDITION_ALIASES[name]):
                    continue
                observation = header_observation(label, cell)
                candidate = f"{decimal_text(observation[0])} {observation[1]}" if observation else cell
                labelled_values.add(_experiment_key(name, candidate))
            if labelled_values:
                return labelled_values == {expected}
            # A unique explicit physical condition in ordinary prose may also
            # support an equivalent unit. Multiple possible temperatures/loads
            # stay ambiguous instead of borrowing a convenient one.
            if name in _CONDITION_UNITS:
                values = {_experiment_key(name, f"{decimal_text(amount)} {unit}")
                          for amount, unit in quantities("\n".join(physical_fields))} - {None}
                if values == {expected}:
                    return True
        if name in _CONDITION_UNITS:
            if expected is not None:
                return False
            # Keep a literally reported qualitative/ranged setting for audit,
            # but _experiment_key still returns None so it cannot establish
            # comparability. Preserve scientific unit case in this fallback.
            literal = _notation(value).strip()
            phrase = re.escape(literal).replace(r"\ ", r"\s+")
            flags = re.IGNORECASE if normalize_text(value) in {
                "room temperature", "ambient temperature", "rt", "ambient pressure", "室温", "常温", "常压"} else 0
            boundary = r"(?![a-zA-Z0-9.])" if literal[-1:].isdigit() else r"(?![a-zA-Z0-9])"
            return bool(re.search(r"(?<![a-zA-Z0-9.])" + phrase + boundary, _notation(text), flags))
    aliases = {
        "China": ("中国",), "global": ("全球", "worldwide"),
        "current": ("现价", "current prices", "nominal"),
        "actual": ("实际", "实绩", "actual"), "forecast": ("预测", "预计", "forecast"),
        "preliminary": ("初步", "preliminary"), "final": ("最终", "final"),
        "annualized": ("annual rate", "年化"),
        "electrolyte": ("电解质",), "electrode": ("电极",), "cell": ("电芯", "电池"),
        "full_cell": ("full cell", "full-cell", "全电池"),
        "half_cell": ("half cell", "half-cell", "半电池"),
        "symmetric_cell": ("symmetric cell", "symmetric-cell", "对称电池"),
        "active_material": ("active material", "active-material", "活性材料", "活性物质"),
        "total_cell": ("total cell mass", "cell mass", "电芯总质量"),
    }
    explicit_phrases = {
        "price_basis": {"nominal", "real", "current prices", "constant prices", "现价", "不变价", "不变价格"},
        "rate_basis": {"year-on-year", "year over year", "yoy", "month-on-month", "mom", "同比", "环比", "annual rate"},
        "value_kind": {"actual", "forecast", "estimate", "preliminary", "final", "预测", "初步", "最终"},
    }
    if normalize_text(value) in explicit_phrases.get(name, set()) and normalize_text(value) in normalized:
        return True
    if name not in {"price_basis", "rate_basis", "value_kind"}:
        pattern = re.escape(normalize_text(value))
        if re.search(r"[a-z0-9]", value, re.IGNORECASE):
            pattern = r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            return True
    return any(normalize_text(alias) in normalized for alias in aliases.get(value, ()))


def observed_quantities(text: str, name: str) -> list[tuple[Decimal, str]]:
    """Bind the first amount after this metric label, stopping at another metric.

    A quote mentioning TAM=100 and revenue=200 cannot support TAM=200 merely by
    containing both the word TAM and the number 200.
    """
    normalized = _notation(text)
    positions = []
    for candidate, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", normalized, re.IGNORECASE):
                positions.append((match.start(), match.end(), candidate))
    # A generic 'conductivity' inside 'electronic conductivity' must not erase
    # the stated measurement type; likewise current/critical current density.
    positions = list({position for position in positions if not any(
        other[0] <= position[0] and other[1] >= position[1]
        and (other[0], other[1]) != (position[0], position[1]) for other in positions)})
    found = []
    for start, end, candidate in positions:
        if candidate != name:
            continue
        stop = min((other_start for other_start, _, _ in positions if other_start >= end), default=len(normalized))
        segment = normalized[end:stop]
        values = quantities(segment)
        if name in SCIENTIFIC_METRICS:
            values = [(amount, unit) for amount, unit in values
                      if parse_unit(unit).dimension in _SCIENTIFIC_DIMENSIONS[name]]
            # Table renderers repeat the exact column label on the same anchored
            # row, e.g. 'Ionic conductivity (mS cm−1): 1.2; Temperature (°C): 25'.
            field_text = re.split(_ROW_FIELDS, normalized[start:stop], maxsplit=1)[0]
            if stop < len(normalized):
                field_text = field_text.rstrip().removesuffix("|").rstrip()
            labelled = re.match(r"\s*(.*?)\s*[:=：]\s*(.*?)\s*$", field_text)
            if labelled:
                observed = header_observation(labelled[1], labelled[2])
                if observed and parse_unit(observed[1]).dimension in _SCIENTIFIC_DIMENSIONS[name]:
                    values = [observed, *values]
        if values:
            found.append(values[0])
    return found


def proposal_from_claim(claim: Claim) -> dict | None:
    if claim.metric:
        return dict(claim.metric)
    found = quantities(claim.value or claim.text)
    name = normalize_text(claim.attribute or "")
    if name not in METRIC_ALIASES:
        name = next((candidate for candidate in METRIC_ALIASES if mentions_metric(claim.text, candidate)
                     and (candidate not in SCIENTIFIC_METRICS or any(
                         parse_unit(unit).dimension in _SCIENTIFIC_DIMENSIONS[candidate]
                         for _, unit in observed_quantities(claim.text, candidate)))), "")
    if not name:
        return None
    if name in SCIENTIFIC_METRICS:
        found = observed_quantities(claim.text, name)
    # Multiple amounts in one unstructured claim are ambiguous; require explicit slots.
    if len(found) != 1:
        return None
    value, unit = found[0]
    metadata = explicit_metadata(claim.text)
    period = claim.time_scope or metadata.get("period")
    if not period:
        years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", claim.text)
        period = years[0] if len(set(years)) == 1 else None
    return {"name": name, "value": decimal_text(value), "unit": unit,
            "entity": claim.entity or metadata.get("entity"), "period": period,
            **{k: v for k, v in metadata.items() if k not in {"entity", "period", *_CONDITION_ALIASES}},
            "conditions": {k: v for k, v in metadata.items() if k in _CONDITION_ALIASES}}


def _matched_condition_evidence(metric: Metric, record: SentenceCitation, source: Source,
                                proposals: dict) -> dict[str, dict]:
    from core_engine.report.scientific import anchored_table_candidate

    version = source.content_sha256
    if (metric.entity is None or record.source_sha256 != version
            or type(record.start_char) is not int or type(record.end_char) is not int
            or not 0 <= record.start_char < record.end_char <= len(source.text)
            or source.text[record.start_char:record.end_char] != record.quote):
        return {}
    try:
        candidate = anchored_table_candidate(source, record.start_char, record.end_char,
                                             metric.name, metric.entity, metric.value, metric.unit)
    except (ValueError, TypeError, KeyError):
        # A damaged snapshot cannot add Methods conditions to a valid value row.
        return {}
    if (candidate is None or candidate.metric is None
            or candidate.metric.get("conditions", {}) != metric.conditions
            or candidate.metric.get("boundary") != metric.boundary):
        return {}
    derived = candidate.metric.get("condition_evidence", {})
    if not isinstance(derived, dict):
        return {}
    matched = {}
    for key, witness in derived.items():
        if (not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,48}", key)
                or not isinstance(witness, dict) or metric.conditions.get(key) is not None
                or proposals.get(key) != witness):
            continue
        condition = witness.get("value")
        if not isinstance(condition, str) or not condition.strip() or len(condition) > 500:
            continue
        start, end = witness.get("start"), witness.get("end")
        if (witness.get("source_url") != source.url or witness.get("source_version") != version
                or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source.text)
                or source.text[start:end] != witness.get("quote")
                or not (end <= record.start_char or start >= record.end_char)):
            continue
        matched[key] = witness
    return matched


def catalog_from_claims(claims: list[Claim], citations: list[SentenceCitation], *,
                        sources: Iterable[Source] = ()) -> tuple[list[Metric], list[dict]]:
    metrics: list[Metric] = []
    issues: list[dict] = []
    source_by_url = {source.url: source for source in sources}
    for claim in claims:
        if claim.reporting_status in {"disputed", "lead_only"}:
            issues.append({"claim_id": claim.id, "reason": "unsettled_or_unanchored_publication_status"})
            continue
        try:
            proposal = proposal_from_claim(claim)
        except (MetricError, TypeError) as exc:
            issues.append({"claim_id": claim.id, "reason": str(exc)})
            continue
        if not proposal:
            continue
        records = [c for c in citations if c.claim_id == claim.id]
        try:
            name = normalize_text(str(proposal.get("name", "")))
            if name not in METRIC_ALIASES or not mentions_metric(claim.text, name):
                raise MetricError("Metric name is not supported by the claim")
            value, unit = decimal_value(proposal.get("value")), parse_unit(str(proposal.get("unit", "")))
            supported = []
            fields = {}
            for key in ("entity", "period", "geography", "scope", "price_basis", "rate_basis", "value_kind", "part_of", "boundary"):
                value_text = proposal.get(key)
                if value_text is not None:
                    if not isinstance(value_text, str) or not value_text.strip():
                        raise MetricError(f"Invalid metric metadata: {key}")
                    fields[key] = value_text
            conditions = proposal.get("conditions", {})
            if not isinstance(conditions, dict) or len(conditions) > 32:
                raise MetricError("Invalid experimental conditions")
            if any(not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,48}", key)
                   or item is not None and (not isinstance(item, str) or not item.strip() or len(item) > 500)
                   for key, item in conditions.items()):
                raise MetricError("Experimental conditions must be explicit short text or unknown")
            for record in records:
                passages = split_sentences(record.quote)
                if name in SCIENTIFIC_METRICS:
                    row_lines = [line for line in record.quote.splitlines() if len(re.findall(
                        r"(?:^|\s+\|\s+|[;；])\s*[^:：|]{1,120}[:：]", line)) >= 2]
                    # Table captions/notes are admissible local context only
                    # when this anchored quote contains exactly one labelled
                    # data row. Never pool two rows to fill a missing condition.
                    if len(row_lines) == 1:
                        passages = [record.quote]
                for passage in passages:
                    observed = observed_quantities(passage, name)
                    if name in SCIENTIFIC_METRICS and (len(observed) != 1 or len(re.findall(
                            r"(?:\b(?:sample|material|specimen)\b|样品|材料)\s*[:=：]", passage,
                            re.IGNORECASE)) > 1):
                        continue
                    observations = set(observed)
                    if len(observations) != 1:
                        continue
                    number, source_unit = next(iter(observations))
                    try:
                        amount_matches = convert_value(number, source_unit, unit.symbol) == value
                    except MetricError:
                        continue
                    if not amount_matches:
                        continue
                    # A pooled multi-year quote cannot bind one year's value to
                    # another year. Require an atomic passage or explicit period.
                    years = set(re.findall(r"(?<!\d)(?:19|20|21)\d{2}(?!\d)", passage))
                    if (fields.get("period") and len(years) > 1
                            and explicit_metadata(passage).get("period") != fields["period"]):
                        continue
                    if (all(_field_supported(key, field_value, passage) for key, field_value in fields.items())
                            and all(item is None or _field_supported(key, item, passage)
                                    for key, item in conditions.items())):
                        supported.append(record)
                        break
            if not supported:
                raise MetricError("No anchored atomic passage supports this metric, amount and metadata")
            metric = Metric(id=f"m{len(metrics) + 1}", name=name, value=value, unit=unit.symbol,
                            claim_ids=[claim.id], evidence_ids=[r.id for r in supported],
                            source_versions=sorted({r.source_sha256 for r in supported}),
                            entity=fields.get("entity"), period=fields.get("period"),
                            geography=fields.get("geography"), scope=fields.get("scope"),
                            price_basis=fields.get("price_basis"), rate_basis=fields.get("rate_basis"),
                            value_kind=fields.get("value_kind"), part_of=fields.get("part_of"),
                            boundary=fields.get("boundary"), conditions=dict(conditions))
            condition_proposals = proposal.get("condition_evidence")
            if name in SCIENTIFIC_METRICS and isinstance(condition_proposals, dict) and condition_proposals:
                for record in supported:
                    source = source_by_url.get(record.source_url)
                    if source is None:
                        continue
                    matched = _matched_condition_evidence(metric, record, source, condition_proposals)
                    if matched:
                        # The conditions describe this row's measurement. A
                        # different source's value citation is not its witness.
                        metric = replace(metric, evidence_ids=[record.id], source_versions=[record.source_sha256],
                                         conditions={**metric.conditions,
                                                     **{key: witness["value"] for key, witness in matched.items()}},
                                         condition_evidence=matched)
                        break
            metrics.append(metric)
        except (MetricError, TypeError) as exc:
            issues.append({"claim_id": claim.id, "reason": str(exc)})
    return metrics, issues
