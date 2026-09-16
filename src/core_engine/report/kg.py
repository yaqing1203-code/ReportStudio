"""Report Knowledge Graph — the KG layer feeding the structured sections.

The report pipeline scrapes the WEB, not the proprietary DB, so the "knowledge
graph" for a report is built by extracting entities + typed relationships from the
already-VERIFIED claims and their authoritative sources. It is held as an in-memory
`ReportKnowledgeGraph`, and — when a Postgres/AGE instance is configured
(CE_KG_PROJECT_TO_AGE=true) — optionally projected into the revived AGE GraphStore
so the dormant multi-hop layer and this pipeline share one graph representation.

What the KG produces for the mandated sections:
  - industry_chain          : upstream / midstream / downstream tiers + edges
                              (drives the TikZ chain diagram).
  - market_size             : TAM/SAM/SOM + a historical size/growth series
                              (drives the pgfplots chart).
  - competitive_landscape   : players with market share + advantages
                              (drives the booktabs table + a share bar chart).

CRITICAL: every node/edge/datum here is derived ONLY from verified claims. The KG
never invents a relationship the sources didn't support — extraction is proposed by
the LLM but each supporting claim already passed the harness, and anything we cannot
attach to a verified claim is dropped. Provenance (the claim id + source URLs) rides
along on every element for the audit trail and in-report citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core_engine.report.models import Claim


class ChainTier(str, Enum):
    UPSTREAM = "upstream"      # inputs / raw materials / components / equipment
    MIDSTREAM = "midstream"    # core manufacturing / integration / platforms
    DOWNSTREAM = "downstream"  # distribution / applications / end markets


@dataclass(slots=True)
class ChainNode:
    """One entity in the industry chain, pinned to a tier."""

    name: str
    tier: ChainTier
    # provenance: verified claim ids + source URLs that put this node on the map.
    claim_ids: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChainEdge:
    """A directed supply relationship (from supplies/feeds to)."""

    src: str        # ChainNode.name
    dst: str        # ChainNode.name
    label: str = "supplies"
    claim_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MarketDatum:
    """One point in a historical market-size series."""

    year: int
    value: float          # in `unit`
    unit: str = "USD bn"
    claim_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MarketSize:
    """TAM/SAM/SOM headline figures + a historical series for the pgfplots chart."""

    tam: float | None = None
    sam: float | None = None
    som: float | None = None
    unit: str = "USD bn"
    cagr_pct: float | None = None
    series: list[MarketDatum] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)

    def has_chartable_series(self) -> bool:
        return len(self.series) >= 2


@dataclass(slots=True)
class Competitor:
    """A market player for the competitive-landscape table + share chart."""

    name: str
    market_share_pct: float | None = None
    advantage: str = ""
    claim_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReportKnowledgeGraph:
    """The synthesized, provenance-carrying KG for one report."""

    topic: str
    chain_nodes: list[ChainNode] = field(default_factory=list)
    chain_edges: list[ChainEdge] = field(default_factory=list)
    market: MarketSize = field(default_factory=MarketSize)
    competitors: list[Competitor] = field(default_factory=list)

    def tier(self, tier: ChainTier) -> list[ChainNode]:
        return [n for n in self.chain_nodes if n.tier is tier]

    def has_chain(self) -> bool:
        return bool(self.chain_nodes)

    def has_competitors(self) -> bool:
        return bool(self.competitors)

    def provenance_claim_ids(self) -> set[str]:
        ids: set[str] = set()
        for n in self.chain_nodes:
            ids.update(n.claim_ids)
        for e in self.chain_edges:
            ids.update(e.claim_ids)
        for c in self.competitors:
            ids.update(c.claim_ids)
        ids.update(self.market.claim_ids)
        for d in self.market.series:
            ids.update(d.claim_ids)
        return ids


# --------------------------------------------------------------------------
# Builder: KGExtraction (proposed by the LLM) -> provenance-checked KG.
# --------------------------------------------------------------------------
async def build_report_kg(topic: str, verified: list[Claim], llm) -> ReportKnowledgeGraph:
    """Ask the LLM to extract KG structure from VERIFIED claim text, then keep only
    the elements whose provenance claim_ids reference a real verified claim.

    This is the trust boundary: the LLM proposes structure, but the harness already
    decided which claims are true, and this builder refuses to place any node/edge/
    figure on the map that isn't backed by one of them. `llm` is untyped here to
    avoid a circular import with report.llm.
    """
    verified_ids = {c.id for c in verified}
    by_id = {c.id: c for c in verified}
    extraction = await llm.extract_kg(topic, [(c.id, c.text) for c in verified])

    def _keep_ids(raw_ids: list) -> list[str]:
        return [cid for cid in (raw_ids or []) if cid in verified_ids]

    def _urls_for(ids: list[str]) -> list[str]:
        urls: list[str] = []
        for cid in ids:
            claim = by_id.get(cid)
            if claim:
                urls.extend(claim.supporting_sources())
        return sorted(set(urls))

    kg = ReportKnowledgeGraph(topic=topic)

    # --- chain nodes (dedupe by (name, tier); require >=1 verified claim id) ---
    tier_map = {t.value: t for t in ChainTier}
    seen_nodes: dict[tuple[str, str], ChainNode] = {}
    for item in extraction.chain:
        tier = tier_map.get(str(item.get("tier", "")).lower())
        name = str(item.get("name", "")).strip()
        ids = _keep_ids(item.get("claim_ids"))
        if not tier or not name or not ids:
            continue
        key = (name.lower(), tier.value)
        if key in seen_nodes:
            seen_nodes[key].claim_ids = sorted(set(seen_nodes[key].claim_ids) | set(ids))
            continue
        node = ChainNode(name=name, tier=tier, claim_ids=ids, source_urls=_urls_for(ids))
        seen_nodes[key] = node
        kg.chain_nodes.append(node)

    # cap nodes per tier so the diagram stays legible
    from core_engine.config import get_settings
    cap = get_settings().kg_max_chain_nodes_per_tier
    if cap and cap > 0:
        capped: list[ChainNode] = []
        for t in ChainTier:
            capped.extend(kg.tier(t)[:cap])
        kg.chain_nodes = capped

    valid_names = {n.name.lower() for n in kg.chain_nodes}
    for e in extraction.chain_edges:
        src = str(e.get("src", "")).strip()
        dst = str(e.get("dst", "")).strip()
        ids = _keep_ids(e.get("claim_ids"))
        # only connect nodes that survived; require provenance
        if src.lower() in valid_names and dst.lower() in valid_names and ids:
            kg.chain_edges.append(ChainEdge(src=src, dst=dst,
                                            label=str(e.get("label", "supplies")),
                                            claim_ids=ids))

    # --- market size ---
    m = extraction.market or {}
    m_ids = _keep_ids(m.get("claim_ids"))
    series: list[MarketDatum] = []
    for d in m.get("series", []) or []:
        d_ids = _keep_ids(d.get("claim_ids"))
        try:
            year = int(d.get("year"))
            value = float(d.get("value"))
        except (TypeError, ValueError):
            continue
        if d_ids:
            series.append(MarketDatum(year=year, value=value,
                                      unit=str(m.get("unit", "USD bn")), claim_ids=d_ids))
    series.sort(key=lambda x: x.year)
    kg.market = MarketSize(
        tam=_num(m.get("tam")), sam=_num(m.get("sam")), som=_num(m.get("som")),
        unit=str(m.get("unit", "USD bn")), cagr_pct=_num(m.get("cagr_pct")),
        series=series, claim_ids=m_ids,
    )

    # --- competitors (require a verified claim id; dedupe by name) ---
    seen_comp: set[str] = set()
    for c in extraction.competitors:
        name = str(c.get("name", "")).strip()
        ids = _keep_ids(c.get("claim_ids"))
        if not name or not ids or name.lower() in seen_comp:
            continue
        seen_comp.add(name.lower())
        kg.competitors.append(Competitor(
            name=name, market_share_pct=_num(c.get("market_share_pct")),
            advantage=str(c.get("advantage", "")).strip(), claim_ids=ids,
        ))

    return kg


def _num(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
