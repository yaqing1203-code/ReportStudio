"""Standardized LaTeX chart/table generation from the report KG.

Turns the provenance-checked ReportKnowledgeGraph into LaTeX fragments:

  - industry_chain         -> TikZ flowchart (upstream -> midstream -> downstream)
  - market_size            -> pgfplots line chart (historical series) + TAM/SAM/SOM
  - competitive_landscape  -> booktabs table + pgfplots bar chart (market share)

Two safety rules, same as the renderer (latex.py):
  1. Every string taken from scraped/synthesized data is LaTeX-escaped before it
     lands in a fragment. Node names, competitor names, and advantages are all
     untrusted text — they cannot be allowed to inject LaTeX.
  2. Numbers are formatted by us from parsed floats, never interpolated as raw
     strings, so a malformed figure can't smuggle markup into a coordinate.

Each generator returns "" when it lacks enough data, so the template can decide to
render a graceful "insufficient verified data" note instead of a broken figure.
"""
from __future__ import annotations

from core_engine.report.kg import ChainTier, ReportKnowledgeGraph
from core_engine.report.latex import tex_escape


def _fmt(value: float) -> str:
    """Format a number for LaTeX: drop the trailing .0 on integers."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------
# Industry Chain Map -> TikZ
# --------------------------------------------------------------------------
def industry_chain_tikz(kg: ReportKnowledgeGraph) -> str:
    """A three-column TikZ flowchart: upstream | midstream | downstream, with the
    extracted supply edges drawn between named nodes."""
    if not kg.has_chain():
        return ""

    tiers = [
        (ChainTier.UPSTREAM, "Upstream"),
        (ChainTier.MIDSTREAM, "Midstream"),
        (ChainTier.DOWNSTREAM, "Downstream"),
    ]
    lines: list[str] = [
        r"\begin{tikzpicture}[",
        r"    node distance=0.6cm and 1.8cm,",
        r"    tiernode/.style={draw, rounded corners, align=center, "
        r"fill=primary!8, draw=primary, text width=2.6cm, minimum height=0.9cm, font=\small},",
        r"    tierhead/.style={font=\bfseries\color{primary}},",
        r"    supply/.style={-{Latex[length=2mm]}, draw=accent, thick},",
        r"]",
    ]

    # Track the LaTeX node id for each chain-node name so edges can reference them.
    node_id: dict[str, str] = {}
    col_x = {ChainTier.UPSTREAM: 0.0, ChainTier.MIDSTREAM: 4.2, ChainTier.DOWNSTREAM: 8.4}
    counter = 0
    for tier, label in tiers:
        nodes = kg.tier(tier)
        x = col_x[tier]
        # column header
        lines.append(
            rf"\node[tierhead] (head-{tier.value}) at ({_fmt(x)}, 1.2) {{{tex_escape(label)}}};"
        )
        prev = f"head-{tier.value}"
        for n in nodes:
            counter += 1
            nid = f"n{counter}"
            node_id[n.name.lower()] = nid
            y_anchor = "below=of " + prev if prev else ""
            lines.append(
                rf"\node[tiernode, {y_anchor}] ({nid}) {{{tex_escape(n.name)}}};"
            )
            prev = nid

    # edges: only those whose endpoints both resolved to a drawn node
    for e in kg.chain_edges:
        sid = node_id.get(e.src.lower())
        did = node_id.get(e.dst.lower())
        if sid and did:
            lines.append(rf"\draw[supply] ({sid}) -- ({did});")

    # If no explicit edges survived, draw tier-to-tier guide arrows so the flow reads.
    if not any(node_id.get(e.src.lower()) and node_id.get(e.dst.lower())
               for e in kg.chain_edges):
        lines.append(r"\draw[supply, dashed] (head-upstream) -- (head-midstream);")
        lines.append(r"\draw[supply, dashed] (head-midstream) -- (head-downstream);")

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Market Size -> pgfplots line chart + TAM/SAM/SOM callout
# --------------------------------------------------------------------------
def market_size_plot(kg: ReportKnowledgeGraph) -> str:
    """A pgfplots line chart of the historical market-size series. Returns "" if we
    don't have at least two verified data points to plot."""
    m = kg.market
    if not m.has_chartable_series():
        return ""

    coords = " ".join(f"({d.year},{_fmt(d.value)})" for d in m.series)
    unit = tex_escape(m.series[0].unit if m.series else m.unit)
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.9\textwidth, height=6cm,",
        r"    xlabel={Year}, ylabel={Market size (" + unit + r")},",
        r"    xtick=data, tick label style={font=\small},",
        r"    grid=major, grid style={accent!25},",
        r"    every axis plot/.append style={primary, thick, mark=*},",
        r"    /pgf/number format/1000 sep={},",
        r"]",
        r"\addplot coordinates {" + coords + r"};",
        r"\end{axis}",
        r"\end{tikzpicture}",
    ]
    return "\n".join(lines)


def market_size_callout(kg: ReportKnowledgeGraph) -> str:
    """A small booktabs table of the TAM/SAM/SOM headline figures + CAGR. Returns ""
    if none of the headline figures were verified."""
    m = kg.market
    rows: list[tuple[str, float | None]] = [
        ("TAM (Total Addressable Market)", m.tam),
        ("SAM (Serviceable Addressable Market)", m.sam),
        ("SOM (Serviceable Obtainable Market)", m.som),
    ]
    present = [(label, v) for label, v in rows if v is not None]
    if not present and m.cagr_pct is None:
        return ""

    unit = tex_escape(m.unit)
    # Content-only (no float env): the template wraps every block in a figure[H],
    # and a `table` float nested inside a figure float is a LaTeX error.
    out = [
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value (" + unit + r")} \\",
        r"\midrule",
    ]
    for label, v in present:
        out.append(rf"{tex_escape(label)} & {_fmt(v)} \\")
    if m.cagr_pct is not None:
        out.append(rf"CAGR & {_fmt(m.cagr_pct)}\% \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Competitive Landscape -> booktabs table + pgfplots bar chart
# --------------------------------------------------------------------------
def competitive_table(kg: ReportKnowledgeGraph) -> str:
    """A booktabs table of players, market share, and competitive advantage."""
    if not kg.has_competitors():
        return ""
    # Content-only (no float env): the template wraps this in a figure[H]. Uses a
    # tabularx-free fixed-width layout that wraps long advantage text across lines so
    # a verbose competitor row can't overflow the page.
    out = [
        r"\begin{tabular}{@{}p{0.24\textwidth}r p{0.50\textwidth}@{}}",
        r"\toprule",
        r"\textbf{Player} & \textbf{Share} & \textbf{Competitive advantage} \\",
        r"\midrule",
    ]
    # Sort by share desc where known, so the table reads like a ranking.
    comps = sorted(kg.competitors,
                   key=lambda c: (c.market_share_pct is None, -(c.market_share_pct or 0)))
    for c in comps:
        share = f"{_fmt(c.market_share_pct)}\\%" if c.market_share_pct is not None else "n/a"
        out.append(
            rf"{tex_escape(c.name)} & {share} & {tex_escape(c.advantage or '—')} \\"
        )
        out.append(r"\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def competitive_share_bar(kg: ReportKnowledgeGraph) -> str:
    """A pgfplots horizontal bar chart of market share. Returns "" if fewer than two
    competitors have a known share."""
    withshare = [c for c in kg.competitors if c.market_share_pct is not None]
    if len(withshare) < 2:
        return ""
    withshare.sort(key=lambda c: -(c.market_share_pct or 0))
    names = ",".join(tex_escape(c.name) for c in withshare)
    coords = " ".join(f"({_fmt(c.market_share_pct)},{i})"
                      for i, c in enumerate(withshare))
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.9\textwidth, height=" + _fmt(1.2 + 0.6 * len(withshare)) + r"cm,",
        r"    xbar, xmin=0,",
        r"    xlabel={Market share (\%)},",
        r"    ytick={" + ",".join(str(i) for i in range(len(withshare))) + r"},",
        r"    yticklabels={" + names + r"},",
        r"    tick label style={font=\small}, bar width=10pt,",
        r"    nodes near coords, nodes near coords align={horizontal},",
        r"    every axis plot/.append style={primary, fill=primary!30},",
        r"]",
        r"\addplot coordinates {" + coords + r"};",
        r"\end{axis}",
        r"\end{tikzpicture}",
    ]
    return "\n".join(lines)


def build_all(kg: ReportKnowledgeGraph) -> dict[str, str]:
    """Render every chart fragment once, keyed for the template. Empty strings mean
    'not enough verified data' and the template renders a fallback note."""
    return {
        "chain_tikz": industry_chain_tikz(kg),
        "market_plot": market_size_plot(kg),
        "market_callout": market_size_callout(kg),
        "competitive_table": competitive_table(kg),
        "competitive_bar": competitive_share_bar(kg),
    }
