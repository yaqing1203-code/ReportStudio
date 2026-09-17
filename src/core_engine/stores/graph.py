"""Knowledge Graph store over Apache AGE — the multi-hop reasoning engine.

This is the layer you prioritized: 'how does X connect to Y'. It exposes three
primitives the retriever composes:

  1. resolve_entities()  : anchor free-text mentions to graph nodes (by label /
                           searchable properties). These anchors seed traversal.
  2. expand()            : bounded breadth-first multi-hop expansion from anchor
                           nodes, returning the connected sub-graph + node URIs.
  3. find_paths()        : shortest / all paths between two entities — the direct
                           answer to a 'how is X related to Y' question.

All Cypher runs through AGE's cypher() SQL function. RLS still applies because the
graph tables live in the same tenant-scoped database; we ALSO carry tenant_id as a
node property and filter on it defensively inside every query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core_engine.config import get_settings

if TYPE_CHECKING:  # driver only needed to actually connect; annotations are strings
    import psycopg


@dataclass(slots=True)
class GraphNode:
    uri: str
    label: str
    properties: dict[str, Any]


@dataclass(slots=True)
class GraphPath:
    """An ordered chain of nodes + the edge labels connecting them."""

    nodes: list[GraphNode]
    edges: list[str]

    def describe(self) -> str:
        """Human-readable path, e.g. 'Acme -[SUPPLIES]-> WidgetCo -[REGULATED_BY]-> GDPR'."""
        parts: list[str] = []
        for i, node in enumerate(self.nodes):
            label = node.properties.get("name") or node.properties.get("uri") or node.label
            parts.append(str(label))
            if i < len(self.edges):
                parts.append(f"-[{self.edges[i]}]->")
        return " ".join(parts)


@dataclass(slots=True)
class Subgraph:
    nodes: list[GraphNode] = field(default_factory=list)
    # URIs are the join key into the vector store (graph-augmented retrieval).
    def uris(self) -> list[str]:
        return [n.uri for n in self.nodes]


class GraphStore:
    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self._conn = conn
        self._graph = get_settings().age_graph_name

    async def _cypher(self, query: str, columns: str) -> list[tuple[Any, ...]]:
        """Run a Cypher statement via AGE. `columns` declares the agtype output
        columns AGE requires, e.g. '(n agtype)' or '(u agtype, l agtype)'.

        NOTE: AGE does not support host-parameter binding inside cypher(); callers
        MUST pass values that are already validated against the ontology (entity
        labels, property names) or literal-escaped. resolve_entities/expand/find_paths
        only ever interpolate ontology-validated identifiers + escaped string values.
        """
        sql = f"SELECT * FROM cypher('{self._graph}', $$ {query} $$) AS {columns};"
        async with self._conn.cursor() as cur:
            await cur.execute(sql)
            return await cur.fetchall()

    async def resolve_entities(
        self,
        tenant_id: str,
        mentions: list[str],
        *,
        limit_per_mention: int = 5,
    ) -> list[GraphNode]:
        """Anchor text mentions to nodes by matching the label/name property.

        Uses a case-insensitive CONTAINS match; entity resolution/canonicalization
        at ingestion (§1.3) keeps this cheap at query time.
        """
        anchors: list[GraphNode] = []
        for mention in mentions:
            safe = _escape(mention)
            tenant = _escape(tenant_id)
            rows = await self._cypher(
                f"""
                MATCH (n)
                WHERE n.tenant_id = '{tenant}'
                  AND toLower(n.name) CONTAINS toLower('{safe}')
                RETURN n
                LIMIT {int(limit_per_mention)}
                """,
                "(n agtype)",
            )
            anchors.extend(_node_from_agtype(r[0]) for r in rows)
        return anchors

    async def expand(
        self,
        tenant_id: str,
        anchor_uris: list[str],
        *,
        max_hops: int | None = None,
        edge_types: list[str] | None = None,
    ) -> Subgraph:
        """Bounded multi-hop BFS from anchors. Returns every node reachable within
        max_hops, optionally restricted to specific edge labels."""
        if not anchor_uris:
            return Subgraph()
        max_hops = max_hops or get_settings().max_hops
        tenant = _escape(tenant_id)
        uri_list = ", ".join(f"'{_escape(u)}'" for u in anchor_uris)
        edge_filter = ""
        if edge_types:
            labels = "|".join(_escape(e) for e in edge_types)
            edge_filter = f":{labels}"

        rows = await self._cypher(
            f"""
            MATCH (a)
            WHERE a.uri IN [{uri_list}] AND a.tenant_id = '{tenant}'
            MATCH (a)-[{edge_filter}*1..{int(max_hops)}]-(m)
            WHERE m.tenant_id = '{tenant}'
            RETURN DISTINCT m
            """,
            "(m agtype)",
        )
        nodes = [_node_from_agtype(r[0]) for r in rows]
        return Subgraph(nodes=nodes)

    async def find_paths(
        self,
        tenant_id: str,
        from_uri: str,
        to_uri: str,
        *,
        max_hops: int | None = None,
    ) -> list[GraphPath]:
        """Directly answer 'how is X connected to Y' — the multi-hop core.

        Returns the shortest path(s) between the two anchored entities so the
        Report agent can explain the connection with the actual relationship chain.
        """
        max_hops = max_hops or get_settings().max_hops
        tenant = _escape(tenant_id)
        rows = await self._cypher(
            f"""
            MATCH (a {{uri: '{_escape(from_uri)}', tenant_id: '{tenant}'}}),
                  (b {{uri: '{_escape(to_uri)}', tenant_id: '{tenant}'}}),
                  p = shortestPath((a)-[*1..{int(max_hops)}]-(b))
            RETURN p
            """,
            "(p agtype)",
        )
        return [_path_from_agtype(r[0]) for r in rows]


# --- agtype parsing helpers ------------------------------------------------
# AGE returns agtype (JSON-ish). psycopg gives it back as a string with a type
# annotation suffix like '::vertex'. We strip and json-parse.
def _strip_agtype(raw: Any) -> Any:
    import json

    if raw is None:
        return None
    s = str(raw)
    for suffix in ("::vertex", "::edge", "::path"):
        s = s.removesuffix(suffix)
    return json.loads(s)


def _node_from_agtype(raw: Any) -> GraphNode:
    obj = _strip_agtype(raw)
    props = obj.get("properties", {})
    return GraphNode(
        uri=props.get("uri", ""),
        label=obj.get("label", ""),
        properties=props,
    )


def _path_from_agtype(raw: Any) -> GraphPath:
    """A path agtype is a list alternating [vertex, edge, vertex, edge, ...]."""
    seq = _strip_agtype(raw) or []
    nodes: list[GraphNode] = []
    edges: list[str] = []
    for i, element in enumerate(seq):
        if i % 2 == 0:  # vertex
            props = element.get("properties", {})
            nodes.append(GraphNode(props.get("uri", ""), element.get("label", ""), props))
        else:  # edge
            edges.append(element.get("label", ""))
    return GraphPath(nodes=nodes, edges=edges)


def _escape(value: str) -> str:
    """Escape single quotes for safe Cypher string-literal interpolation.

    WARNING — defense-in-depth ONLY. Apache AGE's cypher() does not support
    parameter binding, so every value reaching a query string is interpolated via
    this hand-rolled escaper. Hand-rolled escaping is a known-fragile pattern:
    before enabling this (dormant) layer, audit EVERY interpolation point that
    funnels into cypher() and prefer restructuring queries so untrusted free-form
    text never reaches them. Callers are expected to pass ontology-validated
    identifiers only.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")
