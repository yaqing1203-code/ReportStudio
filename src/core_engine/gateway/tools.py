"""Built-in tool set — the capabilities agents invoke through the gateway.

Each tool is a thin adapter: it opens an RLS-scoped session from the request
context, calls into the retriever / graph store, and returns JSON-serialisable
data. Agents see only the ToolSpec contract (name + schema); the implementation
behind it is swappable.

To add a capability (e.g. a Report-render tool, a compliance policy lookup),
register another ToolSpec here — the router and agents need no changes.
"""
from __future__ import annotations

from typing import Any

from psycopg_pool import AsyncConnectionPool

from core_engine.embeddings import Embedder, get_embedder
from core_engine.gateway.base import ToolRegistry, ToolSpec
from core_engine.retrieval.rerank import CrossEncoderReranker, Reranker
from core_engine.retrieval.retriever import MultiHopRetriever
from core_engine.security.context import RequestContext, rls_session
from core_engine.stores.graph import GraphStore


def build_default_registry(
    pool: AsyncConnectionPool,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> ToolRegistry:
    """Wire the retriever + graph store into a registry of MCP tools.

    The pool/embedder/reranker are captured in closures so handlers stay stateless
    from the agent's perspective — they just receive (ctx, args).
    """
    embedder = embedder or get_embedder()
    reranker = reranker or CrossEncoderReranker()
    registry = ToolRegistry()

    # -- rag_search: hybrid dense+keyword retrieval with optional graph expansion --
    async def rag_search(ctx: RequestContext, args: dict[str, Any]) -> Any:
        async with pool.connection() as conn, rls_session(conn, ctx):
            retriever = MultiHopRetriever(conn, embedder, reranker)
            result = await retriever.retrieve(
                ctx,
                args["query"],
                mentions=args.get("mentions"),
                metadata_filter=args.get("metadata_filter"),
                top_n=args.get("top_n"),
            )
            if result.abstain:
                return {"abstain": True,
                        "reason": "Insufficient grounded evidence to answer."}
            return {
                "abstain": False,
                "contexts": [
                    {"uri": c.uri, "chunk_id": c.chunk_id,
                     "content": c.content, "score": c.score}
                    for c in result.contexts
                ],
                "citations": result.citations(),
            }

    registry.register(ToolSpec(
        name="rag_search",
        description=(
            "Retrieve grounded passages for a natural-language query using hybrid "
            "semantic + keyword search. Optionally pass `mentions` (entity names) to "
            "pull in KG-connected context. Returns passages with citations, or "
            "{abstain:true} when evidence is too thin to answer safely."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to retrieve for."},
                "mentions": {"type": "array", "items": {"type": "string"},
                             "description": "Entity names to anchor into the knowledge graph."},
                "metadata_filter": {"type": "object",
                                    "description": "JSONB containment pre-filter, e.g. {\"sensitivity\":\"public\"}."},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ))

    # -- kg_connect: 'how is X related to Y' — the multi-hop path finder --
    async def kg_connect(ctx: RequestContext, args: dict[str, Any]) -> Any:
        async with pool.connection() as conn, rls_session(conn, ctx):
            graph = GraphStore(conn)
            x = await graph.resolve_entities(ctx.tenant_id, [args["from_entity"]],
                                             limit_per_mention=1)
            y = await graph.resolve_entities(ctx.tenant_id, [args["to_entity"]],
                                             limit_per_mention=1)
            if not x or not y:
                return {"connected": False,
                        "reason": "Could not resolve one or both entities in the KG."}
            paths = await graph.find_paths(
                ctx.tenant_id, x[0].uri, y[0].uri, max_hops=args.get("max_hops"),
            )
            return {
                "connected": bool(paths),
                "paths": [{"description": p.describe(),
                           "uris": [n.uri for n in p.nodes],
                           "edges": p.edges} for p in paths],
            }

    registry.register(ToolSpec(
        name="kg_connect",
        description=(
            "Find how two entities are connected in the knowledge graph. Returns the "
            "actual relationship path(s) between them (e.g. 'Acme -[SUPPLIES]-> WidgetCo "
            "-[AFFILIATED_WITH]-> RegBody'). Use for 'how is X related to Y' questions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "from_entity": {"type": "string"},
                "to_entity": {"type": "string"},
                "max_hops": {"type": "integer", "minimum": 1, "maximum": 6},
            },
            "required": ["from_entity", "to_entity"],
            "additionalProperties": False,
        },
    ))

    # -- kg_neighbors: expand N hops from an entity (subgraph exploration) --
    async def kg_neighbors(ctx: RequestContext, args: dict[str, Any]) -> Any:
        async with pool.connection() as conn, rls_session(conn, ctx):
            graph = GraphStore(conn)
            anchors = await graph.resolve_entities(
                ctx.tenant_id, [args["entity"]], limit_per_mention=3)
            if not anchors:
                return {"found": False, "neighbors": []}
            sub = await graph.expand(
                ctx.tenant_id, [a.uri for a in anchors],
                max_hops=args.get("max_hops"), edge_types=args.get("edge_types"),
            )
            return {
                "found": True,
                "neighbors": [{"uri": n.uri, "label": n.label,
                               "name": n.properties.get("name")} for n in sub.nodes],
            }

    registry.register(ToolSpec(
        name="kg_neighbors",
        description=(
            "Expand outward from an entity up to N hops in the knowledge graph, "
            "optionally restricted to specific relationship types. Returns the "
            "connected entities. Use to explore what an entity relates to."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "max_hops": {"type": "integer", "minimum": 1, "maximum": 6},
                "edge_types": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["entity"],
            "additionalProperties": False,
        },
    ))

    return registry
