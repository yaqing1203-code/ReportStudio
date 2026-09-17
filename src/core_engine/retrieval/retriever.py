"""Multi-hop RAG retriever — fuses four legs into one grounded context.

Pipeline (all legs share the same RLS/ACL pre-filters via the session context):

    query ─┬─ dense (pgvector)        ┐
           ├─ keyword (full-text)     ├─ Reciprocal Rank Fusion ─┐
           │                          ┘                          │
           └─ GRAPH multi-hop:                                   │
                resolve mentions → anchor nodes                  ├─ cross-encoder
                expand N hops (Subgraph)                         │   re-rank
                fetch chunks for connected node URIs ────────────┘        │
                                                                          ▼
                                                          top_n grounded context
                                                          + relationship paths
                                                          + citations (URIs)

The graph leg is what makes 'how does X connect to Y' work: we don't just retrieve
text that mentions X and Y, we traverse the KG to find the *actual path* between
them and pull the chunks attached to every node on that path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core_engine.config import get_settings
from core_engine.embeddings import Embedder
from core_engine.retrieval.rerank import Reranker
from core_engine.security.context import RequestContext
from core_engine.stores.graph import GraphPath, GraphStore
from core_engine.stores.vector import Candidate, VectorStore

if TYPE_CHECKING:  # driver only needed to actually connect; annotations are strings
    import psycopg


@dataclass(slots=True)
class RetrievalResult:
    """Everything the generation layer needs to answer AND to ground the answer."""

    query: str
    contexts: list[Candidate] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)  # explicit X→Y connections
    abstain: bool = False  # true when evidence is too thin to answer safely (§5.3)

    def citations(self) -> list[str]:
        return sorted({c.uri for c in self.contexts if c.uri})

    def path_explanations(self) -> list[str]:
        return [p.describe() for p in self.paths]


class MultiHopRetriever:
    def __init__(
        self,
        conn: psycopg.AsyncConnection,
        embedder: Embedder,
        reranker: Reranker,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._reranker = reranker
        self._vec = VectorStore(conn)
        self._graph = GraphStore(conn)
        self._s = get_settings()

    async def retrieve(
        self,
        ctx: RequestContext,
        query: str,
        *,
        mentions: list[str] | None = None,
        connect: tuple[str, str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
        max_hops: int | None = None,
    ) -> RetrievalResult:
        """Run all legs and fuse.

        Args:
            mentions: entity mentions to anchor into the KG. If None, we still run
                dense+keyword; graph leg is skipped. (An upstream Query-Planner agent
                normally extracts these; passing them explicitly keeps the retriever
                deterministic and testable.)
            connect: an (X, Y) pair to find explicit relationship paths for. This is
                the multi-hop 'how are these related' entry point.
            metadata_filter: additional JSONB pre-filter (sensitivity, type, date...).
        """
        top_k = top_k or self._s.retrieval_top_k
        top_n = top_n or self._s.rerank_top_n
        max_hops = max_hops or self._s.max_hops

        # --- Leg 1 + 2: dense & keyword (independent, run concurrently) ---
        # embed_query returns an Embedding (dense + sparse + version); the dense
        # leg needs only the dense vector. The sparse half is available on
        # query_emb.sparse for a future sparse-vector leg without re-embedding.
        query_emb = await self._embedder.embed_query(query)
        import anyio

        dense: list[Candidate] = []
        keyword: list[Candidate] = []

        async with anyio.create_task_group() as tg:
            async def _dense() -> None:
                nonlocal dense
                dense = await self._vec.dense_search(
                    query_emb.dense, top_k=top_k, metadata_filter=metadata_filter
                )

            async def _keyword() -> None:
                nonlocal keyword
                keyword = await self._vec.keyword_search(
                    query, top_k=top_k, metadata_filter=metadata_filter
                )

            tg.start_soon(_dense)
            tg.start_soon(_keyword)

        # --- Leg 3: graph multi-hop expansion ---
        graph_candidates: list[Candidate] = []
        paths: list[GraphPath] = []
        if mentions:
            anchors = await self._graph.resolve_entities(ctx.tenant_id, mentions)
            anchor_uris = [a.uri for a in anchors if a.uri]
            if anchor_uris:
                subgraph = await self._graph.expand(
                    ctx.tenant_id, anchor_uris, max_hops=max_hops
                )
                graph_candidates = await self._vec.fetch_by_uris(subgraph.uris())

        # --- Leg 4: explicit path finding for 'how is X connected to Y' ---
        if connect:
            paths = await self._resolve_and_connect(ctx, *connect, max_hops=max_hops)
            # Pull chunks for every node on the discovered paths too.
            path_uris = {n.uri for p in paths for n in p.nodes if n.uri}
            if path_uris:
                graph_candidates += await self._vec.fetch_by_uris(list(path_uris))

        # --- Fuse dense + keyword + graph via RRF ---
        fused = _reciprocal_rank_fusion(
            [dense, keyword, graph_candidates], k=self._s.rrf_k
        )

        # --- Abstain if the best evidence is too weak (anti-hallucination) ---
        if not fused or (dense and max(c.score for c in dense) < self._s.min_retrieval_score
                         and not paths):
            return RetrievalResult(query=query, contexts=[], paths=paths, abstain=True)

        # --- Re-rank to precise top_n ---
        reranked = await self._reranker.rerank(query, fused, top_n=top_n)
        return RetrievalResult(query=query, contexts=reranked, paths=paths, abstain=False)

    async def _resolve_and_connect(
        self, ctx: RequestContext, x: str, y: str, *, max_hops: int
    ) -> list[GraphPath]:
        x_nodes = await self._graph.resolve_entities(ctx.tenant_id, [x], limit_per_mention=1)
        y_nodes = await self._graph.resolve_entities(ctx.tenant_id, [y], limit_per_mention=1)
        if not x_nodes or not y_nodes:
            return []
        return await self._graph.find_paths(
            ctx.tenant_id, x_nodes[0].uri, y_nodes[0].uri, max_hops=max_hops
        )


def _reciprocal_rank_fusion(
    leg_results: list[list[Candidate]], *, k: int
) -> list[Candidate]:
    """Merge ranked lists with RRF: score = Σ 1/(k + rank). Tuning-free and robust to
    the different score scales of dense/keyword/graph legs.

    Dedupe on chunk_id, keeping the fused score and remembering which legs hit it.
    """
    fused: dict[str, Candidate] = {}
    scores: dict[str, float] = {}
    for leg in leg_results:
        for rank, cand in enumerate(leg):
            key = cand.chunk_id or cand.uri
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in fused:
                fused[key] = cand
            elif cand.source not in fused[key].source:
                fused[key].source += f"+{cand.source}"
    for key, cand in fused.items():
        cand.score = scores[key]
    return sorted(fused.values(), key=lambda c: c.score, reverse=True)
