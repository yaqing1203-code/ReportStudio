"""Vector store over pgvector.

Chunks live in the same Postgres instance as the relational tables and the AGE
graph, so RLS applies here too (see db/init/002_rls.sql). Every chunk row carries:
  - uri            : the Universal Resource Identifier (§1.1) linking back to the
                     KG node and the canonical relational row it was derived from.
  - tenant_id      : mandatory isolation dimension (RLS pre-filter).
  - allowed_roles  : document-level ACL enforced as a HARD pre-filter, not a prompt.
  - metadata       : arbitrary JSONB for additional pre-filtering.
  - embedding      : dense vector (bge-m3, 1024-d by default).
  - sparse         : optional sparse/lexical weights for hybrid (stored as JSONB).

Hybrid search = dense (pgvector) + keyword (Postgres full-text / BM25-ish) fused
by Reciprocal Rank Fusion in the retriever. This module exposes the two legs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from core_engine.config import get_settings

if TYPE_CHECKING:  # driver only needed to actually connect; annotations are strings
    import psycopg


@dataclass(slots=True)
class Candidate:
    """A single retrieval hit, from either the dense or keyword leg."""

    uri: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any]
    source: str  # "dense" | "keyword" — for debugging fusion

    # populated after fusion / re-rank
    rank: int | None = None


class VectorStore:
    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self._conn = conn
        self._s = get_settings()

    async def dense_search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        """Cosine-distance ANN over pgvector. RLS + ACL applied automatically by the
        session GUCs; metadata_filter is an additional JSONB containment pre-filter."""
        top_k = top_k or self._s.retrieval_top_k
        where, params = _metadata_where(metadata_filter)
        # `<=>` is cosine distance in pgvector; 1 - distance ≈ similarity.
        sql = f"""
            SELECT uri, chunk_id, content, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM ce_chunks
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vec = list(embedding)
        from psycopg.rows import dict_row  # runtime-only; keeps import off module load

        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, [vec, *params, vec, top_k])
            rows = await cur.fetchall()
        return [
            Candidate(
                uri=r["uri"], chunk_id=r["chunk_id"], content=r["content"],
                score=float(r["score"]), metadata=r["metadata"] or {}, source="dense",
            )
            for r in rows
        ]

    async def keyword_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        """Lexical leg via Postgres full-text search (ts_rank). Catches exact IDs,
        part numbers, and proper nouns that dense embeddings miss."""
        top_k = top_k or self._s.retrieval_top_k
        where, params = _metadata_where(metadata_filter)
        connector = "AND" if where else "WHERE"
        sql = f"""
            SELECT uri, chunk_id, content, metadata,
                   ts_rank(content_tsv, websearch_to_tsquery('english', %s)) AS score
            FROM ce_chunks
            {where}
            {connector} content_tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """
        from psycopg.rows import dict_row  # runtime-only; keeps import off module load

        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, [query, *params, query, top_k])
            rows = await cur.fetchall()
        return [
            Candidate(
                uri=r["uri"], chunk_id=r["chunk_id"], content=r["content"],
                score=float(r["score"]), metadata=r["metadata"] or {}, source="keyword",
            )
            for r in rows
        ]

    async def fetch_by_uris(self, uris: Sequence[str]) -> list[Candidate]:
        """Pull chunks for a set of KG-discovered node URIs (graph-augmented leg)."""
        if not uris:
            return []
        from psycopg.rows import dict_row

        sql = """
            SELECT uri, chunk_id, content, metadata
            FROM ce_chunks
            WHERE uri = ANY(%s)
        """
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, [list(uris)])
            rows = await cur.fetchall()
        return [
            Candidate(
                uri=r["uri"], chunk_id=r["chunk_id"], content=r["content"],
                score=0.0, metadata=r["metadata"] or {}, source="graph",
            )
            for r in rows
        ]


def _metadata_where(
    metadata_filter: dict[str, Any] | None,
) -> tuple[str, list[Any]]:
    """Build a JSONB containment pre-filter. RLS/ACL are handled by policies, so we
    only translate caller-supplied metadata constraints here."""
    if not metadata_filter:
        return "", []
    # @> containment: metadata @> '{"sensitivity":"public"}'
    return "WHERE metadata @> %s::jsonb", [_as_jsonb(metadata_filter)]


def _as_jsonb(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d)
