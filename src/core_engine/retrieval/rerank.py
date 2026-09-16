"""Cross-encoder re-ranker — the highest-ROI accuracy lever in the pipeline.

Retrieve wide (top_k ~50 from fused legs), then re-rank to a precise top_n (~8)
with a cross-encoder that scores (query, chunk) jointly. Unlike the bi-encoder
used for embeddings, the cross-encoder sees both texts at once, so it resolves
subtle relevance the ANN leg cannot.

The concrete model is swappable; the interface is what the retriever depends on.
"""
from __future__ import annotations

from typing import Protocol

from core_engine.stores.vector import Candidate


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[Candidate], top_n: int
    ) -> list[Candidate]:
        ...


class CrossEncoderReranker:
    """Default implementation backed by a sentence-transformers cross-encoder.

    Lazy-loads the model so importing the package stays cheap and test doubles can
    replace it without pulling torch.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # heavy import, deferred

            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(
        self, query: str, candidates: list[Candidate], top_n: int
    ) -> list[Candidate]:
        if not candidates:
            return []
        model = self._ensure()
        pairs = [(query, c.content) for c in candidates]
        # CrossEncoder.predict is sync/CPU-GPU bound; run in a thread to avoid
        # blocking the event loop.
        import anyio

        scores = await anyio.to_thread.run_sync(lambda: model.predict(pairs))
        for cand, score in zip(candidates, scores):
            cand.score = float(score)
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
        for i, cand in enumerate(ranked):
            cand.rank = i
        return ranked


class NoopReranker:
    """Fallback used in tests / environments without the reranker model. Preserves
    the incoming fused order and just truncates."""

    async def rerank(
        self, query: str, candidates: list[Candidate], top_n: int
    ) -> list[Candidate]:
        ranked = candidates[:top_n]
        for i, cand in enumerate(ranked):
            cand.rank = i
        return ranked
