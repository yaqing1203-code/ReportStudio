"""Embedding provider abstraction.

BGE-M3 gives dense + sparse (lexical) vectors from ONE model, which is why it's
the default: hybrid search needs both and we avoid running two models. The
Embedder is an interface so you can swap in a commercial API later without
touching the retriever.

`embedding_model_version` is returned alongside every vector and MUST be stored
in metadata — model upgrades require re-embedding and running old/new indexes in
parallel during migration.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from core_engine.config import get_settings


@dataclass(slots=True)
class Embedding:
    dense: list[float]
    # token_id -> weight; the sparse/lexical half of hybrid search.
    sparse: dict[int, float]
    model_version: str


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[Embedding]: ...
    async def embed_query(self, text: str) -> Embedding: ...


class BGEM3Embedder:
    """Local BGE-M3. Lazy-loads the model on first use.

    Kept deliberately thin: the FlagEmbedding call is wrapped so the rest of the
    engine only ever sees the Embedding dataclass.
    """

    def __init__(self, model_name: str, dim: int) -> None:
        self._model_name = model_name
        self._dim = dim
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily so the package imports without the heavy dep present.
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self._model_name, use_fp16=True)
        return self._model

    async def embed(self, texts: list[str]) -> list[Embedding]:
        model = self._ensure_model()
        out = model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        results: list[Embedding] = []
        for dense, sparse in zip(out["dense_vecs"], out["lexical_weights"]):
            results.append(
                Embedding(
                    dense=list(map(float, dense)),
                    sparse={int(k): float(v) for k, v in sparse.items()},
                    model_version=self._model_name,
                )
            )
        return results

    async def embed_query(self, text: str) -> Embedding:
        return (await self.embed([text]))[0]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return BGEM3Embedder(s.embedding_model, s.embedding_dim)
