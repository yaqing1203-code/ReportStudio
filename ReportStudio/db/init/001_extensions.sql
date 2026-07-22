-- Core-engine database bootstrap: one Postgres instance, three access patterns.
-- Run these in order (001 -> 002 -> 003). Idempotent where possible.

-- Apache AGE: the Knowledge Graph lives here (multi-hop reasoning, prioritized).
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, public;

-- pgvector: dense embeddings for the RAG semantic leg.
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_trgm: fuzzy text matching for entity resolution / keyword fallback.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
