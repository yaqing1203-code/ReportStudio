-- The RAG chunk store (pgvector). Every chunk links back to its KG node and
-- canonical relational row via `uri` (the Universal Resource Identifier, §1.1).
--
-- Embedding dimension defaults to 1024 (bge-m3). If you change CE_EMBEDDING_DIM,
-- change the vector(1024) below to match, or the dense leg will error on insert.

CREATE TABLE IF NOT EXISTS ce_chunks (
    chunk_id      text PRIMARY KEY,
    uri           text NOT NULL,               -- urn:{tenant}:{domain}:{type}:{id}
    tenant_id     text NOT NULL,               -- mandatory isolation dimension (RLS)
    content       text NOT NULL,
    -- document-level ACL: caller must share a role in allowed_roles (HARD pre-filter).
    -- empty array => readable by any authenticated caller in the tenant.
    allowed_roles text[] NOT NULL DEFAULT '{}',
    metadata      jsonb  NOT NULL DEFAULT '{}',
    embedding     vector(1024),                -- dense (bge-m3)
    sparse        jsonb  NOT NULL DEFAULT '{}', -- token_id -> weight (lexical leg)
    -- generated full-text column for the keyword leg (BM25-ish via ts_rank).
    content_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding_model_version text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- ANN index for dense search. HNSW = fast recall; cosine to match `<=>` in queries.
CREATE INDEX IF NOT EXISTS idx_ce_chunks_embedding
    ON ce_chunks USING hnsw (embedding vector_cosine_ops);

-- Keyword leg.
CREATE INDEX IF NOT EXISTS idx_ce_chunks_tsv
    ON ce_chunks USING gin (content_tsv);

-- Metadata pre-filter (@> containment) + tenant scoping.
CREATE INDEX IF NOT EXISTS idx_ce_chunks_metadata ON ce_chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_ce_chunks_tenant   ON ce_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ce_chunks_uri      ON ce_chunks (uri);

-- Immutable audit log (§5.4). Append-only; every tool call + retrieval lands here.
CREATE TABLE IF NOT EXISTS ce_audit (
    id          bigserial PRIMARY KEY,
    tenant_id   text NOT NULL,
    user_id     text NOT NULL,
    roles       text[] NOT NULL DEFAULT '{}',
    tool        text,
    outcome     text NOT NULL,
    arg_keys    text[] NOT NULL DEFAULT '{}',
    latency_ms  double precision,
    at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ce_audit_tenant_at ON ce_audit (tenant_id, at DESC);
