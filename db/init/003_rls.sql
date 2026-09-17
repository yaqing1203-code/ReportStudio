-- Row-Level Security: the locked-in isolation model (shared schema, no physical
-- per-tenant separation). Isolation is enforced HERE, at the data layer — never in
-- application code and NEVER by prompting the model.
--
-- The application binds three GUCs per unit of work (see security/context.py):
--   ce.tenant_id  - the caller's tenant
--   ce.user_id    - the caller
--   ce.roles      - comma-separated role list
-- LOCAL-scoped, so they clear on commit/rollback and never leak across the pool.
--
-- Two policies stack on ce_chunks:
--   1. tenant isolation   : row.tenant_id must equal ce.tenant_id
--   2. document-level ACL : allowed_roles empty, OR intersects the caller's roles
-- A row is visible only if BOTH pass. This is the HARD pre-filter the retriever
-- relies on — a caller physically cannot read what they aren't entitled to, so the
-- model cannot leak it.

-- Helper: parse the comma-separated ce.roles GUC into a text[].
CREATE OR REPLACE FUNCTION ce_current_roles() RETURNS text[]
    LANGUAGE sql STABLE AS $$
    SELECT CASE
        WHEN current_setting('ce.roles', true) IS NULL
          OR current_setting('ce.roles', true) = '' THEN '{}'::text[]
        ELSE string_to_array(current_setting('ce.roles', true), ',')
    END
$$;

CREATE OR REPLACE FUNCTION ce_current_tenant() RETURNS text
    LANGUAGE sql STABLE AS $$
    SELECT current_setting('ce.tenant_id', true)
$$;

ALTER TABLE ce_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ce_chunks FORCE ROW LEVEL SECURITY;   -- applies even to the table owner

DROP POLICY IF EXISTS p_chunks_tenant_read ON ce_chunks;
CREATE POLICY p_chunks_tenant_read ON ce_chunks
    FOR SELECT
    USING (
        tenant_id = ce_current_tenant()
        AND (
            cardinality(allowed_roles) = 0
            OR allowed_roles && ce_current_roles()   -- array overlap
        )
    );

-- Writes are tenant-scoped too; ingestion runs under a service context.
-- IMPORTANT: these are scoped to INSERT/UPDATE/DELETE, NOT `FOR ALL`. A `FOR ALL`
-- write policy would also apply to SELECT, and since permissive policies are OR'd
-- together it would let a caller read ANY tenant-matching row — silently bypassing
-- the allowed_roles ACL in p_chunks_tenant_read. Keeping SELECT out of the write
-- policy means reads are governed solely by the ACL-aware read policy above.
DROP POLICY IF EXISTS p_chunks_tenant_write ON ce_chunks;      -- legacy FOR ALL policy
DROP POLICY IF EXISTS p_chunks_tenant_insert ON ce_chunks;
CREATE POLICY p_chunks_tenant_insert ON ce_chunks
    FOR INSERT
    WITH CHECK (tenant_id = ce_current_tenant());

DROP POLICY IF EXISTS p_chunks_tenant_update ON ce_chunks;
CREATE POLICY p_chunks_tenant_update ON ce_chunks
    FOR UPDATE
    USING (tenant_id = ce_current_tenant())
    WITH CHECK (tenant_id = ce_current_tenant());

DROP POLICY IF EXISTS p_chunks_tenant_delete ON ce_chunks;
CREATE POLICY p_chunks_tenant_delete ON ce_chunks
    FOR DELETE
    USING (tenant_id = ce_current_tenant());

-- Audit is tenant-scoped on read; append-only in practice (no UPDATE/DELETE policy).
ALTER TABLE ce_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE ce_audit FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS p_audit_tenant_read ON ce_audit;
CREATE POLICY p_audit_tenant_read ON ce_audit
    FOR SELECT USING (tenant_id = ce_current_tenant());

DROP POLICY IF EXISTS p_audit_insert ON ce_audit;
CREATE POLICY p_audit_insert ON ce_audit
    FOR INSERT WITH CHECK (tenant_id = ce_current_tenant());

-- NOTE on the AGE graph: AGE stores vertices/edges in per-label tables under the
-- graph's schema. We ALSO carry tenant_id as a node property and filter on it in
-- every Cypher query (see stores/graph.py) as defense-in-depth, since RLS on AGE's
-- internal tables is not ergonomic. When you send your schema we'll decide whether
-- any graph label needs its own hardened policy.
