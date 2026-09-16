"""Request-scoped security context.

Every operation in the engine flows through a RequestContext. It carries the
tenant and the caller's roles/attributes, and those values are pushed into the
Postgres session as GUCs so Row-Level Security policies enforce isolation at the
data layer — NOT in application code, and NEVER by prompting the model.

Decision locked in: shared-schema + RLS (no physical per-tenant isolation).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # driver only needed to actually connect; annotations are strings
    import psycopg


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Immutable identity for a single inbound request."""

    tenant_id: str
    user_id: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    # ABAC attributes layered on top of roles (dept, clearance, region, ...).
    attributes: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def attr(self, key: str, default: str | None = None) -> str | None:
        for k, v in self.attributes:
            if k == key:
                return v
        return default

    @classmethod
    def build(
        cls,
        tenant_id: str,
        user_id: str,
        roles: Sequence[str] = (),
        attributes: dict[str, str] | None = None,
    ) -> "RequestContext":
        return cls(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=tuple(roles),
            attributes=tuple((attributes or {}).items()),
        )


@asynccontextmanager
async def rls_session(conn: psycopg.AsyncConnection, ctx: RequestContext):
    """Bind the RequestContext to the connection for the duration of a unit of work.

    Sets LOCAL GUCs consumed by the RLS policies in db/init/002_rls.sql. LOCAL means
    they are scoped to the current transaction and cleared automatically on commit /
    rollback — no leakage across pooled connections.
    """
    async with conn.transaction():
        await conn.execute(
            "SELECT set_config('ce.tenant_id', %s, true),"
            "       set_config('ce.user_id',   %s, true),"
            "       set_config('ce.roles',     %s, true)",
            (ctx.tenant_id, ctx.user_id, ",".join(ctx.roles)),
        )
        yield conn
