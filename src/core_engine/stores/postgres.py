"""Async Postgres connection pool. Single instance backs all three access
patterns: relational (source of truth), AGE (graph), pgvector (embeddings)."""
from __future__ import annotations

from collections.abc import AsyncIterator

import psycopg
from psycopg_pool import AsyncConnectionPool

from core_engine.config import get_settings

_pool: AsyncConnectionPool | None = None


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = AsyncConnectionPool(
            conninfo=s.pg_dsn,
            min_size=s.pg_pool_min,
            max_size=s.pg_pool_max,
            open=False,
            # Load AGE + set search_path on every fresh connection so Cypher works.
            configure=_configure_connection,
        )
        await _pool.open()
    return _pool


async def _configure_connection(conn: psycopg.AsyncConnection) -> None:
    await conn.execute("LOAD 'age';")
    await conn.execute("SET search_path = ag_catalog, public;")
    await conn.set_autocommit(True)


async def acquire() -> AsyncIterator[psycopg.AsyncConnection]:
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
