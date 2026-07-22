"""Ontology loader + KG projector.

Two responsibilities:
  1. load()          -> parse & validate ontology.yaml into the Ontology model.
  2. apply_to_graph  -> project the ontology onto Apache AGE: ensure the graph
                        exists, create property indexes, and (later) drive the
                        structured-ingestion mapping from your relational tables
                        into graph nodes/edges.

When you send your database schema, we generate/fill the `mappings:` section and
call ingest_relational() to populate the graph — no engine code changes needed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from core_engine.config import get_settings
from core_engine.ontology.schema import Ontology

if TYPE_CHECKING:  # driver only needed to actually connect; annotations are strings
    import psycopg

log = logging.getLogger(__name__)


def load(path: Path | None = None) -> Ontology:
    path = path or get_settings().ontology_path
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    onto = Ontology.model_validate(raw)
    log.info(
        "Loaded ontology '%s' v%d: %d entities, %d relationships, %d mappings",
        onto.name, onto.version, len(onto.entities),
        len(onto.relationships), len(onto.mappings),
    )
    return onto


async def ensure_graph(conn: psycopg.AsyncConnection) -> None:
    graph = get_settings().age_graph_name
    # create_graph is not idempotent (AGE errors if the graph already exists),
    # so guard on the catalog first.
    (exists,) = await (await conn.execute(
        "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s", (graph,)
    )).fetchone()
    if not exists:
        await conn.execute("SELECT create_graph(%s);", (graph,))
        log.info("Created AGE graph '%s'", graph)


async def apply_to_graph(conn: psycopg.AsyncConnection, onto: Ontology) -> None:
    """Register vertex/edge labels and create indexes on indexed properties."""
    graph = get_settings().age_graph_name
    await ensure_graph(conn)

    for entity in onto.entities:
        if entity.abstract:
            continue
        # AGE errors on duplicate labels; guard with catalog check.
        await _ensure_vlabel(conn, graph, entity.type)
        for prop in entity.properties:
            if prop.indexed:
                await _ensure_property_index(conn, graph, entity.type, prop.name)

    for rel in onto.relationships:
        await _ensure_elabel(conn, graph, rel.type)


async def _ensure_vlabel(conn, graph: str, label: str) -> None:
    row = await (await conn.execute(
        "SELECT count(*) FROM ag_catalog.ag_label "
        "WHERE name = %s AND graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)",
        (label, graph),
    )).fetchone()
    if not row[0]:
        await conn.execute("SELECT create_vlabel(%s, %s);", (graph, label))


async def _ensure_elabel(conn, graph: str, label: str) -> None:
    row = await (await conn.execute(
        "SELECT count(*) FROM ag_catalog.ag_label "
        "WHERE name = %s AND graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)",
        (label, graph),
    )).fetchone()
    if not row[0]:
        await conn.execute("SELECT create_elabel(%s, %s);", (graph, label))


async def _ensure_property_index(conn, graph: str, label: str, prop: str) -> None:
    # AGE stores properties as agtype; a btree expression index accelerates lookups.
    idx = f"idx_{graph}_{label}_{prop}"
    await conn.execute(
        f'CREATE INDEX IF NOT EXISTS "{idx}" '
        f'ON "{graph}"."{label}" USING btree (agtype_access_operator(properties, %s::agtype));',
        (f'"{prop}"',),
    )
