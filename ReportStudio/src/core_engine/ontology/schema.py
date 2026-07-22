"""Ontology data model.

This is the machine-readable subset of RDFS/OWL that drives the ENTIRE data-layer
domain adaptation. The engine has zero hard-coded entity types — it reads these
Pydantic models from adapters/<vertical>/ontology.yaml at runtime.

To pivot verticals you replace the YAML. You never touch this file.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DType(str, Enum):
    string = "string"
    text = "text"
    integer = "integer"
    number = "number"
    boolean = "boolean"
    datetime = "datetime"
    enum = "enum"
    uuid = "uuid"


class PropertyDef(BaseModel):
    name: str
    dtype: DType
    indexed: bool = False
    required: bool = False
    values: list[str] = Field(default_factory=list)  # only for dtype == enum

    @model_validator(mode="after")
    def _check_enum(self) -> "PropertyDef":
        if self.dtype is DType.enum and not self.values:
            raise ValueError(f"enum property '{self.name}' must declare values")
        return self


class EntityDef(BaseModel):
    type: str
    abstract: bool = False
    parent: str | None = None  # single inheritance for shared property sets
    properties: list[PropertyDef] = Field(default_factory=list)
    # Which property should be embedded / used as the human label.
    label_property: str | None = None
    searchable_properties: list[str] = Field(default_factory=list)

    def property_map(self) -> dict[str, PropertyDef]:
        return {p.name: p for p in self.properties}


Cardinality = Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]


class RelationshipDef(BaseModel):
    type: str                       # edge label, e.g. AUTHORED_BY
    from_: str = Field(alias="from")
    to: str
    cardinality: Cardinality = "many_to_many"
    directed: bool = True
    properties: list[PropertyDef] = Field(default_factory=list)
    # If set, this edge can be traversed transitively during multi-hop expansion.
    transitive: bool = False

    model_config = {"populate_by_name": True}


class MappingRule(BaseModel):
    """Declarative rule mapping a relational source (table) onto the graph.

    Populated against YOUR upcoming schema. Deterministic — no LLM involved for
    structured ingestion.
    """

    source_table: str
    entity_type: str
    # column -> entity property
    column_map: dict[str, str]
    primary_key: str
    # Foreign keys become typed edges.
    edges: list["EdgeMappingRule"] = Field(default_factory=list)


class EdgeMappingRule(BaseModel):
    relationship_type: str
    fk_column: str            # column on source_table holding the reference
    target_entity: str
    target_key: str = "id"    # column on the target the fk points to


class ExtractionRule(BaseModel):
    """How to pull an entity out of UNSTRUCTURED text (PDFs, reports)."""

    entity: str
    strategy: Literal["ner", "ner_plus_llm", "llm"] = "ner_plus_llm"
    canonicalization_threshold: float = 0.88


class Ontology(BaseModel):
    version: int = 1
    name: str = "default"
    entities: list[EntityDef]
    relationships: list[RelationshipDef] = Field(default_factory=list)
    mappings: list[MappingRule] = Field(default_factory=list)
    extraction: list[ExtractionRule] = Field(default_factory=list)

    # ---- convenience indexes, built once on load ----
    def entity(self, type_: str) -> EntityDef:
        for e in self.entities:
            if e.type == type_:
                return e
        raise KeyError(f"Unknown entity type: {type_}")

    def relationship(self, type_: str) -> RelationshipDef:
        for r in self.relationships:
            if r.type == type_:
                return r
        raise KeyError(f"Unknown relationship type: {type_}")

    def transitive_edges(self) -> list[str]:
        return [r.type for r in self.relationships if r.transitive]

    @model_validator(mode="after")
    def _validate_references(self) -> "Ontology":
        types = {e.type for e in self.entities}
        for r in self.relationships:
            for endpoint in (r.from_, r.to):
                if endpoint not in types:
                    raise ValueError(
                        f"Relationship '{r.type}' references unknown entity '{endpoint}'"
                    )
        for m in self.mappings:
            if m.entity_type not in types:
                raise ValueError(
                    f"Mapping for table '{m.source_table}' references unknown entity "
                    f"'{m.entity_type}'"
                )
        return self
