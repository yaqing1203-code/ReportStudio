"""Ontology = the machine-readable domain adapter. Swap adapters/<vertical>/ontology.yaml
to pivot verticals; the engine reads it at runtime and never hard-codes entity types."""

from core_engine.ontology.schema import Ontology  # noqa: F401
