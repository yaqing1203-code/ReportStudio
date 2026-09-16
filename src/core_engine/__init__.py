"""Domain-agnostic AI-agent core engine.

Invariant core. Vertical adaptation lives entirely under adapters/<vertical>/
(ontology.yaml, chunking profiles, agent manifests, RBAC policies, golden sets).
If you find yourself editing this package to onboard a vertical, a domain
assumption has leaked into the core — push it back up into the adapter layer.
"""

__version__ = "0.1.0"
