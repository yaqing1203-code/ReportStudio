"""Tool contract + registry for the MCP tool-gateway.

Every capability the agents can invoke — query the KG, run RAG retrieval, look up
a policy — is a Tool. Agents NEVER touch the database, the graph, or an external
API directly. They emit a tool call; the gateway (see router.py) is the single
enforcement point for auth, RBAC scoping, input validation, and audit logging.

This mirrors the MCP model: each Tool declares a JSON-Schema input contract and a
set of required roles. The gateway is the MCP host; tools are what MCP servers
expose. Swapping a tool's implementation (e.g. a different DB) never changes the
agent-facing contract.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from core_engine.security.context import RequestContext

# A tool handler receives the request context (identity) + validated args and
# returns a JSON-serialisable result. The context lets the handler open an
# RLS-scoped session so isolation is enforced at the data layer.
ToolHandler = Callable[[RequestContext, dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    """The agent-facing contract for one tool. This IS the API — invest in the
    description and schema, because it is literally what the model reasons over."""

    name: str
    description: str
    # JSON Schema for the input arguments. Validated by the gateway before dispatch.
    input_schema: dict[str, Any]
    handler: ToolHandler
    # RBAC: caller must hold at least one of these roles. Empty => any authenticated.
    required_roles: tuple[str, ...] = ()
    # Whether this tool mutates state. Mutating tools get dry-run + confirm handling
    # in the router (never expose raw writes to an agent without a human gate).
    mutating: bool = False
    # Cap runaway agents: max invocations of THIS tool per request trajectory.
    max_calls_per_request: int = 16

    def to_mcp(self) -> dict[str, Any]:
        """Render as an MCP tool descriptor for discovery/advertisement."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "requiredRoles": list(self.required_roles),
                "mutating": self.mutating,
            },
        }


class ToolError(Exception):
    """Raised by handlers for expected, agent-recoverable failures (bad args,
    not-found). The router converts these into structured error results rather
    than crashing the trajectory."""


class Authorizer(Protocol):
    def can_call(self, ctx: RequestContext, spec: ToolSpec) -> bool: ...


@dataclass(slots=True)
class RoleAuthorizer:
    """Default RBAC check: role membership. ABAC attribute rules layer on top via
    the policy engine (adapter layer) — kept out of the core so verticals swap it."""

    def can_call(self, ctx: RequestContext, spec: ToolSpec) -> bool:
        if not spec.required_roles:
            return True
        return any(r in spec.required_roles for r in ctx.roles)


@dataclass
class ToolRegistry:
    """Holds the available tools. An agent manifest names a SUBSET of these; adding
    a new capability means registering a ToolSpec, not editing the router."""

    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"Unknown tool: {name}")

    def manifest(self, allowed: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """MCP tool list, optionally filtered to the subset an agent may use."""
        specs = self._tools.values()
        if allowed is not None:
            specs = [s for s in specs if s.name in allowed]
        return [s.to_mcp() for s in specs]
