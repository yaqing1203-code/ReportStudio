"""MCP tool-gateway: the single enforcement point for all tool routing.

Agents speak the tool contract; the router enforces RBAC, validation, call budgets,
mutation gating, and audit. Tools are swappable implementations behind that contract.
"""

from core_engine.gateway.base import ToolRegistry, ToolSpec  # noqa: F401
from core_engine.gateway.router import CallBudget, ToolRouter, ToolResult  # noqa: F401
