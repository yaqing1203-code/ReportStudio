"""The tool-gateway router — the single enforcement point.

Every tool call from every agent funnels through dispatch(). Nothing else in the
system is allowed to call a ToolSpec.handler directly. In order, dispatch:

  1. resolves the tool from the registry
  2. checks RBAC (Authorizer) — reject before any work happens
  3. enforces the per-request call budget (loop/runaway protection)
  4. validates arguments against the tool's JSON Schema
  5. gates mutating tools behind an explicit confirmation token (dry-run first)
  6. invokes the handler
  7. writes an immutable audit record (who, what, when, args, outcome)

The result is always a structured ToolResult — agents get typed success/error,
never a raw exception, so a bad call is recoverable within the trajectory.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from core_engine.gateway.base import (
    Authorizer,
    RoleAuthorizer,
    ToolError,
    ToolRegistry,
    ToolSpec,
)
from core_engine.security.context import RequestContext

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolResult:
    ok: bool
    tool: str
    result: Any = None
    error: str | None = None
    # Set for mutating tools when awaiting confirmation: the agent must re-call
    # with confirm_token to actually execute.
    requires_confirmation: bool = False
    confirm_token: str | None = None
    preview: Any = None
    latency_ms: float = 0.0


class AuditSink:
    """Override to persist to the immutable audit store. Default logs structurally."""

    async def record(self, event: dict[str, Any]) -> None:
        log.info("audit %s", event)


@dataclass
class CallBudget:
    """Per-request trajectory counters — the loop/runaway guard for §3.3."""

    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total: int = 0
    max_total: int = 64

    def check_and_increment(self, spec: ToolSpec) -> None:
        if self.total >= self.max_total:
            raise ToolError(f"Request tool-call budget exhausted ({self.max_total})")
        if self.counts[spec.name] >= spec.max_calls_per_request:
            raise ToolError(
                f"Per-tool budget exhausted for '{spec.name}' "
                f"({spec.max_calls_per_request})"
            )
        self.counts[spec.name] += 1
        self.total += 1


class ToolRouter:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        authorizer: Authorizer | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self._registry = registry
        self._authorizer = authorizer or RoleAuthorizer()
        self._audit = audit or AuditSink()
        # confirm_token -> (tool_name, args) for pending mutations.
        self._pending: dict[str, tuple[str, dict[str, Any]]] = {}

    async def dispatch(
        self,
        ctx: RequestContext,
        tool_name: str,
        args: dict[str, Any],
        *,
        budget: CallBudget,
        confirm_token: str | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        outcome = "ok"
        try:
            spec = self._registry.get(tool_name)

            # 1. RBAC — fail closed, before any work or logging of args.
            if not self._authorizer.can_call(ctx, spec):
                outcome = "denied"
                return ToolResult(
                    ok=False, tool=tool_name,
                    error=f"Role check failed: requires one of {spec.required_roles}",
                )

            # 2. Budget / loop guard.
            budget.check_and_increment(spec)

            # 3. Schema validation.
            _validate(spec.input_schema, args)

            # 4. Mutation gate: first call returns a preview + token; the agent must
            #    re-call with the token to execute. Human-in-the-loop hooks here.
            if spec.mutating and confirm_token is None:
                token = _make_token(tool_name, args)
                self._pending[token] = (tool_name, args)
                outcome = "awaiting_confirmation"
                return ToolResult(
                    ok=True, tool=tool_name, requires_confirmation=True,
                    confirm_token=token,
                    preview={"action": tool_name, "args": args,
                             "note": "Re-call with confirm_token to execute."},
                )
            if spec.mutating and confirm_token is not None:
                pending = self._pending.pop(confirm_token, None)
                if pending != (tool_name, args):
                    outcome = "bad_token"
                    return ToolResult(
                        ok=False, tool=tool_name,
                        error="Invalid or mismatched confirmation token.",
                    )

            # 5. Execute.
            result = await spec.handler(ctx, args)
            return ToolResult(ok=True, tool=tool_name, result=result)

        except ToolError as e:  # expected, agent-recoverable
            outcome = "tool_error"
            return ToolResult(ok=False, tool=tool_name, error=str(e))
        except ValidationError as e:
            outcome = "invalid_args"
            return ToolResult(ok=False, tool=tool_name,
                              error=f"Argument validation failed: {e.message}")
        except Exception:  # unexpected — do not leak internals to the agent
            outcome = "internal_error"
            log.exception("tool %s crashed", tool_name)
            return ToolResult(ok=False, tool=tool_name, error="Internal tool error.")
        finally:
            latency = (time.perf_counter() - started) * 1000.0
            await self._audit.record(
                {
                    "tenant_id": ctx.tenant_id,
                    "user_id": ctx.user_id,
                    "roles": list(ctx.roles),
                    "tool": tool_name,
                    "outcome": outcome,
                    "latency_ms": round(latency, 2),
                    # Log arg KEYS only by default — values may carry sensitive data.
                    "arg_keys": sorted(args.keys()),
                }
            )


def _validate(schema: dict[str, Any], args: dict[str, Any]) -> None:
    Draft202012Validator(schema).validate(args)


def _make_token(tool_name: str, args: dict[str, Any]) -> str:
    import hashlib
    import json
    import uuid

    digest = hashlib.sha256(
        (tool_name + json.dumps(args, sort_keys=True)).encode()
    ).hexdigest()[:8]
    return f"{tool_name}:{digest}:{uuid.uuid4().hex[:8]}"
