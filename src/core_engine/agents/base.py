"""Multi-agent orchestration seam.

Decision locked in: supervisor (orchestrator) pattern with stateless specialist
workers, NOT a free-for-all mesh. The orchestrator owns control flow; workers are
pure functions of (input + tools + blackboard). Adding an agent = dropping a
manifest file + a handler, never rewiring the graph.

This module defines:
  - Blackboard    : the shared, replayable run state every agent reads/writes.
  - AgentManifest : the declarative contract (loaded from adapters/<v>/agents/*.yaml).
  - Agent         : the worker interface.
  - Orchestrator  : plans → routes to specialists → aggregates → terminates.

Everything an agent needs to touch the world goes through the ToolRouter, so RBAC,
audit, and call-budget enforcement are uniform (see gateway/router.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from core_engine.gateway.router import CallBudget, ToolRouter
from core_engine.security.context import RequestContext


@dataclass
class Blackboard:
    """Shared run state — the single source of truth for a trajectory.

    Workers hold NO private hidden state; everything lives here so the run is
    replayable and auditable (§3.3). Sections are namespaced by agent name.
    """

    query: str
    ctx: RequestContext
    # agent_name -> that agent's structured outputs
    sections: dict[str, Any] = field(default_factory=dict)
    # accumulated evidence with citations, shared across agents
    evidence: list[dict[str, Any]] = field(default_factory=list)
    # ordered trace of every step for audit / debugging / eval
    trace: list[dict[str, Any]] = field(default_factory=list)

    def write(self, agent: str, data: Any) -> None:
        self.sections[agent] = data
        self.trace.append({"agent": agent, "data": data})

    def add_evidence(self, items: list[dict[str, Any]]) -> None:
        self.evidence.extend(items)


@dataclass(slots=True)
class AgentManifest:
    """Declarative agent contract — the unit of modular extensibility (§3.2)."""

    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...]                 # subset of the registry this agent may use
    handoff_targets: tuple[str, ...] = ()  # who it may hand control back to
    max_tool_calls: int = 8
    model: str = "claude-opus-4-8"

    @classmethod
    def from_yaml(cls, path: Path) -> AgentManifest:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        prompt_ref = raw.get("system_prompt_ref")
        prompt = raw.get("system_prompt", "")
        if prompt_ref:  # resolve prompt file relative to the manifest
            prompt = (path.parent / prompt_ref).read_text(encoding="utf-8")
        return cls(
            name=raw["name"],
            description=raw["description"],
            system_prompt=prompt,
            tools=tuple(raw.get("tools", [])),
            handoff_targets=tuple(raw.get("handoff_targets", [])),
            max_tool_calls=int(raw.get("max_tool_calls", 8)),
            model=raw.get("model", "claude-opus-4-8"),
        )


@dataclass(slots=True)
class AgentResult:
    """Typed handoff envelope (§3.3). `next` names the target or 'DONE'."""

    status: str            # "ok" | "error" | "abstain"
    result: Any
    next: str              # target agent name, or "DONE"
    confidence: float = 1.0


class Agent(Protocol):
    manifest: AgentManifest

    async def run(
        self, board: Blackboard, router: ToolRouter, budget: CallBudget
    ) -> AgentResult: ...


class Orchestrator:
    """Supervisor. Plans the route, invokes specialists in turn, and enforces the
    global termination budget so a trajectory can never loop forever (§3.3).

    This is deliberately a thin, deterministic control loop — the intelligence
    lives in the specialist agents and the planner. Swap the routing policy without
    touching workers.
    """

    def __init__(
        self,
        agents: dict[str, Agent],
        router: ToolRouter,
        *,
        entrypoint: str,
        max_steps: int = 12,
    ) -> None:
        self._agents = agents
        self._router = router
        self._entrypoint = entrypoint
        self._max_steps = max_steps

    async def run(self, ctx: RequestContext, query: str) -> Blackboard:
        board = Blackboard(query=query, ctx=ctx)
        budget = CallBudget()  # shared across the whole trajectory
        current = self._entrypoint
        steps = 0

        while current != "DONE" and steps < self._max_steps:
            agent = self._agents.get(current)
            if agent is None:
                board.trace.append({"error": f"unknown agent '{current}'"})
                break

            result = await agent.run(board, self._router, budget)
            board.write(current, result.result)

            # Enforce the manifest's declared handoff graph — a worker cannot route
            # to an arbitrary agent, only to a declared target (or DONE).
            allowed = set(agent.manifest.handoff_targets) | {"DONE"}
            if result.next not in allowed:
                board.trace.append(
                    {"error": f"'{current}' attempted illegal handoff to "
                              f"'{result.next}' (allowed: {sorted(allowed)})"}
                )
                break
            current = result.next
            steps += 1

        if steps >= self._max_steps:
            board.trace.append({"warning": "max_steps reached — forced termination"})
        return board


def load_agents(adapter_dir: Path) -> dict[str, AgentManifest]:
    """Load every agent manifest from adapters/<vertical>/agents/*.yaml.

    Adding a specialist to a vertical is a file drop here — no engine change.
    """
    agents_dir = adapter_dir / "agents"
    if not agents_dir.exists():
        return {}
    manifests: dict[str, AgentManifest] = {}
    for path in sorted(agents_dir.glob("*.yaml")):
        m = AgentManifest.from_yaml(path)
        manifests[m.name] = m
    return manifests
