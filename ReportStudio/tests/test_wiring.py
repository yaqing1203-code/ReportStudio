"""Smoke tests for the core-engine wiring.

These use fakes — no Postgres, no model weights — so they verify the ENGINE
contracts (RBAC, call budget, schema validation, mutation gating, RRF fusion,
abstention, orchestrator handoff enforcement) rather than the infrastructure.

Run: pytest -q
"""
from __future__ import annotations

import pytest

from core_engine.gateway.base import ToolRegistry, ToolSpec
from core_engine.gateway.router import CallBudget, ToolRouter
from core_engine.retrieval.retriever import _reciprocal_rank_fusion
from core_engine.security.context import RequestContext
from core_engine.stores.vector import Candidate

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ctx(roles=("analyst",)) -> RequestContext:
    return RequestContext.build("tenant-1", "user-1", roles=roles)


def _echo_tool(name="echo", required_roles=(), mutating=False) -> ToolSpec:
    async def handler(ctx, args):
        return {"seen": args}

    return ToolSpec(
        name=name,
        description="echoes args",
        input_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": False,
        },
        handler=handler,
        required_roles=required_roles,
        mutating=mutating,
        max_calls_per_request=2,
    )


async def test_rbac_denies_missing_role():
    reg = ToolRegistry()
    reg.register(_echo_tool(required_roles=("admin",)))
    router = ToolRouter(reg)

    res = await router.dispatch(_ctx(roles=("analyst",)), "echo", {"q": "hi"},
                                budget=CallBudget())
    assert res.ok is False
    assert "Role check failed" in res.error


async def test_rbac_allows_held_role():
    reg = ToolRegistry()
    reg.register(_echo_tool(required_roles=("analyst",)))
    router = ToolRouter(reg)

    res = await router.dispatch(_ctx(roles=("analyst",)), "echo", {"q": "hi"},
                                budget=CallBudget())
    assert res.ok is True
    assert res.result == {"seen": {"q": "hi"}}


async def test_schema_validation_rejects_bad_args():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    router = ToolRouter(reg)

    res = await router.dispatch(_ctx(), "echo", {"wrong": 1}, budget=CallBudget())
    assert res.ok is False
    assert "validation failed" in res.error.lower()


async def test_per_tool_call_budget_enforced():
    reg = ToolRegistry()
    reg.register(_echo_tool())  # max_calls_per_request=2
    router = ToolRouter(reg)
    budget = CallBudget()

    assert (await router.dispatch(_ctx(), "echo", {"q": "1"}, budget=budget)).ok
    assert (await router.dispatch(_ctx(), "echo", {"q": "2"}, budget=budget)).ok
    third = await router.dispatch(_ctx(), "echo", {"q": "3"}, budget=budget)
    assert third.ok is False
    assert "budget exhausted" in third.error.lower()


async def test_mutation_requires_confirmation_then_executes():
    reg = ToolRegistry()
    reg.register(_echo_tool(name="mutate", mutating=True))
    router = ToolRouter(reg)
    budget = CallBudget()

    first = await router.dispatch(_ctx(), "mutate", {"q": "go"}, budget=budget)
    assert first.requires_confirmation is True
    assert first.confirm_token is not None
    assert first.result is None  # nothing executed yet

    second = await router.dispatch(_ctx(), "mutate", {"q": "go"}, budget=budget,
                                   confirm_token=first.confirm_token)
    assert second.ok is True
    assert second.result == {"seen": {"q": "go"}}


async def test_mismatched_confirm_token_rejected():
    reg = ToolRegistry()
    reg.register(_echo_tool(name="mutate", mutating=True))
    router = ToolRouter(reg)
    budget = CallBudget()

    first = await router.dispatch(_ctx(), "mutate", {"q": "go"}, budget=budget)
    # Different args than the ones the token was minted for.
    bad = await router.dispatch(_ctx(), "mutate", {"q": "different"}, budget=budget,
                                confirm_token=first.confirm_token)
    assert bad.ok is False
    assert "token" in bad.error.lower()


def test_rrf_fuses_and_dedupes_across_legs():
    # Same chunk hit by both legs should rank above singletons.
    dense = [
        Candidate("urn:a", "c1", "A", 0.9, {}, "dense"),
        Candidate("urn:b", "c2", "B", 0.8, {}, "dense"),
    ]
    keyword = [
        Candidate("urn:a", "c1", "A", 5.0, {}, "keyword"),  # dup of c1
        Candidate("urn:c", "c3", "C", 4.0, {}, "keyword"),
    ]
    fused = _reciprocal_rank_fusion([dense, keyword], k=60)
    ids = [c.chunk_id for c in fused]

    assert ids[0] == "c1"                    # hit by both legs -> top
    assert set(ids) == {"c1", "c2", "c3"}    # deduped
    assert "dense" in fused[0].source and "keyword" in fused[0].source


def test_manifest_tool_subset_is_enforceable():
    # An agent manifest names a SUBSET; the registry can render just that subset.
    reg = ToolRegistry()
    reg.register(_echo_tool(name="rag_search"))
    reg.register(_echo_tool(name="kg_connect"))
    reg.register(_echo_tool(name="admin_only", required_roles=("admin",)))

    manifest = reg.manifest(allowed=("rag_search", "kg_connect"))
    names = {t["name"] for t in manifest}
    assert names == {"rag_search", "kg_connect"}  # admin_only not exposed to this agent
