"""LLM interface for the report pipeline.

Three call sites use the LLM, all behind this one interface:
  1. claim extraction  : source text -> list of atomic factual claims
  2. entailment check   : (claim, source span) -> supported? + verbatim quote
  3. drafting           : verified claims -> section prose (NO new facts)

Design rule that makes 'zero-hallucination' meaningful: the LLM is NEVER the
source of truth. It proposes claims and drafts prose, but every claim must be
independently confirmed by the entailment check against a REAL quoted span
(verify.py), and drafting is constrained to only-verified claims. The LLM cannot
introduce a fact that isn't already grounded.

The FakeLLM makes the whole pipeline runnable and deterministic in tests with no
API key. The Anthropic client is a thin real implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol

from core_engine.config import get_settings
from core_engine.report.text import (
    content_terms,
    lexical_tokens,
    numbers_supported,
    numeric_tokens,
    split_sentences,
)

log = logging.getLogger(__name__)

# A bounded per-call override. Context-local so concurrent reports keep their own
# configured budget even while a composition retries a truncated review.
_completion_budget: ContextVar[int | None] = ContextVar("completion_budget", default=None)


class LLMRateLimitError(Exception):
    """Raised when the provider keeps returning 429/5xx after all retries are spent.

    Carries the last HTTP status so the pipeline can surface an actionable message
    (rate limit vs. server error) instead of a raw transport exception.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LLMResponseError(ValueError):
    """An incomplete model response is not evidence that a document has no facts."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Model response unavailable ({reason}); check output budget and thinking mode")


def completion_text(raw: object, finish_reason: str | None = None) -> str:
    if finish_reason in {"length", "max_tokens", "model_context_window_exceeded"}:
        raise LLMResponseError("output_limit")
    if finish_reason in {"content_filter", "refusal"}:
        raise LLMResponseError("provider_refusal")
    text = strip_think_tags(raw) if isinstance(raw, str) else ""
    if not text.strip():
        raise LLMResponseError("empty_final_answer")
    return text


def _status_of(exc: Exception) -> int | None:
    """Best-effort HTTP status extraction that works for BOTH transports we use:
    httpx.HTTPStatusError (OpenAI-compatible) and the anthropic SDK's APIStatusError.
    Returns None for non-HTTP errors (e.g. connection resets)."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_retryable_network_error(exc: Exception) -> bool:
    """A statusless error (no HTTP response) that is still worth retrying — transient
    connectivity: timeouts, connection resets/drops. A bare 'name not found' or SSL
    error is NOT retried. Matched by class name so we don't hard-import httpx here."""
    name = type(exc).__name__
    return name in {
        "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
        "PoolTimeout", "ReadError", "RemoteProtocolError", "TimeoutException",
        "APIConnectionError", "APITimeoutError",
    }


def _retry_after_seconds(exc: Exception) -> float | None:
    """Parse a Retry-After header (delta-seconds form) off the error's response, if
    present. Honouring the server's own hint is more polite and effective than a blind
    backoff. Only the integer-seconds form is handled; a malformed value is ignored."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class ExtractedClaim:
    text: str
    candidate_source_urls: list[str]
    # Optional structured slots (entity-attribute-value dual-track output). The
    # extraction prompt asks for them; any slot the model cannot fill stays None and
    # the claim degrades to a plain-text statement downstream.
    entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    qualifier: str | None = None
    time_scope: str | None = None
    metric: dict | None = None
    source_windows: dict[str, list[list[int]]] = field(default_factory=dict)
    source_quote: str | None = None


@dataclass(slots=True)
class EntailmentJudgement:
    supports: bool
    quote: str          # verbatim span from the source; empty if not supported
    note: str = ""


@dataclass(slots=True)
class TraceJudgement:
    """Semantic-support judgement (verify_mode='traceable').

    `supported`   — the source STATES, IMPLIES, or is CONSISTENT WITH the claim's core
                    fact (semantic/circumstantial; verbatim NOT required).
    `hallucinated`— now means the source DIRECTLY CONTRADICTS the claim (states the
                    opposite). This is the ONLY hard fail. A source that simply doesn't
                    mention the claim is neither supported nor a contradiction.
    `evidence_span` — best-effort supporting excerpt for the audit trail (advisory).
    """

    supported: bool
    hallucinated: bool
    evidence_span: str = ""
    confidence: float = 0.0
    note: str = ""


@dataclass(slots=True)
class ContradictionJudgement:
    """Conflict-only verdict between TWO claims (conflict-resolution model).

    `contradict` — the two claims make DIRECTLY INCOMPATIBLE assertions about the same
                   subject (e.g. different values for the same metric/date/entity), so
                   both cannot be true at once. Claims that merely differ in topic, or
                   that are consistent but not identical, do NOT contradict.
    `note`       — one short line explaining the incompatibility (audit trail).
    """

    contradict: bool
    note: str = ""


@dataclass(slots=True)
class KGExtraction:
    """Structured KG payload extracted from verified claims for the chart sections.

    Everything here is PROPOSED by the LLM over verified-claim text only; the builder
    (kg.py) drops any element it cannot tie back to a verified claim id, so the KG
    cannot introduce a relationship or figure the harness never cleared.

    Shapes (all lists may be empty):
      chain      : [{"name","tier","claim_ids"}]  tier in upstream|midstream|downstream
      chain_edges: [{"src","dst","label","claim_ids"}]
      market     : {"tam","sam","som","unit","cagr_pct","series":[{"year","value","claim_ids"}],
                    "claim_ids"}
      competitors: [{"name","market_share_pct","advantage","claim_ids"}]
    """

    chain: list[dict] = None  # type: ignore[assignment]
    chain_edges: list[dict] = None  # type: ignore[assignment]
    market: dict = None  # type: ignore[assignment]
    competitors: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.chain = self.chain or []
        self.chain_edges = self.chain_edges or []
        self.market = self.market or {}
        self.competitors = self.competitors or []


@dataclass(slots=True)
class InsightCandidate:
    """A high-value finding the model flags as worth a deep-dive analysis.

    Proposed by the LLM over VERIFIED-claim text only. `claim_ids` must reference
    real verified claims; the pipeline drops any candidate that doesn't, so a
    deep-dive can never be built on unverified ground.
    """

    title: str
    kind: str                          # "bottleneck" | "policy_impact" | "competitor_edge" | other
    section_id: str                    # which mandated section it belongs to
    claim_ids: list[str] = None  # type: ignore[assignment]
    rationale: str = ""

    def __post_init__(self) -> None:
        self.claim_ids = self.claim_ids or []


class LLM(Protocol):
    async def complete(self, system: str, user: str) -> str: ...
    async def extract_claims(self, source_url: str, text: str) -> list[ExtractedClaim]: ...
    async def check_entailment(self, claim: str, source_text: str) -> EntailmentJudgement: ...
    async def check_traceability(self, claim: str, source_text: str) -> TraceJudgement: ...
    async def check_contradiction(self, claim_a: str, claim_b: str) -> ContradictionJudgement: ...
    async def draft_section(
        self, heading: str, claims: list[str], instructions: str = ""
    ) -> str: ...
    async def propose_outline(self, topic: str, claims: list[str]) -> list[str]: ...
    async def extract_kg(
        self, topic: str, verified_claims: list[tuple[str, str]]
    ) -> KGExtraction: ...
    async def identify_insights(
        self, topic: str, verified_claims: list[tuple[str, str]]
    ) -> list[InsightCandidate]: ...
    async def draft_deep_dive(
        self, title: str, kind: str, claims: list[str]
    ) -> list[str]: ...


# --------------------------------------------------------------------------
# Fake LLM: deterministic, rule-based. Enough to exercise the full pipeline.
# --------------------------------------------------------------------------
class FakeLLM:
    """Rule-based stand-in. Extraction splits sentences; entailment is a literal
    substring/overlap check (so it behaves like a strict grounding oracle);
    drafting concatenates claim text. No creativity — which is exactly what we
    want when testing the GATE rather than the prose quality.

    Test extension points (offline determinism):
      - set_structured_claims(url, items): pin the exact ExtractedClaim list (with
        structured slots) returned for a source URL, bypassing the rule-based extractor.
      - set_complete(marker, response): pin the raw string returned by complete() when
        the system prompt contains `marker` (e.g. inject an "inconsistent" verdict or a
        fixed deviation score for the alignment layer).
    """

    def __init__(self) -> None:
        # url -> list of ExtractedClaim | dict payload (see set_structured_claims).
        self._structured: dict[str, list] = {}
        # (marker substring of the system prompt) -> raw response for complete().
        self._complete_overrides: dict[str, str] = {}

    def set_structured_claims(self, source_url: str, items: list) -> None:
        """Pin the structured extraction output for one source URL. Items may be
        ExtractedClaim objects or dicts with text/entity/attribute/value/qualifier/
        time_scope keys (missing slots default to None)."""
        self._structured[source_url] = list(items)

    def set_complete(self, marker: str, response: str) -> None:
        """Pin complete()'s raw response whenever the system prompt contains `marker`."""
        self._complete_overrides[marker] = response

    async def complete(self, system: str, user: str) -> str:
        """Deterministic offline completion.

        Test-injected overrides win first. Then the intent-classifier prompt and the
        alignment-layer prompts (entity normalization, consistency, deviation, source
        note, query generation) get safe rule-based defaults so the aligned verify
        strategy runs fully offline. For a plain chat prompt, echoes a canned reply.
        """
        for marker, response in self._complete_overrides.items():
            if marker in system:
                return response
        sys_l = system.lower()
        if "research planner" in sys_l:
            return '{"questions":[]}'
        if "research coverage assessor" in sys_l:
            return '{"assessments":[]}'
        if "evidence-based analyst" in sys_l:
            return '{"findings":[]}'  # Extractive baseline never pretends to reason.
        if "analysis critic" in sys_l:
            return '{"acceptable":false,"unsupported_assertions":["No semantic reviewer in offline baseline"]}'
        if "intent classifier" in sys_l or "report_generation" in sys_l:
            u = user.lower()
            report_kw = ("report", "analyze", "analysis", "research", "market",
                         "industry", "write a", "generate", "supply chain",
                         "landscape", "overview of", "study of")
            is_report = any(k in u for k in report_kw)
            intent = "report_generation" if is_report else "general_chat"
            return json.dumps({
                "intent": intent,
                "confidence": "high",
                "reasoning": f"[fake] matched={is_report}",
            })
        # Alignment-layer defaults (verify_strategy="aligned"). Fail-open and neutral:
        # identity alias map, consistent, zero deviation, canned note, no queries.
        if "entity alias normalizer" in sys_l:
            try:
                payload = json.loads(user)
                names = payload.get("entities", []) if isinstance(payload, dict) else []
            except Exception:
                names = []
            return json.dumps({"mapping": {str(n): str(n) for n in names}})
        if "internal consistency reviewer" in sys_l:
            return json.dumps({"consistent": True, "note": "[fake] consistent"})
        if "deviation assessor" in sys_l:
            return json.dumps({"deviation": 0.0, "note": "[fake] within baseline"})
        if "source note writer" in sys_l:
            return "[fake] single-source note: provenance recorded; publication recency is not inferred from fetch time."
        if "verification query generator" in sys_l:
            return "[]"
        # Plain chat fallback.
        return ("This is the offline demo assistant. Configure an Anthropic or "
                "OpenAI-compatible provider in Settings for real answers.")

    async def extract_claims(self, source_url: str, text: str) -> list[ExtractedClaim]:
        if source_url in self._structured:
            out: list[ExtractedClaim] = []
            for it in self._structured[source_url]:
                if isinstance(it, ExtractedClaim):
                    if not it.candidate_source_urls:
                        it.candidate_source_urls.append(source_url)
                    out.append(it)
                elif isinstance(it, dict) and it.get("text"):
                    out.append(ExtractedClaim(
                        text=str(it["text"]),
                        candidate_source_urls=[source_url],
                        entity=_slot(it.get("entity")),
                        attribute=_slot(it.get("attribute")),
                        value=_slot(it.get("value")),
                        qualifier=_slot(it.get("qualifier")),
                        time_scope=_slot(it.get("time_scope")),
                        metric=it.get("metric") if isinstance(it.get("metric"), dict) else None,
                    ))
            return out
        sentences = _split_sentences(text)
        claims: list[ExtractedClaim] = []
        for sent in sentences:
            # Only treat sentences with a factual shape (a number, %, date, or
            # a 'is/are/was/were/reported') as claims — keeps noise down.
            if _looks_factual(sent):
                slots = _extract_slots(sent)
                claims.append(ExtractedClaim(
                    text=sent, candidate_source_urls=[source_url], **slots))
        return claims

    async def check_entailment(self, claim: str, source_text: str) -> EntailmentJudgement:
        # Strict: the claim is supported only if a near-verbatim span exists in the
        # source. We find the best-overlap sentence and require high token overlap.
        best, score = _best_overlap(claim, source_text)
        if score >= 0.85 and best and numbers_supported(claim, best):
            return EntailmentJudgement(supports=True, quote=best, note=f"overlap={score:.2f}")
        return EntailmentJudgement(supports=False, quote="", note=f"overlap={score:.2f}")

    async def check_traceability(self, claim: str, source_text: str) -> TraceJudgement:
        # Semantic stand-in: accept at a LOWER overlap bar (the source need only be
        # consistent with the claim's gist), and NEVER flag contradiction from overlap
        # alone — a keyword heuristic can't detect a real contradiction, and treating
        # low overlap as contradiction is what wrongly dropped claims. 'hallucinated'
        # (contradiction) stays False here; low overlap just means 'not supported'.
        best, score = _best_overlap(claim, source_text)
        if score >= 0.35:
            return TraceJudgement(supported=True, hallucinated=False,
                                  evidence_span=best, confidence=score,
                                  note=f"overlap={score:.2f}")
        return TraceJudgement(supported=False, hallucinated=False,
                              evidence_span="", confidence=score,
                              note=f"overlap={score:.2f}")

    async def check_contradiction(self, claim_a: str, claim_b: str) -> ContradictionJudgement:
        # Deterministic offline heuristic: two claims contradict if they share enough
        # subject terms (same topic) BUT carry different numeric/date values — the shape
        # of "X was 12%" vs "X was 20%". No overlap, or identical numbers, => no conflict.
        # A keyword stand-in can't reason about semantics, so this only fires on the clear
        # same-subject-different-number case, mirroring how the real model is prompted.
        a_terms, b_terms = _content_terms(claim_a), _content_terms(claim_b)
        if not a_terms or not b_terms:
            return ContradictionJudgement(contradict=False, note="[fake] empty")
        overlap = len(a_terms & b_terms) / min(len(a_terms), len(b_terms))
        a_nums, b_nums = _numbers(claim_a), _numbers(claim_b)
        if overlap >= 0.5 and a_nums and b_nums and a_nums != b_nums:
            return ContradictionJudgement(
                contradict=True,
                note=f"[fake] same-subject (overlap={overlap:.2f}) diverging values "
                     f"{sorted(a_nums)} vs {sorted(b_nums)}")
        return ContradictionJudgement(contradict=False,
                                      note=f"[fake] overlap={overlap:.2f}")

    async def propose_outline(self, topic: str, claims: list[str]) -> list[str]:
        # Deterministic 3-section skeleton.
        return ["Overview", "Key Findings", "Details and Data"]

    async def draft_section(
        self, heading: str, claims: list[str], instructions: str = ""
    ) -> str:
        # Concatenate verified claims into multi-paragraph prose (two sentences per
        # paragraph, roughly). No new facts introduced. Hedging markers such as
        # "[UNVERIFIED - single source]" are part of the claim text and are preserved
        # verbatim in the output so tests can assert on them. `instructions` is a
        # real-LLM-only prompt refinement; the fake stays purely deterministic.
        if not claims:
            return ""
        if all(re.match(r"^\[\[[A-Za-z0-9_, -]+\]\]", claim) for claim in claims):
            sentences = []
            for i, claim in enumerate(claims):
                marker = re.match(r"^\[\[([A-Za-z0-9_, -]+)\]\]\s*(.*)$", claim, re.DOTALL)
                for text in split_sentences(marker[2]):
                    sentences.append({"text": text, "claim_ids": marker[1].split(","),
                                      "paragraph_index": i // 2})
            return json.dumps({"sentences": sentences}, ensure_ascii=False)
        paras = []
        for i in range(0, len(claims), 2):
            chunk = claims[i : i + 2]
            paras.append(" ".join(c.rstrip(".") + "." for c in chunk))
        return "\n\n".join(paras)

    async def extract_kg(
        self, topic: str, verified_claims: list[tuple[str, str]]
    ) -> KGExtraction:
        # Deterministic, rule-based extraction from verified-claim text. Recognizes
        # simple structured markers so tests/fixtures can drive the chart sections
        # without a real LLM:
        #   "UPSTREAM: <name>" / "MIDSTREAM: <name>" / "DOWNSTREAM: <name>"
        #   "<A> supplies <B>"
        #   "TAM is 120 USD bn" / "SAM is 40 USD bn" / "SOM is 8 USD bn" / "CAGR is 12%"
        #   "In 2021 the market was 90 USD bn"
        #   "COMPETITOR: <name> share 25% advantage <text>"
        chain: list[dict] = []
        chain_edges: list[dict] = []
        market: dict = {"series": [], "claim_ids": []}
        competitors: list[dict] = []

        for cid, text in verified_claims:
            for tier in ("upstream", "midstream", "downstream"):
                m = re.search(rf"\b{tier}\s*[:\-]\s*([A-Za-z0-9 &/]+)", text, re.IGNORECASE)
                if m:
                    chain.append({"name": m.group(1).strip(), "tier": tier,
                                  "claim_ids": [cid]})
            m = re.search(r"([A-Za-z0-9 &/]+?)\s+supplies\s+([A-Za-z0-9 &/]+)", text, re.IGNORECASE)
            if m:
                chain_edges.append({"src": m.group(1).strip(), "dst": m.group(2).strip(),
                                    "label": "supplies", "claim_ids": [cid]})
            for key in ("tam", "sam", "som"):
                m = re.search(rf"\b{key}\b\D*([\d.]+)\s*([A-Za-z$ ]*bn|[A-Za-z$ ]*billion)?",
                              text, re.IGNORECASE)
                if m:
                    market[key] = m.group(1)
                    if m.group(2):
                        market["unit"] = m.group(2).strip()
                    market.setdefault("claim_ids", []).append(cid)
            m = re.search(r"\bcagr\b\D*([\d.]+)\s*%", text, re.IGNORECASE)
            if m:
                market["cagr_pct"] = m.group(1)
                market.setdefault("claim_ids", []).append(cid)
            m = re.search(r"\b(19|20)\d{2}\b.*?([\d.]+)\s*(USD\s*bn|billion|bn)", text, re.IGNORECASE)
            if m:
                year = int(re.search(r"\b((?:19|20)\d{2})\b", text).group(1))
                market["series"].append({"year": year, "value": m.group(2), "unit": m.group(3),
                                         "claim_ids": [cid]})
            m = re.search(r"competitor\s*[:\-]\s*([A-Za-z0-9 &/.]+?)\s+share\s+([\d.]+)\s*%"
                          r"(?:\s+advantage\s+(.+))?", text, re.IGNORECASE)
            if m:
                competitors.append({
                    "name": m.group(1).strip(),
                    "market_share_pct": m.group(2),
                    "advantage": (m.group(3) or "").strip(),
                    "claim_ids": [cid],
                })
        return KGExtraction(chain=chain, chain_edges=chain_edges,
                            market=market, competitors=competitors)

    async def identify_insights(
        self, topic: str, verified_claims: list[tuple[str, str]]
    ) -> list[InsightCandidate]:
        # Rule-based high-value-finding detection. Expanded to cover all five mandated
        # sections so the fallback per-section deep-dive coverage still leaves a
        # deterministic signal for tests. Recognizes bottlenecks, policy impacts,
        # competitor edges, market drivers, chain structure, and industry context.
        signals = {
            "bottleneck": (
                ["bottleneck", "shortage", "constraint", "capacity", "supply risk",
                 "dependency", "chokepoint", "lead time", "limited"],
                "industry_chain", "Supply Bottleneck",
            ),
            "policy_impact": (
                ["policy", "regulation", "subsidy", "tariff", "directive", "mandate",
                 "ban", "incentive", "sanction", "law"],
                "policy_analysis", "Core Policy Impact",
            ),
            "competitor_edge": (
                ["market share", "advantage", "leader", "dominant", "moat", "patent",
                 "competitive", "leading player"],
                "competitive_landscape", "Key Competitor Advantage",
            ),
            "market_driver": (
                ["growth", "CAGR", "forecast", "demand", "adoption", "expansion", "TAM",
                 "revenue", "market size"],
                "market_size", "Market Growth Driver",
            ),
            "structure": (
                ["upstream", "midstream", "downstream", "supply chain", "tier", "supplier",
                 "value chain", "integration"],
                "industry_chain", "Industry Structure",
            ),
            "overview": (
                ["industry", "sector", "market", "technology", "application"],
                "industry_overview", "Industry Context",
            ),
        }
        out: list[InsightCandidate] = []
        for kind, (kws, section_id, title) in signals.items():
            matched = [cid for cid, text in verified_claims
                       if any(kw in text.lower() for kw in kws)]
            if matched:
                out.append(InsightCandidate(
                    title=title, kind=kind, section_id=section_id,
                    claim_ids=matched, rationale=f"{len(matched)} verified claim(s) signal {kind}",
                ))
        return out

    async def draft_deep_dive(
        self, title: str, kind: str, claims: list[str]
    ) -> list[str]:
        # Deterministic multi-paragraph analysis. No new facts: paragraph 1 states the
        # finding from the claims; paragraph 2 draws implications by connecting them.
        if not claims:
            return []
        if all(claim.startswith("[[") for claim in claims):
            return [await self.draft_section(title, claims)]
        facts = " ".join(c.rstrip(".") + "." for c in claims)
        p1 = f"{title}: {facts}"
        p2 = ("Taken together, these verified data points indicate a material factor "
              f"for the topic. {facts} The convergence of these findings across "
              "independent authoritative sources underlines the significance of this "
              "issue for stakeholders.")
        return [p1, p2]


# --------------------------------------------------------------------------
# Real Anthropic client (thin). Imports deferred.
# --------------------------------------------------------------------------
class AnthropicLLM:
    # ONE global limiter per LLM instance. The pipeline holds a single shared instance
    # (ReportPipeline._llm), so this semaphore caps TOTAL concurrent requests to the
    # provider across EVERY phase — extraction, verification, KG, drafting — at once.
    # Per-phase semaphores still shape each stage, but this is the hard ceiling that
    # actually keeps us under the provider's rate limit. Created lazily so its size can
    # follow the loaded settings and so it binds to the running event loop.
    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        self._key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._sem: asyncio.Semaphore | None = None
        self._pace_lock: asyncio.Lock | None = None
        self._next_request_at = 0.0

    async def _pace(self) -> None:
        interval = get_settings().llm_min_interval_s
        if interval <= 0:
            return
        if self._pace_lock is None:
            self._pace_lock = asyncio.Lock()
        async with self._pace_lock:
            delay = self._next_request_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_request_at = time.monotonic() + interval

    def _gate(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(max(1, get_settings().llm_max_concurrency))
        return self._sem

    def _client(self):
        from anthropic import AsyncAnthropic

        # Hard timeout so a stalled request can never hang the pipeline (this was the
        # cause of the 'stuck on verifying' freeze — the SDK default is effectively
        # unbounded on a stalled connection).
        return AsyncAnthropic(api_key=self._key, timeout=get_settings().llm_timeout_s)

    async def _transport(self, system: str, user: str) -> str:
        """Provider-specific wire call — the ONLY method a subclass overrides. Must not
        implement concurrency/retry policy; that lives in `_complete` so every provider
        shares one rate-limit strategy."""
        client = self._client()
        msg = await client.messages.create(
            model=self._model,
            max_tokens=_completion_budget.get() or self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(block.text for block in msg.content if block.type == "text")
        # Strip model reasoning tags at the boundary so nothing downstream sees them.
        return completion_text(raw, getattr(msg, "stop_reason", None))

    async def _complete(self, system: str, user: str) -> str:
        """Rate-limited, retrying entry point that EVERY report method funnels through.

        Policy (single chokepoint for all phases):
          - global semaphore caps concurrent in-flight requests (avoids bursting 429),
          - 429 and 5xx are retried with exponential backoff + full jitter, honouring a
            server Retry-After hint when present,
          - auth/other 4xx are permanent -> fail fast (retrying won't help),
          - after the retry budget is spent we raise LLMRateLimitError with the status,
            so the pipeline reports 'rate limited' rather than dying on a raw 429.
        """
        s = get_settings()
        attempts = max(1, s.llm_max_retries)
        last_status: int | None = None
        async with self._gate():
            for attempt in range(1, attempts + 1):
                try:
                    await self._pace()
                    return await self._transport(system, user)
                except Exception as e:
                    status = _status_of(e)
                    transient = status in (429, 500, 502, 503, 504) or (
                        status is None and _is_retryable_network_error(e))
                    if not transient:
                        raise  # auth / bad-request / non-retryable — surface immediately
                    last_status = status
                    if attempt >= attempts:
                        break
                    # Prefer the server's Retry-After; otherwise exp backoff + FULL jitter.
                    hinted = _retry_after_seconds(e)
                    if hinted is not None:
                        delay = min(hinted, s.llm_retry_max_s)
                    else:
                        ceil = min(s.llm_retry_base_s * (2 ** (attempt - 1)),
                                   s.llm_retry_max_s)
                        delay = random.uniform(0, ceil)   # full jitter
                    log.warning("llm: status=%s on attempt %d/%d — backing off %.2fs",
                                status, attempt, attempts, delay)
                    await asyncio.sleep(delay)
            raise LLMRateLimitError(
                f"LLM provider rate-limited/unavailable after {attempts} attempt(s) "
                f"(last status {last_status}). Lower CE_LLM_MAX_CONCURRENCY or check "
                f"your provider quota.", status=last_status)

    async def complete(self, system: str, user: str) -> str:
        """Public raw-completion entry point (part of the LLM protocol). Used by the
        chat/intent layer; report methods call _complete directly."""
        return await self._complete(system, user)

    async def complete_with_budget(self, system: str, user: str, *, max_tokens: int) -> str:
        token = _completion_budget.set(max_tokens)
        try:
            return await self._complete(system, user)
        finally:
            _completion_budget.reset(token)

    async def extract_research_claims(self, source_url: str, text: str, questions: list[str], *, max_claims=None) -> list[ExtractedClaim]:
        return await self.extract_claims(source_url, text, research_questions=questions, max_claims=max_claims)

    async def extract_claims(self, source_url: str, text: str, *, research_questions=None, max_claims=None) -> list[ExtractedClaim]:
        system = (
            "Extract ATOMIC, checkable factual claims from the text. "
            f"Return at most {get_settings().extraction_max_claims_per_source} high-value claims; "
            "prioritize substantive observations over background and repeated wording. Each claim must "
            "Keep experimental conditions, chemistry, full-cell versus half-cell scope, and limitations "
            "with the observation. Include substantive mechanisms and failure modes as well as numbers. "
            "Never infer results from a bibliography, paper title, author list or cited reference alone. "
            "be a single verifiable statement, self-contained, no opinions. Return a "
            "JSON array only. Each item is an object: {\"text\": <the claim>, "
            "\"entity\": <the subject the claim is about, e.g. a company or market, or "
            "null>, \"attribute\": <the measured property, e.g. \"market_share\" or "
            "\"TAM\", or null>, \"value\": <the claimed value as stated, e.g. \"32%\", "
            "or null>, \"qualifier\": <scope/calibre qualifiers such as \"GAAP\" or "
            "\"domestic only\", or null>, \"time_scope\": <the period the claim refers "
            "to, e.g. \"FY2023\", or null>}. Omit a slot (or set it null) whenever the "
            "text does not state it. For numeric observations, also return an optional "
            '\"metric\": {\"name\":\"revenue|net_profit|market_size|market_share|gdp|growth_rate|cpi|installed_capacity|capacity_addition|tam|sam|som|cagr|ionic_conductivity|electronic_conductivity|conductivity|gravimetric_energy_density|volumetric_energy_density|specific_capacity|areal_capacity|current_density|critical_current_density|cycle_life|capacity_retention|coulombic_efficiency|interfacial_impedance|impedance\",'
            '\"value\":\"decimal string, never binary float\",\"unit\":\"explicit source unit\",'
            '\"entity\":str|null,\"period\":str|null,\"geography\":str|null,\"scope\":str|null,'
            '\"price_basis\":str|null,\"rate_basis\":str|null,\"value_kind\":str|null,'
            '\"boundary\":str|null,\"conditions\":{\"temperature\":str|null,\"pressure\":str|null,'
            '\"test_method\":str|null,\"sample_form\":str|null,\"cell_type\":str|null,'
            '\"mass_basis\":str|null,\"volume_basis\":str|null,\"current_density\":str|null,'
            '\"c_rate\":str|null,\"cycle_count\":str|null,\"retention_threshold\":str|null}}. '
            "Every metric field must be supported by the same quoted observation; "
            "preserve original metadata wording. Missing metadata MUST be null, never guessed. "
            "Separate multiple metrics into separate atomic claims. Keep full metric context in claim text."
            "For scientific tables, bind each value to its exact row, specimen, column unit, caption and applicable footnotes. "
            "Keep electronic and ionic conductivity distinct, and active-material, electrode, half-cell, full-cell and pack boundaries separate. "
            "Do not turn ranges, error bars, inequalities or approximate numbers into exact scalar measurements. "
            "Only measured and explicitly specified conditions belong to an observation; missing temperature, pressure, rate, "
            "loading or cell geometry remains null. A material property cannot establish complete-cell performance."
        )
        if research_questions:
            system += (" This is a question-directed full-text reading task. Prioritize passages addressing the supplied "
                "research questions and their missing evidence over generic advantages, background or bibliometrics. "
                "Questions are retrieval goals only, NOT evidence. Extract only what SOURCE states. Include concrete "
                "route distinctions, failure mechanisms, manufacturing constraints and testing comparability when present. "
                "Preserve experimental conditions and negative findings; do not fill any missing requirement from memory.")
            system += (" Also return a quote string for each claim: an EXACT verbatim substring of SOURCE, preserving "
                "line breaks, sufficient to support the observation. Do not translate or normalize the quote. "
                "The quote will be checked against SOURCE and does not itself establish semantic support.")
            if max_claims is not None:
                system += (f" For this focused deficit return at most {max_claims} SHORT atomic claims. "
                    "Only extract observations directly addressing the missing requirement. Return [] if absent. "
                    "Split independent facts into separate items; do not combine multiple claims into one sentence.")
            user = json.dumps({"research_questions": research_questions,
                               "SOURCE": text[:get_settings().llm_context_chars]}, ensure_ascii=False)
        else:
            user = text[:get_settings().llm_context_chars]
        raw = await self._complete(system, user)
        items = _safe_json_list(raw)
        out: list[ExtractedClaim] = []
        for it in items[:get_settings().extraction_max_claims_per_source]:
            # Back-compat: a bare string item is a claim with no structured slots.
            if isinstance(it, str):
                out.append(ExtractedClaim(text=it, candidate_source_urls=[source_url]))
                continue
            if not isinstance(it, dict) or not it.get("text"):
                continue
            out.append(ExtractedClaim(
                text=str(it["text"]),
                candidate_source_urls=[source_url],
                entity=_slot(it.get("entity")),
                attribute=_slot(it.get("attribute")),
                value=_slot(it.get("value")),
                qualifier=_slot(it.get("qualifier")),
                time_scope=_slot(it.get("time_scope")),
                metric=it.get("metric") if isinstance(it.get("metric"), dict) else None,
                source_quote=it["quote"] if isinstance(it.get("quote"), str) and it["quote"] and it["quote"] in text else None,
            ))
        return out

    async def check_entailment(self, claim: str, source_text: str) -> EntailmentJudgement:
        system = (
            "You are a strict fact-checker. Decide if the SOURCE explicitly supports "
            "the CLAIM. You MUST quote the exact verbatim span from the source that "
            "supports it. If no such span exists, the claim is NOT supported. Respond "
            'ONLY as JSON: {"supports": bool, "quote": "<verbatim span or empty>"}.'
        )
        user = f"CLAIM:\n{claim}\n\nSOURCE:\n{source_text[:get_settings().llm_context_chars]}"
        raw = await self._complete(system, user)
        obj = _safe_json_obj(raw)
        supports = bool(obj.get("supports", False))
        quote = str(obj.get("quote", "") or "")
        # Defensive: a claimed 'support' whose quote is NOT actually in the source
        # is downgraded. The model does not get to invent its own evidence.
        if supports and (not quote.strip() or quote.strip() not in source_text
                         or not numbers_supported(claim, quote)):
            supports = False
            quote = ""
        return EntailmentJudgement(supports=supports, quote=quote)

    async def check_traceability(self, claim: str, source_text: str) -> TraceJudgement:
        # Relaxed contract: strict TRACEABILITY + factual accuracy, but NO mandatory
        # verbatim span. The model must judge whether the source fully supports the
        # (possibly synthesized/paraphrased) claim, and MUST flag any fabricated fact.
        system = (
            "You are a fact-checker assessing SEMANTIC support, not verbatim matching. "
            "Read the SOURCE and judge the CLAIM's underlying meaning.\n"
            "- Set supported=true only if the source supports ALL material details "
            "of the claim, including number, sign, unit, entity, scope and period. "
            "Paraphrase is allowed, mere consistency or topic overlap is insufficient.\n"
            "- Set hallucinated=true ONLY if the source DIRECTLY CONTRADICTS the claim "
            "(states the opposite). If the source simply doesn't mention the claim, "
            "that is NOT a contradiction: supported=false, hallucinated=false.\n"
            "- confidence in [0,1] reflects how strongly the source supports the gist.\n"
            "Provide a short supporting/most-relevant span. Respond ONLY as JSON: "
            '{"supported": bool, "hallucinated": bool, "evidence_span": "<span>", '
            '"confidence": <float>}.'
        )
        user = f"CLAIM:\n{claim}\n\nSOURCE:\n{source_text[:get_settings().llm_context_chars]}"
        raw = await self._complete(system, user)
        obj = _safe_json_obj(raw)
        span = str(obj.get("evidence_span", "") or "").strip()
        return TraceJudgement(
            supported=bool(obj.get("supported", False)) and bool(span)
                      and span in source_text and numbers_supported(claim, span),
            hallucinated=bool(obj.get("hallucinated", False)),
            evidence_span=span,
            confidence=float(obj.get("confidence", 0.0) or 0.0),
        )

    async def check_contradiction(self, claim_a: str, claim_b: str) -> ContradictionJudgement:
        # Conflict-only model: we no longer verify a claim against sources. The ONLY LLM
        # judgement in verify is whether two same-subject claims are mutually exclusive,
        # so the harness can drop/keep among conflicting claims. Keep the prompt tight —
        # this fires only on candidate pairs the cheap prefilter already flagged as similar.
        system = (
            "You compare TWO factual claims and decide if they DIRECTLY CONTRADICT — "
            "i.e. they describe the SAME subject/metric/event but assert MUTUALLY "
            "EXCLUSIVE values, so both cannot be true at once (e.g. 'revenue was $2B in "
            "2023' vs 'revenue was $5B in 2023').\n"
            "- contradict=true ONLY for a genuine same-subject incompatibility.\n"
            "- contradict=false if they discuss different subjects, are consistent, are "
            "about different time periods/entities, or one merely adds detail.\n"
            "Respond ONLY as JSON: {\"contradict\": bool, \"note\": \"<one short line>\"}."
        )
        user = f"CLAIM A:\n{claim_a}\n\nCLAIM B:\n{claim_b}"
        raw = await self._complete(system, user)
        obj = _safe_json_obj(raw)
        return ContradictionJudgement(
            contradict=bool(obj.get("contradict", False)),
            note=str(obj.get("note", "") or ""),
        )

    async def propose_outline(self, topic: str, claims: list[str]) -> list[str]:
        system = (
            "Propose 3-6 report section headings for the topic, ordered logically. "
            "Return a JSON array of strings only."
        )
        raw = await self._complete(system, f"TOPIC: {topic}\nCLAIMS:\n" + "\n".join(claims[:50]))
        out = [c for c in _safe_json_list(raw) if isinstance(c, str)]
        return out or ["Overview", "Key Findings", "Details and Data"]

    async def draft_section(
        self, heading: str, claims: list[str], instructions: str = ""
    ) -> str:
        s = get_settings()
        system = (
            "You are a senior industry analyst writing a section of a formal research "
            "report. Write a substantive, well-structured analysis of at least "
            f"{s.section_min_paragraphs} paragraphs (~{s.section_target_words} words) "
            "for the section below.\n"
            "GROUNDING RULES (strict): use ONLY the provided verified claims. You MAY "
            "organize, contextualize, compare, and explain the significance of these "
            "facts and how they relate to each other. You MUST NOT introduce any fact, "
            "figure, name, or date not present in the claims, MUST NOT speculate, and "
            "MUST NOT pad with filler to reach the length — if the verified material is "
            "limited, write a shorter but accurate section. Separate paragraphs with a "
            "blank line. Preserve the language requested below.\n"
            "CITATION CONTRACT: each input claim begins with [[claim_id]]. Return ONLY "
            'JSON: {"sentences":[{"text":"one complete sentence",'
            '"claim_ids":["c1"],"paragraph_index":0}]}. '
            "Cite only supplied claim IDs that support that sentence. Preserve signs, "
            "units, scope and periods. Do not add facts or source IDs.\n"
            "HEDGING RULES (strict): any claim line marked [UNVERIFIED - single source] "
            "rests on ONE source and has NOT been independently verified. When you draw "
            "on it you MUST hedge explicitly (e.g. \"According to a single source, "
            "...\", \"... has not been independently verified\") and you MUST NOT state "
            "it as established fact. Never use categorical phrasing such as \"The data "
            "shows\" or \"It is proven\" for such material."
        )
        if instructions.strip():
            system += f"\nSECTION-SPECIFIC REQUIREMENTS: {instructions.strip()}"
        raw = await self._complete(system, f"SECTION: {heading}\nVERIFIED CLAIMS:\n"
                                   + "\n".join(f"- {c}" for c in claims))
        return raw.strip()

    async def extract_kg(
        self, topic: str, verified_claims: list[tuple[str, str]]
    ) -> KGExtraction:
        # Structured KG extraction over VERIFIED-claim text only. Each claim is
        # passed as "[id] text" so the model can attach provenance claim_ids to
        # every element; the builder (kg.py) drops anything whose claim_ids do not
        # reference a real verified claim, so the model cannot smuggle in a fact.
        system = (
            "You extract a structured industry knowledge graph from a list of "
            "already-VERIFIED factual claims (each prefixed with its [claim_id]). "
            "Use ONLY information present in these claims — do NOT add outside "
            "knowledge. For every element, list the claim_id(s) it came from.\n"
            "Return ONLY JSON with this shape:\n"
            '{"chain":[{"name":str,"tier":"upstream|midstream|downstream",'
            '"claim_ids":[str]}],'
            '"chain_edges":[{"src":str,"dst":str,"label":str,"claim_ids":[str]}],'
            '"market":{"tam":decimal_string|null,"sam":decimal_string|null,"som":decimal_string|null,"unit":str,'
            '"cagr_pct":decimal_string|null,"series":[{"year":int,"value":decimal_string,"unit":str,"claim_ids":[str]}],'
            '"claim_ids":[str]},'
            '"competitors":[{"name":str,"market_share_pct":decimal_string|null,"advantage":str,'
            '"claim_ids":[str]}]}'
            "\nDecimal values must be JSON strings, preserving source precision. Never assume a currency, "
            "scope or period, or compute growth/share yourself. Omit unsupported values."
        )
        body = "\n".join(f"[{cid}] {text}" for cid, text in verified_claims)
        raw = await self._complete(system, f"TOPIC: {topic}\nCLAIMS:\n{body[:14000]}")
        obj = _safe_json_obj(raw)
        return KGExtraction(
            chain=obj.get("chain") or [],
            chain_edges=obj.get("chain_edges") or [],
            market=obj.get("market") or {},
            competitors=obj.get("competitors") or [],
        )

    async def identify_insights(
        self, topic: str, verified_claims: list[tuple[str, str]]
    ) -> list[InsightCandidate]:
        # Ask the model to nominate the highest-value findings worth a deep-dive,
        # tying each to its verified claim_ids and one of the mandated sections.
        system = (
            "You are a senior industry analyst. From the list of already-VERIFIED "
            "claims (each prefixed with its [claim_id]), identify the highest-value "
            "findings that merit a comprehensive multi-paragraph analysis rather than "
            "a one-line summary. Cover the full report where the evidence supports it: "
            "industry structure and trends, core policy/regulatory impacts, "
            "supply-chain dependencies and bottlenecks, market sizing and growth "
            "drivers, and key competitor advantages. Aim for one strong insight per "
            "report section wherever the claims allow it. Use ONLY the provided claims; "
            "do NOT add outside facts. For each insight list the claim_id(s) it rests "
            "on and the section it belongs to (one of: industry_overview, "
            "policy_analysis, industry_chain, market_size, competitive_landscape).\n"
            "Return ONLY JSON: {\"insights\":[{\"title\":str,\"kind\":"
            "\"bottleneck|policy_impact|competitor_edge|structure|market_driver|other\","
            "\"section_id\":str,\"claim_ids\":[str],\"rationale\":str}]}"
        )
        body = "\n".join(f"[{cid}] {text}" for cid, text in verified_claims)
        raw = await self._complete(system, f"TOPIC: {topic}\nCLAIMS:\n{body[:14000]}")
        obj = _safe_json_obj(raw)
        out: list[InsightCandidate] = []
        for it in obj.get("insights", []) or []:
            if not isinstance(it, dict):
                continue
            out.append(InsightCandidate(
                title=str(it.get("title", "")).strip() or "Key Insight",
                kind=str(it.get("kind", "other")).strip() or "other",
                section_id=str(it.get("section_id", "")).strip(),
                claim_ids=[str(x) for x in (it.get("claim_ids") or [])],
                rationale=str(it.get("rationale", "")).strip(),
            ))
        return out

    async def draft_deep_dive(
        self, title: str, kind: str, claims: list[str]
    ) -> list[str]:
        # Comprehensive, multi-paragraph analysis grounded ONLY in the given verified
        # claims. The model may INTERPRET and CONNECT facts but must not introduce new
        # ones. Returns a list of paragraphs (split on blank lines).
        s = get_settings()
        system = (
            "You are a senior industry analyst writing an in-depth analytical insight "
            "for a formal report. Using ONLY the provided verified claims, write a "
            f"comprehensive analysis of at least {s.deep_dive_min_paragraphs} "
            f"paragraphs (~{s.deep_dive_target_words} words) on the finding. You MAY "
            "interpret, contextualize, and connect the verified facts, and discuss "
            "implications — but you MUST NOT introduce any fact not present in the "
            "claims, and MUST NOT speculate beyond what they support. Return ONLY "
            'JSON: {"sentences":[{"text":"one sentence","claim_ids":["c1"],'
            '"paragraph_index":0}]}. Claim IDs come from the [[id]] prefixes. '
            f"Write in {'Chinese' if s.report_locale == 'zh' else 'English'}."
        )
        user = (f"INSIGHT TITLE: {title}\nKIND: {kind}\nVERIFIED CLAIMS:\n"
                + "\n".join(f"- {c}" for c in claims))
        raw = await self._complete(system, user)
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw.strip()) if p.strip()]
        return paras


class OpenAILLM(AnthropicLLM):
    """Any OpenAI-COMPATIBLE chat endpoint (OpenAI, Azure OpenAI, vLLM, Ollama,
    Together, Groq, ...). Reuses every prompt from AnthropicLLM — only the transport
    (`_complete`) differs — so the two providers stay behaviourally identical and we
    don't fork the prompt engineering.

    base_url points at the `/v1`-style root; we call `{base_url}/chat/completions`
    with the standard Bearer-key + messages payload.
    """

    def __init__(self, api_key: str, model: str, max_tokens: int, base_url: str) -> None:
        super().__init__(api_key, model, max_tokens)
        self._base_url = base_url.rstrip("/")

    async def _transport(self, system: str, user: str) -> str:
        # Transport ONLY — concurrency + 429/5xx retry policy live in the base
        # AnthropicLLM._complete, shared by every provider. We call raise_for_status()
        # so an httpx.HTTPStatusError (which carries .response, hence status +
        # Retry-After) propagates to that policy for classification.
        import httpx

        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        payload = {
            "model": self._model,
            "max_tokens": _completion_budget.get() or self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        from urllib.parse import urlsplit

        settings = get_settings()
        minimax = urlsplit(self._base_url).hostname in {"api.minimaxi.com", "api.minimax.cn", "api.minimax.io"}
        if minimax and self._model.startswith("MiniMax-M3"):
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            payload["reasoning_split"] = True
            if settings.llm_thinking_mode != "provider_default":
                if settings.llm_thinking_mode == "disabled" and self._model != "MiniMax-M3":
                    raise LLMResponseError("thinking_disable_unsupported_by_model")
                payload["thinking"] = {"type": settings.llm_thinking_mode}
        async with httpx.AsyncClient(timeout=get_settings().llm_timeout_s) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMResponseError("invalid_completion_envelope")
        choice = choices[0]
        message = choice.get("message")
        raw = message.get("content") if isinstance(message, dict) else None
        # Strip <think>...</think> reasoning tags (DeepSeek-R1, QwQ, etc.) at the
        # boundary — OpenAI-compatible endpoints often front these models.
        return completion_text(raw, choice.get("finish_reason"))


def get_llm() -> LLM:
    s = get_settings()
    if s.llm_provider == "fake":
        return FakeLLM()
    if s.llm_provider == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError("CE_ANTHROPIC_API_KEY required for anthropic provider")
        return AnthropicLLM(s.anthropic_api_key, s.llm_model, s.llm_max_tokens)
    if s.llm_provider == "openai":
        if not s.llm_base_url:
            raise RuntimeError("CE_LLM_BASE_URL required for openai-compatible provider")
        if not s.llm_api_key:
            raise RuntimeError("CE_LLM_API_KEY required for openai-compatible provider")
        return OpenAILLM(s.llm_api_key, s.llm_model, s.llm_max_tokens, s.llm_base_url)
    raise RuntimeError(f"Unknown llm_provider: {s.llm_provider}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    return split_sentences(text)


def _looks_factual(sentence: str) -> bool:
    if re.search(r"\d", sentence):
        return True
    if re.search(r"增长|下降|发布|成立|增加|减少|制造|供应|监管|市场|行业", sentence):
        return True
    if re.search(r"\b(?:we|this study|this research|this review|the study)\s+(?:created|report|reports|show|shows|describe|describes|demonstrate|demonstrates)\b", sentence, re.IGNORECASE):
        return True
    return bool(re.search(r"\b(is|are|was|were|reported|announced|found|rose|fell|increased|decreased)\b",
                          sentence, re.IGNORECASE))


def _tokens(s: str) -> set[str]:
    return set(lexical_tokens(s))


# Discriminating content words for the offline contradiction heuristic — same intent as
# verify._content_terms, kept local so llm.py has no dependency on the harness module.
_FAKE_STOP = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "is",
    "are", "was", "were", "be", "been", "by", "at", "as", "that", "this", "it",
    "from", "has", "have", "had", "will", "would", "which", "than", "then", "its",
    "into", "over", "about", "more", "most", "such", "also", "can", "may", "these",
    "those", "their", "they", "we", "our", "but", "not", "per", "via", "vs",
})


def _content_terms(text: str) -> set[str]:
    return content_terms(text, _FAKE_STOP)


def _numbers(text: str) -> set[str]:
    """Distinct numeric tokens (integers, decimals, percentages) in a claim, normalized
    so '12' and '12.0' compare equal — the divergence signal the fake conflict check uses."""
    return numeric_tokens(text)


def _best_overlap(claim: str, source_text: str) -> tuple[str, float]:
    """Find the source sentence with the highest token-overlap (Jaccard-ish) with
    the claim. Returns (sentence, score)."""
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return "", 0.0
    best_sent, best_score = "", 0.0
    for sent in _split_sentences(source_text):
        st = _tokens(sent)
        if not st:
            continue
        overlap = len(claim_tokens & st) / len(claim_tokens)
        if overlap > best_score:
            best_sent, best_score = sent, overlap
    return best_sent, best_score


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks that some models (e.g. DeepSeek-R1,
    QwQ) emit. Applied at the LLM boundary so NO downstream consumer — chat, intent
    classification, or report drafting — ever sees them. Also tidies the whitespace
    left behind. Never relies on the frontend to hide them."""
    if not text:
        return text
    cleaned = _THINK_RE.sub("", text)
    # A model may emit an unterminated <think> (no closing tag) if truncated; drop
    # from a lone opening tag to end-of-string as a safety net.
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _safe_json_list(raw: str) -> list:
    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else []
    except Exception:
        return []


def _safe_json_obj(raw: str) -> dict:
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def _slot(v) -> str | None:
    """Coerce an LLM-supplied slot value to a stripped string or None (missing/empty/
    explicit null all degrade to None)."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _extract_slots(sentence: str) -> dict:
    """Rule-based structured-slot extraction for the FakeLLM dual-track output.

    Recognizes a small set of deterministic shapes so offline tests exercise the
    alignment layer without a model:
      "COMPETITOR: <name> share 30% ..."   -> entity=name, attribute=market_share
      "ENTITY: <name> ..."                 -> entity=name (explicit marker)
      "<Name>'s <attr> ..."                -> entity=<Name> (possessive)
      "TAM/SAM/SOM/CAGR is 200 USD bn"     -> attribute + value
      "market share of 25%" / "revenue ..."-> attribute + value
      "GAAP" / "domestic" / "global"       -> qualifier
      a year or FYxxxx                     -> time_scope
    Anything unrecognized stays None (the claim degrades to plain-text handling).
    """
    slots: dict[str, str | None] = {
        "entity": None, "attribute": None, "value": None,
        "qualifier": None, "time_scope": None,
    }
    m = re.search(r"\bcompetitor\s*[:\-]\s*([A-Za-z0-9 &/.]+?)\s+share\s+([\d.]+\s*%)",
                  sentence, re.IGNORECASE)
    if m:
        slots["entity"] = m.group(1).strip()
        slots["attribute"] = "market_share"
        slots["value"] = m.group(2).strip()
    else:
        m = re.search(r"\bENTITY\s*[:\-]\s*([A-Za-z0-9 &/.]+)", sentence)
        if m:
            slots["entity"] = m.group(1).strip().rstrip(".")
        else:
            m = re.match(r"([A-Z][\w&]*(?:\s+[A-Z][\w&]*){0,3})'s\s+", sentence)
            if m:
                slots["entity"] = m.group(1).strip()
        m = re.search(r"\b(TAM|SAM|SOM|CAGR)\b[^\d%]*([\d.]+\s*(?:USD\s*bn|bn|billion|%)?)",
                      sentence, re.IGNORECASE)
        if m:
            slots["attribute"] = m.group(1).upper()
            slots["value"] = m.group(2).strip() or None
        else:
            m = re.search(r"\b(market share|revenue|capacity)\b[^\d%]*([\d.]+\s*"
                          r"(?:USD\s*bn|bn|billion|%|GW))?", sentence, re.IGNORECASE)
            if m:
                slots["attribute"] = m.group(1).lower().replace(" ", "_")
                if m.group(2):
                    slots["value"] = m.group(2).strip()
    if slots["value"] is None:
        m = re.search(r"([\d.]+\s*(?:USD\s*bn|bn|billion|%|percent))", sentence,
                      re.IGNORECASE)
        if m:
            slots["value"] = m.group(1).strip()
    m = re.search(r"\b(GAAP|non-GAAP|domestic(?:\s+only)?|global(?:\s+only)?)\b",
                  sentence, re.IGNORECASE)
    if m:
        slots["qualifier"] = m.group(1).strip()
    m = re.search(r"\bFY\s?(\d{4})\b", sentence, re.IGNORECASE)
    if m:
        slots["time_scope"] = f"FY{m.group(1)}"
    else:
        m = re.search(r"\b((?:19|20)\d{2})\b", sentence)
        if m:
            slots["time_scope"] = m.group(1)
    return slots
