"""Bounded, opt-in standard/deep report comparison on one immutable research plan.

Replay supplies identical saved source bodies and forbids retrieval. Live collection
uses separate cold caches per arm and records changing web inputs; it is an
observational case comparison, not a causal before/after or scientific truth score.
No saved settings or environment variables are changed by this script.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, fields
from datetime import UTC, date, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_engine.app.runtime import load_settings
from core_engine.config import Settings, get_settings, settings_for_run
from core_engine.report import llm as llm_module
from core_engine.report import metrics as metric_module
from core_engine.report.evidence_quality import origin_groups
from core_engine.report.generation import GenerationDepth, profile_audit, report_profile
from core_engine.report.grounding import evidence_packet
from core_engine.report.llm import get_llm
from core_engine.report.models import CredibilityLevel, Source, SourceKind
from core_engine.report.pipeline import ReportPipeline
from core_engine.report.quality_eval import check_packet
from core_engine.report.research import KINDS, ResearchPlan, ResearchQuestion
from core_engine.report.scholarly import FetchedDocument
from core_engine.report.scrape import get_search_provider
from core_engine.report.search_cache import SearchCache
from scripts.compare_openalex import SnapshotFetcher, json_default, write

DEFAULT_PLAN = str(Path(__file__).resolve().parents[1] / "evals/research_quality/solid_state_depth_plan.json")
SCIENTIFIC_DIMENSIONS = {"conductivity", "specific_energy", "volumetric_energy", "specific_capacity",
                         "areal_capacity", "current_density", "resistance", "resistance_area",
                         "cycles"}
SCIENTIFIC_NAMES = {"ionic_conductivity", "electronic_conductivity", "energy_density", "cycle_life",
                    "capacity_retention", "coulombic_efficiency", "critical_current_density",
                    "interfacial_resistance", "specific_capacity", "areal_capacity"}
PUBLIC_TOKEN_BUDGET_KEYS = {"llm_max_tokens", "extraction_max_tokens", "thinking_max_tokens"}
_ACTIVE_LEDGER: ContextVar[Ledger | None] = ContextVar("depth_benchmark_arm", default=None)


class BenchmarkStop(BaseException):
    """Bypass per-source soft failures when a benchmark's hard cost boundary fires."""


class RequestBudgetExceeded(BenchmarkStop):
    pass


class ReplayNetworkDenied(BenchmarkStop):
    pass


class ModelCacheMiss(BenchmarkStop):
    pass


class ArmScopeClosed(BenchmarkStop):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    default=json_default).encode()).hexdigest()


def configured_settings() -> Settings:
    """Use the GUI's configured values without apply_settings_to_env/save_settings."""
    saved = {key: value for key, value in load_settings().items() if key in Settings.model_fields}
    return Settings.model_validate({**get_settings().model_dump(), **saved})


def model_hostname(settings: Settings) -> str:
    if settings.llm_provider == "anthropic":
        return "api.anthropic.com"
    return urlsplit(settings.llm_base_url).hostname or ""


def safe_settings(settings: Settings) -> dict:
    result = {}
    for key, value in settings.model_dump().items():
        if re.search(r"key|token|password|secret|dsn", key, re.IGNORECASE) and key not in PUBLIC_TOKEN_BUDGET_KEYS:
            continue
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            url = urlsplit(value)
            value = urlunsplit((url.scheme, url.hostname or "", url.path, "", ""))
        result[key] = value
    return result


def code_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    package = Path(llm_module.__file__).resolve().parents[1]
    paths = [*package.rglob("*.py"), *package.rglob("*.j2"), Path(__file__), Path(__file__).with_name("compare_openalex.py")]
    hashes = {str(path.relative_to(package)) if path.is_relative_to(package) else path.name:
              hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        head = None
    return {"recorded_at": now(), "git_head": head, "package_path": str(package),
            "files": hashes, "sha256": digest(hashes),
            "scientific_units_present": "S/cm" in Path(metric_module.__file__).read_text(encoding="utf-8")}


def load_plan(path: Path) -> ResearchPlan:
    saved = json.loads(path.read_text(encoding="utf-8"))
    row = saved.get("research_plan") or saved.get("report", {}).get("research_plan") or saved
    questions = [ResearchQuestion(**question) for question in row["questions"]]
    if (len(questions) != 6 or len({q.id for q in questions}) != 6
            or not row.get("objective") or questions[0].question != row["objective"]
            or questions[0].required is not True or any(q.kind not in KINDS for q in questions)):
        raise ValueError("The benchmark requires the unchanged historical six-question plan")
    allowed = {item.name for item in fields(ResearchPlan)} - {"questions"}
    return ResearchPlan(**{key: value for key, value in row.items() if key in allowed}, questions=questions)


def before_cutoff(value, cutoff: date) -> bool:
    """Exclude explicitly future-dated records; never invent dates for unknowns."""
    if not value:
        return True
    match = re.match(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", str(value))
    if not match:
        return True
    try:
        return date(int(match[1]), int(match[2] or 1), int(match[3] or 1)) <= cutoff
    except ValueError:
        return True


def publication_date(metadata: dict):
    return next((metadata.get(key) for key in ("published_at", "publication_date", "publication_year", "year")
                 if metadata.get(key)), None)


def restore_sources(path: Path, cutoff: date) -> list[Source]:
    row = json.loads(path.read_text(encoding="utf-8"))
    rows = row.get("report", row).get("sources", [])
    allowed = {item.name for item in fields(Source)}
    sources = []
    for saved in rows:
        data = {key: value for key, value in saved.items() if key in allowed}
        data["kind"] = SourceKind(data["kind"])
        data["credibility"] = CredibilityLevel(data.get("credibility", 3))
        source = Source(**data)
        if before_cutoff(source.published_at or publication_date(source.metadata), cutoff):
            sources.append(source)
    if not sources or len({source.url for source in sources}) != len(sources):
        raise ValueError("Replay needs non-empty, distinct saved source bodies")
    return sources


def corpus_manifest(sources: list[Source]) -> dict:
    rows = [{"url": source.url, "sha256": source.content_sha256, "characters": len(source.text),
             "format": source.document_format, "published_at": source.published_at,
             "full_text_accessed": bool(source.metadata.get("full_text_accessed"))} for source in sources]
    return {"sources": rows, "sha256": digest(rows), "count": len(rows)}


def historical_summary(path: Path, plan: ResearchPlan) -> dict | None:
    saved = json.loads(path.read_text(encoding="utf-8"))
    report = saved.get("report")
    if not isinstance(report, dict):
        return None
    sources = report.get("sources", [])
    return {"source_file": str(path.resolve()), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "coverage": coverage_summary(plan, report.get("research_audit") or saved.get("research_audit", {})),
            "sources": len(sources), "metrics": len(report.get("metrics", [])),
            "full_text_documents": sum(bool(source.get("metadata", {}).get("full_text_accessed")) for source in sources),
            "semantic_citation_correctness": None, "live_cost": None, "live_elapsed_s": None,
            "interpretation": "Unchanged saved historical coverage labels, not reassessed evidence. Different code/profile/retrieval period; descriptive only, not a causal third A/B arm."}


class Ledger:
    def __init__(self, directory: Path, *, max_live_requests: int, mode: str, model_host: str):
        self.directory = directory
        self.max_live_requests = max_live_requests
        self.mode = mode
        self.model_host = model_host
        self.counts: Counter[str] = Counter()
        self.seconds: dict[str, float] = defaultdict(float)
        self.events: list[dict] = []
        self.budget_exhausted = False
        self.closed = False

    def ensure_open(self):
        if self.closed:
            raise ArmScopeClosed("Benchmark arm has already closed")

    @asynccontextmanager
    async def owned_tasks(self) -> AsyncIterator[None]:
        """Drain only tasks born in this arm before restoring transport accounting.

        An early gather exception can leave sibling requests in flight. Tracking
        task creation by inherited context (not all_tasks()) excludes unrelated
        event-loop work and keeps late usage out of the next arm's counters.
        """
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()
        owned: set[asyncio.Task[Any]] = set()

        def factory(event_loop, coroutine, **kwargs):
            context = kwargs.get("context")
            active = context.get(_ACTIVE_LEDGER) if context is not None else _ACTIVE_LEDGER.get()
            task = (previous(event_loop, coroutine, **kwargs) if previous else
                    asyncio.Task(coroutine, loop=event_loop, **kwargs))
            if active is self:
                owned.add(task)
                task.add_done_callback(owned.discard)
            return task

        loop.set_task_factory(factory)
        try:
            yield
        finally:
            self.closed = True
            waiting = {task for task in owned if not task.done()}
            try:
                if waiting:
                    for task in waiting:
                        task.cancel()
                    finished, _ = await asyncio.wait(waiting, timeout=2)
                    for task in finished:
                        if not task.cancelled():
                            task.exception()  # Consume a sibling's already-logged error.
                    self.counts["background_tasks_cancelled"] += sum(task.cancelled() for task in finished)
                    unfinished = {task for task in owned if not task.done()}
                    self.counts["background_cleanup_pending"] = len(unfinished)
                    for task in unfinished:
                        task.cancel()
                    self.event("arm_task_cleanup", requested=len(waiting),
                               pending=self.counts["background_cleanup_pending"])
            finally:
                loop.set_task_factory(previous)

    def event(self, kind: str, **detail):
        self.events.append({"at": now(), "kind": kind, **detail})
        write(self.directory / "requests.json", self.events)
        write(self.directory / "accounting-live.json", self.snapshot())

    def model_guard(self):
        self.ensure_open()
        if self.counts["model_http_attempts"] >= self.max_live_requests:
            self.budget_exhausted = True
            raise RequestBudgetExceeded("Per-arm real model HTTP attempt budget reached")

    def snapshot(self) -> dict:
        return {"counts": dict(self.counts), "elapsed_sums_s": {key: round(value, 4) for key, value in self.seconds.items()},
                "budget_exhausted": self.budget_exhausted, "max_live_model_http_attempts": self.max_live_requests,
                "provider_reported_usage": ({key: self.counts["provider_" + key] for key in
                    ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
                    if "provider_" + key in self.counts} if self.counts["provider_usage_response_count"] else None),
                "billed_tokens": None, "monetary_cost": None,
                "accounting_note": "HTTP attempts include retries/cancelled sends observed through httpx; non-httpx transports such as DDGS internals are not counted as observed HTTP. Logical provider calls, fetch operations and cache reads are separate; summed concurrent request durations are not wall latency. Billed usage/currency are unavailable."}

    def record_model_response(self, response: httpx.Response):
        try:
            data = response.json()
        except (ValueError, httpx.ResponseNotRead):
            return
        if not isinstance(data, dict):
            return
        supplied = data.get("usage")
        usage = {key: value for key, value in supplied.items()
                 if key in {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"}
                 and type(value) is int and value >= 0} if isinstance(supplied, dict) else {}
        if usage:
            self.counts["provider_usage_response_count"] += 1
            for key, value in usage.items():
                self.counts["provider_" + key] += value
        choices = data.get("choices")
        choice: dict = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        supplied_message = choice.get("message")
        message: dict = supplied_message if isinstance(supplied_message, dict) else {}
        finish = choice.get("finish_reason", data.get("stop_reason"))
        allowed = {"stop", "length", "max_tokens", "content_filter", "refusal", "tool_calls", "end_turn"}
        self.event("model_response_metadata", usage=usage or None,
                   finish_reason=finish if isinstance(finish, str) and finish in allowed else "other" if finish is not None else None,
                   content_characters=len(message["content"]) if isinstance(message.get("content"), str) else None,
                   reasoning_characters=len(message["reasoning_content"]) if isinstance(message.get("reasoning_content"), str) else None)

    @contextmanager
    def http_accounting(self):
        original = httpx.AsyncClient.send

        async def send(client, request, *args, **kwargs):
            active = _ACTIVE_LEDGER.get()
            if active is None:
                return await original(client, request, *args, **kwargs)
            if active is not self:
                raise ArmScopeClosed("Late HTTP request belongs to another benchmark arm")
            self.ensure_open()
            host = request.url.host
            model = host == self.model_host
            if self.mode == "replay" and not model:
                self.counts["blocked_replay_http_attempts"] += 1
                raise ReplayNetworkDenied("Frozen-corpus replay forbids retrieval HTTP requests")
            if model:
                self.model_guard()
                role = "model"
            elif any(part in host for part in ("tavily", "duckduckgo", "firecrawl", "jina.ai")):
                role = "search"
            elif host in {"api.openalex.org", "api.crossref.org", "www.ebi.ac.uk", "europepmc.org"}:
                role = "academic_metadata"
            else:
                role = "source_or_auxiliary"
            self.counts[role + "_http_attempts"] += 1
            started = time.monotonic()
            self.event("http_start", role=role, host=host, path=request.url.path,
                       method=request.method)  # No auth headers/query strings/bodies.
            try:
                response = await original(client, request, *args, **kwargs)
                self.counts[role + "_http_responses"] += 1
                if response.is_error:
                    self.counts[role + "_http_error_responses"] += 1
                if model:
                    self.record_model_response(response)
                self.event("http_end", role=role, host=host, status=response.status_code)
                return response
            except BaseException as exc:
                self.counts[role + "_http_failures"] += 1
                self.event("http_failure", role=role, host=host, error_type=type(exc).__name__)
                raise
            finally:
                self.seconds[role + "_http"] += time.monotonic() - started
                write(self.directory / "accounting-live.json", self.snapshot())

        token = _ACTIVE_LEDGER.set(self)
        try:
            with patch.object(httpx.AsyncClient, "send", send):
                yield
        finally:
            self.closed = True
            _ACTIVE_LEDGER.reset(token)


def cache_llm(llm, directory: Path, ledger: Ledger, *, cache_only=False):
    """Exact-prompt replay at the same seam as compare_openalex.record_llm.

    Unlike that older helper, include effective per-call output budget in the key,
    and count provider HTTP attempts separately from logical completion operations.
    """
    seam = "_complete" if hasattr(llm, "_complete") else "complete"
    original = getattr(llm, seam)
    locks: dict[str, asyncio.Lock] = {}

    async def complete(system, user):
        ledger.ensure_open()
        settings = get_settings()
        budget = llm_module._completion_budget.get() or settings.llm_max_tokens
        identity = [settings.llm_provider, settings.llm_base_url, settings.llm_model,
                    budget, settings.llm_thinking_mode, system, user]
        key = digest(identity)
        path = directory / (key + ".json")
        async with locks.setdefault(key, asyncio.Lock()):
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("response"):
                    ledger.counts["model_cache_replays"] += 1
                    ledger.event("model_replay", request_sha256=key)
                    return cached["response"]
            if cache_only:
                raise ModelCacheMiss("No exact saved model response; live model requests disabled")
            ledger.model_guard()
            ledger.counts["model_completion_operations"] += 1
            started = time.monotonic()
            try:
                response = await original(system, user)
            except BaseException as exc:
                ledger.counts["model_completion_failures"] += 1
                ledger.event("model_completion_failure", request_sha256=key, error_type=type(exc).__name__,
                             reason=exc.reason if isinstance(exc, llm_module.LLMResponseError) else None,
                             effective_max_tokens=budget)
                raise
            elapsed = time.monotonic() - started
            ledger.seconds["model_completion_operations"] += elapsed
            write(path, {"request_sha256": key, "recorded_at": now(), "model": settings.llm_model,
                         "effective_max_tokens": budget, "system": system, "user": user,
                         "response": response, "original_elapsed_s": elapsed})
            return response

    setattr(llm, seam, complete)
    return llm


class RecordedFetcher(SnapshotFetcher):
    def __init__(self, directory, ledger: Ledger, cutoff: date | None = None):
        super().__init__(directory)
        self.ledger = ledger
        self.cutoff = cutoff

    async def _fetch(self, url, *, fulltext):
        self.ledger.ensure_open()
        started = time.monotonic()
        try:
            result = await super()._fetch(url, fulltext=fulltext)
            if (self.cutoff and isinstance(result, FetchedDocument)
                    and not before_cutoff(result.published_at or publication_date(result.metadata), self.cutoff)):
                self.failures[url] = "published_after_benchmark_cutoff"
                self.ledger.counts["future_dated_documents_excluded"] += 1
                return None
            return result
        finally:
            # SnapshotFetcher's counters are shared by concurrent fetch tasks;
            # adding per-call deltas would count another task's operation twice.
            self.ledger.counts["fetcher_live_operations"] = self.network_reads
            self.ledger.counts["fetcher_snapshot_replays"] = self.snapshot_reads
            self.ledger.seconds["fetcher_operations"] += time.monotonic() - started
            self.ledger.event("fetch_operation_end", url_sha256=hashlib.sha256(url.encode()).hexdigest(),
                              fulltext=fulltext)


class FrozenFetcher:
    def __init__(self, sources: list[Source], ledger: Ledger):
        self.sources = {source.url: source for source in sources}
        self.ledger = ledger
        self.failures: dict[str, str] = {}

    async def fetch(self, url):
        self.ledger.ensure_open()
        source = self.sources.get(url)
        if source is None:
            self.failures[url] = "not_in_frozen_corpus"
            self.ledger.counts["frozen_fetch_misses"] += 1
            return None
        self.ledger.counts["fetcher_snapshot_replays"] += 1
        return FetchedDocument(source.text, source.document_format, source.published_at, copy.deepcopy(source.metadata))

    async def fetch_fulltext(self, url):
        return await self.fetch(url)


class RecordedSearchCache(SearchCache):
    def __init__(self, directory: Path, ledger: Ledger):
        super().__init__(directory)
        self.ledger = ledger

    def get(self, backend, query):
        hits = super().get(backend, query)
        if hits:
            self.ledger.counts["search_cache_replays"] += 1
            self.ledger.event("search_replay", backend=backend, query=query, hits=len(hits))
        return hits


@contextmanager
def search_accounting(ledger: Ledger, directory: Path, cutoff: date):
    """Instrument real backend calls without replacing provider types/cache policy."""
    original = get_search_provider
    cache = RecordedSearchCache(directory, ledger)

    def provider(name=None):
        active = _ACTIVE_LEDGER.get()
        if active is None:
            return original(name)
        if active is not ledger:
            raise ArmScopeClosed("Late search request belongs to another benchmark arm")
        ledger.ensure_open()
        if ledger.mode == "replay":
            class NoSearch:
                async def search(self, *_args, **_kwargs):
                    raise ReplayNetworkDenied("Frozen corpus forbids search backend calls")
            return NoSearch()
        backend = name or get_settings().search_provider
        instance = original(name)
        raw_search = instance.search

        async def search(query, *, max_results):
            ledger.ensure_open()
            ledger.counts["search_provider_operations"] += 1
            ledger.counts["search_provider_" + backend] += 1
            started = time.monotonic()
            ledger.event("search_start", backend=backend, query=query, max_results=max_results)
            try:
                hits = await raw_search(query, max_results=max_results)
                kept = [hit for hit in hits if before_cutoff(publication_date(hit.metadata), cutoff)]
                ledger.event("search_end", backend=backend, query=query, hits=len(hits),
                             admitted_hits=len(kept), future_dated_excluded=len(hits) - len(kept))
                return kept
            except BaseException as exc:
                ledger.counts["search_provider_failures"] += 1
                ledger.event("search_failure", backend=backend, error_type=type(exc).__name__)
                raise
            finally:
                ledger.seconds["search_provider_operations"] += time.monotonic() - started
        instance.search = search
        return instance

    with ExitStack() as stack:
        for target in ("core_engine.report.scrape.get_search_provider", "core_engine.report.router.get_search_provider",
                       "core_engine.report.pipeline.get_search_provider"):
            stack.enter_context(patch(target, provider))
        stack.enter_context(patch("core_engine.report.scrape.SearchCache",
                                  lambda *args, **kwargs: cache if _ACTIVE_LEDGER.get() is ledger else SearchCache(*args, **kwargs)))
        yield


def coverage_summary(plan: ResearchPlan, audit: dict) -> dict:
    rows = audit.get("questions", [])
    if (len({row.get("id") for row in rows}) != len(rows)
            or any(row.get("id") not in {q.id for q in plan.questions} for row in rows)):
        raise ValueError("Coverage has duplicate or unexpected question identifiers")
    indexed = {row.get("id"): row for row in rows}
    result = []
    for question in plan.questions:
        row = indexed.get(question.id)
        status = row.get("status") if row else "not_evaluated"
        if row and row.get("question") != question.question:
            raise ValueError("Coverage changed a frozen question")
        mapped = {"supported": "complete", "partial": "partial", "missing": "unsupported",
                  "undetermined": "not_evaluated", "not_evaluated": "not_evaluated"}.get(status)
        if mapped is None:
            raise ValueError("Unknown coverage status")
        result.append({"id": question.id, "question": question.question, "status": mapped,
                       "claim_ids": row.get("claim_ids", []) if row else []})
    counts = Counter(row["status"] for row in result)
    denominator = len(plan.questions)
    return {"denominator": denominator, **{key: counts[key] for key in
            ("complete", "partial", "unsupported", "not_evaluated")},
            "complete_rate": counts["complete"] / denominator, "questions": result}


def scientific_pairs(rows: list[dict]) -> dict:
    metrics, invalid, condition_observations = [], [], 0
    for row in rows:
        try:
            metric = metric_module.Metric(**row)
            dimension = metric_module.parse_unit(metric.unit).dimension
            if dimension in {"temperature", "pressure"}:
                condition_observations += 1
            elif dimension in SCIENTIFIC_DIMENSIONS or metric.name in SCIENTIFIC_NAMES:
                metrics.append(metric)
        except (TypeError, ValueError) as exc:
            invalid.append({"id": row.get("id"), "error_type": type(exc).__name__})
    accepted, rejected = [], []
    for left, right in combinations(metrics, 2):
        if left.name != right.name:
            continue
        errors = metric_module.comparison_errors(left, right, allow_entity=True, same_period=True)
        entry = {"left": left.id, "right": right.id, "metric": left.name}
        if errors:
            rejected.append({**entry, "reasons": errors})
        else:
            accepted.append(entry)
    return {"scientific_observations": len(metrics), "comparable_pairs": len(accepted),
            "condition_observations": condition_observations,
            "accepted": accepted, "incomparable_candidates": rejected, "invalid_metrics": invalid,
            "rule": "Production comparison_errors(allow_entity=True, same_period=True); no missing conditions are guessed."}


def semantic_samples(report, limit=5) -> list[dict]:
    citations = {citation.id: citation for citation in report.citations}
    sources = {source.url: source for source in report.sources}
    samples: list[dict] = []
    eligible = [sentence for sentence in report.all_sentences()
                if sentence.status == "supported" and sentence.citation_ids]
    # Deliberately expose quantitative/condition-bearing sentences first rather
    # than filling the review with easy introductory definitions. This is a
    # deterministic spot sample, not a random representative accuracy estimate.
    eligible.sort(key=lambda sentence: -int(bool(re.search(
        r"\d|S/cm|mS|Wh|mAh|MPa|°C|温度|压力|电芯|半电池", sentence.text))))
    for sentence in eligible:
        if sentence.status != "supported" or not sentence.citation_ids:
            continue
        links = []
        for identifier in sentence.citation_ids:
            citation = citations.get(identifier)
            if citation is None or citation.source_url not in sources:
                continue
            source = sources[citation.source_url]
            links.append({"citation_id": identifier, "url": citation.source_url, "quote": citation.quote,
                          "source_context": source.text[max(0, citation.start_char - 180):citation.end_char + 180],
                          "quote_matches_source": source.text[citation.start_char:citation.end_char] == citation.quote})
        if links:
            samples.append({"id": "sample_" + str(len(samples) + 1), "sentence": sentence.text,
                            "attribution": sentence.attribution,
                            "displayed_sentence": sentence.attribution + sentence.text,
                            "claim_ids": sentence.claim_ids, "citations": links,
                            "metrics": [metric for metric in report.metrics
                                        if set(metric.get("claim_ids", [])) & set(sentence.claim_ids)]})
        if len(samples) >= limit:
            break
    return samples


async def semantic_review(report, llm, count: int) -> dict:
    samples = semantic_samples(report, max(5, count))
    review = {"status": "pending_manual_review", "semantic_citation_correctness": None,
              "reviewer": None, "samples": samples,
              "note": "Deterministic numeric/conditions-first spot sample, not random. Exact quote integrity is separate from semantic correctness. No score is invented when nobody reviewed the samples."}
    if count == 0 or not samples:
        return review
    raw = await llm.complete(
        "Audit whether each displayed_sentence, including its displayed attribution, follows from the quoted source context "
        "without changing material/cell scope, test conditions or strength of the claim. "
        "Return JSON {reviews:[{id,verdict,note}]}; verdict is supported, overstated, contradicted or uncertain. Do not use outside knowledge.",
        json.dumps(samples[:count], ensure_ascii=False))
    rows = json.loads(raw).get("reviews", [])
    expected = {row["id"] for row in samples[:count]}
    if (len(rows) != len(expected) or {row.get("id") for row in rows} != expected
            or any(row.get("verdict") not in {"supported", "overstated", "contradicted", "uncertain"} for row in rows)):
        raise ValueError("Incomplete semantic review contract")
    review.update(status="model_spot_review_completed", reviewer="same_configured_model_not_independent",
                  reviews=rows, semantic_citation_correctness=sum(row["verdict"] == "supported" for row in rows) / len(rows))
    return review


def summarize_result(result, plan: ResearchPlan) -> dict:
    report = result.report
    audit = (report.research_audit or result.research_audit) if report else result.research_audit
    summary = {"pipeline_status": result.status.value, "coverage": coverage_summary(plan, audit),
               "message": result.message, "report_available": report is not None}
    if report is None:
        return summary
    packet = evidence_packet(report)
    groups = origin_groups(report.sources)
    summary.update(source_count=len(report.sources), origin_groups=len(set(groups.values())),
                   full_text_documents=sum(bool(source.metadata.get("full_text_accessed")) for source in report.sources),
                   body_characters=sum(len(sentence.text) for sentence in report.all_sentences() if sentence.status == "supported"),
                   claims=len(report.claims), metrics=len(report.metrics), metric_issues=len(report.metric_issues),
                   scientific=scientific_pairs(report.metrics), integrity=check_packet(packet),
                   corpus=corpus_manifest(report.sources), quality_status=report.quality_status,
                   pdf=result.pdf_path, evidence=result.evidence_path, tex=result.tex_path)
    return summary


def snapshot_inventory(directory: Path) -> dict:
    rows = [{"path": str(path.relative_to(directory)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "bytes": path.stat().st_size} for path in sorted(directory.rglob("*.json"))]
    return {"files": rows, "sha256": digest(rows)}


async def run_arm(arm: str, *, base: Settings, plan: ResearchPlan, directory: Path, mode: str,
                  sources: list[Source], timeout: float, max_live_requests: int, as_of: date,
                  pdf=False, cache_only=False, review_count=0, pipeline_factory=ReportPipeline,
                  llm_factory=get_llm) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    profile = report_profile(base, cast(GenerationDepth, arm)).model_copy(update={"output_dir": directory / "report"})
    model_host = model_hostname(profile)
    ledger = Ledger(directory, max_live_requests=max_live_requests, mode=mode, model_host=model_host)
    frozen_plan = plan.to_dict()
    manifest = {"arm": arm, "started_at": now(), "mode": mode, "as_of_date": as_of.isoformat(),
                "plan_sha256": digest(frozen_plan), "profile": profile_audit(profile),
                "settings": safe_settings(profile), "code": code_state(),
                "incoming_corpus": corpus_manifest(sources) if mode == "replay" else None,
                "temporal_rule": "Exclude known future publication dates; unknown dates stay unknown. Same collection calendar date, not identical live web results."}
    write(directory / "manifest.json", manifest)
    write(directory / "frozen-plan.json", frozen_plan)
    result, status, error_type = None, "running", None
    semantic = {"status": "not_reached", "semantic_citation_correctness": None}
    started = time.monotonic()
    progress_events = []

    def progress(stage, detail):
        progress_events.append({"at": now(), "stage": stage, "detail": detail})
        write(directory / "progress.json", progress_events)
        if not stage.endswith("_progress"):
            print(f"[{arm}] {stage}: {detail[:160]}", flush=True)

    try:
        with settings_for_run(profile), ledger.http_accounting(), search_accounting(ledger, directory / "cache/search", as_of):
            async with ledger.owned_tasks(), asyncio.timeout(timeout):
                llm = cache_llm(llm_factory(), directory / "cache/llm", ledger, cache_only=cache_only)
                fetcher = (FrozenFetcher(copy.deepcopy(sources), ledger) if mode == "replay" else
                           RecordedFetcher(directory / "cache/source-snapshots", ledger, as_of))
                replay_plan = copy.deepcopy(plan)
                result = await pipeline_factory(llm=llm, fetcher=fetcher).run(
                    plan.objective, compile_to_pdf=pdf, research_plan=replay_plan,
                    extra_sources=copy.deepcopy(sources) if mode == "replay" else None,
                    skip_search=mode == "replay", on_progress=progress)
                if replay_plan.to_dict() != frozen_plan:
                    raise ValueError("An arm mutated the frozen research plan")
                if result.research_plan and result.research_plan != frozen_plan:
                    raise ValueError("An arm changed the historical research plan")
                write(directory / "result.json", asdict(result))
                if result.report:
                    semantic = await semantic_review(result.report, llm, review_count)
                status = "pipeline_error" if result.status.value == "error" else "completed"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, asyncio.CancelledError):
            status, error_type = "cancelled", type(exc).__name__
            raise
        status = ("budget_exhausted" if ledger.budget_exhausted else "timeout" if isinstance(exc, TimeoutError)
                  else "failed")
        error_type = type(exc).__name__
        ledger.event("arm_failure", error_type=error_type, status=status)
    finally:
        summary = {"arm": arm, "status": status, "error_type": error_type,
                   "started_at": manifest["started_at"], "finished_at": now(),
                   "elapsed_wall_s": round(time.monotonic() - started, 3),
                   "plan_sha256": manifest["plan_sha256"], "profile": manifest["profile"],
                   "mode": mode, "accounting": ledger.snapshot(), "semantic_review": semantic}
        if result is not None:
            try:
                summary.update(summarize_result(result, plan))
            except Exception as exc:
                # The raw result was already retained. A broken metric/review
                # must not erase this arm or prevent the independent next arm.
                summary.update(analysis_error_type=type(exc).__name__, report_available=result.report is not None,
                               pipeline_status=result.status.value, coverage=coverage_summary(plan, {}))
                if status == "completed":
                    summary["status"] = "analysis_failed"
        else:
            summary.update(report_available=False, pipeline_status=None, coverage=coverage_summary(plan, {}))
        summary["snapshots"] = snapshot_inventory(directory / "cache")
        write(directory / "summary.json", summary)
        write(directory / "partial-manifest.json", {"status": status, "snapshots": summary["snapshots"],
                                                    "progress_events": len(progress_events), "accounting": ledger.snapshot()})
    return summary


def compare_arms(summaries: dict[str, dict]) -> dict:
    standard, deep = summaries.get("standard"), summaries.get("deep")
    if not standard or not deep:
        return {"paired": False, "reason": "Both arm records are required"}
    same = standard["plan_sha256"] == deep["plan_sha256"]
    if not same or standard["coverage"]["denominator"] != deep["coverage"]["denominator"]:
        raise ValueError("Coverage denominators/questions differ; refusing an A/B rate")
    delta = {name: deep["coverage"][name] - standard["coverage"][name]
             for name in ("complete", "partial", "unsupported", "not_evaluated")}
    completed = all(summary["status"] == "completed" and summary.get("report_available") is True
                    and summary.get("pipeline_status") == "completed" for summary in (standard, deep))
    return {"paired": completed, "same_plan": same, "denominator": standard["coverage"]["denominator"],
            "deep_minus_standard": delta, "elapsed_wall_delta_s": deep["elapsed_wall_s"] - standard["elapsed_wall_s"],
            "pair_status": "both_reports_available" if completed else "incomplete_pair_do_not_rank_quality",
            "interpretation": "One case. A longer report, traceable quote or model coverage verdict is not proof of factual truth. Live arms may collect different documents; replay holds source bodies constant."}


async def probe(base: Settings, root: Path, timeout: float) -> dict:
    """One tiny model response, one primary query, one specified public source read."""
    profile = base.model_copy(update={"llm_max_tokens": 64, "llm_max_retries": 1,
                                     "llm_timeout_s": min(25.0, timeout)})
    ledger = Ledger(root, max_live_requests=1, mode="live", model_host=model_hostname(profile))
    results: dict[str, Any] = {"provider": profile.llm_provider, "model_id": profile.llm_model,
               "primary_search": profile.search_provider, "recorded_at": now()}
    value: Any
    with settings_for_run(profile), ledger.http_accounting():
        for label in ("model", "search", "source"):
            started = time.monotonic()
            try:
                async with asyncio.timeout(timeout):
                    if label == "model":
                        value = await get_llm().complete("Reply with OK only.", "Connectivity check.")
                        results[label] = {"status": "ok", "nonempty_response": bool(value.strip())}
                    elif label == "search":
                        value = await get_search_provider().search("solid state battery electrolyte", max_results=1)
                        results[label] = {"status": "ok", "hits": len(value)}
                    else:
                        fetcher = RecordedFetcher(root / "source-snapshots", ledger)
                        value = await fetcher.fetch("https://pubs.rsc.org/en/content/articlehtml/2024/ma/d4ma00619d")
                        results[label] = {"status": "ok" if value else "unavailable",
                                          "characters": len(value.text if isinstance(value, FetchedDocument) else value or ""),
                                          "failure_reason": fetcher.failures.get("https://pubs.rsc.org/en/content/articlehtml/2024/ma/d4ma00619d")}
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                results[label] = {"status": "failed", "error_type": type(exc).__name__}
            results[label]["elapsed_s"] = round(time.monotonic() - started, 3)
    results["accounting"] = ledger.snapshot()
    write(root / "probe.json", results)
    return results


async def run(args, *, base: Settings | None = None, pipeline_factory=ReportPipeline, llm_factory=get_llm) -> dict:
    base = base or configured_settings()
    if args.locale:
        base = base.model_copy(update={"report_locale": args.locale})
    if args.thinking_mode:
        base = base.model_copy(update={"llm_thinking_mode": args.thinking_mode})
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.probe:
        if base.llm_provider == "fake":
            raise ValueError("Readiness probing requires a real configured provider")
        return await probe(base, root / "probe", min(args.timeout, 40.0))
    if args.mode == "live" and not args.allow_live:
        raise ValueError("Live collection must be explicitly requested with --allow-live")
    if base.llm_provider == "fake" and llm_factory is get_llm:
        raise ValueError("A real configured model is required; fake output cannot be called a real benchmark")
    plan = load_plan(Path(args.plan_file))
    sources = restore_sources(Path(args.corpus_result), args.as_of_date) if args.mode == "replay" and args.corpus_result else []
    if args.mode == "replay" and not sources:
        raise ValueError("Replay requires --corpus-result with real saved source bodies")
    arms = args.arms.split(",")
    if not arms or len(arms) != len(set(arms)) or any(arm not in {"standard", "deep"} for arm in arms):
        raise ValueError("Use distinct --arms standard,deep")
    for arm in arms:
        if (root / arm / "manifest.json").exists() and not args.resume:
            raise ValueError("Output arm already exists; choose a fresh output or explicitly --resume (cache replays remain separately counted)")
    state = code_state()
    historical = historical_summary(Path(args.plan_file), plan)
    if historical is not None:
        write(root / "historical-baseline.json", historical)
    write(root / "run-manifest.json", {"started_at": now(), "code": state, "mode": args.mode,
                                       "as_of_date": args.as_of_date.isoformat(), "plan": plan.to_dict(),
                                       "plan_sha256": digest(plan.to_dict()), "settings": safe_settings(base),
                                       "timeout_per_arm_s": args.timeout, "max_live_requests_per_arm": args.max_live_requests,
                                       "requested_arms": arms, "resumed": args.resume,
                                       "task_overrides": {"locale": args.locale, "thinking_mode": args.thinking_mode,
                                                          "saved_user_settings_modified": False},
                                       "historical_baseline": historical})
    if sources:
        write(root / "frozen-corpus.json", {"manifest": corpus_manifest(sources), "sources": [asdict(s) for s in sources]})
    summaries = {}
    for arm in arms:
        summaries[arm] = await run_arm(arm, base=base, plan=plan, directory=root / arm, mode=args.mode,
                                       sources=sources, timeout=args.timeout, max_live_requests=args.max_live_requests,
                                       as_of=args.as_of_date, pdf=args.pdf, cache_only=args.llm_cache_only,
                                       review_count=args.semantic_review, pipeline_factory=pipeline_factory,
                                       llm_factory=llm_factory)
        comparison = {"arms": summaries, "comparison": compare_arms(summaries), "mode": args.mode,
                      "plan_sha256": digest(plan.to_dict()), "finished_at": now()}
        write(root / "comparison.json", comparison)
        print(f"[{arm}] {summaries[arm]['status']}: {summaries[arm]['coverage']}", flush=True)
        if summaries[arm]["accounting"]["counts"].get("background_cleanup_pending"):
            comparison["not_started_arms"] = [remaining for remaining in arms if remaining not in summaries]
            comparison["comparison"]["paired"] = False
            comparison["comparison"]["reason"] = "An arm left uncooperative background work; run any remaining arm in a separate CLI process"
            write(root / "comparison.json", comparison)
            break
    if code_state()["sha256"] != state["sha256"]:
        comparison["code_changed_during_benchmark"] = True
        comparison["comparison"]["paired"] = False
        comparison["comparison"]["reason"] = "Source code changed during collection; rerun from a stable snapshot for interpretation"
        write(root / "comparison.json", comparison)
    return comparison


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "replay"), default="replay")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--plan-file", default=DEFAULT_PLAN)
    parser.add_argument("--corpus-result", help="Saved result supplying identical Source bodies to both replay arms")
    parser.add_argument("--output", default="output/depth-quality-live-20261105/post-fix-live")
    parser.add_argument("--arms", default="standard,deep")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--max-live-requests", type=int, default=180, help="Hard per-arm model HTTP-attempt ceiling, including provider retries")
    parser.add_argument("--as-of-date", type=date.fromisoformat, default=date(2026, 11, 5))
    parser.add_argument("--locale", choices=("zh", "en"))
    parser.add_argument("--thinking-mode", choices=("provider_default", "enabled", "disabled"),
                        help="Task-local shared model setting for both arms; recorded in manifest, never saved to GUI settings")
    parser.add_argument("--semantic-review", type=int, default=0, help="0 saves samples for manual review; 1..5 invokes the same configured model, not an independent expert")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--llm-cache-only", action="store_true", help="Forbid uncached model completions; no silent fallback")
    parser.add_argument("--resume", action="store_true", help="Explicitly reuse an arm's existing caches; replay counts are not live cost")
    parser.add_argument("--probe", action="store_true", help="Only one lightweight model/primary-search/public-source readiness probe")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.max_live_requests < 0 or not 0 <= args.semantic_review <= 5:
        parser.error("Use positive --timeout, nonnegative --max-live-requests and --semantic-review 0..5")
    return args


def main():
    result = asyncio.run(run(parse_args()))
    if "provider" in result:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
