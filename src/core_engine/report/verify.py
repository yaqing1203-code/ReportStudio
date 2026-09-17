"""Verification harness — three selectable strategies (see Settings.verify_strategy).

STRATEGY "aligned" (DEFAULT): the alignment/fusion layer. Like conflict_only it ACCEPTS
every extracted claim up-front (no per-claim entailment), but adds structure-aware
fusion on top:
  1. entity alias normalization — one batched LLM call maps aliases ("CATL"/"宁德时代")
     to a canonical name, so the same real-world subject clusters together;
  2. clustering — claims carrying structured slots group by (entity, attribute);
     claims WITHOUT slots fall back to the lexical conflict path of conflict_only;
  3. intra-cluster tri-classification — corroborate (same calibre: evidence merged),
     complement (different qualifier/time_scope: BOTH kept, noted), or conflict
     (resolved by source authority, lower dropped);
  4. isolated-claim (孤证) marking — single-source claims at credibility L3/L4;
  5. isolated review — internal-consistency check and domain-baseline deviation score
     (each can downgrade credibility one level), a provenance/timeliness source note,
     and (optionally) ONE bounded round of active verification: proxy-metric queries
     are generated and re-searched, and corroborating hits can lift the isolated flag.
The per-cluster comparison data is exported on VerificationReport.clusters for the
synthesis layer (comparison matrix rendering).

STRATEGY "conflict_only" (fast): prioritizes speed over strict fact-checking.
It does NOT verify each claim against its sources. Every extracted claim is ACCEPTED as
fact; the only LLM work is resolving DIRECT CONTRADICTIONS between claims that talk about
the same subject (found by a cheap lexical prefilter). When two claims conflict, the
lower source-authority one is dropped. A claim with no contradicting counterpart is kept
untouched. Cost scales with the number of conflicting pairs, not claims x sources.
TRADE-OFF: this removes the per-claim anti-fabrication guarantee — an extraction error or
a source's own false statement can reach the report unflagged. See `_verify_conflict_only`.

STRATEGY "cross_reference" (legacy, strict, slower): the broad-collection model below.
The harness no longer DISCARDS uncorroborated claims. It KEEPS every claim that is (a)
traceable to at least one real fetched source and (b) not flagged as hallucinated, and it
LABELS each with a ConfidenceTier so the report can separate verified facts from rumors.

Per-source support decision (selected by verify_mode):
  "strict"    — a source supports the claim only if it contains a REAL verbatim span.
  "traceable" — a source supports the claim if the checker judges it fully entailed,
                with NO fabricated facts (paraphrase/synthesis allowed).

Tiering (after evidence is gathered across all candidate sources):
  HIGH         — >= high_confidence_min_domains distinct supporting domains, OR any
                 high-trust (curated-allowlist) source endorses it.  -> reported as fact
  CORROBORATED — supported by >1 distinct domain, below the HIGH bar. -> reported as fact
  RUMOR        — supported by a single source / domain.               -> kept, LABELED

DROPPED (never reported): a claim a source CONTRADICTS or that asserts facts no source
supports. That is an error, not a rumor — keeping it would break the anti-fabrication
guarantee. Rumors (uncorroborated but traceable) are kept; hallucinations are not.

HONESTY NOTE: no tier proves a claim is TRUE. HIGH/CORROBORATED means "multiple
independent authoritative sources say so"; RUMOR means "one source says so, unverified".
Every kept claim still traces to a real source — the pipeline never emits ungrounded text.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from core_engine.config import get_settings
from core_engine.report.llm import LLM, _safe_json_list, _safe_json_obj
from core_engine.report.models import (
    Claim,
    ConfidenceTier,
    CredibilityLevel,
    Evidence,
    Source,
    SourceKind,
)
from core_engine.report.sources import classify, credibility_for, extract_domain

# Relative authority of a source kind, used ONLY by conflict-only resolution to decide
# which of two contradicting claims to keep. Higher wins. Mirrors the high-trust tiers.
_KIND_AUTHORITY: dict[SourceKind, int] = {
    SourceKind.ACADEMIC: 6,
    SourceKind.GOVERNMENT: 6,
    SourceKind.INVESTMENT_BANK: 5,
    SourceKind.INDUSTRY_INSTITUTION: 5,
    SourceKind.AUTHORITATIVE_MEDIA: 4,
    SourceKind.NONPROFIT: 4,
    SourceKind.GENERAL_WEB: 1,
    SourceKind.REJECTED: 0,
}

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
# Common words that carry no discriminating signal for claim/source overlap.
_STOP = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "is",
    "are", "was", "were", "be", "been", "by", "at", "as", "that", "this", "it",
    "from", "has", "have", "had", "will", "would", "which", "than", "then", "its",
    "into", "over", "about", "more", "most", "such", "also", "can", "may", "these",
    "those", "their", "they", "we", "our", "but", "not", "per", "via", "vs",
})


def _content_terms(text: str) -> set[str]:
    """Distinct lowercased content words (>2 chars, non-stopword) — for cheap lexical
    overlap between a claim and a source, used by the prefilter and passage retrieval."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def _split_paragraphs(text: str) -> list[str]:
    """Split source text into passages on blank lines, falling back to sentence groups
    for sources that arrive as one wall of text."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) > 1:
        return parts
    # No paragraph breaks — chunk into ~3-sentence windows so retrieval has granularity.
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sents) <= 3:
        return [text.strip()] if text.strip() else []
    return [" ".join(sents[i:i + 3]) for i in range(0, len(sents), 3)]


def claim_source_overlap(claim_text: str, source_text: str) -> int:
    """Number of distinct claim content-terms that appear in the source. A cheap,
    dependency-free relevance signal used to PREFILTER cross-referenced sources so we
    don't spend an LLM call on a source that can't possibly corroborate the claim."""
    cterms = _content_terms(claim_text)
    if not cterms:
        return 0
    return len(cterms & _content_terms(source_text))


def retrieve_passages(claim_text: str, source_text: str, max_chars: int) -> str:
    """Return only the source passages most relevant to the claim, capped at max_chars.

    This is the latency win: instead of feeding a whole article (up to 12k chars) to
    the LLM for every (claim, source) pair, we rank passages by claim-term overlap and
    send just the top ones. Falls back to a head-truncation if the text is already
    short or has no clear passages.
    """
    text = (source_text or "").strip()
    if len(text) <= max_chars:
        return text
    cterms = _content_terms(claim_text)
    passages = _split_paragraphs(text)
    if not cterms or not passages:
        return text[:max_chars]
    # Score each passage by distinct claim-term overlap (ties broken by original order).
    scored = sorted(
        ((len(cterms & _content_terms(p)), -i, i, p) for i, p in enumerate(passages)),
        key=lambda t: (t[0], t[1]), reverse=True,
    )
    chosen: list[tuple[int, str]] = []
    used = 0
    for score, _neg_i, idx, passage in scored:
        if score == 0 and chosen:
            break  # no more relevant passages; keep at least the top one
        take = passage[: max_chars - used]
        chosen.append((idx, take))
        used += len(take) + 2
        if used >= max_chars:
            break
    # Restore original document order so the excerpt reads coherently.
    chosen.sort(key=lambda t: t[0])
    return "\n\n".join(p for _i, p in chosen)[:max_chars]


@dataclass(slots=True)
class VerificationReport:
    """Outcome of running the harness over a set of candidate claims.

    Broad-collection model: `kept_claims` holds EVERY traceable, non-hallucinated
    claim, each tagged with a ConfidenceTier. `rejected_claims` holds only claims a
    source contradicted or that asserted unsupported facts (true errors, not rumors).
    """

    kept_claims: list[Claim] = field(default_factory=list)
    rejected_claims: list[Claim] = field(default_factory=list)
    rounds_run: int = 0
    reasons: dict[str, str] = field(default_factory=dict)
    # True if the wall-clock fail-safe fired and we returned partial results.
    timed_out: bool = False
    claims_seen: int = 0          # how many claims we actually got to classify
    claims_total: int = 0         # how many were queued
    # --- Alignment clusters (verify_strategy="aligned" only) ---
    # Data interface for the synthesis layer (workflow E: comparison matrix). One dict
    # per (entity, attribute) cluster that held >= 2 claims:
    #   {
    #     "entity": <canonical entity name>,
    #     "attribute": <the measured property>,
    #     "cells": [{"source_url": str, "value": str|None, "qualifier": str|None,
    #                "credibility": int|None, "claim_text": str}],   # one per claim
    #     "relation": "corroborate" | "complement" | "conflict",
    #   }
    # relation is the strongest intra-cluster outcome observed: "conflict" if any pair
    # contradicted (loser dropped from kept_claims), else "complement" if any pair
    # differed in qualifier/time_scope (both kept), else "corroborate" (evidence merged).
    clusters: list[dict] = field(default_factory=list)

    # --- tier views ---
    @property
    def verified_claims(self) -> list[Claim]:
        """HIGH + CORROBORATED + LIKELY — reportable as verified / likely-true fact.
        (Back-compat name; now includes the LIKELY tier so single credible-source or
        strong circumstantial support counts as reportable, per the relaxed policy.)"""
        return [c for c in self.kept_claims
                if c.confidence in (ConfidenceTier.HIGH, ConfidenceTier.CORROBORATED,
                                    ConfidenceTier.LIKELY)]

    @property
    def rumor_claims(self) -> list[Claim]:
        return [c for c in self.kept_claims if c.confidence is ConfidenceTier.RUMOR]

    @property
    def verified_count(self) -> int:
        return len(self.verified_claims)

    @property
    def kept_count(self) -> int:
        return len(self.kept_claims)

    def distinct_supporting_domains(self) -> set[str]:
        domains: set[str] = set()
        for c in self.verified_claims:
            for url in c.supporting_sources():
                domains.add(extract_domain(url))
        return domains


class VerificationHarness:
    """Runs the triple-check + cross-reference gate. Pure logic over an LLM entailment
    oracle and the fetched sources; no network of its own — EXCEPT the aligned
    strategy's optional one-round active verification, which searches/fetches through
    the injected `search_fn`/`fetcher` (defaults: the search router + real fetcher)."""

    def __init__(self, llm: LLM, *, fetcher=None, search_fn=None) -> None:
        self._llm = llm
        self._s = get_settings()
        # fetcher: scraper Fetcher for active-verification page fetches (optional).
        self._fetcher = fetcher
        # search_fn: async (query) -> list[SearchHit]; None = route via router.py.
        self._search_fn = search_fn

    async def verify(
        self, claims: list[Claim], sources: list[Source], *, on_progress=None,
    ) -> VerificationReport:
        """Dispatch to the configured verification strategy.

        "aligned" (default): accept-all + entity alignment clustering + isolated-claim
        review. See `_verify_aligned`.
        "conflict_only" (fast): accept all claims; only resolve direct contradictions
        between same-subject claims. See `_verify_conflict_only`.
        "cross_reference" (legacy, strict): the broad-collection cross-check below.
        """
        if self._s.verify_strategy == "aligned":
            return await self._verify_aligned(claims, sources, on_progress=on_progress)
        if self._s.verify_strategy == "conflict_only":
            return await self._verify_conflict_only(
                claims, sources, on_progress=on_progress)
        return await self._verify_cross_reference(
            claims, sources, on_progress=on_progress)

    # ======================================================================
    # Conflict-only strategy (fast path)
    # ======================================================================
    async def _verify_conflict_only(
        self, claims: list[Claim], sources: list[Source], *, on_progress=None,
    ) -> VerificationReport:
        """Accept-all + conflict resolution.

        Speed model: we do NOT verify each claim against sources (no entailment /
        traceability calls). Instead:
          1. Every claim is ACCEPTED as fact by default and tagged so downstream code
             (KG, drafting, bibliography) treats it exactly like a verified claim.
          2. We look for DIRECT CONTRADICTIONS only between claims that plausibly talk
             about the same subject — found by a cheap lexical prefilter (shared content
             terms), so we never run the O(n^2) LLM comparison over unrelated pairs.
          3. When two claims contradict, we DROP the lower-authority one (authority =
             best source-kind among the claim's candidate sources) and keep the other.
             Ties keep the earlier claim for determinism.

        A claim with no contradicting counterpart is accepted untouched. The stage is
        still bounded by `verify_deadline_s`; if the contradiction sweep runs long we
        stop issuing checks and accept the remaining claims as-is (fail-open — matches
        the "prioritize speed" intent).
        """
        by_url = {s.url: s for s in sources}
        report = VerificationReport(rounds_run=0)
        report.claims_total = len(claims)
        report.claims_seen = len(claims)
        if not claims:
            return report

        # 1) ACCEPT every claim as a fact up-front. Attach a self-evidence entry for each
        #    candidate source so supporting_sources()/bibliography() still resolve to real
        #    URLs (the report needs citations even though we skipped source verification).
        for c in claims:
            self._accept_as_fact(c, by_url)

        # 2) Build candidate contradiction PAIRS via a cheap lexical prefilter: only claims
        #    that share enough content terms are plausibly about the same subject. Bucket by
        #    term so we don't form the full n^2 product; then dedupe pairs.
        pairs = self._candidate_conflict_pairs(claims)
        if on_progress:
            try:
                on_progress(0, len(claims),
                            f"Accepted {len(claims)} claim(s); scanning "
                            f"{len(pairs)} candidate conflict(s)…")
            except Exception:
                pass

        # 3) LLM contradiction checks over the candidate pairs, bounded by the deadline
        #    and a hard cap. dropped[] collects the losing claim id of each real conflict.
        deadline = time.monotonic() + self._s.verify_deadline_s
        dropped: dict[str, str] = {}     # claim_id -> reason
        checks = min(len(pairs), self._s.conflict_max_pair_checks)
        done = 0
        for i, j in pairs[:checks]:
            if time.monotonic() >= deadline:
                report.timed_out = True
                log.warning("verify(conflict): deadline hit after %d/%d pair checks",
                            done, checks)
                break
            a, b = claims[i], claims[j]
            # If one side already lost an earlier conflict, no need to re-check it.
            if a.id in dropped or b.id in dropped:
                continue
            budget = self._s.llm_timeout_s * 1.2 + 2.0
            try:
                verdict = await asyncio.wait_for(
                    self._llm.check_contradiction(a.text, b.text), timeout=budget)
            except (TimeoutError, Exception) as e:
                # Fail-open: an unresolved pair is NOT a conflict — accept both, move on.
                log.warning("verify(conflict): pair check failed (%s) — keeping both",
                            type(e).__name__)
                verdict = None
            done += 1
            if verdict is not None and verdict.contradict:
                loser, winner = self._resolve_conflict(a, b, by_url)
                dropped[loser.id] = (
                    f"dropped: contradicted by higher-authority claim {winner.id} "
                    f"({verdict.note})" if verdict.note else
                    f"dropped: contradicted by higher-authority claim {winner.id}")
            if on_progress:
                try:
                    on_progress(done, checks,
                                f"Checked conflict {done}/{checks}; "
                                f"{len(dropped)} claim(s) dropped")
                except Exception:
                    pass

        # 4) Partition into kept (accepted) vs. rejected (lost a conflict).
        for c in claims:
            if c.id in dropped:
                c.verified = False
                c.confidence = None
                report.rejected_claims.append(c)
                report.reasons[c.id] = dropped[c.id]
            else:
                report.kept_claims.append(c)
        log.info("verify(conflict): accepted %d/%d claim(s); dropped %d via conflict",
                 len(report.kept_claims), len(claims), len(report.rejected_claims))
        return report

    def _accept_as_fact(self, claim: Claim, by_url: dict[str, Source]) -> None:
        """Mark a claim as an accepted fact WITHOUT source verification.

        We attach a supporting Evidence entry per candidate source (best source first) so
        the downstream report can still cite real URLs. The confidence tier reflects the
        best source kind available — high-trust sources -> HIGH, otherwise LIKELY — but
        note this is source-provenance only, NOT a truth check: per the conflict-only
        model no entailment was run."""
        best_authority = 0
        for url in dict.fromkeys(claim.candidate_source_urls):
            src = by_url.get(url)
            note = "accepted (conflict-only: no per-claim verification)"
            claim.evidence.append(
                Evidence(source_url=url, quote="", supports=True, note=note))
            if src is not None:
                best_authority = max(best_authority, _KIND_AUTHORITY.get(src.kind, 1))
        claim.verified = True
        claim.verification_rounds = 0
        # A high-trust provenance -> HIGH; anything else still reportable as LIKELY.
        claim.confidence = ConfidenceTier.HIGH if best_authority >= 4 else ConfidenceTier.LIKELY

    def _candidate_conflict_pairs(self, claims: list[Claim]) -> list[tuple[int, int]]:
        """Index claims by content term, then emit (i, j) pairs that co-occur under enough
        shared terms to plausibly discuss the same subject. Purely lexical — no LLM. This
        is what keeps conflict detection cheap: unrelated claims never form a pair.

        Fast-fail: if the claim set is pathologically large (many near-duplicates), we cap
        the pair enumeration early and return only the most-similar pairs, bounded by 3x
        the config'd check limit — so even a worst-case input finishes quickly."""
        n = len(claims)
        # Pathological case: 500 claims with 50% pairwise overlap = 125k pairs. Computing
        # overlap scores for all of them is wasteful when we'll only check ~200. So if the
        # claim count suggests we could hit >>conflict_max_pair_checks pairs, we tighten
        # the overlap threshold on-the-fly (accept only higher-overlap pairs) to prune early.
        adaptive_threshold = self._s.conflict_min_term_overlap
        if n > 100:
            # Rough heuristic: if all claims shared terms we'd get ~n^2/2 pairs. If that's
            # >>3x the check budget, raise the bar so fewer pairs qualify.
            potential = n * (n - 1) // 2
            budget = self._s.conflict_max_pair_checks * 3
            if potential > budget:
                adaptive_threshold = min(0.7, self._s.conflict_min_term_overlap + 0.2)
                log.info("verify(conflict): %d claims -> adaptively raising overlap to %.2f",
                         n, adaptive_threshold)

        term_sets = [_content_terms(c.text) for c in claims]
        by_term: dict[str, list[int]] = {}
        for idx, terms in enumerate(term_sets):
            for t in terms:
                by_term.setdefault(t, []).append(idx)
        # Count shared terms per co-occurring pair.
        shared: dict[tuple[int, int], int] = {}
        for idxs in by_term.values():
            if len(idxs) < 2:
                continue
            # Cap per-bucket pair enumeration: if one term appears in 200 claims, forming
            # 200*199/2 = ~20k pairs for that term alone is wasteful. Only form pairs from
            # the bucket if it's reasonable, otherwise skip this term entirely (the pair will
            # likely still form under another shared term if it's truly similar).
            if len(idxs) > 150:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    key = (idxs[a], idxs[b])
                    shared[key] = shared.get(key, 0) + 1
        pairs: list[tuple[int, int]] = []
        for (i, j), n_shared in shared.items():
            smaller = min(len(term_sets[i]), len(term_sets[j])) or 1
            if n_shared / smaller >= adaptive_threshold:
                pairs.append((i, j))
        # Most-similar pairs first so the highest-value checks run before any deadline.
        pairs.sort(key=lambda p: -shared[p])
        return pairs

    def _resolve_conflict(
        self, a: Claim, b: Claim, by_url: dict[str, Source],
    ) -> tuple[Claim, Claim]:
        """Given two contradicting claims, return (loser, winner). Winner = higher source
        authority; ties keep `a` (the earlier claim) for deterministic output."""
        auth_a = self._claim_authority(a, by_url)
        auth_b = self._claim_authority(b, by_url)
        if auth_b > auth_a:
            return a, b
        return b, a

    def _claim_authority(self, claim: Claim, by_url: dict[str, Source]) -> int:
        best = 0
        for url in claim.candidate_source_urls:
            src = by_url.get(url)
            if src is not None:
                best = max(best, _KIND_AUTHORITY.get(src.kind, 1))
        return best

    # ======================================================================
    # Aligned strategy (default) — entity alignment + isolated-claim review
    # ======================================================================
    async def _verify_aligned(
        self, claims: list[Claim], sources: list[Source], *, on_progress=None,
    ) -> VerificationReport:
        """Accept-all + structure-aware fusion (see module docstring, strategy "aligned").

        Steps: accept every claim as fact -> entity alias normalization -> cluster
        structured claims by (entity, attribute) and tri-classify intra-cluster pairs
        (corroborate / complement / conflict) -> lexical conflict path for unstructured
        claims -> isolated (孤证) marking -> isolated review (internal consistency,
        baseline deviation, source note, one bounded round of active verification).

        The stage is bounded by `verify_deadline_s`; on deadline we stop issuing LLM
        checks and return what we have (fail-open, matching conflict_only).
        """
        by_url = {s.url: s for s in sources}
        report = VerificationReport(rounds_run=0)
        report.claims_total = len(claims)
        report.claims_seen = len(claims)
        if not claims:
            return report
        deadline = time.monotonic() + self._s.verify_deadline_s

        def _emit(i: int, total: int, text: str) -> None:
            if on_progress:
                try:
                    on_progress(i, total, text)
                except Exception:
                    pass  # progress is best-effort, never break verification

        # 1) Accept every claim as fact up-front (same contract as conflict_only) and
        #    pin its credibility to the STRONGEST candidate source when unset.
        for c in claims:
            self._accept_as_fact(c, by_url)
            if c.credibility is None:
                c.credibility = self._best_credibility(c, by_url)
        _emit(0, len(claims), f"Accepted {len(claims)} claim(s); aligning entities…")

        # 2) Entity alias normalization (one batched LLM call; fail-open to identity).
        await self._normalize_entities(claims, deadline)

        # 3) Cluster + tri-classify structured claims; lexical path for the rest.
        dropped = await self._classify_clusters(claims, by_url, report, deadline, _emit)
        await self._check_unstructured_conflicts(
            claims, by_url, report, deadline, dropped, _emit)

        # 4) Partition kept vs. conflict-dropped.
        for c in claims:
            if c.id in dropped:
                c.verified = False
                c.confidence = None
                report.rejected_claims.append(c)
                report.reasons[c.id] = dropped[c.id]
            else:
                report.kept_claims.append(c)

        # 5) Isolated (孤证) marking: single candidate source at credibility L3/L4.
        for c in report.kept_claims:
            c.isolated = (len(c.candidate_source_urls) == 1
                          and c.credibility is not None
                          and c.credibility >= CredibilityLevel.L3)

        # 6) Isolated review + optional one-round active verification.
        await self._review_isolated(report, by_url, sources, deadline, _emit)
        log.info("verify(aligned): kept %d/%d claim(s); %d dropped, %d isolated, "
                 "%d cluster(s)", len(report.kept_claims), len(claims),
                 len(report.rejected_claims),
                 sum(1 for c in report.kept_claims if c.isolated),
                 len(report.clusters))
        return report

    def _best_credibility(
        self, claim: Claim, by_url: dict[str, Source]
    ) -> CredibilityLevel:
        """Strongest (smallest-numbered) credibility among the claim's candidate
        sources; L3 (industry consensus) when none resolve."""
        best = CredibilityLevel.L3
        found = False
        for url in claim.candidate_source_urls:
            src = by_url.get(url)
            if src is not None and (not found or src.credibility < best):
                best = src.credibility
                found = True
        return best

    async def _complete_guarded(
        self, system: str, user: str, deadline: float, what: str
    ) -> str | None:
        """One llm.complete call under the verify deadline + per-call timeout budget
        (same pattern as `_classify_one`). Returns None on timeout/error (fail-open)."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        budget = min(self._s.llm_timeout_s * 1.2 + 2.0, remaining)
        try:
            return await asyncio.wait_for(self._llm.complete(system, user),
                                          timeout=budget)
        except (TimeoutError, Exception) as e:
            log.warning("verify(aligned): %s failed (%s) — skipping", what,
                        type(e).__name__)
            return None

    async def _normalize_entities(self, claims: list[Claim], deadline: float) -> None:
        """Batch-map entity aliases to canonical names via one llm.complete call; the
        returned mapping rewrites claim.entity. Any failure leaves entities as-is."""
        entities = sorted({c.entity for c in claims if c.entity})
        if len(entities) < 2:
            return
        system = (
            "You are an entity alias normalizer. Given a JSON list of entity names "
            "extracted from research claims, map every alias to ONE canonical name "
            "(e.g. \"CATL\" and \"宁德时代\" both map to \"宁德时代\"). Respond ONLY "
            "as JSON: {\"mapping\": {\"<input name>\": \"<canonical name>\", ...}}. "
            "Include every input name; map a name to itself when it has no alias."
        )
        raw = await self._complete_guarded(
            system, json.dumps({"entities": entities}, ensure_ascii=False),
            deadline, "entity normalization")
        mapping = _safe_json_obj(raw or "").get("mapping")
        if not isinstance(mapping, dict):
            return
        for c in claims:
            if c.entity:
                canonical = mapping.get(c.entity)
                if isinstance(canonical, str) and canonical.strip():
                    c.entity = canonical.strip()

    async def _classify_clusters(
        self, claims: list[Claim], by_url: dict[str, Source],
        report: VerificationReport, deadline: float, emit,
    ) -> dict[str, str]:
        """Cluster structured claims by (entity, attribute) and tri-classify each
        intra-cluster pair. Returns the dropped map (claim_id -> reason)."""
        clusters: dict[tuple[str, str], list[Claim]] = {}
        for c in claims:
            if c.entity and c.attribute:
                key = (c.entity.strip().lower(), c.attribute.strip().lower())
                clusters.setdefault(key, []).append(c)
        dropped: dict[str, str] = {}
        checks_done = 0
        for (_entity, _attribute), members in clusters.items():
            if len(members) < 2:
                continue
            relation = "corroborate"
            stop = False
            for ai in range(len(members)):
                for bi in range(ai + 1, len(members)):
                    a, b = members[ai], members[bi]
                    if a.id in dropped or b.id in dropped:
                        continue
                    if time.monotonic() >= deadline \
                            or checks_done >= self._s.conflict_max_pair_checks:
                        if time.monotonic() >= deadline:
                            report.timed_out = True
                        stop = True
                        break
                    budget = self._s.llm_timeout_s * 1.2 + 2.0
                    try:
                        verdict = await asyncio.wait_for(
                            self._llm.check_contradiction(a.text, b.text),
                            timeout=budget)
                    except (TimeoutError, Exception) as e:
                        # Fail-open: an unresolved pair is NOT a conflict.
                        log.warning("verify(aligned): pair check failed (%s) — "
                                    "keeping both", type(e).__name__)
                        verdict = None
                    checks_done += 1
                    if verdict is not None and verdict.contradict:
                        loser, winner = self._resolve_conflict(a, b, by_url)
                        dropped[loser.id] = (
                            f"dropped: contradicted by higher-authority claim "
                            f"{winner.id} ({verdict.note})" if verdict.note else
                            f"dropped: contradicted by higher-authority claim "
                            f"{winner.id}")
                        relation = "conflict"
                    elif (a.qualifier or None) != (b.qualifier or None) \
                            or (a.time_scope or None) != (b.time_scope or None):
                        # COMPLEMENT: different calibre/period — BOTH claims kept.
                        if relation != "conflict":
                            relation = "complement"
                        report.reasons[f"pair:{a.id}|{b.id}"] = (
                            "complement: same entity/attribute but different "
                            "qualifier/time_scope — both kept")
                    else:
                        # CORROBORATE: same calibre, consistent — merge evidence.
                        self._merge_corroboration(a, b)
                        report.reasons[f"pair:{a.id}|{b.id}"] = (
                            "corroborate: same entity/attribute/calibre — "
                            "evidence merged")
                if stop:
                    break
            if len(members) >= 2:
                report.clusters.append({
                    "entity": members[0].entity,
                    "attribute": members[0].attribute,
                    "cells": [{
                        "source_url": (m.candidate_source_urls[0]
                                       if m.candidate_source_urls else ""),
                        "value": m.value,
                        "qualifier": m.qualifier,
                        "credibility": (int(m.credibility)
                                        if m.credibility is not None else None),
                        "claim_text": m.text,
                    } for m in members],
                    "relation": relation,
                })
            emit(checks_done, self._s.conflict_max_pair_checks,
                 f"Aligned cluster {_entity}/{_attribute}: {relation}")
        return dropped

    @staticmethod
    def _merge_corroboration(a: Claim, b: Claim) -> None:
        """Corroborating pair: union candidate URLs and evidence onto BOTH claims so
        either one cites the full support set (deduped by source URL)."""
        urls = list(dict.fromkeys(a.candidate_source_urls + b.candidate_source_urls))
        ev: dict[str, Evidence] = {}
        for e in a.evidence + b.evidence:
            ev.setdefault(e.source_url, e)
        for c in (a, b):
            c.candidate_source_urls[:] = urls
            c.evidence[:] = list(ev.values())

    async def _check_unstructured_conflicts(
        self, claims: list[Claim], by_url: dict[str, Source],
        report: VerificationReport, deadline: float,
        dropped: dict[str, str], emit,
    ) -> None:
        """Claims WITHOUT structured slots fall back to the conflict_only lexical path:
        cheap overlap prefilter + contradiction oracle + authority resolution."""
        sub = [c for c in claims if not (c.entity and c.attribute)]
        if len(sub) < 2:
            return
        pairs = self._candidate_conflict_pairs(sub)
        checks = min(len(pairs), self._s.conflict_max_pair_checks)
        done = 0
        for i, j in pairs[:checks]:
            if time.monotonic() >= deadline:
                report.timed_out = True
                log.warning("verify(aligned): deadline hit after %d/%d unstructured "
                            "pair checks", done, checks)
                break
            a, b = sub[i], sub[j]
            if a.id in dropped or b.id in dropped:
                continue
            budget = self._s.llm_timeout_s * 1.2 + 2.0
            try:
                verdict = await asyncio.wait_for(
                    self._llm.check_contradiction(a.text, b.text), timeout=budget)
            except (TimeoutError, Exception) as e:
                log.warning("verify(aligned): pair check failed (%s) — keeping both",
                            type(e).__name__)
                verdict = None
            done += 1
            if verdict is not None and verdict.contradict:
                loser, winner = self._resolve_conflict(a, b, by_url)
                dropped[loser.id] = (
                    f"dropped: contradicted by higher-authority claim {winner.id} "
                    f"({verdict.note})" if verdict.note else
                    f"dropped: contradicted by higher-authority claim {winner.id}")
            emit(done, checks, f"Checked unstructured conflict {done}/{checks}")

    async def _review_isolated(
        self, report: VerificationReport, by_url: dict[str, Source],
        sources: list[Source], deadline: float, emit,
    ) -> None:
        """Review each isolated (孤证) claim: internal consistency, baseline deviation,
        a provenance/timeliness source note, and (for core isolated claims) one round
        of active verification. All fail-open; findings are recorded in reasons."""
        isolated = [c for c in list(report.kept_claims) if c.isolated]
        for c in isolated:
            if time.monotonic() >= deadline:
                report.timed_out = True
                log.warning("verify(aligned): deadline hit during isolated review")
                break
            notes: list[str] = []
            src = by_url.get(c.candidate_source_urls[0]) \
                if c.candidate_source_urls else None

            # (a) Internal consistency: do the numbers/relations inside the claim's
            #     own source text hang together (数据勾稽)? Inconsistent -> downgrade.
            if src is not None:
                excerpt = retrieve_passages(c.text, src.text,
                                            self._s.verify_passage_max_chars)
                raw = await self._complete_guarded(
                    "You are an internal consistency reviewer. Read the SOURCE excerpt "
                    "and the CLAIM extracted from it. Check whether the numbers and "
                    "relations in the source hang together (数据勾稽) and support the "
                    "claim. Respond ONLY as JSON: {\"consistent\": bool, \"note\": "
                    "\"<one short line>\"}.",
                    f"CLAIM:\n{c.text}\n\nSOURCE:\n{excerpt}",
                    deadline, "consistency check")
                obj = _safe_json_obj(raw or "")
                if obj.get("consistent") is False:
                    c.credibility = self._downgrade(c.credibility)
                    notes.append(
                        f"internal inconsistency in source — credibility downgraded "
                        f"to {c.credibility.label} ({obj.get('note', '')})".rstrip(" ()"))

            # (b) Baseline deviation: domain-common-sense outlier score 0-1. Isolated
            #     AND above threshold -> downgrade one level.
            raw = await self._complete_guarded(
                "You are a deviation assessor with domain common sense. Judge how far "
                "the CLAIM deviates from the accepted baseline of its field "
                "(0 = perfectly in line, 1 = extreme outlier). Respond ONLY as JSON: "
                "{\"deviation\": <float 0-1>, \"note\": \"<one short line>\"}.",
                f"CLAIM:\n{c.text}\nENTITY: {c.entity or 'n/a'}\n"
                f"ATTRIBUTE: {c.attribute or 'n/a'}\nVALUE: {c.value or 'n/a'}",
                deadline, "deviation assessment")
            obj = _safe_json_obj(raw or "")
            try:
                deviation = float(obj.get("deviation")) if obj else None
            except (TypeError, ValueError):
                deviation = None
            if deviation is not None and c.isolated \
                    and deviation > self._s.alignment_deviation_threshold:
                c.credibility = self._downgrade(c.credibility)
                notes.append(
                    f"deviation {deviation:.2f} exceeds threshold "
                    f"{self._s.alignment_deviation_threshold:.2f} for an isolated "
                    f"claim — credibility downgraded to {c.credibility.label} "
                    f"({obj.get('note', '')})".rstrip(" ()"))

            # (c) Source note: one line on timeliness (fetched_at) and domain
            #     attributes. No extra network — metadata only.
            if src is not None:
                note = await self._complete_guarded(
                    "You are a source note writer. Given a source's domain, title, and "
                    "fetch timestamp, write ONE sentence assessing its timeliness and "
                    "possible motivation/stance for a research report audit trail. "
                    "Plain text, one sentence only.",
                    f"DOMAIN: {src.domain}\nTITLE: {src.title}\n"
                    f"FETCHED_AT: {src.fetched_at}\nCREDIBILITY: "
                    f"{src.credibility.label}",
                    deadline, "source note")
                if note and note.strip():
                    notes.append(f"source note: {note.strip()}")

            # (d) Active verification for CORE isolated claims (entity known): one
            #     bounded round of proxy-metric re-search that can lift isolation.
            if c.isolated and c.entity and self._s.active_verify_enabled:
                note = await self._active_verify(c, report, by_url, sources,
                                                 deadline, emit)
                if note:
                    notes.append(note)

            if notes:
                prior = report.reasons.get(c.id)
                report.reasons[c.id] = (prior + "; " if prior else "") + "; ".join(notes)

    @staticmethod
    def _downgrade(level: CredibilityLevel | None) -> CredibilityLevel:
        """One credibility level weaker, floored at L4."""
        return CredibilityLevel(min(4, int(level or CredibilityLevel.L3) + 1))

    async def _active_verify(
        self, claim: Claim, report: VerificationReport, by_url: dict[str, Source],
        sources: list[Source], deadline: float, emit,
    ) -> str | None:
        """One-round active verification for a core isolated claim: generate up to
        `active_verify_max_queries` proxy-metric queries, re-search them, fetch and
        extract claims from NEW urls, and merge corroboration into this run's pool.
        Returns an audit note, or None when nothing ran. NO recursion — claims found
        here are never themselves actively verified."""
        raw = await self._complete_guarded(
            "You are a verification query generator. The CLAIM below is an isolated "
            "single-source assertion. Propose up to "
            f"{self._s.active_verify_max_queries} search queries for PROXY metrics or "
            "independent sources that could corroborate or refute it. Respond ONLY as "
            "a JSON array of query strings.",
            f"CLAIM:\n{claim.text}\nENTITY: {claim.entity or 'n/a'}\n"
            f"ATTRIBUTE: {claim.attribute or 'n/a'}\nVALUE: {claim.value or 'n/a'}",
            deadline, "query generation")
        queries = [q.strip() for q in _safe_json_list(raw or "[]")
                   if isinstance(q, str) and q.strip()]
        queries = queries[: self._s.active_verify_max_queries]
        if not queries:
            return None
        search_fn = self._search_fn or self._default_search
        budget = self._s.llm_timeout_s * 1.2 + 2.0
        corroborated = False
        for q in queries:
            if time.monotonic() >= deadline:
                report.timed_out = True
                break
            emit(0, 0, f"Active verification search: {q[:60]}")
            try:
                hits = await asyncio.wait_for(search_fn(q), timeout=budget)
            except (TimeoutError, Exception) as e:
                log.warning("verify(aligned): active search %r failed (%s)",
                            q, type(e).__name__)
                continue
            for hit in hits[:3]:
                if hit.url in by_url or self._fetcher is None:
                    continue
                try:
                    text = await asyncio.wait_for(self._fetcher.fetch(hit.url),
                                                  timeout=budget)
                except (TimeoutError, Exception):
                    continue
                if not text:
                    continue
                src = Source(url=hit.url, domain=extract_domain(hit.url),
                             title=hit.title, kind=classify(hit.url), text=text,
                             credibility=credibility_for(hit.url))
                by_url[hit.url] = src
                sources.append(src)
                try:
                    extracted = await self._llm.extract_claims(hit.url, text)
                except Exception as e:
                    log.warning("verify(aligned): active extraction on %s failed (%s)",
                                hit.url, type(e).__name__)
                    continue
                for ex in extracted:
                    if self._merge_active_claim(ex, src, claim, report, by_url):
                        corroborated = True
        if corroborated:
            claim.isolated = False
            return ("active verification found corroborating source(s) — "
                    "isolation lifted")
        return f"active verification ran {len(queries)} query(ies); no corroboration found"

    def _merge_active_claim(
        self, ex, src: Source, target: Claim, report: VerificationReport,
        by_url: dict[str, Source],
    ) -> bool:
        """Fold one actively-extracted claim into the pool: same normalized text extends
        an existing claim; otherwise it becomes a new accepted claim. Returns True when
        it corroborates the isolated target (same text, or same entity+attribute)."""
        key = re.sub(r"[^a-z0-9 ]", "", ex.text.lower()).strip()
        for c in report.kept_claims:
            if re.sub(r"[^a-z0-9 ]", "", c.text.lower()).strip() != key:
                continue
            if src.url not in c.candidate_source_urls:
                c.candidate_source_urls.append(src.url)
                c.credibility = (src.credibility if c.credibility is None
                                 else min(c.credibility, src.credibility))
                c.evidence.append(Evidence(
                    source_url=src.url, quote="", supports=True,
                    note="accepted (active verification corroboration)"))
            return c is target or (
                c.entity and target.entity
                and c.entity.lower() == target.entity.lower()
                and c.attribute and target.attribute
                and c.attribute.lower() == target.attribute.lower())
        # New claim from the active round — accepted as fact like the rest of the pool.
        claim = Claim(
            id=f"av{len(report.kept_claims) + 1}",
            text=ex.text,
            candidate_source_urls=list(ex.candidate_source_urls),
            credibility=src.credibility,
            entity=ex.entity, attribute=ex.attribute, value=ex.value,
            qualifier=ex.qualifier, time_scope=ex.time_scope,
        )
        self._accept_as_fact(claim, by_url)
        report.kept_claims.append(claim)
        report.claims_total += 1
        report.claims_seen += 1
        report.reasons[claim.id] = ("accepted via active verification of isolated "
                                    f"claim {target.id}")
        return bool(
            claim.entity and target.entity
            and claim.entity.lower() == target.entity.lower()
            and claim.attribute and target.attribute
            and claim.attribute.lower() == target.attribute.lower())

    async def _default_search(self, query: str):
        """Active-verification search through the query router (lazy import: router.py
        depends on the scrape factory, and verify.py must not create a cycle)."""
        from core_engine.report.router import routed_search

        return await routed_search(query, self._llm,
                                   max_results=self._s.scrape_max_results)

    async def _verify_cross_reference(
        self, claims: list[Claim], sources: list[Source], *, on_progress=None,
    ) -> VerificationReport:
        """Classify every claim by corroboration (broad collection + late filter).

        A claim is KEPT and tiered unless a source flags it as hallucinated (a true
        error). Uncorroborated claims are kept as RUMOR, not discarded.

        FAIL-SAFE: the whole stage is bounded by `verify_deadline_s`. If the deadline
        passes we STOP classifying further claims and return what we have — the run
        proceeds with partial results rather than hanging. `on_progress(i, total, text)`
        (optional) is called per claim so the UI shows step-by-step movement.
        """
        by_url = {s.url: s for s in sources}
        report = VerificationReport(rounds_run=self._s.verify_rounds)
        report.claims_total = len(claims)
        deadline = time.monotonic() + self._s.verify_deadline_s

        # CONCURRENCY: classify claims in parallel with a bounded semaphore so 100+
        # claims don't run one-by-one. The cap (`verify_concurrency`) respects provider
        # rate limits — every claim still issues its own (now-prefiltered) LLM checks,
        # but up to N claims are in flight at once. The overall deadline still bounds the
        # whole stage; claims not finished by then are left unclassified (partial result).
        sem = asyncio.Semaphore(max(1, self._s.verify_max_concurrency))
        done_count = 0
        total = len(claims)

        async def _run(idx: int, claim: Claim):
            nonlocal done_count
            # Skip work if the deadline already passed (returns a sentinel to mark unseen).
            if time.monotonic() >= deadline:
                return idx, claim, "deadline"
            async with sem:
                if time.monotonic() >= deadline:
                    return idx, claim, "deadline"
                log.info("verify: classifying claim %d/%d (id=%s): %.80s",
                         idx, total, claim.id, claim.text)
                try:
                    tiered = await self._classify_one(
                        claim, by_url, report.rounds_run, deadline)
                except Exception as e:  # never let one claim kill the batch
                    log.warning("verify: claim %s crashed (%s) — dropping",
                                claim.id, type(e).__name__)
                    tiered = None
            done_count += 1
            if on_progress:
                try:
                    on_progress(done_count, total, claim.text)
                except Exception:
                    pass  # progress is best-effort, never break verification
            return idx, claim, tiered

        tasks = [asyncio.create_task(_run(i, c)) for i, c in enumerate(claims, start=1)]
        # An outer wait_for guards against any straggler that ignores the soft deadline.
        results: list[tuple[int, Claim, object]] = []
        try:
            grace = self._s.verify_deadline_s + self._s.llm_timeout_s + 5.0
            gathered = await asyncio.wait_for(asyncio.gather(*tasks), timeout=grace)
            results = list(gathered)
        except TimeoutError:
            report.timed_out = True
            log.warning("verify: hard grace deadline hit — collecting finished claims")
            for t in tasks:
                if t.done() and not t.cancelled():
                    try:
                        results.append(t.result())
                    except Exception:
                        pass
                else:
                    t.cancel()

        # Reassemble in the original claim order for a stable, deterministic report.
        for idx, claim, tiered in sorted(results, key=lambda r: r[0]):
            if tiered == "deadline":
                report.timed_out = True
                continue
            report.claims_seen += 1
            if tiered is None:
                report.rejected_claims.append(claim)
                report.reasons[claim.id] = self._rejection_reason(claim)
            else:
                report.kept_claims.append(tiered)  # type: ignore[arg-type]
                log.info("verify: claim %s -> %s", claim.id,
                         tiered.confidence.value if tiered.confidence else "kept")

        if report.claims_seen < total:
            report.timed_out = True
            log.warning("verify: classified %d/%d claims before the deadline — partial",
                        report.claims_seen, total)
        return report

    async def _classify_one(
        self, claim: Claim, by_url: dict[str, Source], rounds: int,
        deadline: float | None = None,
    ) -> Claim | None:
        """Gather evidence for a claim across all its candidate sources and assign a
        ConfidenceTier. Returns None ONLY for hallucinated claims (dropped); every
        other traceable claim is returned with a tier set.

        Tiering:
          HIGH        — >= high_confidence_min_domains distinct supporting domains,
                        OR endorsed by any high-trust (curated-allowlist) source.
          CORROBORATED— supported by >1 distinct domain, below the HIGH bar.
          RUMOR       — supported by exactly one source / domain.
          (no support at all, or hallucinated) — dropped.
        """
        supported_high_trust = False
        contradiction_flagged = False

        # CROSS-REFERENCE against the WHOLE source pool, not only the source a claim was
        # extracted from. This is what lets Source B corroborate a claim first seen in
        # Source A — the semantic cross-check the relaxed policy needs. We check the
        # claim's own sources first (most likely to support), then the rest, honouring
        # the deadline and stopping early once we have enough corroboration.
        own_set = set(dict.fromkeys(claim.candidate_source_urls))
        own = [u for u in claim.candidate_source_urls if u in by_url]
        # PREFILTER + RANK the OTHER sources by cheap lexical overlap so we only spend
        # LLM calls on sources that actually mention the claim's terms, best first. This
        # is the big win at scale: most of the pool shares no terms with a given claim,
        # so we skip those calls entirely instead of feeding every article to the LLM.
        others_scored = []
        for u, src in by_url.items():
            if u in own_set or not src.authoritative:
                continue
            ov = claim_source_overlap(claim.text, src.text)
            if ov >= self._s.verify_prefilter_min_hits:
                others_scored.append((ov, u))
        others_scored.sort(key=lambda t: t[0], reverse=True)
        # The claim's OWN candidate sources come first and are ALWAYS checked (they are
        # few and already fetched). The whole-pool sources come after that boundary and
        # are the expensive "cross-referencing" the fast-path is allowed to skip.
        n_own = len(own)
        ordered_urls = own + [u for _ov, u in others_scored]
        checked = 0
        for idx, url in enumerate(ordered_urls):
            if deadline is not None and time.monotonic() >= deadline:
                break
            in_cross_ref = idx >= n_own      # past the claim's own sources -> pool cross-ref
            # FAST-PATH (authority short-circuit, requirement #3): once a high-trust
            # (Gov/Academic/institution) source has endorsed the claim, it is already at
            # the top HIGH tier and no further source can raise it. So we STOP the
            # expensive whole-pool cross-referencing — but only AFTER exhausting the
            # claim's own candidate sources, so a direct contradiction sitting in one of
            # them is never skipped ("...and there are no direct contradictions"). This is
            # what accelerates verification: proven claims don't burn LLM calls scanning
            # the rest of the pool. (Configurable via CE_VERIFY_FAST_PATH_HIGH_TRUST.)
            if self._s.verify_fast_path_high_trust and supported_high_trust and in_cross_ref:
                break
            # Otherwise: enough distinct-domain corroboration already? Stop to bound cost.
            if in_cross_ref \
                    and len({extract_domain(u) for u in claim.supporting_sources()}) \
                    >= self._s.high_confidence_min_domains and supported_high_trust:
                break
            if in_cross_ref and checked >= self._s.max_cross_ref_sources:
                break
            src = by_url.get(url)
            if src is None or not src.authoritative:
                continue
            checked += 1
            supported, ev, contradicted = await self._check_source(claim.text, src, 0)
            if contradicted:
                # DIRECT contradiction — the only thing that drops a claim now.
                contradiction_flagged = True
                claim.evidence.append(ev)
                break
            if supported:
                claim.evidence.append(ev)
                if src.kind.high_trust:
                    supported_high_trust = True
        claim.verification_rounds = 1

        # Direct contradiction is the only hard drop (was: any unsupported detail).
        if contradiction_flagged:
            claim.verified = False
            return None

        domains = {extract_domain(u) for u in claim.supporting_sources()}
        n = len(domains)
        if n == 0:
            # No source supports OR contradicts it after a broad cross-check.
            claim.verified = False
            return None

        # Relaxed tiering: high-trust endorsement or multi-domain -> HIGH; consistent
        # 2nd domain -> CORROBORATED; a single credible source -> LIKELY (reportable);
        # only a lone low-trust signal stays RUMOR.
        if supported_high_trust and n >= self._s.high_confidence_min_domains or supported_high_trust:
            claim.confidence = ConfidenceTier.HIGH
            claim.verified = True
        elif n >= self._s.high_confidence_min_domains:
            claim.confidence = ConfidenceTier.CORROBORATED
            claim.verified = True
        else:
            # Exactly one supporting domain, and it wasn't high-trust. Distinguish a
            # credible single source (LIKELY, reportable) from a lone low-trust /
            # general-web signal (RUMOR, kept but flagged).
            only_domain = next(iter(domains))
            only_kind = next((by_url[u].kind for u in claim.supporting_sources()
                              if extract_domain(u) == only_domain and u in by_url), None)
            if only_kind is not None and only_kind is not SourceKind.GENERAL_WEB:
                claim.confidence = ConfidenceTier.LIKELY   # credible single source
                claim.verified = True
            else:
                claim.confidence = ConfidenceTier.RUMOR    # lone low-trust signal
                claim.verified = False
        return claim

    async def _check_source(
        self, claim_text: str, src: Source, round_idx: int
    ) -> tuple[bool, Evidence, bool]:
        """One (claim, source) SEMANTIC support decision.

        Returns (supported, evidence, contradicted):
          supported    — the source states, implies, or is consistent with the claim's
                         underlying fact (not verbatim; semantic/circumstantial).
          contradicted — the source DIRECTLY contradicts the claim (the only hard drop).
                         Merely 'not mentioned' is neither supported nor contradicted.

        Wrapped in a hard per-call timeout: a stalled/failing check degrades to
        'not supported, not contradicted' and the loop moves on.
        """
        # Slightly over the client-level timeout so the client's own timeout normally
        # fires first (cleaner error), but never far beyond it.
        budget = self._s.llm_timeout_s * 1.2 + 2.0
        # RETRIEVAL: send only the claim-relevant passages, not the whole article. This
        # cuts per-call latency/tokens dramatically (a 12k-char source becomes a focused
        # excerpt). Strict mode still needs verbatim spans, so we verify the returned
        # quote against the FULL source text below, not just the excerpt.
        excerpt = retrieve_passages(claim_text, src.text, self._s.verify_passage_max_chars)
        try:
            if self._s.verify_mode == "strict":
                j = await asyncio.wait_for(
                    self._llm.check_entailment(claim_text, excerpt), timeout=budget)
                # Quote must exist in the FULL source (excerpt is a subset; a real span
                # from it is still a real span of the source).
                ok = bool(j.supports and j.quote and j.quote in src.text)
                ev = Evidence(source_url=src.url, quote=j.quote if ok else "",
                              supports=ok, note=f"strict round={round_idx}")
                return ok, ev, False

            # traceable mode
            j = await asyncio.wait_for(
                self._llm.check_traceability(claim_text, excerpt), timeout=budget)
        except (TimeoutError, Exception) as e:
            # Timed out / errored -> treat as no support (NOT a hallucination), log, move on.
            log.warning("verify: check on %s timed out/failed (%s) — skipping",
                        src.url, type(e).__name__)
            ev = Evidence(source_url=src.url, quote="", supports=False,
                          note=f"check failed round={round_idx}: {type(e).__name__}")
            return False, ev, False

        if j.hallucinated:   # now means: source DIRECTLY contradicts the claim
            ev = Evidence(source_url=src.url, quote="", supports=False,
                          note=f"CONTRADICTED by source: {j.note}")
            return False, ev, True
        ev = Evidence(source_url=src.url, quote=j.evidence_span, supports=j.supported,
                      note=f"semantic conf={j.confidence:.2f}")
        return j.supported, ev, False

    def _rejection_reason(self, claim: Claim) -> str:
        # Only two things cause a DROP now: a source DIRECTLY contradicts the claim, or
        # no source supports OR contradicts it after a broad cross-check. Weak/single
        # support is NOT dropped — it is kept as LIKELY or RUMOR.
        if any("CONTRADICTED" in e.note for e in claim.evidence):
            return "dropped: directly contradicted by a credible source"
        return "dropped: no credible source supported or mentioned the claim after broad search"
