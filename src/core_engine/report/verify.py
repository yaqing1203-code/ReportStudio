"""Verification harness — two selectable strategies (see Settings.verify_strategy).

STRATEGY "conflict_only" (DEFAULT, fast): prioritizes speed over strict fact-checking.
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

import logging
from dataclasses import dataclass, field

import asyncio
import re
import time

from core_engine.config import get_settings
from core_engine.report.llm import LLM
from core_engine.report.models import (
    Claim,
    ConfidenceTier,
    Evidence,
    Source,
    SourceKind,
)
from core_engine.report.sources import extract_domain

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
    oracle and the fetched sources; no network of its own."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self._s = get_settings()

    async def verify(
        self, claims: list[Claim], sources: list[Source], *, on_progress=None,
    ) -> VerificationReport:
        """Dispatch to the configured verification strategy.

        "conflict_only" (default, fast): accept all claims; only resolve direct
        contradictions between same-subject claims. See `_verify_conflict_only`.
        "cross_reference" (legacy, strict): the broad-collection cross-check below.
        """
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
            except (asyncio.TimeoutError, Exception) as e:
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
        except asyncio.TimeoutError:
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
        if supported_high_trust and n >= self._s.high_confidence_min_domains:
            claim.confidence = ConfidenceTier.HIGH
            claim.verified = True
        elif supported_high_trust:
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
        except (asyncio.TimeoutError, Exception) as e:
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
