"""Source filter — the allowlist gate (EXPANDED for industry reports).

Authoritative categories, in priority order:
  1. GOVERNMENT           — .gov / ministries / IGOs / .edu / .mil (by suffix)
  2. INVESTMENT_BANK       — IBs, research houses, market-data providers (TAM/SAM/SOM,
                            market share, industry-chain data the charts need)
  3. INDUSTRY_INSTITUTION  — trade associations / standards bodies / IGO research
  4. NONPROFIT             — NGO / non-profit research orgs with methodological rigor
  5. AUTHORITATIVE_MEDIA   — allowlisted mainstream outlets with editorial standards

Design stance: still a hard ALLOWLIST, not a blocklist. A domain is authoritative
only if it positively matches one of the allowlists above. Everything else is
REJECTED by default — we would rather drop a borderline-good source than admit a bad
one. A separate tabloid/gossip DENYLIST (小道媒体) is rejected unconditionally, even
if it would otherwise look like general media.

Nothing here trusts the LLM: classification is pure string logic on the registrable
domain, so it is deterministic and auditable.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from core_engine.config import get_settings
from core_engine.report.models import SearchHit, SourceKind

# User-generated / social / low-signal hosts. With the GENERAL_WEB tier admitting
# most sites, this denylist becomes the real gatekeeper, so it is broader: social
# media, forums, UGC blog platforms, Q&A, content farms, and aggregators. These are
# always rejected regardless of tier.
_HARD_DENY_HOSTS = frozenset({
    # social
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "linkedin.com", "youtube.com", "pinterest.com", "reddit.com", "threads.net",
    # UGC blog / publishing platforms
    "medium.com", "substack.com", "blogspot.com", "wordpress.com", "tumblr.com",
    "wixsite.com", "weebly.com", "ghost.io",
    # Q&A / forums / wikis (tertiary, not primary)
    "quora.com", "stackexchange.com", "stackoverflow.com", "wikipedia.org",
    "wikihow.com", "answers.com", "fandom.com",
    # content farms / aggregators
    "pinterest.com", "slideshare.net", "scribd.com", "coursehero.com",
})


def extract_domain(url: str) -> str:
    """Return the lowercased host without a leading 'www.'. Empty string if unparseable."""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _registrable(host: str) -> str:
    """Best-effort registrable domain (last two labels). Good enough for allowlist
    membership on the media list; government matching uses full-suffix logic instead."""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _in(host: str, allowlist: tuple[str, ...]) -> bool:
    """True if the host or its registrable domain is in an exact-host allowlist."""
    return host in allowlist or _registrable(host) in allowlist


def classify(url: str) -> SourceKind:
    """Classify a URL against the allowlists. REJECTED unless it positively matches.

    Order matters: the tabloid denylist and the user-generated hard-deny set win over
    every allowlist, then government (by suffix), then the exact-host allowlists in
    priority order. A host that matches nothing is REJECTED.
    """
    s = get_settings()
    host = extract_domain(url)
    if not host:
        return SourceKind.REJECTED

    # Hard deny wins over everything: user-generated hosts + tabloid/gossip (小道媒体).
    if host in _HARD_DENY_HOSTS or _registrable(host) in _HARD_DENY_HOSTS:
        return SourceKind.REJECTED
    if _in(host, s.tabloid_denylist):
        return SourceKind.REJECTED

    # Academic / peer-reviewed databases FIRST (highest authority for factual/scientific
    # claims). Checked before the .gov suffix so e.g. ncbi.nlm.nih.gov reads as ACADEMIC.
    if _in(host, s.academic_allowlist):
        return SourceKind.ACADEMIC

    # Government / IGO / .edu / .mil by suffix. We check dotted suffixes against the
    # host with a leading dot so 'notagov.com' cannot match '.gov'.
    dotted = "." + host
    for suffix in s.gov_domain_suffixes:
        if dotted.endswith(suffix):
            return SourceKind.GOVERNMENT

    # Exact-host allowlists, in priority order.
    if _in(host, s.investment_bank_allowlist):
        return SourceKind.INVESTMENT_BANK
    if _in(host, s.industry_institution_allowlist):
        return SourceKind.INDUSTRY_INSTITUTION
    if _in(host, s.nonprofit_allowlist):
        return SourceKind.NONPROFIT
    if _in(host, s.media_allowlist):
        return SourceKind.AUTHORITATIVE_MEDIA

    # TIERED TRUST: a host that matched no curated allowlist but survived the denylist
    # is admitted as GENERAL_WEB (when enabled). This is what stops valid topics from
    # starving on the strict list — real search results (trade press, IEEE, uni news,
    # analyst sites) become usable, at a LOWER trust weight than the curated tiers.
    # Grounding is preserved elsewhere: every claim still traces to a fetched source.
    if s.allow_general_web:
        return SourceKind.GENERAL_WEB

    return SourceKind.REJECTED


def filter_hits(hits: list[SearchHit]) -> tuple[list[SearchHit], list[SearchHit]]:
    """Split search hits into (authoritative, rejected). Dedupe authoritative by domain
    is NOT done here — we want multiple pages from one gov site — but we drop exact URL
    duplicates."""
    seen: set[str] = set()
    kept: list[SearchHit] = []
    rejected: list[SearchHit] = []
    for hit in hits:
        if hit.url in seen:
            continue
        seen.add(hit.url)
        if classify(hit.url) is SourceKind.REJECTED:
            rejected.append(hit)
        else:
            kept.append(hit)
    return kept, rejected


def distinct_authoritative_domains(urls: list[str]) -> set[str]:
    """Count independent authoritative domains — used by the scope gate to decide if we
    have enough corroboration to proceed at all."""
    domains: set[str] = set()
    for url in urls:
        if classify(url) is not SourceKind.REJECTED:
            domains.add(extract_domain(url))
    return domains
