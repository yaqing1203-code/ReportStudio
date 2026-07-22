"""Standalone network / search diagnostic for Report Studio.

Run:  python test_network.py         (uses the project's py311 env)

Tests, in order:
  1. Raw connectivity to google.com / bing.com / duckduckgo.
  2. The ACTUAL configured search provider end-to-end (does a real query return hits?).
  3. Whether the search/LLM API keys are loaded from settings/.env.
  4. Proxy / firewall environment settings that could block outbound requests.

Writes a plain-text report to network_diag.txt (avoids console-encoding issues on
Windows cp950) and prints a short summary.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, "src")

LINES: list[str] = []


def log(msg: str = "") -> None:
    LINES.append(msg)


async def test_raw_connectivity() -> None:
    log("=" * 60)
    log("TEST 1 — Raw outbound connectivity")
    log("=" * 60)
    try:
        import httpx
    except Exception as e:
        log(f"  [FAIL] httpx not importable: {e}")
        return

    targets = [
        ("https://www.google.com", "Google"),
        ("https://www.bing.com", "Bing"),
        ("https://html.duckduckgo.com/html/", "DuckDuckGo HTML (our default search)"),
    ]
    for url, name in targets:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
                r = await c.get(url)
            log(f"  [OK]   {name}: HTTP {r.status_code} ({len(r.content)} bytes)")
        except Exception as e:
            log(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    log("")


async def test_search_provider() -> None:
    log("=" * 60)
    log("TEST 2 — Configured search provider (end-to-end)")
    log("=" * 60)
    try:
        from core_engine.config import get_settings
        from core_engine.report.scrape import get_search_provider

        s = get_settings()
        log(f"  search_provider = {s.search_provider!r}")
        provider = get_search_provider()
        log(f"  provider class  = {type(provider).__name__}")
        hits = await provider.search("solid-state battery supply chain", max_results=8)
        log(f"  query returned  = {len(hits)} hit(s)")
        for h in hits[:5]:
            log(f"      - {h.domain}  {h.url[:80]}")
        if not hits:
            log("  [WARN] Zero hits. If DuckDuckGo, the HTML layout may have changed or "
                "the endpoint is blocked. If tavily, the key/endpoint may be wrong.")
        else:
            log("  [OK]   Search is returning results.")
    except Exception as e:
        log(f"  [FAIL] {type(e).__name__}: {e}")
    log("")


def test_key_loading() -> None:
    log("=" * 60)
    log("TEST 3 — API key / settings loading")
    log("=" * 60)
    try:
        from core_engine.config import get_settings

        s = get_settings()
        def shown(v: str) -> str:
            return f"set ({len(v)} chars)" if v else "MISSING"
        log(f"  llm_provider        = {s.llm_provider!r}")
        log(f"  llm_base_url        = {s.llm_base_url or '(none)'}")
        log(f"  llm_api_key         = {shown(s.llm_api_key)}")
        log(f"  anthropic_api_key   = {shown(s.anthropic_api_key)}")
        log(f"  search_provider     = {s.search_provider!r}")
        log(f"  search_api_key      = {shown(s.search_api_key)}")
        # Where would persisted GUI settings live?
        try:
            from core_engine.app import runtime
            sp = runtime.user_data_dir() / "settings.json"
            log(f"  settings.json       = {sp} (exists: {sp.exists()})")
        except Exception as e:
            log(f"  settings.json probe failed: {e}")
        if s.search_provider == "tavily" and not s.search_api_key:
            log("  [FAIL] provider=tavily but no search_api_key -> search will error.")
        elif s.search_provider == "duckduckgo":
            log("  [OK]   duckduckgo needs no key (keyless default).")
    except Exception as e:
        log(f"  [FAIL] {type(e).__name__}: {e}")
    log("")


def test_proxy_env() -> None:
    log("=" * 60)
    log("TEST 4 — Proxy / firewall environment")
    log("=" * 60)
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy"]
    any_set = False
    for k in keys:
        v = os.environ.get(k)
        if v:
            any_set = True
            log(f"  {k} = {v}")
    if not any_set:
        log("  No proxy environment variables set.")
    log("")


def main() -> int:
    asyncio.run(test_raw_connectivity())
    asyncio.run(test_search_provider())
    test_key_loading()
    test_proxy_env()

    report = "\n".join(LINES)
    with open("network_diag.txt", "w", encoding="utf-8") as fh:
        fh.write(report)

    # Console summary (ascii-safe).
    oks = sum(1 for x in LINES if "[OK]" in x)
    fails = sum(1 for x in LINES if "[FAIL]" in x)
    warns = sum(1 for x in LINES if "[WARN]" in x)
    print(f"network diagnostic complete: {oks} OK, {warns} WARN, {fails} FAIL")
    print("full report written to network_diag.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
