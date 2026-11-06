"""Bounded public-web reads, including DNS pinning and redirect validation."""
from __future__ import annotations

import asyncio
import ipaddress
import socket

import httpx


class UnsafeURL(ValueError):
    """A URL, redirect or response crossed the public-web boundary."""


class PublicDNSUnavailable(UnsafeURL):
    """Public DNS transport failed; retryable without relaxing address validation."""


def _benchmark_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return address in ipaddress.ip_network("198.18.0.0/15") if address.version == 4 else address in ipaddress.ip_network("2001:2::/48")


async def _public_dns_lookup(host: str) -> str:
    """Bootstrap a fixed public resolver without trusting a local Fake-IP mapping.

    No original benchmark/private address is ever connected to. TLS still verifies
    the resolver name; redirects, compression, oversized replies and private answers
    are rejected. Only invoked for all-benchmark answers to a non-literal hostname.
    """
    import json

    # A transient failure of one fixed bootstrap must not make every source look
    # unavailable. Both attempts together retain the old five-second budget.
    # Invalid responses are rejected immediately, never retried into acceptance.
    async with asyncio.timeout(5), httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
        for index, address in enumerate(("1.1.1.1", "1.0.0.1")):
            try:
                async with client.stream("GET", f"https://{address}/dns-query", params={"name": host, "type": "A"},
                        headers={"Host": "cloudflare-dns.com", "Accept": "application/dns-json", "Accept-Encoding": "identity"},
                        extensions={"sni_hostname": "cloudflare-dns.com"}, timeout=2.5) as response:
                    if response.status_code != 200 or response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise UnsafeURL("Public DNS resolver returned an invalid response")
                    raw = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(raw) + len(chunk) > 32768:
                            raise UnsafeURL("Public DNS response exceeds the limit")
                        raw.extend(chunk)
                break
            except httpx.TransportError:
                if index:
                    raise
    data = json.loads(raw)
    if data.get("Status") != 0 or not any(q.get("name", "").rstrip(".").lower() == host.lower() for q in data.get("Question", [])):
        raise UnsafeURL("Public DNS did not answer the requested hostname")
    answers = [str(a["data"]) for a in data.get("Answer", []) if a.get("type") == 1]
    if not answers or any(not ipaddress.ip_address(address).is_global or ipaddress.ip_address(address).is_multicast for address in answers):
        raise UnsafeURL("Public DNS did not return exclusively public addresses")
    return answers[0]


async def _public_address(url: httpx.URL) -> str:
    port = url.port or (443 if url.scheme == "https" else 80)
    if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
            or port not in {80, 443} or len(str(url)) > 8192):
        raise UnsafeURL("Only public HTTP(S) URLs on standard ports are supported")
    host = url.host.lower().rstrip(".")
    if "%" in host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise UnsafeURL("Local hosts are not public sources")
    records = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(str(record[4][0]) for record in records))
    if not addresses:
        raise UnsafeURL("No public address found")
    if all(_benchmark_address(address) for address in addresses):
        from core_engine.config import get_settings

        try:
            ipaddress.ip_address(host)
            literal_ip = True
        except ValueError:
            literal_ip = False
        if get_settings().public_dns_fallback and not literal_ip:
            try:
                address = await _public_dns_lookup(url.raw_host.decode("ascii"))
                if not ipaddress.ip_address(address).is_global or ipaddress.ip_address(address).is_multicast:
                    raise UnsafeURL("Fallback DNS returned a non-public address")
                return address
            except Exception as exc:
                if isinstance(exc, (httpx.TransportError, TimeoutError)):
                    raise PublicDNSUnavailable("Public DNS transport unavailable") from exc
                raise UnsafeURL(f"Proxy-style DNS mapping detected; public DNS verification failed ({type(exc).__name__})") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise UnsafeURL("Non-public network destinations are blocked")
    return addresses[0]


async def read_public_text(
    client: httpx.AsyncClient, url: str, *, max_bytes: int, timeout: float,
    allow_public_redirects: bool = False, before_request=None,
) -> httpx.Response:
    """Read bounded, DNS-pinned public data and return its final logical URL.

    Callers must use a direct client (trust_env=False), so a proxy cannot substitute
    a different destination. Host and TLS SNI retain the original hostname while
    the socket connects to the checked IP. Responses must honor identity encoding.
    Cross-site document mode requires a credential-free client and an optional
    per-hop robots gate; callers must use the returned URL for attribution.
    """
    current = httpx.URL(url)
    if allow_public_redirects and (client.auth is not None or any(client.headers.get(key) for key in
            ("authorization", "cookie", "proxy-authorization")) or len(client.cookies)):
        raise UnsafeURL("Public document redirects require a credential-free client")
    original_host = current.host.lower().rstrip(".").removeprefix("www.")
    for _ in range(6):
        address = await _public_address(current)
        if not allow_public_redirects and current.host.lower().rstrip(".").removeprefix("www.") != original_host:
            raise UnsafeURL("Cross-site redirects cannot retain the original source attribution")
        if before_request is not None and not await before_request(str(current)):
            raise UnsafeURL("robots_disallowed_redirect")
        pinned = current.copy_with(host=address)
        async with client.stream(
            "GET", pinned, follow_redirects=False, timeout=timeout,
            headers={"Host": current.netloc.decode("ascii"), "Accept-Encoding": "identity"},
            extensions={"sni_hostname": current.host},
        ) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise UnsafeURL("Redirect has no destination")
                current = current.join(location)
                continue
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise UnsafeURL("Compressed responses are not accepted for bounded source reads")
            length = response.headers.get("content-length")
            if length is not None and int(length) > max_bytes:
                raise UnsafeURL("Source exceeds the download limit")
            body = bytearray()
            async for chunk in response.aiter_raw():
                if len(body) + len(chunk) > max_bytes:
                    raise UnsafeURL("Source exceeds the download limit")
                body.extend(chunk)
            return httpx.Response(
                response.status_code, headers=response.headers, content=bytes(body),
                request=httpx.Request("GET", current),
            )
    raise UnsafeURL("Too many redirects")
