"""Public-web requests must not reach private hosts, including via redirects/DNS."""
from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from core_engine.security import network

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def mock_dns(monkeypatch, addresses):
    async def resolve(host, port, **kwargs):
        return [(socket.AF_INET6 if ":" in address else socket.AF_INET,
                 socket.SOCK_STREAM, 6, "", (address, port)) for address in addresses]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1",
                                     "0.0.0.0", "::1", "fc00::1", "224.0.0.1", "::ffff:127.0.0.1"])
async def test_non_public_dns_answers_blocked(monkeypatch, address):
    mock_dns(monkeypatch, [address])
    with pytest.raises(network.UnsafeURL):
        await network._public_address(httpx.URL("https://source.example/article"))


async def test_mixed_public_private_dns_is_rejected(monkeypatch):
    mock_dns(monkeypatch, ["8.8.8.8", "127.0.0.1"])
    with pytest.raises(network.UnsafeURL):
        await network._public_address(httpx.URL("https://source.example/article"))


async def test_proxy_benchmark_mapping_is_re_resolved_without_connecting_to_it(monkeypatch):
    mock_dns(monkeypatch, ["198.18.1.60", "2001:2::126"])
    requested = []

    async def public_lookup(host):
        requested.append(host)
        return "8.8.8.8"

    monkeypatch.setattr(network, "_public_dns_lookup", public_lookup)
    assert await network._public_address(httpx.URL("https://api.crossref.org/works")) == "8.8.8.8"
    assert requested == ["api.crossref.org"]


@pytest.mark.parametrize("addresses", [["198.18.1.60", "127.0.0.1"], ["198.18.1.60", "10.0.0.1"]])
async def test_proxy_fallback_does_not_relax_mixed_private_dns(monkeypatch, addresses):
    mock_dns(monkeypatch, addresses)

    async def forbidden(host):
        pytest.fail("Mixed private DNS must never invoke public fallback")

    monkeypatch.setattr(network, "_public_dns_lookup", forbidden)
    with pytest.raises(network.UnsafeURL):
        await network._public_address(httpx.URL("https://source.example/article"))


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "198.18.1.60"])
async def test_proxy_fallback_still_rejects_nonpublic_answers(monkeypatch, address):
    mock_dns(monkeypatch, ["198.18.1.60"])

    async def lookup(host):
        return address

    monkeypatch.setattr(network, "_public_dns_lookup", lookup)
    with pytest.raises(network.UnsafeURL):
        await network._public_address(httpx.URL("https://source.example/article"))


async def test_proxy_fallback_never_resolves_a_benchmark_ip_literal(monkeypatch):
    mock_dns(monkeypatch, ["198.18.1.60"])
    with pytest.raises(network.UnsafeURL):
        await network._public_address(httpx.URL("https://198.18.1.60/article"))


async def test_proxy_fallback_can_be_disabled(monkeypatch):
    from core_engine.config import get_settings
    mock_dns(monkeypatch, ["198.18.1.60"])
    monkeypatch.setenv("CE_PUBLIC_DNS_FALLBACK", "false")
    get_settings.cache_clear()
    try:
        with pytest.raises(network.UnsafeURL):
            await network._public_address(httpx.URL("https://source.example/article"))
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(("status", "question", "answer", "valid"), [
    (0, "source.example", "8.8.8.8", True), (0, "other.example", "8.8.8.8", False),
    (0, "source.example", "127.0.0.1", False), (3, "source.example", "8.8.8.8", False),
])
async def test_public_resolver_uses_fixed_tls_bootstrap_and_validates_answers(monkeypatch, status, question, answer, valid):
    import json
    actual_client = httpx.AsyncClient
    requests = []

    def handler(request):
        requests.append(request)
        body = json.dumps({"Status": status, "Question": [{"name": question}], "Answer": [{"type": 1, "data": answer}]}).encode()
        return httpx.Response(200, stream=Chunks(body))

    monkeypatch.setattr(network.httpx, "AsyncClient", lambda **kwargs: actual_client(transport=httpx.MockTransport(handler), **kwargs))
    if valid:
        assert await network._public_dns_lookup("source.example") == "8.8.8.8"
    else:
        with pytest.raises(network.UnsafeURL):
            await network._public_dns_lookup("source.example")
    assert requests[0].url.host == "1.1.1.1" and requests[0].headers["host"] == "cloudflare-dns.com"
    assert requests[0].extensions["sni_hostname"] == "cloudflare-dns.com"


async def test_public_resolver_retries_alternate_bootstrap_only_on_transport_failure(monkeypatch):
    import json
    actual_client = httpx.AsyncClient
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "1.1.1.1":
            raise httpx.ConnectError("temporary bootstrap failure", request=request)
        return httpx.Response(200, stream=Chunks(json.dumps({"Status": 0,
            "Question": [{"name": "source.example"}], "Answer": [{"type": 1, "data": "8.8.8.8"}]}).encode()))

    monkeypatch.setattr(network.httpx, "AsyncClient", lambda **kwargs: actual_client(transport=httpx.MockTransport(handler), **kwargs))
    assert await network._public_dns_lookup("source.example") == "8.8.8.8"
    assert [r.url.host for r in requests] == ["1.1.1.1", "1.0.0.1"]
    assert all(r.headers["host"] == r.extensions["sni_hostname"] == "cloudflare-dns.com" for r in requests)


async def test_public_resolver_does_not_retry_private_answers(monkeypatch):
    import json
    actual_client = httpx.AsyncClient
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=Chunks(json.dumps({"Status": 0,
            "Question": [{"name": "source.example"}], "Answer": [{"type": 1, "data": "127.0.0.1"}]}).encode()))

    monkeypatch.setattr(network.httpx, "AsyncClient", lambda **kwargs: actual_client(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(network.UnsafeURL):
        await network._public_dns_lookup("source.example")
    assert len(requests) == 1


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://localhost/x", "https://service.local/x",
                                 "https://user:pass@example.com/x", "http://example.com:8080/x"])
async def test_unsafe_url_shapes_blocked(url):
    with pytest.raises(network.UnsafeURL):
        await network._public_address(httpx.URL(url))


class Chunks(httpx.AsyncByteStream):
    def __init__(self, *parts):
        self.parts = parts

    async def __aiter__(self):
        for part in self.parts:
            yield part


async def test_dns_is_pinned_with_original_host_and_tls_name(monkeypatch):
    mock_dns(monkeypatch, ["8.8.8.8"])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=Chunks(b"<p>Public text</p>"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        response = await network.read_public_text(client, "https://source.example/path",
                                                  max_bytes=100, timeout=1)
    assert response.text == "<p>Public text</p>"
    assert calls[0].url.host == "8.8.8.8"
    assert calls[0].headers["host"] == "source.example"
    assert calls[0].extensions["sni_hostname"] == "source.example"
    assert response.url.host == "source.example"


async def test_open_document_redirect_keeps_actual_target_and_checks_each_hop(monkeypatch):
    mock_dns(monkeypatch, ["8.8.8.8"])
    gated, requests = [], []

    async def robots(url):
        gated.append(url)
        return True

    def handler(request):
        requests.append(request)
        if request.headers["host"] == "repo.example":
            return httpx.Response(302, headers={"location": "https://files.example/paper.pdf"}, stream=Chunks())
        return httpx.Response(200, stream=Chunks(b"%PDF-paper"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        response = await network.read_public_text(client, "https://repo.example/download", max_bytes=100,
            timeout=1, allow_public_redirects=True, before_request=robots)
    assert str(response.url) == "https://files.example/paper.pdf"
    assert gated == ["https://repo.example/download", "https://files.example/paper.pdf"]
    assert all(r.url.host == "8.8.8.8" for r in requests)
    assert requests[-1].extensions["sni_hostname"] == "files.example"


@pytest.mark.parametrize("headers", [{"Authorization": "secret"}, {"Cookie": "session=secret"}])
async def test_document_redirect_mode_refuses_credentialed_clients(headers):
    async with httpx.AsyncClient(headers=headers, trust_env=False) as client:
        with pytest.raises(network.UnsafeURL, match="credential-free"):
            await network.read_public_text(client, "https://repo.example/download", max_bytes=100,
                timeout=1, allow_public_redirects=True)


async def test_open_document_redirect_still_blocks_private_network(monkeypatch):
    async def resolve(host, port, **kwargs):
        address = "127.0.0.1" if host == "private.example" else "8.8.8.8"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://private.example/paper.pdf"}, stream=Chunks())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(network.UnsafeURL):
            await network.read_public_text(client, "https://repo.example/download", max_bytes=100,
                timeout=1, allow_public_redirects=True)
    assert len(requests) == 1


async def test_open_document_redirect_honors_target_robots(monkeypatch):
    mock_dns(monkeypatch, ["8.8.8.8"])

    async def robots(url):
        return False

    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(network.UnsafeURL, match="robots_disallowed"):
            await network.read_public_text(client, "https://repo.example/download", max_bytes=100,
                timeout=1, allow_public_redirects=True, before_request=robots)


async def test_redirect_to_private_address_never_connected(monkeypatch):
    async def resolve(host, port, **kwargs):
        address = "127.0.0.1" if host == "127.0.0.1" else "8.8.8.8"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"}, stream=Chunks())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with pytest.raises(network.UnsafeURL):
            await network.read_public_text(client, "https://source.example/path", max_bytes=100, timeout=1)
    assert len(seen) == 1


async def test_relative_redirect_uses_original_hostname(monkeypatch):
    mock_dns(monkeypatch, ["8.8.8.8"])
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new"}, stream=Chunks())
        return httpx.Response(200, stream=Chunks(b"body"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        response = await network.read_public_text(client, "https://source.example/old", max_bytes=10, timeout=1)
    assert paths == ["/old", "/new"]
    assert response.url == "https://source.example/new"


@pytest.mark.parametrize("headers,chunks", [({"Content-Length": "101"}, [b"small"]),
                                            ({}, [b"x" * 60, b"y" * 41]),
                                            ({"Content-Encoding": "gzip"}, [b"compressed"])])
async def test_response_size_and_encoding_limits(monkeypatch, headers, chunks):
    mock_dns(monkeypatch, ["8.8.8.8"])
    transport = httpx.MockTransport(lambda request: httpx.Response(200, headers=headers, stream=Chunks(*chunks)))
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        with pytest.raises(network.UnsafeURL):
            await network.read_public_text(client, "https://source.example/path", max_bytes=100, timeout=1)


async def test_robots_rules_cached_per_host_but_evaluated_per_path(monkeypatch):
    from core_engine.config import get_settings
    from core_engine.report import scrape

    monkeypatch.setenv("CE_SCRAPE_RESPECT_ROBOTS", "true")
    get_settings.cache_clear()
    calls = []

    async def read_robots(*args, **kwargs):
        calls.append(args)
        return httpx.Response(200, text="User-agent: *\nDisallow: /private\nAllow: /public\n")

    monkeypatch.setattr(scrape, "read_public_text", read_robots)
    try:
        fetcher = scrape.HttpFetcher()
        assert await fetcher._allowed(None, "https://source.example/private") is False
        assert await fetcher._allowed(None, "https://source.example/public") is True
        assert len(calls) == 1
    finally:
        get_settings.cache_clear()
