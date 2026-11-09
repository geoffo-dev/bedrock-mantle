"""Helpers shared by the route modules."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import socket
from collections.abc import AsyncIterator, Callable
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, Request

from ..catalog import Catalog
from ..config import Settings
from ..sse import DONE, iter_sse, sse_data
from ..translate.common import TranslationError
from ..upstream import Upstream

log = logging.getLogger(__name__)

STREAM_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}


def state(request: Request) -> tuple[Settings, Catalog, Upstream]:
    app_state = request.app.state
    return app_state.settings, app_state.catalog, app_state.upstream


async def read_json(request: Request) -> dict:
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return body


async def translate_stream(
    resp: httpx.Response, translator, events: AsyncIterator[tuple[str | None, str]] | None = None
) -> AsyncIterator[bytes]:
    """Re-emit an upstream stream (SSE unless `events` is given) as OpenAI chat.completion.chunk events."""
    try:
        async for event, data in events or iter_sse(resp.aiter_lines()):
            if data.strip() == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            for chunk in translator.feed(event, obj):
                yield sse_data(chunk)
            if translator.done:
                break
        for chunk in translator.finish():
            yield sse_data(chunk)
    except httpx.HTTPError as exc:
        log.warning("Upstream stream interrupted: %s", exc)
        yield sse_data(
            {"error": {"message": f"Upstream stream interrupted: {exc}", "type": "upstream_error"}}
        )
    finally:
        await resp.aclose()
    yield DONE


async def relay_bytes(resp: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()


def _is_public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast:
            return False
    return True


async def inline_remote_images(
    blocks: list[dict],
    client: httpx.AsyncClient,
    settings: Settings,
    set_source: Callable[[dict, str, str], None],
) -> None:
    """Download http(s) images and inline them as base64 (only when FETCH_REMOTE_IMAGES is on).

    Private, loopback and link-local addresses are refused so clients can't reach internal endpoints.
    """
    if not blocks or not settings.fetch_remote_images:
        return
    for block in blocks:
        url = block["source"]["url"]
        host = urlparse(url).hostname or ""
        if not await asyncio.to_thread(_is_public_host, host):
            raise TranslationError(f"Refusing to fetch image from non-public host {host!r}")
        try:
            r = await client.get(url, follow_redirects=False, timeout=30)
        except httpx.HTTPError as exc:
            raise TranslationError(f"Could not fetch image {url}: {exc}") from exc
        if r.status_code >= 400:
            raise TranslationError(f"Could not fetch image {url}: HTTP {r.status_code}")
        if len(r.content) > settings.max_image_bytes:
            raise TranslationError(f"Image {url} exceeds {settings.max_image_bytes} bytes")
        media_type = (r.headers.get("content-type") or "image/png").split(";")[0].strip()
        set_source(block, media_type, base64.b64encode(r.content).decode())
