"""Native passthroughs: Anthropic Messages (/v1/messages) and OpenAI Responses (/v1/responses).

These let tools such as Claude Code (ANTHROPIC_BASE_URL) or Codex CLI use the same gateway and key.
Bodies are forwarded as-is apart from model-alias resolution; responses are relayed verbatim.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from ..auth import require_api_key
from ..upstream import Upstream
from .common import STREAM_HEADERS, read_json, relay_bytes, state

log = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])

_RELAYED_RESPONSE_HEADERS = ("request-id", "x-amzn-requestid", "anthropic-ratelimit-requests-remaining")


async def _proxy(upstream: Upstream, url: str, body: dict, headers: dict, params, stream: bool) -> Response:
    resp = await upstream.send("POST", url, json_body=body, headers=headers, params=params, stream=stream)
    out_headers = {k: resp.headers[k] for k in _RELAYED_RESPONSE_HEADERS if k in resp.headers}
    media_type = resp.headers.get("content-type", "text/event-stream" if stream else "application/json")
    if stream and resp.status_code < 400:
        return StreamingResponse(
            relay_bytes(resp),
            status_code=resp.status_code,
            media_type=media_type,
            headers={**STREAM_HEADERS, **out_headers},
        )
    content = await resp.aread() if stream else resp.content
    if stream:
        await resp.aclose()
    return Response(content=content, status_code=resp.status_code, media_type=media_type, headers=out_headers)


@router.post("/v1/messages")
@router.post("/v1/messages/count_tokens")
async def anthropic_messages(request: Request):
    settings, catalog, upstream = state(request)
    body = await read_json(request)
    if isinstance(body.get("model"), str):
        body["model"] = catalog.resolve(body["model"]).upstream_id
    suffix = request.url.path.removeprefix("/v1/messages")
    url = upstream.mantle(f"/anthropic/v1/messages{suffix}")
    headers = {"anthropic-version": request.headers.get("anthropic-version") or settings.anthropic_version}
    if request.headers.get("anthropic-beta"):
        headers["anthropic-beta"] = request.headers["anthropic-beta"]
    stream = bool(body.get("stream"))
    log.info("passthrough anthropic path=%s model=%s stream=%s", suffix or "/", body.get("model"), stream)
    return await _proxy(upstream, url, body, headers, list(request.query_params.multi_items()), stream)


@router.post("/v1/responses")
async def openai_responses(request: Request):
    _, catalog, upstream = state(request)
    body = await read_json(request)
    base = "v1"
    if isinstance(body.get("model"), str):
        model = catalog.resolve(body["model"])
        body["model"] = model.upstream_id
        base = model.base
    url = upstream.mantle(f"/{base}/responses")
    stream = bool(body.get("stream"))
    log.info("passthrough responses base=%s model=%s stream=%s", base, body.get("model"), stream)
    return await _proxy(upstream, url, body, {}, list(request.query_params.multi_items()), stream)
