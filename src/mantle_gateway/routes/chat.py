"""POST /v1/chat/completions: route each model to the right Mantle API surface."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import require_api_key
from ..catalog import NUMBERED_GPT_RE, ResolvedModel
from ..config import Settings
from ..translate.chat_to_messages import (
    MessagesStreamTranslator,
    build_messages_request,
    messages_response_to_chat,
    remote_image_urls,
)
from ..translate.chat_to_responses import (
    ResponsesStreamTranslator,
    build_responses_request,
    responses_to_chat,
)
from ..upstream import Upstream, error_from_body
from .common import STREAM_HEADERS, inline_remote_images, read_json, relay_bytes, state, translate_stream

log = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    settings, catalog, upstream = state(request)
    body = await read_json(request)
    name = body.get("model")
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="'model' is required")
    if not catalog.is_allowed(name):
        raise HTTPException(status_code=404, detail=f"Model {name!r} is not enabled on this gateway")
    model = catalog.resolve(name)
    if model.type == "embedding":
        raise HTTPException(status_code=400, detail=f"{name!r} is an embedding model; use /v1/embeddings")
    stream = bool(body.get("stream"))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    log.info(
        "chat model=%s upstream=%s api=%s base=%s stream=%s",
        name,
        model.upstream_id,
        model.api,
        model.base,
        stream,
    )
    if model.api == "messages":
        return await _via_messages(body, model, settings, upstream, stream, include_usage)
    if model.api == "responses":
        return await _via_responses(body, model, settings, upstream, stream, include_usage)
    return await _via_chat(body, model, upstream, stream)


async def _via_messages(
    body: dict,
    model: ResolvedModel,
    settings: Settings,
    upstream: Upstream,
    stream: bool,
    include_usage: bool,
):
    req = build_messages_request(body, model, settings, stream)

    def set_source(block: dict, media_type: str, data: str) -> None:
        block["source"] = {"type": "base64", "media_type": media_type, "data": data}

    await inline_remote_images(remote_image_urls(req), upstream.client, settings, set_source)
    url = upstream.mantle("/anthropic/v1/messages")
    headers = {"anthropic-version": settings.anthropic_version}
    if not stream:
        data = await upstream.request_json("POST", url, json_body=req, headers=headers)
        return JSONResponse(messages_response_to_chat(data, model.public_id, settings))
    resp = await upstream.open_stream("POST", url, json_body=req, headers=headers)
    translator = MessagesStreamTranslator(model.public_id, settings, include_usage)
    return StreamingResponse(
        translate_stream(resp, translator), media_type="text/event-stream", headers=STREAM_HEADERS
    )


async def _via_responses(
    body: dict,
    model: ResolvedModel,
    settings: Settings,
    upstream: Upstream,
    stream: bool,
    include_usage: bool,
):
    req = build_responses_request(body, model, settings, stream)
    url = upstream.mantle(f"/{model.base}/responses")
    if not stream:
        data = await upstream.request_json("POST", url, json_body=req)
        return JSONResponse(responses_to_chat(data, model.public_id, settings))
    resp = await upstream.open_stream("POST", url, json_body=req)
    translator = ResponsesStreamTranslator(model.public_id, settings, include_usage)
    return StreamingResponse(
        translate_stream(resp, translator), media_type="text/event-stream", headers=STREAM_HEADERS
    )


async def _via_chat(body: dict, model: ResolvedModel, upstream: Upstream, stream: bool):
    payload = {**body, "model": model.upstream_id}
    if not model.allow_sampling:
        for key in ("temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty"):
            payload.pop(key, None)
    if model.max_tokens and not (payload.get("max_tokens") or payload.get("max_completion_tokens")):
        payload["max_tokens"] = model.max_tokens
    if NUMBERED_GPT_RE.search(model.upstream_id) and "max_tokens" in payload:
        payload.setdefault("max_completion_tokens", payload.pop("max_tokens"))
    if model.default_effort and "reasoning_effort" not in payload:
        payload["reasoning_effort"] = model.default_effort
    url = upstream.mantle(f"/{model.base}/chat/completions")
    if not stream:
        resp = await upstream.send("POST", url, json_body=payload)
        if resp.status_code >= 400:
            raise error_from_body(resp.status_code, resp.content)
        data = resp.json()
        if isinstance(data, dict):
            data["model"] = model.public_id
        return JSONResponse(data)
    resp = await upstream.open_stream("POST", url, json_body=payload)
    return StreamingResponse(relay_bytes(resp), media_type="text/event-stream", headers=STREAM_HEADERS)
