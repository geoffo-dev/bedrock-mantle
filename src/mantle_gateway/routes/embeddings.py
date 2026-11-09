"""POST /v1/embeddings -> bedrock-runtime InvokeModel (Titan / Cohere)."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import require_api_key
from ..translate.embeddings import (
    build_calls,
    embedding_family,
    normalise_input,
    openai_response,
    parse_result,
)
from .common import read_json, state

log = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    settings, catalog, upstream = state(request)
    body = await read_json(request)
    name = body.get("model")
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="'model' is required")
    if not catalog.is_allowed(name):
        raise HTTPException(status_code=404, detail=f"Model {name!r} is not enabled on this gateway")
    model = catalog.resolve(name)
    family = embedding_family(model.upstream_id)
    texts = normalise_input(body.get("input"))
    calls = build_calls(
        family, texts, body.get("dimensions") or model.dimensions, body.get("input_type") or model.input_type
    )
    url = upstream.runtime(f"/model/{quote(model.upstream_id, safe='')}/invoke")
    log.info(
        "embeddings model=%s upstream=%s inputs=%d calls=%d", name, model.upstream_id, len(texts), len(calls)
    )

    sem = asyncio.Semaphore(max(1, settings.embedding_concurrency))

    async def run(call):
        async with sem:
            return await upstream.request_json("POST", url, json_body=call.body)

    results = await asyncio.gather(*(run(c) for c in calls))
    vectors: list[list[float]] = []
    tokens = 0
    for result in results:
        vecs, used = parse_result(family, result)
        vectors.extend(vecs)
        tokens += used
    if len(vectors) != len(texts):
        raise HTTPException(status_code=502, detail="Bedrock returned an unexpected number of embeddings")
    if not tokens:  # Cohere doesn't always report usage; rough estimate keeps clients happy
        tokens = sum(max(1, len(t) // 4) for t in texts)
    return openai_response(vectors, tokens, model.public_id, body.get("encoding_format"))
