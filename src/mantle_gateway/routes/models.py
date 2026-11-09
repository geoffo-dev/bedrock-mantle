"""GET /v1/models."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import require_api_key
from .common import state

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/v1/models")
async def list_models(request: Request):
    _, catalog, _ = state(request)
    return {"object": "list", "data": await catalog.list_models()}


@router.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, request: Request):
    _, catalog, _ = state(request)
    for m in await catalog.list_models():
        if m["id"] == model_id:
            return m
    if not catalog.is_allowed(model_id):
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")
    rm = catalog.resolve(model_id)
    return {
        "id": rm.public_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": rm.owned_by,
        "name": rm.name,
    }
