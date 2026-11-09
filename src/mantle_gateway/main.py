"""FastAPI application factory.

Run with:  uvicorn --factory mantle_gateway.main:create_app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .auth import CredentialsError, TokenProvider
from .catalog import Catalog
from .config import ModelsFile, Settings, load_models_file
from .routes import chat, embeddings, models, passthrough
from .translate.common import TranslationError
from .translate.embeddings import EmbeddingError
from .upstream import Upstream, UpstreamError, openai_error

log = logging.getLogger("mantle_gateway")

_HTTP_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
}


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    minter=None,
    models_file: ModelsFile | None = None,
) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    if settings.log_level.upper() != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per upstream call is noise
    models_file = models_file if models_file is not None else load_models_file(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout, connect=15.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            transport=transport,
        )
        tokens = TokenProvider(settings, minter)
        upstream = Upstream(settings, client, tokens)
        app.state.upstream = upstream
        app.state.catalog = Catalog(settings, models_file, upstream)
        if not settings.api_keys and not settings.allow_no_auth:
            log.warning("API_KEY is not set: every /v1 request will be rejected until it is configured")
        log.info(
            "Mantle gateway %s ready: mantle=%s runtime=%s bedrock-auth=%s discovery=%s",
            __version__,
            settings.mantle_url,
            settings.runtime_url,
            tokens.mode,
            app.state.catalog.discovery_enabled,
        )
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(
        title="Mantle Gateway", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.state.settings = settings

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        err_type = _HTTP_ERROR_TYPES.get(exc.status_code, "api_error")
        return JSONResponse(status_code=exc.status_code, content=openai_error(str(exc.detail), err_type))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content=openai_error(str(exc), "invalid_request_error"))

    @app.exception_handler(UpstreamError)
    async def upstream_error(_: Request, exc: UpstreamError):
        log.warning("Upstream error %s: %s", exc.status_code, exc.message)
        return exc.to_response()

    @app.exception_handler(TranslationError)
    @app.exception_handler(EmbeddingError)
    async def bad_request(_: Request, exc: Exception):
        return JSONResponse(status_code=400, content=openai_error(str(exc), "invalid_request_error"))

    @app.exception_handler(CredentialsError)
    async def credentials_error(_: Request, exc: CredentialsError):
        log.error("%s", exc)
        return JSONResponse(status_code=500, content=openai_error(str(exc), "credentials_error"))

    @app.get("/health", include_in_schema=False)
    async def health():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def root():
        return {"name": "mantle-gateway", "version": __version__, "openai_base_url": "/v1"}

    for router in (models.router, chat.router, embeddings.router, passthrough.router):
        app.include_router(router)
    return app
