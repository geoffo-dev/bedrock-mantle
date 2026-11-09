"""HTTP client wrapper for Bedrock Mantle and bedrock-runtime."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastapi.responses import JSONResponse

from .auth import TokenProvider
from .config import Settings

log = logging.getLogger(__name__)


class UpstreamError(Exception):
    def __init__(self, status_code: int, message: str, err_type: str = "upstream_error", code: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type
        self.code = code

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code, content=openai_error(self.message, self.err_type, self.code)
        )


def openai_error(message: str, err_type: str = "upstream_error", code: Any = None) -> dict:
    return {"error": {"message": message, "type": err_type, "param": None, "code": code}}


def error_from_body(status_code: int, body: bytes) -> UpstreamError:
    """Build an UpstreamError from whatever error shape Mantle / Bedrock returned."""
    message = body.decode("utf-8", "replace").strip() or f"Upstream returned HTTP {status_code}"
    err_type, code = "upstream_error", None
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            message = err.get("message") or message
            err_type = err.get("type") or err_type
            code = err.get("code")
        elif isinstance(err, str):
            message = err
        else:
            message = data.get("message") or data.get("Message") or message
            err_type = data.get("type") or data.get("__type") or err_type
    return UpstreamError(status_code, message, err_type, code)


class Upstream:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, tokens: TokenProvider) -> None:
        self.settings = settings
        self.client = client
        self.tokens = tokens

    def mantle(self, path: str) -> str:
        return f"{self.settings.mantle_url}/{path.lstrip('/')}"

    def runtime(self, path: str) -> str:
        return f"{self.settings.runtime_url}/{path.lstrip('/')}"

    async def send(
        self,
        method: str,
        url: str,
        *,
        json_body: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        params: Any = None,
        stream: bool = False,
    ) -> httpx.Response:
        """Send a request with the Bedrock bearer token.

        A minted (IAM) token is refreshed and the request retried once on 401/403. With stream=True the
        caller owns the response and must close it.
        """
        if json_body is not None:
            content = json.dumps(json_body, ensure_ascii=False).encode()
        base_headers = {"content-type": "application/json", "accept": "application/json"}
        if stream:
            base_headers["accept"] = "text/event-stream"
        base_headers.update(headers or {})

        for attempt in range(2):
            token = await self.tokens.get(force_refresh=attempt > 0)
            req = self.client.build_request(
                method,
                url,
                content=content,
                params=params,
                headers={**base_headers, "authorization": f"Bearer {token}"},
            )
            try:
                resp = await self.client.send(req, stream=stream)
            except httpx.TimeoutException as exc:
                raise UpstreamError(504, f"Timed out calling Bedrock: {exc}", "timeout") from exc
            except httpx.HTTPError as exc:
                raise UpstreamError(
                    502, f"Could not reach Bedrock ({url}): {exc}", "connection_error"
                ) from exc
            if resp.status_code in (401, 403) and attempt == 0 and not self.tokens.is_static:
                await resp.aclose()
                log.info("Upstream returned %s; refreshing IAM bearer token and retrying", resp.status_code)
                continue
            return resp
        return resp  # pragma: no cover - loop always returns

    async def request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        resp = await self.send(method, url, **kwargs)
        if resp.status_code >= 400:
            raise error_from_body(resp.status_code, resp.content)
        try:
            return resp.json()
        except ValueError as exc:
            raise UpstreamError(502, "Bedrock returned a non-JSON response") from exc

    async def open_stream(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Open a streaming request; raise UpstreamError (after closing) if it failed."""
        resp = await self.send(method, url, stream=True, **kwargs)
        if resp.status_code >= 400:
            body = await resp.aread()
            await resp.aclose()
            raise error_from_body(resp.status_code, body)
        return resp
