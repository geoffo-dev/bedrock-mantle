"""Inbound API-key checks and the outbound Bedrock bearer-token provider."""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
from collections.abc import Callable
from datetime import timedelta

from fastapi import HTTPException, Request

from .config import Settings

log = logging.getLogger(__name__)


def extract_client_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key") or request.headers.get("api-key")


async def require_api_key(request: Request) -> None:
    """FastAPI dependency: accept `Authorization: Bearer <key>` or `x-api-key: <key>`."""
    settings: Settings = request.app.state.settings
    keys = settings.api_keys
    if not keys:
        if settings.allow_no_auth:
            return
        raise HTTPException(
            status_code=500, detail="Gateway misconfigured: set API_KEY (or ALLOW_NO_AUTH=true)"
        )
    presented = extract_client_key(request)
    if presented and any(hmac.compare_digest(presented.encode(), k.encode()) for k in keys):
        return
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


class CredentialsError(RuntimeError):
    pass


def _default_minter(region: str) -> Callable[[], str]:
    """Mint short-term Bedrock API keys from the default AWS credential chain."""
    from aws_bedrock_token_generator import provide_token
    from botocore.session import Session

    session = Session()

    class _SessionCredentials:
        # Duck-typed CredentialProvider: provide_token() only calls .load().
        def load(self):
            return session.get_credentials()

    provider = _SessionCredentials()

    def mint() -> str:
        return provide_token(region=region, aws_credentials_provider=provider, expiry=timedelta(hours=1))

    return mint


class TokenProvider:
    """Returns the Bearer token for Mantle and bedrock-runtime.

    Uses AWS_BEARER_TOKEN_BEDROCK when set. Otherwise it mints a short-term token from IAM credentials
    (env keys, profile, SSO, IRSA, EKS Pod Identity, ECS/EC2 roles) and re-mints it periodically.
    """

    def __init__(self, settings: Settings, minter: Callable[[], str] | None = None) -> None:
        self._static = settings.aws_bearer_token_bedrock or None
        self._region = settings.aws_region
        self._refresh_after = settings.token_refresh_seconds
        self._minter = minter
        self._token: str | None = None
        self._minted_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def is_static(self) -> bool:
        return self._static is not None

    @property
    def mode(self) -> str:
        return "api-key" if self.is_static else "iam"

    async def get(self, force_refresh: bool = False) -> str:
        if self._static:
            return self._static
        async with self._lock:
            fresh = self._token and (time.monotonic() - self._minted_at) < self._refresh_after
            if fresh and not force_refresh:
                return self._token  # type: ignore[return-value]
            if self._minter is None:
                self._minter = _default_minter(self._region)
            try:
                token = await asyncio.to_thread(self._minter)
            except Exception as exc:  # noqa: BLE001 - surface any credential-chain failure clearly
                raise CredentialsError(
                    "No Bedrock credentials: set AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key) or provide "
                    f"AWS credentials for IAM token minting ({exc})"
                ) from exc
            self._token, self._minted_at = token, time.monotonic()
            log.debug("Minted new Bedrock bearer token from IAM credentials")
            return token
