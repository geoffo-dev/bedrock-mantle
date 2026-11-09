from __future__ import annotations

import contextlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from mantle_gateway.config import ModelsFile, Settings
from mantle_gateway.main import create_app
from tests.mock_mantle import create_mock_app

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


def make_settings(**overrides) -> Settings:
    base = {
        "api_key": API_KEY,
        "aws_bearer_token_bedrock": "good-token",
        "mantle_base_url": "http://mantle.test",
        "bedrock_runtime_url": "http://runtime.test",
        "aws_region": "us-east-1",
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def mock():
    return create_mock_app()


@pytest.fixture
def gateway(mock):
    """Factory: gateway(models=..., **settings) -> TestClient wired to the mock Mantle."""
    stack = contextlib.ExitStack()

    def _make(models: dict | None = None, minter=None, **settings_overrides) -> TestClient:
        app = create_app(
            make_settings(**settings_overrides),
            transport=httpx.ASGITransport(app=mock),
            minter=minter,
            models_file=ModelsFile.model_validate(models or {}),
        )
        return stack.enter_context(TestClient(app))

    yield _make
    stack.close()


def sse_events(text: str) -> list:
    """Parse an SSE body into a list of decoded JSON payloads ('[DONE]' kept as a string)."""
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                payload = line[6:]
                out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def last_call(mock, path_prefix: str) -> dict:
    calls = [c for c in mock.state.calls if c["path"].startswith(path_prefix)]
    assert calls, f"no upstream call to {path_prefix}; saw {[c['path'] for c in mock.state.calls]}"
    return calls[-1]
