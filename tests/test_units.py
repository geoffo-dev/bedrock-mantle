"""Catalog rules, SSE parsing, embeddings helpers and token provider."""

import base64
import struct

import pytest

from mantle_gateway.auth import CredentialsError, TokenProvider
from mantle_gateway.catalog import (
    auto_api,
    auto_base,
    auto_forced_tool_choice,
    auto_thinking,
    claude_version,
    resolve_entry,
)
from mantle_gateway.config import ModelEntry, Settings
from mantle_gateway.sse import iter_sse
from mantle_gateway.translate.embeddings import (
    EmbeddingError,
    build_calls,
    embedding_family,
    encode,
    normalise_input,
    parse_result,
)


@pytest.mark.parametrize(
    ("model", "api", "base"),
    [
        ("anthropic.claude-opus-5-5", "messages", "v1"),
        ("openai.gpt-5.5", "responses", "openai/v1"),
        ("openai.gpt-6.1-sol", "responses", "openai/v1"),
        ("openai.gpt-oss-120b", "chat", "v1"),
        ("google.gemma-4-31b", "chat", "openai/v1"),
        ("xai.grok-4.6", "chat", "openai/v1"),
        ("qwen.qwen3-32b", "chat", "v1"),
    ],
)
def test_auto_routes(model, api, base):
    assert auto_api(model) == api
    assert auto_base(model) == base


@pytest.mark.parametrize(
    ("model", "thinking", "forced"),
    [
        ("anthropic.claude-opus-5-5", "adaptive", False),
        ("anthropic.claude-sonnet-5-5", "adaptive", False),
        ("anthropic.claude-haiku-5-5", "adaptive", True),
        ("anthropic.claude-opus-5", "adaptive", True),
        ("anthropic.claude-sonnet-4-6", "adaptive", True),
        ("anthropic.claude-haiku-4-5", "budget", True),
        ("anthropic.claude-sonnet-4-20250514-v1:0", "budget", True),
        ("anthropic.claude-3-7-sonnet", "budget", True),
        ("anthropic.claude-fable-5-1", "adaptive", False),
        ("anthropic.claude-mythos-preview", "adaptive", True),
    ],
)
def test_claude_capabilities(model, thinking, forced):
    assert auto_thinking(model) == thinking
    assert auto_forced_tool_choice(model) is forced


def test_claude_version_ignores_date_suffix():
    v = claude_version("claude-sonnet-4-20250514")
    assert (v.family, v.major, v.minor) == ("sonnet", 4, 0)


def test_entry_overrides():
    rm = resolve_entry(
        "x", ModelEntry(id="openai.gpt-5.5", alias="gpt", name="GPT", api="chat", allow_sampling=True)
    )
    assert (rm.public_id, rm.upstream_id, rm.name, rm.api, rm.base, rm.allow_sampling) == (
        "gpt",
        "openai.gpt-5.5",
        "GPT",
        "chat",
        "openai/v1",
        True,
    )


async def _lines(items):
    for i in items:
        yield i


async def test_iter_sse():
    lines = [": comment", "event: a", 'data: {"x":1}', "", "data: line1", "data: line2", "", "data: [DONE]"]
    out = [e async for e in iter_sse(_lines(lines))]
    assert out == [("a", '{"x":1}'), (None, "line1\nline2"), (None, "[DONE]")]


def test_embedding_helpers():
    assert embedding_family("amazon.titan-embed-text-v2:0") == "titan-v2"
    assert embedding_family("amazon.titan-embed-text-v1") == "titan-v1"
    assert embedding_family("cohere.embed-multilingual-v3") == "cohere-v3"
    assert embedding_family("us.cohere.embed-v4:0") == "cohere-v4"
    with pytest.raises(EmbeddingError):
        embedding_family("openai.text-embedding-3")
    with pytest.raises(EmbeddingError):
        normalise_input([1, 2, 3])
    assert normalise_input("a") == ["a"]

    calls = build_calls("cohere-v4", [f"t{i}" for i in range(100)], 512, None)
    assert [c.count for c in calls] == [96, 4]
    assert calls[0].body["output_dimension"] == 512 and calls[0].body["input_type"] == "search_document"
    titan = build_calls("titan-v2", ["a", "b"], 256, None)
    assert titan[0].body == {"inputText": "a", "normalize": True, "dimensions": 256}

    assert parse_result("cohere-v4", {"embeddings": {"float": [[1.0]]}}) == ([[1.0]], 0)
    assert parse_result("titan-v2", {"embedding": [1.0], "inputTextTokenCount": 2}) == ([[1.0]], 2)
    b64 = encode([1.0, 2.0], "base64")
    assert struct.unpack("<2f", base64.b64decode(b64)) == (1.0, 2.0)


async def test_token_provider_static_and_minted():
    static = TokenProvider(Settings(api_key="k", aws_bearer_token_bedrock="abc"))
    assert await static.get() == "abc" and static.is_static

    counter = {"n": 0}

    def mint():
        counter["n"] += 1
        return f"tok{counter['n']}"

    minted = TokenProvider(Settings(api_key="k", aws_bearer_token_bedrock=None), minter=mint)
    assert await minted.get() == "tok1"
    assert await minted.get() == "tok1"  # cached
    assert await minted.get(force_refresh=True) == "tok2"

    def broken():
        raise RuntimeError("no creds")

    with pytest.raises(CredentialsError):
        await TokenProvider(Settings(api_key="k", aws_bearer_token_bedrock=None), minter=broken).get()
