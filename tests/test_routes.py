"""End-to-end tests: gateway (FastAPI TestClient) -> mock Mantle (ASGI transport)."""

import base64
import struct

from tests.conftest import AUTH, last_call, sse_events
from tests.mock_mantle import EXPIRED_TOKEN

TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]


def chat(client, model, stream=False, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": "Hello"}], "stream": stream, **extra}
    return client.post("/v1/chat/completions", json=body, headers=AUTH)


def test_health_and_auth(gateway):
    client = gateway()
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").status_code == 401
    r = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
    assert client.get("/v1/models", headers={"x-api-key": "test-key"}).status_code == 200


def test_missing_api_key_config_is_rejected(gateway):
    client = gateway(api_key="")
    assert client.get("/v1/models", headers=AUTH).status_code == 500
    assert gateway(api_key="", allow_no_auth=True).get("/v1/models").status_code == 200


def test_models_merge_aliases_and_hide(gateway, mock):
    client = gateway(
        models={
            "hide": ["*safeguard*"],
            "models": [
                {"id": "anthropic.claude-opus-5-5", "alias": "claude-opus", "name": "Claude Opus 5.5"},
                {"id": "anthropic.claude-sonnet-5-5", "name": "Claude Sonnet 5.5"},
                {"id": "amazon.titan-embed-text-v2:0", "alias": "titan", "type": "embedding"},
                {"id": "qwen.qwen3-32b", "hidden": True},
            ],
        }
    )
    data = client.get("/v1/models", headers=AUTH).json()["data"]
    ids = [m["id"] for m in data]
    assert ids[:2] == ["claude-opus", "anthropic.claude-sonnet-5-5"]
    assert data[0]["name"] == "Claude Opus 5.5" and data[0]["owned_by"] == "anthropic"
    assert "anthropic.claude-opus-5-5" not in ids  # replaced by its alias
    assert "openai.gpt-5.5" in ids and "openai.gpt-oss-120b" in ids  # discovered
    assert not any("safeguard" in i or "embed" in i for i in ids)
    assert "qwen.qwen3-32b" not in ids
    assert last_call(mock, "/v1/models")["headers"]["authorization"] == "Bearer good-token"


def test_allow_list(gateway):
    client = gateway(models={"allow_list": True, "models": [{"id": "openai.gpt-oss-120b", "alias": "oss"}]})
    assert [m["id"] for m in client.get("/v1/models", headers=AUTH).json()["data"]] == ["oss"]
    assert chat(client, "openai.gpt-5.5").status_code == 404
    assert chat(client, "oss").status_code == 200


def test_claude_non_stream(gateway, mock):
    client = gateway(models={"models": [{"id": "anthropic.claude-opus-5-5", "alias": "claude-opus"}]})
    r = chat(client, "claude-opus", temperature=0.5, reasoning_effort="high")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "claude-opus"
    assert body["choices"][0]["message"]["content"] == "Hi from anthropic.claude-opus-5-5"
    assert body["choices"][0]["message"]["reasoning_content"] == "Let me think."
    sent = last_call(mock, "/anthropic/v1/messages")
    assert sent["headers"]["anthropic-version"] == "2023-06-01"
    assert sent["json"]["model"] == "anthropic.claude-opus-5-5"
    assert "temperature" not in sent["json"]
    assert sent["json"]["output_config"] == {"effort": "high"}


def test_claude_stream_with_tools(gateway):
    client = gateway()
    r = chat(
        client, "anthropic.claude-opus-5-5", stream=True, tools=TOOLS, stream_options={"include_usage": True}
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    deltas = [e["choices"][0]["delta"] for e in events[:-1] if e["choices"]]
    assert {"reasoning_content": "Hmm."} in deltas
    args = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas if "tool_calls" in d)
    assert args == '{"city": "Paris"}'
    assert events[-3]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-2]["usage"]["completion_tokens"] == 12


def test_claude_style_ids_are_mapped(gateway, mock):
    client = gateway()
    assert chat(client, "claude-opus-5-5").status_code == 200
    assert last_call(mock, "/anthropic/v1/messages")["json"]["model"] == "anthropic.claude-opus-5-5"


BEDROCK_CLAUDE = {"models": [{"id": "eu.anthropic.claude-sonnet-5-5", "alias": "sonnet", "api": "bedrock"}]}


def test_claude_via_bedrock_runtime(gateway, mock):
    client = gateway(models=BEDROCK_CLAUDE)
    r = chat(client, "sonnet", reasoning_effort="low")
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "Hi from eu.anthropic.claude-sonnet-5-5"
    sent = last_call(mock, "/model/")
    assert sent["path"] == "/model/eu.anthropic.claude-sonnet-5-5/invoke"
    assert sent["json"]["anthropic_version"] == "bedrock-2023-05-31"
    assert sent["json"]["output_config"] == {"effort": "low"}


def test_claude_stream_via_bedrock_runtime(gateway):
    client = gateway(models=BEDROCK_CLAUDE)
    r = chat(client, "sonnet", stream=True, tools=TOOLS, stream_options={"include_usage": True})
    assert r.status_code == 200, r.text
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    deltas = [e["choices"][0]["delta"] for e in events[:-1] if e["choices"]]
    assert {"reasoning_content": "Hmm."} in deltas
    args = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas if "tool_calls" in d)
    assert args == '{"city": "Paris"}'
    assert events[-2]["usage"]["completion_tokens"] == 12


def test_gpt5_stream_via_responses(gateway, mock):
    client = gateway()
    r = chat(client, "openai.gpt-5.5", stream=True, max_tokens=100, temperature=0.3)
    events = sse_events(r.text)
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1] if e["choices"])
    reasoning = "".join(
        e["choices"][0]["delta"].get("reasoning_content", "") for e in events[:-1] if e["choices"]
    )
    assert content == "Yo from openai.gpt-5.5" and reasoning == "Thinking."
    sent = last_call(mock, "/openai/v1/responses")
    assert sent["json"]["max_output_tokens"] == 100 and "temperature" not in sent["json"]


def test_gpt5_non_stream_tool_call(gateway):
    r = chat(gateway(), "openai.gpt-5.5", tools=TOOLS)
    msg = r.json()["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_oss_and_openai_v1_chat_passthrough(gateway, mock):
    client = gateway()
    r = chat(client, "openai.gpt-oss-120b", stream=True, temperature=0.4)
    assert "Hello" in r.text and r.text.rstrip().endswith("[DONE]")
    sent = last_call(mock, "/v1/chat/completions")
    assert sent["json"]["temperature"] == 0.4 and sent["json"]["model"] == "openai.gpt-oss-120b"

    r = chat(client, "google.gemma-4-31b")
    assert r.json()["choices"][0]["message"]["content"] == "Hello from google.gemma-4-31b"
    assert last_call(mock, "/openai/v1/chat/completions")["json"]["model"] == "google.gemma-4-31b"


def test_forced_chat_route_for_gpt_maps_max_tokens(gateway, mock):
    client = gateway(models={"models": [{"id": "openai.gpt-5.5", "alias": "gpt-chat", "api": "chat"}]})
    assert chat(client, "gpt-chat", max_tokens=50).status_code == 200
    sent = last_call(mock, "/openai/v1/chat/completions")["json"]
    assert sent["max_completion_tokens"] == 50 and "max_tokens" not in sent


def test_upstream_error_is_openai_shaped(gateway):
    r = chat(gateway(), "anthropic.claude-nonexistent-9", stream=True)
    assert r.status_code == 404
    assert "not found" in r.json()["error"]["message"]


def test_bad_request_errors(gateway):
    client = gateway()
    r = client.post("/v1/chat/completions", content=b"not json", headers=AUTH)
    assert r.status_code == 400
    assert chat(client, "").status_code == 400
    r = client.post(
        "/v1/chat/completions",
        json={"model": "anthropic.claude-opus-5-5", "messages": [{"role": "wizard", "content": "x"}]},
        headers=AUTH,
    )
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


def test_embeddings_titan_and_cohere(gateway, mock):
    client = gateway(
        models={"models": [{"id": "amazon.titan-embed-text-v2:0", "alias": "titan", "dimensions": 4}]}
    )
    r = client.post("/v1/embeddings", json={"model": "titan", "input": ["a b", "c"]}, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [d["index"] for d in body["data"]] == [0, 1] and body["data"][0]["embedding"] == [0.25] * 4
    assert body["usage"]["prompt_tokens"] == 3 and body["model"] == "titan"
    call = last_call(mock, "/model/")
    assert call["path"] == "/model/amazon.titan-embed-text-v2:0/invoke"
    assert call["headers"]["authorization"] == "Bearer good-token"

    r = client.post(
        "/v1/embeddings",
        json={"model": "cohere.embed-v4:0", "input": "hello", "encoding_format": "base64"},
        headers=AUTH,
    )
    vec = struct.unpack("<2f", base64.b64decode(r.json()["data"][0]["embedding"]))
    assert vec == (0.5, 0.5)
    assert (
        client.post(
            "/v1/embeddings", json={"model": "openai.gpt-5.5", "input": "x"}, headers=AUTH
        ).status_code
        == 400
    )


def test_anthropic_passthrough(gateway, mock):
    client = gateway(models={"models": [{"id": "anthropic.claude-opus-5-5", "alias": "opus"}]})
    r = client.post(
        "/v1/messages?beta=true",
        json={
            "model": "opus",
            "max_tokens": 10,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "foo-2025-01-01",
        },
    )
    assert (
        r.status_code == 200 and "event: message_start" in r.text and "text_delta" in r.text
    )  # raw Anthropic SSE
    sent = last_call(mock, "/anthropic/v1/messages")
    assert sent["query"] == "beta=true"
    assert sent["headers"]["anthropic-beta"] == "foo-2025-01-01"
    assert sent["headers"]["authorization"] == "Bearer good-token"
    assert "x-api-key" not in sent["headers"]
    assert sent["json"]["model"] == "anthropic.claude-opus-5-5"

    r = client.post(
        "/v1/messages/count_tokens",
        json={"model": "opus", "messages": [{"role": "user", "content": "hi"}]},
        headers=AUTH,
    )
    assert r.json() == {"input_tokens": 42}
    assert (
        last_call(mock, "/anthropic/v1/messages/count_tokens")["headers"]["anthropic-version"] == "2023-06-01"
    )

    r = client.post(
        "/v1/messages", json={"model": "anthropic.claude-nonexistent", "messages": []}, headers=AUTH
    )
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"  # relayed verbatim


def test_responses_passthrough(gateway, mock):
    client = gateway()
    r = client.post("/v1/responses", json={"model": "openai.gpt-5.5", "input": "hi"}, headers=AUTH)
    assert r.json()["id"] == "resp_mock"
    assert last_call(mock, "/openai/v1/responses")["json"]["input"] == "hi"
    client.post(
        "/v1/responses", json={"model": "openai.gpt-oss-120b", "input": "hi", "stream": True}, headers=AUTH
    )
    assert last_call(mock, "/v1/responses")["path"] == "/v1/responses"


def test_iam_token_refresh_on_401(gateway, mock):
    tokens = iter([EXPIRED_TOKEN, "fresh-token"])
    client = gateway(aws_bearer_token_bedrock=None, minter=lambda: next(tokens))
    r = chat(client, "openai.gpt-oss-120b")
    assert r.status_code == 200
    auths = [c["headers"]["authorization"] for c in mock.state.calls if c["path"] == "/v1/chat/completions"]
    assert auths == [f"Bearer {EXPIRED_TOKEN}", "Bearer fresh-token"]
