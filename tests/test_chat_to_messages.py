import json

import pytest

from mantle_gateway.catalog import resolve_entry
from mantle_gateway.config import ModelEntry, Settings
from mantle_gateway.translate.chat_to_messages import (
    MessagesStreamTranslator,
    build_messages_request,
    convert_messages,
    messages_response_to_chat,
)
from mantle_gateway.translate.common import TranslationError

S = Settings(api_key="k")
OPUS = resolve_entry("anthropic.claude-opus-5-5", None)
HAIKU45 = resolve_entry("anthropic.claude-haiku-4-5", None)
PNG = "data:image/png;base64,iVBORw0KGgo="


def test_system_and_basic_messages():
    req = build_messages_request(
        {
            "model": "x",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "developer", "content": [{"type": "text", "text": "Use British English."}]},
                {"role": "user", "content": "Hi"},
            ],
        },
        OPUS,
        S,
        stream=True,
    )
    assert req["model"] == "anthropic.claude-opus-5-5"
    assert req["system"] == "Be brief.\n\nUse British English."
    assert req["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    assert req["max_tokens"] == 64000
    assert req["stream"] is True
    assert req["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_sampling_dropped_for_current_claude_and_effort_mapped():
    req = build_messages_request(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "frequency_penalty": 1,
            "seed": 3,
            "reasoning_effort": "minimal",
            "max_tokens": 1000,
        },
        OPUS,
        S,
        stream=False,
    )
    for key in ("temperature", "top_p", "frequency_penalty", "seed"):
        assert key not in req
    assert req["output_config"] == {"effort": "low"}
    assert req["max_tokens"] == 1000


def test_legacy_model_budget_thinking_and_sampling():
    no_effort = build_messages_request(
        {"model": "x", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0.3, "top_p": 0.5},
        HAIKU45,
        S,
        stream=False,
    )
    assert "thinking" not in no_effort
    assert no_effort["temperature"] == 0.3 and "top_p" not in no_effort  # never both
    assert no_effort["max_tokens"] == 8192

    with_effort = build_messages_request(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "high",
            "temperature": 0.3,
        },
        HAIKU45,
        S,
        stream=False,
    )
    assert with_effort["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert with_effort["max_tokens"] > 16384
    assert "temperature" not in with_effort  # sampling not allowed with thinking


def test_images_tools_and_tool_results():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {"type": "image_url", "image_url": {"url": PNG}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                },
                {"id": "call_b", "type": "function", "function": {"name": "time", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "Sunny"},
        {"role": "tool", "tool_call_id": "call_b", "content": "12:00"},
        {"role": "user", "content": "Thanks"},
    ]
    system, msgs = convert_messages(messages)
    assert system == ""
    assert msgs[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
    }
    assert [b["type"] for b in msgs[1]["content"]] == ["tool_use", "tool_use"]
    assert msgs[1]["content"][0]["input"] == {"city": "Paris"}
    # both tool results + the follow-up text merged into one user turn, results first
    assert msgs[2]["role"] == "user"
    assert [b["type"] for b in msgs[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert len(msgs) == 3


def test_tool_pair_repair():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "user", "content": "never mind"},
    ]
    _, msgs = convert_messages(messages)
    results = [b for b in msgs[2]["content"] if b["type"] == "tool_result"]
    assert results and results[0]["tool_use_id"] == "call_a" and results[0]["is_error"] is True

    orphan = [{"role": "tool", "tool_call_id": "zzz", "content": "data"}, {"role": "user", "content": "hi"}]
    _, msgs = convert_messages(orphan)
    assert all(b["type"] == "text" for b in msgs[0]["content"])


def test_tool_definitions_and_choice():
    tools = [
        {
            "type": "function",
            "function": {"name": "weather", "description": "d", "parameters": {"type": "object"}},
        }
    ]
    body = {"model": "x", "messages": [{"role": "user", "content": "Hi"}], "tools": tools}
    req = build_messages_request(
        {**body, "tool_choice": "required", "parallel_tool_calls": False}, OPUS, S, False
    )
    assert req["tools"] == [{"name": "weather", "input_schema": {"type": "object"}, "description": "d"}]
    # Opus 5.5 rejects forced tool use -> downgraded to auto
    assert req["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

    named = {"type": "function", "function": {"name": "weather"}}
    # Sonnet 4.6 accepts forced tool use, but not while (adaptive) thinking is on
    older = resolve_entry("anthropic.claude-sonnet-4-6", None)
    req = build_messages_request({**body, "tool_choice": named}, older, S, False)
    assert req["tool_choice"] == {"type": "auto"} and req["thinking"]["type"] == "adaptive"
    no_thinking = resolve_entry(
        "anthropic.claude-sonnet-4-6", ModelEntry(id="anthropic.claude-sonnet-4-6", thinking="none")
    )
    req = build_messages_request({**body, "tool_choice": named}, no_thinking, S, False)
    assert req["tool_choice"] == {"type": "tool", "name": "weather"} and "thinking" not in req
    # Haiku 4.5 (budget thinking): forced choice kept unless an effort turns thinking on
    req = build_messages_request({**body, "tool_choice": "required"}, HAIKU45, S, False)
    assert req["tool_choice"] == {"type": "any"}
    req = build_messages_request(
        {**body, "tool_choice": "required", "reasoning_effort": "low"}, HAIKU45, S, False
    )
    assert req["tool_choice"] == {"type": "auto"}
    req = build_messages_request({**body, "tool_choice": "none"}, older, S, False)
    assert req["tool_choice"] == {"type": "none"}


def test_response_format_and_stop_and_user():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    req = build_messages_request(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}},
            "stop": "END",
            "user": "alice",
            "reasoning_effort": "high",
        },
        OPUS,
        S,
        False,
    )
    assert req["output_config"] == {"format": {"type": "json_schema", "schema": schema}, "effort": "high"}
    assert req["stop_sequences"] == ["END"]
    assert "metadata" not in req and "user" not in req
    req = build_messages_request(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {"type": "json_object"},
        },
        OPUS,
        S,
        False,
    )
    assert "JSON" in req["system"]


def test_model_overrides_from_config():
    entry = ModelEntry(
        id="anthropic.claude-opus-5-5", max_tokens=1234, default_effort="xhigh", thinking="none"
    )
    rm = resolve_entry(entry.id, entry)
    req = build_messages_request({"model": "x", "messages": [{"role": "user", "content": "Hi"}]}, rm, S, True)
    assert req["max_tokens"] == 1234
    assert "thinking" not in req and "output_config" not in req


def test_bad_inputs():
    with pytest.raises(TranslationError):
        build_messages_request({"model": "x", "messages": []}, OPUS, S, False)
    with pytest.raises(TranslationError):
        convert_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "ftp://x"}}]}]
        )


def test_non_stream_response():
    resp = {
        "id": "msg_1",
        "content": [
            {"type": "thinking", "thinking": "Considering", "signature": "s"},
            {"type": "text", "text": "Answer"},
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
    }
    out = messages_response_to_chat(resp, "claude-opus", S)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Answer"
    assert msg["reasoning_content"] == "Considering"
    assert msg["tool_calls"][0]["function"] == {"name": "f", "arguments": json.dumps({"a": 1})}
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["usage"]["prompt_tokens"] == 15 and out["usage"]["completion_tokens"] == 5
    assert out["model"] == "claude-opus"

    think = messages_response_to_chat(resp, "m", Settings(api_key="k", reasoning_format="think_tags"))
    assert think["choices"][0]["message"]["content"].startswith("<think>\nConsidering\n</think>")


EVENTS = [
    ("message_start", {"type": "message_start", "message": {"id": "msg_9", "usage": {"input_tokens": 7}}}),
    (
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "content_block_start",
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_x", "name": "f"},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"a":'},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": "1}"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 2}),
    (
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
    ),
    ("message_stop", {"type": "message_stop"}),
]


def _run(translator, events):
    chunks = []
    for ev, data in events:
        chunks += translator.feed(ev, data)
    chunks += translator.finish()
    return chunks


def test_stream_translation():
    chunks = _run(MessagesStreamTranslator("m", S, include_usage=True), EVENTS)
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert {"reasoning_content": "hm"} in deltas
    assert {"content": "Hi"} in deltas
    tool_deltas = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert tool_deltas[0]["id"] == "toolu_x" and tool_deltas[0]["function"]["name"] == "f"
    assert "".join(t["function"].get("arguments", "") for t in tool_deltas) == '{"a":1}'
    finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finals[-1]["choices"][0]["finish_reason"] == "tool_calls"
    usage = chunks[-1]
    assert (
        usage["choices"] == []
        and usage["usage"]["prompt_tokens"] == 7
        and usage["usage"]["completion_tokens"] == 9
    )
    assert all(c["id"] == "chatcmpl-msg_9" for c in chunks[1:])


def test_stream_think_tags_and_error():
    s = Settings(api_key="k", reasoning_format="think_tags")
    chunks = _run(MessagesStreamTranslator("m", s, include_usage=False), EVENTS[:7] + EVENTS[-2:])
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert content == "<think>\nhm\n</think>\n\nHi"

    err = MessagesStreamTranslator("m", S, include_usage=False)
    out = err.feed("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    assert out[0]["error"]["message"] == "Overloaded"
    assert err.done and err.finish() == []
