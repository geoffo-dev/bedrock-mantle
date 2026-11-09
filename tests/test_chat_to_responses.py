from mantle_gateway.catalog import resolve_entry
from mantle_gateway.config import Settings
from mantle_gateway.translate.chat_to_responses import (
    ResponsesStreamTranslator,
    build_responses_request,
    responses_to_chat,
)

S = Settings(api_key="k")
GPT = resolve_entry("openai.gpt-5.5", None)


def test_request_mapping():
    body = {
        "model": "gpt",
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            },
            {
                "role": "assistant",
                "content": "Calling tool",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "w", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ],
        "tools": [{"type": "function", "function": {"name": "w", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "w"}},
        "max_tokens": 500,
        "temperature": 0.1,
        "reasoning_effort": "high",
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": {"type": "object"}},
        },
    }
    req = build_responses_request(body, GPT, S, stream=True)
    assert req["model"] == "openai.gpt-5.5"
    assert req["instructions"] == "Be helpful."
    assert req["input"][0] == {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Hi"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
        ],
    }
    assert req["input"][1] == {"role": "assistant", "content": "Calling tool"}
    assert req["input"][2]["type"] == "function_call" and req["input"][2]["call_id"] == "call_1"
    assert req["input"][3] == {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"}
    assert req["tools"] == [
        {"type": "function", "name": "w", "parameters": {"type": "object"}, "strict": False}
    ]
    assert req["tool_choice"] == {"type": "function", "name": "w"}
    assert req["max_output_tokens"] == 500
    assert "temperature" not in req
    assert req["reasoning"] == {"effort": "high", "summary": "auto"}
    assert req["text"]["format"]["type"] == "json_schema" and req["text"]["format"]["name"] == "out"
    assert req["stream"] is True


def test_non_stream_response():
    resp = {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thought"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]},
            {"type": "function_call", "call_id": "call_9", "name": "f", "arguments": '{"x":1}'},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 2}},
    }
    out = responses_to_chat(resp, "gpt", S)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Hello" and msg["reasoning_content"] == "Thought"
    assert msg["tool_calls"][0]["id"] == "call_9"
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2

    incomplete = responses_to_chat(
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
        "gpt",
        S,
    )
    assert incomplete["choices"][0]["finish_reason"] == "length"


def test_stream_translation():
    events = [
        {"type": "response.created", "response": {"id": "resp_7"}},
        {"type": "response.reasoning_summary_part.added", "summary_index": 0},
        {"type": "response.reasoning_summary_text.delta", "delta": "Think"},
        {"type": "response.reasoning_summary_part.added", "summary_index": 1},
        {"type": "response.reasoning_summary_text.delta", "delta": "More"},
        {"type": "response.output_text.delta", "delta": "Hi"},
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "f",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 2,
            "delta": "{}",
        },
        {
            "type": "response.output_item.done",
            "output_index": 2,
            "item": {"type": "function_call", "id": "fc_1", "arguments": "{}"},
        },
        {
            "type": "response.completed",
            "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 2}},
        },
    ]
    t = ResponsesStreamTranslator("gpt", S, include_usage=True)
    chunks = []
    for e in events:
        chunks += t.feed(None, e)
    chunks += t.finish()
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert deltas[0] == {"role": "assistant", "content": ""}
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    assert reasoning == "Think\n\nMore"
    assert {"content": "Hi"} in deltas
    tool = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert tool[0]["id"] == "call_1"
    assert "".join(x["function"].get("arguments", "") for x in tool) == "{}"  # not duplicated by .done
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["usage"]["total_tokens"] == 3
    assert chunks[0]["id"] == "chatcmpl-resp_7"


def test_stream_failure():
    t = ResponsesStreamTranslator("gpt", S, include_usage=False)
    out = t.feed(
        None, {"type": "response.failed", "response": {"error": {"code": "server_error", "message": "boom"}}}
    )
    assert out[-1]["error"]["message"] == "boom"
    assert t.finish() == []
