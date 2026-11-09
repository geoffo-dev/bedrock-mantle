"""A small fake of Bedrock Mantle + bedrock-runtime, for tests and the no-AWS docker compose demo.

It emulates:
  GET  /v1/models
  POST /v1/chat/completions, /openai/v1/chat/completions          (OpenAI chat, JSON or SSE)
  POST /v1/responses, /openai/v1/responses                          (OpenAI Responses, JSON or SSE)
  POST /anthropic/v1/messages[/count_tokens]                        (Anthropic Messages, JSON or SSE)
  POST /model/{model_id}/invoke                                     (Titan / Cohere embeddings)

Every request is recorded in app.state.calls so tests can assert on what the gateway sent.
Run standalone:  uvicorn tests.mock_mantle:app --port 9000
"""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODELS = [
    "openai.gpt-oss-120b",
    "openai.gpt-oss-safeguard-20b",
    "openai.gpt-5.5",
    "anthropic.claude-opus-5-5",
    "anthropic.claude-haiku-4-5",
    "qwen.qwen3-32b",
    "google.gemma-4-31b",
    "amazon.titan-embed-text-v2:0",
]

EXPIRED_TOKEN = "expired-token"


def _sse(obj: dict, event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(obj)}\n\n"


def create_mock_app() -> FastAPI:
    app = FastAPI()
    app.state.calls = []

    @app.middleware("http")
    async def record_and_auth(request: Request, call_next):
        body = await request.body()
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            parsed = None
        app.state.calls.append(
            {
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "json": parsed,
            }
        )
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth == f"Bearer {EXPIRED_TOKEN}":
            return JSONResponse(
                status_code=401, content={"message": "The security token included is invalid."}
            )
        return await call_next(request)

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": m.split(".")[0]} for m in MODELS],
        }

    # ---------------------------------------------------------------- OpenAI chat
    @app.post("/v1/chat/completions")
    @app.post("/openai/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        model = body.get("model")
        if body.get("stream"):

            def gen():
                for i, piece in enumerate(["Hello", " from ", model]):
                    delta = {"role": "assistant", "content": piece} if i == 0 else {"content": piece}
                    yield _sse(
                        {
                            "id": "chatcmpl-mock",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                    )
                yield _sse(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Hello from {model}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    # ---------------------------------------------------------------- Anthropic Messages
    @app.post("/anthropic/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        return {"input_tokens": 42}

    @app.post("/anthropic/v1/messages")
    async def messages(request: Request):
        if not request.headers.get("anthropic-version"):
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "missing version"},
                },
            )
        body = await request.json()
        model = body.get("model", "")
        if "nonexistent" in model:
            return JSONResponse(
                status_code=404,
                content={
                    "type": "error",
                    "error": {"type": "not_found_error", "message": f"model {model} not found"},
                },
            )
        wants_tool = bool(body.get("tools"))
        text = f"Hi from {model}"
        if not body.get("stream"):
            content = [{"type": "thinking", "thinking": "Let me think.", "signature": "sig"}]
            if wants_tool:
                content.append(
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": body["tools"][0]["name"],
                        "input": {"city": "Paris"},
                    }
                )
            else:
                content.append({"type": "text", "text": text})
            return {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": "tool_use" if wants_tool else "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 7, "cache_read_input_tokens": 2},
            }

        def gen():
            yield _sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_mock",
                        "model": model,
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                },
                "message_start",
            )
            yield _sse({"type": "ping"}, "ping")
            yield _sse(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                "content_block_start",
            )
            yield _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Hmm."},
                },
                "content_block_delta",
            )
            yield _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "x"},
                },
                "content_block_delta",
            )
            yield _sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
            if wants_tool:
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": body["tools"][0]["name"],
                            "input": {},
                        },
                    },
                    "content_block_start",
                )
                for part in ['{"city": ', '"Paris"}']:
                    yield _sse(
                        {
                            "type": "content_block_delta",
                            "index": 1,
                            "delta": {"type": "input_json_delta", "partial_json": part},
                        },
                        "content_block_delta",
                    )
                yield _sse({"type": "content_block_stop", "index": 1}, "content_block_stop")
                stop = "tool_use"
            else:
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {"type": "text", "text": ""},
                    },
                    "content_block_start",
                )
                for piece in ["Hi from ", model]:
                    yield _sse(
                        {
                            "type": "content_block_delta",
                            "index": 1,
                            "delta": {"type": "text_delta", "text": piece},
                        },
                        "content_block_delta",
                    )
                yield _sse({"type": "content_block_stop", "index": 1}, "content_block_stop")
                stop = "end_turn"
            yield _sse(
                {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 12}},
                "message_delta",
            )
            yield _sse({"type": "message_stop"}, "message_stop")

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---------------------------------------------------------------- OpenAI Responses
    @app.post("/v1/responses")
    @app.post("/openai/v1/responses")
    async def responses(request: Request):
        body = await request.json()
        model = body.get("model")
        wants_tool = bool(body.get("tools"))
        usage = {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}
        output = [
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Thinking."}]}
        ]
        if wants_tool:
            output.append(
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": body["tools"][0]["name"],
                    "arguments": '{"city": "Paris"}',
                }
            )
        else:
            output.append(
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"Yo from {model}"}],
                }
            )
        final = {
            "id": "resp_mock",
            "object": "response",
            "status": "completed",
            "model": model,
            "output": output,
            "usage": usage,
        }
        if not body.get("stream"):
            return final

        def gen():
            yield _sse({"type": "response.created", "response": {"id": "resp_mock", "status": "in_progress"}})
            yield _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "reasoning", "id": "rs_1", "summary": []},
                }
            )
            yield _sse(
                {"type": "response.reasoning_summary_part.added", "item_id": "rs_1", "summary_index": 0}
            )
            yield _sse(
                {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "Thinking."}
            )
            if wants_tool:
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": 1,
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": body["tools"][0]["name"],
                            "arguments": "",
                        },
                    }
                )
                for part in ['{"city": ', '"Paris"}']:
                    yield _sse(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": "fc_1",
                            "output_index": 1,
                            "delta": part,
                        }
                    )
                yield _sse({"type": "response.output_item.done", "output_index": 1, "item": output[1]})
            else:
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": 1,
                        "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []},
                    }
                )
                for piece in ["Yo from ", model]:
                    yield _sse(
                        {
                            "type": "response.output_text.delta",
                            "item_id": "msg_1",
                            "output_index": 1,
                            "delta": piece,
                        }
                    )
            yield _sse({"type": "response.completed", "response": final})

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---------------------------------------------------------------- bedrock-runtime embeddings
    @app.post("/model/{model_id:path}/invoke")
    async def invoke(model_id: str, request: Request):
        body = await request.json()
        if "titan-embed" in model_id:
            dims = body.get("dimensions") or 4
            return {"embedding": [0.25] * dims, "inputTextTokenCount": len(body["inputText"].split())}
        if "cohere.embed-v4" in model_id:
            return {"embeddings": {"float": [[0.5, 0.5] for _ in body["texts"]]}}
        if "cohere.embed" in model_id:
            return {"embeddings": [[0.5, 0.5] for _ in body["texts"]]}
        return JSONResponse(status_code=400, content={"message": f"Unsupported model {model_id}"})

    return app


app = create_mock_app()
