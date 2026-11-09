"""OpenAI Chat Completions <-> OpenAI Responses API (numbered GPT models on Mantle's /openai/v1)."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ..catalog import ResolvedModel
from ..config import Settings
from .common import TranslationError, clamp_effort_openai, content_parts, reasoning_fields, text_of


def _user_content(content: Any) -> list[dict]:
    out: list[dict] = []
    for part in content_parts(content):
        ptype = part.get("type")
        if ptype in ("text", "input_text"):
            if part.get("text"):
                out.append({"type": "input_text", "text": part["text"]})
        elif ptype == "image_url":
            iu = part.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if url:
                item = {"type": "input_image", "image_url": url}
                if isinstance(iu, dict) and iu.get("detail"):
                    item["detail"] = iu["detail"]
                out.append(item)
        elif ptype == "file":
            f = part.get("file") or {}
            if f.get("file_data"):
                out.append(
                    {
                        "type": "input_file",
                        "file_data": f["file_data"],
                        "filename": f.get("filename") or "file",
                    }
                )
            elif f.get("file_id"):
                out.append({"type": "input_file", "file_id": f["file_id"]})
    return out


def convert_input(messages: list[dict]) -> tuple[str, list[dict]]:
    instructions: list[str] = []
    items: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            text = text_of(msg.get("content"))
            if text:
                instructions.append(text)
        elif role == "user":
            content = _user_content(msg.get("content"))
            if content:
                items.append({"role": "user", "content": content})
        elif role == "assistant":
            text = text_of(msg.get("content")) or msg.get("refusal") or ""
            if text:
                items.append({"role": "assistant", "content": text})
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                args = fn.get("arguments")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "name": fn.get("name") or "unknown_tool",
                        "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                    }
                )
        elif role == "tool":
            content = msg.get("content")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": content if isinstance(content, str) else text_of(content),
                }
            )
        else:
            raise TranslationError(f"Unsupported message role: {role!r}")
    return "\n\n".join(instructions), items


def _convert_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        if t.get("type") not in (None, "function"):
            continue
        fn = t.get("function") or t
        tool = {
            "type": "function",
            "name": fn["name"],
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            # Responses defaults function tools to strict mode; chat clients rarely send strict-ready schemas.
            "strict": bool(fn.get("strict", False)),
        }
        if fn.get("description"):
            tool["description"] = fn["description"]
        out.append(tool)
    return out


def _convert_tool_choice(choice: Any) -> Any:
    if choice in (None, "auto", "none", "required"):
        return choice
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        if name:
            return {"type": "function", "name": name}
    return "auto"


def build_responses_request(body: dict, model: ResolvedModel, settings: Settings, stream: bool) -> dict:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TranslationError("'messages' must be a non-empty list")
    instructions, items = convert_input(messages)
    req: dict[str, Any] = {"model": model.upstream_id, "input": items}
    if instructions:
        req["instructions"] = instructions
    if stream:
        req["stream"] = True

    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or model.max_tokens
    if max_tokens:
        req["max_output_tokens"] = int(max_tokens)

    tools = body.get("tools") or []
    if tools:
        converted = _convert_tools(tools)
        if converted:
            req["tools"] = converted
            choice = _convert_tool_choice(body.get("tool_choice"))
            if choice is not None:
                req["tool_choice"] = choice
            if body.get("parallel_tool_calls") is not None:
                req["parallel_tool_calls"] = bool(body["parallel_tool_calls"])

    reasoning: dict[str, Any] = {}
    effort = clamp_effort_openai(body.get("reasoning_effort") or model.default_effort)
    if effort:
        reasoning["effort"] = effort
    if settings.reasoning_format != "none" and effort != "none":
        reasoning["summary"] = "auto"
    if reasoning:
        req["reasoning"] = reasoning

    text: dict[str, Any] = {}
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        fmt = {"type": "json_schema", "name": js.get("name") or "response", "schema": js.get("schema") or {}}
        if js.get("strict") is not None:
            fmt["strict"] = js["strict"]
        text["format"] = fmt
    elif rf.get("type") == "json_object":
        text["format"] = {"type": "json_object"}
    if body.get("verbosity"):
        text["verbosity"] = body["verbosity"]
    if text:
        req["text"] = text

    if model.allow_sampling:
        for key in ("temperature", "top_p"):
            if body.get(key) is not None:
                req[key] = body[key]
    return req


def _usage(usage: dict | None) -> dict:
    usage = usage or {}
    prompt = usage.get("input_tokens") or 0
    completion = usage.get("output_tokens") or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": usage.get("total_tokens") or prompt + completion,
        "prompt_tokens_details": {
            "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        },
        "completion_tokens_details": {
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        },
    }


def _finish_reason(resp: dict, has_tool_calls: bool) -> str:
    if resp.get("status") == "incomplete":
        reason = (resp.get("incomplete_details") or {}).get("reason")
        return "content_filter" if reason == "content_filter" else "length"
    return "tool_calls" if has_tool_calls else "stop"


def responses_to_chat(resp: dict, public_model: str, settings: Settings) -> dict:
    texts, thoughts, tool_calls = [], [], []
    for item in resp.get("output") or []:
        itype = item.get("type")
        if itype == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    texts.append(c.get("text") or "")
                elif c.get("type") == "refusal":
                    texts.append(c.get("refusal") or "")
        elif itype == "reasoning":
            for s in item.get("summary") or []:
                if s.get("text"):
                    thoughts.append(s["text"])
            if not item.get("summary"):
                for c in item.get("content") or []:
                    if c.get("text"):
                        thoughts.append(c["text"])
        elif itype == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"},
                }
            )
    content = "".join(texts)
    message: dict[str, Any] = {"role": "assistant", "content": content or (None if tool_calls else "")}
    message.update(reasoning_fields("\n\n".join(thoughts), settings.reasoning_format, message))
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{resp.get('id') or uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(resp.get("created_at") or time.time()),
        "model": public_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish_reason(resp, bool(tool_calls)),
                "logprobs": None,
            }
        ],
        "usage": _usage(resp.get("usage")),
    }


class ResponsesStreamTranslator:
    """Turns Responses API SSE events into OpenAI chat.completion.chunk objects."""

    def __init__(self, public_model: str, settings: Settings, include_usage: bool) -> None:
        self.model = public_model
        self.reasoning_format = settings.reasoning_format
        self.include_usage = include_usage
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.tools: dict[str, int] = {}  # item id -> tool index
        self.tools_by_output: dict[int, int] = {}
        self.args_streamed: set[int] = set()
        self.in_think = False
        self.started = False
        self.done = False
        self.summary_parts = 0

    def _chunk(self, delta: dict, finish: str | None = None) -> dict:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
        }

    def _start(self) -> list[dict]:
        if self.started:
            return []
        self.started = True
        return [self._chunk({"role": "assistant", "content": ""})]

    def _close_think(self) -> list[dict]:
        if self.in_think:
            self.in_think = False
            return [self._chunk({"content": "\n</think>\n\n"})]
        return []

    def _reasoning(self, text: str) -> list[dict]:
        if not text or self.reasoning_format == "none":
            return []
        if self.reasoning_format == "reasoning_content":
            return [self._chunk({"reasoning_content": text})]
        prefix = "" if self.in_think else "<think>\n"
        self.in_think = True
        return [self._chunk({"content": prefix + text})]

    def _tool_idx(self, data: dict) -> int | None:
        if data.get("item_id") in self.tools:
            return self.tools[data["item_id"]]
        return self.tools_by_output.get(data.get("output_index", -1))

    def feed(self, event: str | None, data: dict) -> list[dict]:
        etype = data.get("type") or event or ""
        out = self._start() if etype != "error" else []
        if etype == "response.created":
            rid = (data.get("response") or {}).get("id")
            if rid:
                self.id = f"chatcmpl-{rid}"
                for c in out:
                    c["id"] = self.id
            return out
        if etype == "response.output_text.delta" or etype == "response.refusal.delta":
            text = data.get("delta") or ""
            return [*out, *self._close_think(), self._chunk({"content": text})] if text else out
        if etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            return [*out, *self._reasoning(data.get("delta") or "")]
        if etype == "response.reasoning_summary_part.added":
            self.summary_parts += 1
            return [*out, *self._reasoning("\n\n")] if self.summary_parts > 1 else out
        if etype == "response.output_item.added":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                tidx = len(self.tools_by_output)
                self.tools_by_output[data.get("output_index", tidx)] = tidx
                if item.get("id"):
                    self.tools[item["id"]] = tidx
                out += self._close_think()
                call = {
                    "index": tidx,
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments") or ""},
                }
                if item.get("arguments"):
                    self.args_streamed.add(tidx)
                out.append(self._chunk({"tool_calls": [call]}))
            return out
        if etype == "response.function_call_arguments.delta":
            tidx = self._tool_idx(data)
            if tidx is not None and data.get("delta"):
                self.args_streamed.add(tidx)
                out.append(
                    self._chunk({"tool_calls": [{"index": tidx, "function": {"arguments": data["delta"]}}]})
                )
            return out
        if etype == "response.output_item.done":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                tidx = self._tool_idx(
                    {"item_id": item.get("id"), "output_index": data.get("output_index", -1)}
                )
                if tidx is not None and tidx not in self.args_streamed and item.get("arguments"):
                    self.args_streamed.add(tidx)
                    out.append(
                        self._chunk(
                            {"tool_calls": [{"index": tidx, "function": {"arguments": item["arguments"]}}]}
                        )
                    )
            return out
        if etype in ("response.completed", "response.incomplete"):
            resp = data.get("response") or {}
            return [*out, *self._final(_finish_reason(resp, bool(self.tools_by_output)), resp.get("usage"))]
        if etype == "response.failed":
            err = (data.get("response") or {}).get("error") or {}
            self.done = True
            return [
                *out,
                {
                    "error": {
                        "message": err.get("message") or "Response failed",
                        "type": err.get("code") or "upstream_error",
                        "code": err.get("code"),
                    }
                },
            ]
        if etype == "error":
            self.done = True
            return [
                {
                    "error": {
                        "message": data.get("message") or "Upstream stream error",
                        "type": "upstream_error",
                        "code": data.get("code"),
                    }
                }
            ]
        return out

    def _final(self, finish: str, usage: dict | None) -> list[dict]:
        if self.done:
            return []
        self.done = True
        out = [*self._close_think(), self._chunk({}, finish)]
        if self.include_usage:
            out.append(
                {
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": _usage(usage),
                }
            )
        return out

    def finish(self) -> list[dict]:
        return self._final("tool_calls" if self.tools_by_output else "stop", None)
