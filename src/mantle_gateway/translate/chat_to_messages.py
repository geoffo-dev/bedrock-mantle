"""OpenAI Chat Completions <-> Anthropic Messages (Claude on Bedrock Mantle)."""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

from ..catalog import ResolvedModel
from ..config import Settings
from .common import (
    TranslationError,
    clamp_effort_claude,
    content_parts,
    data_uri_parts,
    reasoning_fields,
    stop_list,
    text_of,
)

SAMPLING_PARAMS = ("temperature", "top_p", "top_k")
BUDGET_BY_EFFORT = {"low": 2048, "medium": 8192, "high": 16384, "xhigh": 32000, "max": 32000}
DEFAULT_MAX_TOKENS_STREAM = 64000
DEFAULT_MAX_TOKENS = 16000
LEGACY_DEFAULT_MAX_TOKENS = 8192
JSON_OBJECT_INSTRUCTION = (
    "Respond with a single valid JSON object only, with no surrounding text or code fences."
)

STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


# ----------------------------------------------------------------------------------------------------
# Request
# ----------------------------------------------------------------------------------------------------


def _image_block(url: str) -> dict:
    parsed = data_uri_parts(url)
    if parsed:
        media_type, data = parsed
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    if url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise TranslationError("Unsupported image_url: expected a data: URI or an http(s) URL")


def _user_blocks(content: Any) -> list[dict]:
    blocks: list[dict] = []
    for part in content_parts(content):
        ptype = part.get("type")
        if ptype in ("text", "input_text"):
            if part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
        elif ptype == "image_url":
            iu = part.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if url:
                blocks.append(_image_block(url))
        elif ptype == "file":
            f = part.get("file") or {}
            parsed = data_uri_parts(f.get("file_data") or "")
            if parsed and parsed[0] == "application/pdf":
                blocks.append(
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": parsed[0], "data": parsed[1]},
                    }
                )
            elif parsed and parsed[0].startswith("text/"):
                text = base64.b64decode(parsed[1]).decode("utf-8", "replace")
                blocks.append(
                    {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": text}}
                )
            else:
                raise TranslationError(
                    "Only PDF or text files (as base64 data URIs) are supported for Claude"
                )
        # input_audio and other part types are silently dropped
    return blocks


def _tool_result_block(msg: dict) -> dict:
    content = msg.get("content")
    if isinstance(content, list):
        inner = _user_blocks(content)
        result: Any = [b for b in inner if b["type"] in ("text", "image")] or ""
    else:
        result = text_of(content)
    block = {"type": "tool_result", "tool_use_id": msg.get("tool_call_id") or "", "content": result}
    if result == "":
        block["content"] = "(empty result)"
    return block


def _assistant_blocks(msg: dict) -> list[dict]:
    blocks: list[dict] = []
    text = text_of(msg.get("content"))
    if not text and msg.get("refusal"):
        text = msg["refusal"]
    if text and text.strip():
        blocks.append({"type": "text", "text": text})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            args = {"_raw_arguments": raw}
        if not isinstance(args, dict):
            args = {"value": args}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name") or "unknown_tool",
                "input": args,
            }
        )
    return blocks


def convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return (system_text, anthropic_messages). Thinking from earlier turns is never replayed."""
    system_parts: list[str] = []
    out: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for msg in messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            text = text_of(msg.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            push("user", _user_blocks(msg.get("content")))
        elif role == "assistant":
            push("assistant", _assistant_blocks(msg))
        elif role in ("tool", "function"):
            if role == "function":
                msg = {"tool_call_id": msg.get("name"), "content": msg.get("content")}
            push("user", [_tool_result_block(msg)])
        else:
            raise TranslationError(f"Unsupported message role: {role!r}")

    _repair_tool_pairs(out)
    for m in out:
        if m["role"] == "user":  # tool_result blocks must lead the user turn
            m["content"].sort(key=lambda b: 0 if b["type"] == "tool_result" else 1)
    if out and out[-1]["role"] == "assistant":
        last = out[-1]["content"]
        if last and last[-1]["type"] == "text":
            last[-1]["text"] = last[-1]["text"].rstrip()
    return "\n\n".join(system_parts), out


def _repair_tool_pairs(msgs: list[dict]) -> None:
    """Make tool_use / tool_result pairing valid, as Anthropic requires.

    Missing results get a placeholder error result. Orphan results become plain text.
    """
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant":
            ids = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if ids:
                nxt = msgs[i + 1] if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user" else None
                if nxt is None:
                    nxt = {"role": "user", "content": []}
                    msgs.insert(i + 1, nxt)
                have = {b.get("tool_use_id") for b in nxt["content"] if b["type"] == "tool_result"}
                for tid in ids:
                    if tid not in have:
                        nxt["content"].append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tid,
                                "content": "(no result)",
                                "is_error": True,
                            }
                        )
                valid = set(ids)
                for idx, b in enumerate(nxt["content"]):
                    if b["type"] == "tool_result" and b.get("tool_use_id") not in valid:
                        nxt["content"][idx] = {
                            "type": "text",
                            "text": f"Tool result: {text_of(b.get('content'))}",
                        }
                i += 2
                continue
        if m["role"] == "user":  # not preceded by a tool_use turn: any tool_result here is orphaned
            for idx, b in enumerate(m["content"]):
                if b["type"] == "tool_result":
                    m["content"][idx] = {"type": "text", "text": f"Tool result: {text_of(b.get('content'))}"}
        i += 1


def _convert_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        if t.get("type") not in (None, "function"):
            continue
        fn = t.get("function") or t
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        tool = {"name": fn["name"], "input_schema": params}
        if fn.get("description"):
            tool["description"] = fn["description"]
        if fn.get("strict"):
            tool["strict"] = True
        out.append(tool)
    return out


def _convert_tool_choice(choice: Any, allow_forced: bool, parallel: Any) -> dict | None:
    result: dict | None
    if choice is None:
        result = {"type": "auto"} if parallel is False else None
    elif choice == "auto":
        result = {"type": "auto"}
    elif choice == "none":
        return {"type": "none"}
    elif choice == "required":
        result = {"type": "any"} if allow_forced else {"type": "auto"}
    elif isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        result = {"type": "tool", "name": name} if allow_forced and name else {"type": "auto"}
    else:
        result = {"type": "auto"}
    if result is not None and parallel is False:
        result["disable_parallel_tool_use"] = True
    return result


def build_messages_request(body: dict, model: ResolvedModel, settings: Settings, stream: bool) -> dict:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TranslationError("'messages' must be a non-empty list")
    system, msgs = convert_messages(messages)
    if not msgs:
        raise TranslationError("No user/assistant content to send")

    req: dict[str, Any] = {"model": model.upstream_id, "messages": msgs}

    default_max = model.max_tokens or (
        (DEFAULT_MAX_TOKENS_STREAM if stream else DEFAULT_MAX_TOKENS)
        if model.thinking != "budget"
        else LEGACY_DEFAULT_MAX_TOKENS
    )
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or default_max
    req["max_tokens"] = int(max_tokens)

    output_config: dict[str, Any] = {}
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        schema = (rf.get("json_schema") or {}).get("schema")
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
    elif rf.get("type") == "json_object":
        system = f"{system}\n\n{JSON_OBJECT_INSTRUCTION}" if system else JSON_OBJECT_INSTRUCTION

    if system:
        req["system"] = system

    stops = stop_list(body.get("stop"))
    if stops:
        req["stop_sequences"] = stops

    effort = clamp_effort_claude(body.get("reasoning_effort") or model.default_effort)
    thinking_on = False
    if model.thinking == "adaptive":
        thinking: dict[str, Any] = {"type": "adaptive"}
        if settings.reasoning_format != "none":
            thinking["display"] = "summarized"
        req["thinking"] = thinking
        thinking_on = True
        if effort:
            output_config["effort"] = effort
    elif model.thinking == "budget" and effort:
        budget = BUDGET_BY_EFFORT[effort]
        if req["max_tokens"] <= budget:
            req["max_tokens"] = budget + 4096
        req["thinking"] = {"type": "enabled", "budget_tokens": budget}
        thinking_on = True

    if output_config:
        req["output_config"] = output_config

    tools = body.get("tools") or []
    if body.get("functions") and not tools:  # legacy function-calling field
        tools = [{"type": "function", "function": f} for f in body["functions"]]
    if tools:
        converted = _convert_tools(tools)
        if converted:
            req["tools"] = converted
            # Forced tool use (any/tool) is rejected while thinking is on, and by some models outright.
            allow_forced = model.forced_tool_choice and not thinking_on
            choice = _convert_tool_choice(
                body.get("tool_choice"), allow_forced, body.get("parallel_tool_calls")
            )
            if choice:
                req["tool_choice"] = choice

    if model.allow_sampling and not thinking_on:
        if body.get("temperature") is not None:
            req["temperature"] = body["temperature"]
        elif body.get("top_p") is not None:  # several models reject temperature + top_p together
            req["top_p"] = body["top_p"]
        if body.get("top_k") is not None:
            req["top_k"] = body["top_k"]

    if settings.claude_prompt_caching:
        req["cache_control"] = {"type": "ephemeral"}
    if stream:
        req["stream"] = True
    return req


# ----------------------------------------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------------------------------------


def _usage(usage: dict | None) -> dict:
    usage = usage or {}
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    prompt = (usage.get("input_tokens") or 0) + cache_read + cache_write
    completion = usage.get("output_tokens") or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cache_read},
    }


def messages_response_to_chat(resp: dict, public_model: str, settings: Settings) -> dict:
    texts, thoughts, tool_calls = [], [], []
    for block in resp.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            texts.append(block.get("text") or "")
        elif btype == "thinking" and block.get("thinking"):
            thoughts.append(block["thinking"])
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    content = "".join(texts)
    message: dict[str, Any] = {"role": "assistant", "content": content or (None if tool_calls else "")}
    message.update(reasoning_fields("\n\n".join(thoughts), settings.reasoning_format, message))
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = STOP_REASON_MAP.get(resp.get("stop_reason") or "end_turn", "stop")
    return {
        "id": f"chatcmpl-{resp.get('id') or uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": public_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": _usage(resp.get("usage")),
    }


class MessagesStreamTranslator:
    """Turns Anthropic Messages SSE events into OpenAI chat.completion.chunk objects."""

    def __init__(self, public_model: str, settings: Settings, include_usage: bool) -> None:
        self.model = public_model
        self.reasoning_format = settings.reasoning_format
        self.include_usage = include_usage
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.blocks: dict[int, str] = {}
        self.tool_index: dict[int, int] = {}
        self.usage: dict[str, Any] = {}
        self.finish_reason: str | None = None
        self.in_think = False
        self.done = False

    def _chunk(self, delta: dict, finish: str | None = None) -> dict:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
        }

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

    def feed(self, event: str | None, data: dict) -> list[dict]:
        etype = data.get("type") or event
        if etype == "message_start":
            msg = data.get("message") or {}
            if msg.get("id"):
                self.id = f"chatcmpl-{msg['id']}"
            self.usage.update(msg.get("usage") or {})
            return [self._chunk({"role": "assistant", "content": ""})]
        if etype == "content_block_start":
            idx = data.get("index", 0)
            block = data.get("content_block") or {}
            btype = block.get("type")
            self.blocks[idx] = btype
            if btype == "text":
                out = self._close_think()
                if block.get("text"):
                    out.append(self._chunk({"content": block["text"]}))
                return out
            if btype == "thinking":
                return self._reasoning(block.get("thinking") or "")
            if btype == "tool_use":
                tidx = len(self.tool_index)
                self.tool_index[idx] = tidx
                out = self._close_think()
                out.append(
                    self._chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": tidx,
                                    "id": block.get("id"),
                                    "type": "function",
                                    "function": {"name": block.get("name"), "arguments": ""},
                                }
                            ]
                        }
                    )
                )
                return out
            return []
        if etype == "content_block_delta":
            idx = data.get("index", 0)
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text") or ""
                return [*self._close_think(), self._chunk({"content": text})] if text else []
            if dtype == "thinking_delta":
                return self._reasoning(delta.get("thinking") or "")
            if dtype == "input_json_delta" and idx in self.tool_index:
                part = delta.get("partial_json") or ""
                if not part:
                    return []
                return [
                    self._chunk(
                        {"tool_calls": [{"index": self.tool_index[idx], "function": {"arguments": part}}]}
                    )
                ]
            return []
        if etype == "content_block_stop":
            if self.blocks.get(data.get("index", 0)) == "thinking":
                return self._close_think()
            return []
        if etype == "message_delta":
            delta = data.get("delta") or {}
            if delta.get("stop_reason"):
                self.finish_reason = STOP_REASON_MAP.get(delta["stop_reason"], "stop")
            self.usage.update({k: v for k, v in (data.get("usage") or {}).items() if v is not None})
            return []
        if etype == "message_stop":
            return self.finish()
        if etype == "error":
            err = data.get("error") or {}
            self.done = True
            return [
                {
                    "error": {
                        "message": err.get("message") or "Upstream stream error",
                        "type": err.get("type") or "upstream_error",
                        "code": None,
                    }
                }
            ]
        return []  # ping and unknown events

    def finish(self) -> list[dict]:
        if self.done:
            return []
        self.done = True
        out = self._close_think()
        out.append(self._chunk({}, self.finish_reason or "stop"))
        if self.include_usage:
            out.append(
                {
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": _usage(self.usage),
                }
            )
        return out


def remote_image_urls(req: dict) -> list[dict]:
    """Image blocks with a URL source (to inline before sending, since Mantle expects base64)."""
    found = []
    for m in req.get("messages", []):
        for b in m.get("content", []):
            if b.get("type") == "image" and b.get("source", {}).get("type") == "url":
                found.append(b)
            if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                for inner in b["content"]:
                    if inner.get("type") == "image" and inner.get("source", {}).get("type") == "url":
                        found.append(inner)
    return found
