"""Server-sent events helpers."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

from botocore.eventstream import EventStreamBuffer

DONE = b"data: [DONE]\n\n"


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str | None, str]]:
    """Yield (event, data) pairs from an async iterator of SSE lines."""
    event: str | None = None
    data: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


def sse_data(obj: object) -> bytes:
    return f"data: {json.dumps(obj, separators=(',', ':'), ensure_ascii=False)}\n\n".encode()


async def iter_bedrock_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[tuple[str | None, str]]:
    """Yield (event, data) pairs from a bedrock-runtime InvokeModelWithResponseStream body.

    Each AWS event-stream `chunk` carries one base64-encoded model event (for Claude, the same JSON as an
    Anthropic SSE `data:` line). A mid-stream exception becomes an Anthropic-style `error` event.
    """
    buf = EventStreamBuffer()
    async for chunk in chunks:
        buf.add_data(chunk)
        for msg in buf:
            payload = json.loads(msg.payload or b"{}")
            if msg.headers.get(":message-type") == "exception":
                err = {
                    "type": msg.headers.get(":exception-type") or "upstream_error",
                    "message": payload.get("message") or payload.get("Message") or "Bedrock stream error",
                }
                yield "error", json.dumps({"type": "error", "error": err})
                return
            if "bytes" in payload:
                yield None, base64.b64decode(payload["bytes"]).decode()
