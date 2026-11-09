"""Server-sent events helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

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
