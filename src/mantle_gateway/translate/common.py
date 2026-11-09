"""Helpers shared by the translators."""

from __future__ import annotations

import re
from typing import Any

_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?:;[\w-]+=[^;,]*)*;base64,(?P<data>.*)$", re.DOTALL
)

CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class TranslationError(ValueError):
    """The client request cannot be expressed for the target API (HTTP 400)."""


def content_parts(content: Any) -> list[dict]:
    """Normalise OpenAI message content (str | list | None) to a list of parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                out.append(part)
        return out
    return [{"type": "text", "text": str(content)}]


def text_of(content: Any) -> str:
    """Concatenate the text parts of OpenAI message content."""
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text") or p.get("refusal") or ""
        for p in content_parts(content)
        if p.get("type") in ("text", "input_text", "output_text", "refusal")
    )


def data_uri_parts(url: str) -> tuple[str, str] | None:
    m = _DATA_URI_RE.match(url or "")
    if not m:
        return None
    return (m.group("mime") or "application/octet-stream").lower(), m.group("data").strip()


def stop_list(stop: Any) -> list[str]:
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if isinstance(s, str) and s]


def clamp_effort_claude(effort: Any) -> str | None:
    """Map OpenAI reasoning_effort values onto Claude's effort levels."""
    if not isinstance(effort, str) or not effort:
        return None
    e = effort.lower()
    if e in ("none", "minimal"):
        return "low"
    return e if e in CLAUDE_EFFORTS else None


def clamp_effort_openai(effort: Any) -> str | None:
    if not isinstance(effort, str) or not effort:
        return None
    e = effort.lower()
    return "xhigh" if e == "max" else e


def reasoning_fields(reasoning: str, fmt: str, message: dict) -> dict:
    """Fields to merge into a non-streaming assistant message to expose reasoning."""
    if not reasoning or fmt == "none":
        return {}
    if fmt == "think_tags":
        return {"content": f"<think>\n{reasoning}\n</think>\n\n{message.get('content') or ''}"}
    return {"reasoning_content": reasoning}
