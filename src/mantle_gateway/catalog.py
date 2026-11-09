"""Model catalog: discovery from Mantle, models.yaml merging, aliases and per-model routing."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import time
from dataclasses import dataclass

from .config import ApiKind, BaseKind, ModelEntry, ModelsFile, Settings, ThinkingMode
from .upstream import Upstream, UpstreamError

log = logging.getLogger(__name__)

_CLAUDE_VERSION_RE = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-(\d{1,2})(?:-(\d{1,2}))?(?!\d)")
_CLAUDE_LEGACY_RE = re.compile(r"claude-(\d)(?:-(\d))?-(opus|sonnet|haiku)")
NUMBERED_GPT_RE = re.compile(r"^openai\.gpt-(?!oss)")
_OPENAI_V1_CHAT_RE = re.compile(r"^(google\.gemma-4|xai\.grok)")
_EMBED_RE = re.compile(r"embed", re.IGNORECASE)


@dataclass(frozen=True)
class ClaudeVersion:
    family: str
    major: int
    minor: int


def claude_version(model_id: str) -> ClaudeVersion | None:
    lower = model_id.lower()
    m = _CLAUDE_VERSION_RE.search(lower)
    if m:
        return ClaudeVersion(m.group(1), int(m.group(2)), int(m.group(3) or 0))
    m = _CLAUDE_LEGACY_RE.search(lower)  # e.g. claude-3-7-sonnet
    if m:
        return ClaudeVersion(m.group(3), int(m.group(1)), int(m.group(2) or 0))
    return None


def is_claude(model_id: str) -> bool:
    return "claude" in model_id.lower()


def auto_api(model_id: str) -> ApiKind:
    if is_claude(model_id):
        return "messages"
    if NUMBERED_GPT_RE.search(model_id):
        return "responses"
    return "chat"


def auto_base(model_id: str) -> BaseKind:
    if NUMBERED_GPT_RE.search(model_id) or _OPENAI_V1_CHAT_RE.search(model_id):
        return "openai/v1"
    return "v1"


def auto_thinking(model_id: str) -> ThinkingMode:
    """Adaptive thinking on Claude 4.6+ / 5.x / Fable / Mythos; budget-based on older models."""
    v = claude_version(model_id)
    if v is None:
        return "adaptive"  # unknown / newer naming (e.g. claude-mythos-preview)
    if v.family in ("fable", "mythos") or v.major >= 5:
        return "adaptive"
    if v.major == 4 and v.minor >= 6 and v.family != "haiku":
        return "adaptive"
    return "budget"


def auto_forced_tool_choice(model_id: str) -> bool:
    """Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1 (and later) reject tool_choice any/tool."""
    v = claude_version(model_id)
    if v is None:
        return True
    if v.family in ("opus", "sonnet") and (v.major, v.minor) >= (5, 5):
        return False
    return not (v.family in ("fable", "mythos") and (v.major, v.minor) >= (5, 1))


def auto_allow_sampling(model_id: str, api: ApiKind) -> bool:
    if api == "messages":
        # Current Claude models (4.7+, 5.x) reject temperature/top_p/top_k; older ones accept them
        # unless thinking is on (handled by the translator).
        return auto_thinking(model_id) == "budget"
    # GPT-5.x reasoning models reject non-default sampling
    return not (api == "responses" or NUMBERED_GPT_RE.search(model_id))


@dataclass
class ResolvedModel:
    public_id: str
    upstream_id: str
    name: str
    type: str
    api: ApiKind
    base: BaseKind
    max_tokens: int | None
    default_effort: str | None
    thinking: ThinkingMode
    allow_sampling: bool
    forced_tool_choice: bool
    dimensions: int | None = None
    input_type: str | None = None
    owned_by: str | None = None
    configured: bool = False


def resolve_entry(model_id: str, entry: ModelEntry | None) -> ResolvedModel:
    upstream = entry.id if entry else model_id
    api = (entry.api if entry else None) or auto_api(upstream)
    model_type = entry.type if entry else ("embedding" if _EMBED_RE.search(upstream) else "chat")
    return ResolvedModel(
        public_id=entry.public_id if entry else model_id,
        upstream_id=upstream,
        name=(entry.name if entry and entry.name else (entry.public_id if entry else model_id)),
        type=model_type,
        api=api,
        base=(entry.base if entry else None) or auto_base(upstream),
        max_tokens=entry.max_tokens if entry else None,
        default_effort=entry.default_effort if entry else None,
        thinking=(entry.thinking if entry else None) or auto_thinking(upstream),
        allow_sampling=(
            entry.allow_sampling
            if entry and entry.allow_sampling is not None
            else auto_allow_sampling(upstream, api)
        ),
        forced_tool_choice=(
            entry.forced_tool_choice
            if entry and entry.forced_tool_choice is not None
            else auto_forced_tool_choice(upstream)
        ),
        dimensions=entry.dimensions if entry else None,
        input_type=entry.input_type if entry else None,
        owned_by=(entry.owned_by if entry else None) or owner_of(upstream),
        configured=entry is not None,
    )


def owner_of(model_id: str) -> str:
    head = model_id.split("/")[-1]
    return head.split(".", 1)[0] if "." in head else "bedrock"


class Catalog:
    def __init__(self, settings: Settings, models_file: ModelsFile, upstream: Upstream) -> None:
        self.settings = settings
        self.file = models_file
        self.upstream = upstream
        self._by_public: dict[str, ModelEntry] = {}
        self._by_upstream: dict[str, ModelEntry] = {}
        for entry in models_file.models:
            self._by_public.setdefault(entry.public_id, entry)
            self._by_upstream.setdefault(entry.id, entry)
        self._discovered: list[dict] = []
        self._discovered_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def discovery_enabled(self) -> bool:
        return self.settings.model_discovery and self.file.discovery and not self.file.allow_list

    def resolve(self, name: str) -> ResolvedModel:
        entry = self._by_public.get(name) or self._by_upstream.get(name)
        if entry is None and name.startswith("claude-"):
            # Anthropic-style IDs (e.g. from Claude Code) map to Bedrock IDs:
            # claude-opus-5-5 -> anthropic.claude-opus-5-5
            entry = self._by_upstream.get(f"anthropic.{name}")
            if entry is None:
                resolved = resolve_entry(f"anthropic.{name}", None)
                resolved.public_id = name
                return resolved
        return resolve_entry(name, entry)

    def is_allowed(self, name: str) -> bool:
        if not self.file.allow_list:
            return True
        return name in self._by_public or name in self._by_upstream

    async def discovered(self) -> list[dict]:
        if not self.discovery_enabled:
            return []
        async with self._lock:
            if self._discovered_at and time.monotonic() - self._discovered_at < self.settings.model_cache_ttl:
                return self._discovered
            try:
                data = await self.upstream.request_json("GET", self.upstream.mantle("/v1/models"))
                items = [m for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]
                self._discovered = items
                self._discovered_at = time.monotonic()
                log.info("Discovered %d models from Mantle", len(items))
            except Exception as exc:  # noqa: BLE001 - discovery must never break /v1/models
                msg = exc.message if isinstance(exc, UpstreamError) else str(exc)
                log.warning("Mantle model discovery failed (%s); serving configured models only", msg)
                # retry sooner than the full TTL
                self._discovered_at = time.monotonic() - self.settings.model_cache_ttl + 30
            return self._discovered

    def _hidden(self, model_id: str) -> bool:
        return any(fnmatch.fnmatch(model_id, pat) for pat in self.file.hide)

    async def list_models(self) -> list[dict]:
        created = int(time.time())
        out: list[dict] = []
        seen_upstream: set[str] = set()
        for entry in self.file.models:
            seen_upstream.add(entry.id)
            if entry.hidden or entry.type == "embedding":
                continue
            rm = resolve_entry(entry.id, entry)
            out.append(_model_obj(rm.public_id, rm.name, rm.owned_by or "bedrock", created))
        for item in await self.discovered():
            mid = item["id"]
            if mid in seen_upstream or self._hidden(mid) or _EMBED_RE.search(mid):
                continue
            seen_upstream.add(mid)
            out.append(
                _model_obj(mid, item.get("name") or mid, item.get("owned_by") or owner_of(mid), created)
            )
        return out


def _model_obj(model_id: str, name: str, owned_by: str, created: int) -> dict:
    return {"id": model_id, "object": "model", "created": created, "owned_by": owned_by, "name": name}
