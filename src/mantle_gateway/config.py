"""Environment settings and the optional models.yaml file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

ApiKind = Literal["chat", "responses", "messages", "bedrock"]
BaseKind = Literal["v1", "openai/v1"]
ThinkingMode = Literal["adaptive", "budget", "none"]

DEFAULT_MODELS_CONFIG_PATHS = (Path("/app/config/models.yaml"), Path("config/models.yaml"))


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (case-insensitive)."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    # Inbound auth: comma-separated list of keys clients (e.g. Open WebUI) must present.
    api_key: str = ""
    allow_no_auth: bool = False

    # Upstream (AWS)
    aws_region: str = Field(
        default="us-east-1", validation_alias=AliasChoices("AWS_REGION", "AWS_DEFAULT_REGION", "aws_region")
    )
    aws_bearer_token_bedrock: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_API_KEY", "aws_bearer_token_bedrock"
        ),
    )
    mantle_base_url: str | None = None
    bedrock_runtime_url: str | None = None
    token_refresh_seconds: int = 600

    # Model catalog
    models_config: str | None = None
    model_discovery: bool = True
    model_cache_ttl: int = 600

    # Behaviour
    reasoning_format: Literal["reasoning_content", "think_tags", "none"] = "reasoning_content"
    anthropic_version: str = "2023-06-01"
    claude_prompt_caching: bool = False
    fetch_remote_images: bool = True
    max_image_bytes: int = 20 * 1024 * 1024
    embedding_concurrency: int = 8

    # Server
    request_timeout: float = 600.0
    log_level: str = "INFO"

    @property
    def api_keys(self) -> list[str]:
        return [k.strip() for k in self.api_key.split(",") if k.strip()]

    @property
    def mantle_url(self) -> str:
        return (self.mantle_base_url or f"https://bedrock-mantle.{self.aws_region}.api.aws").rstrip("/")

    @property
    def runtime_url(self) -> str:
        return (
            self.bedrock_runtime_url or f"https://bedrock-runtime.{self.aws_region}.amazonaws.com"
        ).rstrip("/")


class ModelEntry(BaseModel):
    """One model in models.yaml. Every field except `id` is optional."""

    id: str
    alias: str | None = None
    name: str | None = None
    type: Literal["chat", "embedding"] = "chat"
    api: ApiKind | None = None
    base: BaseKind | None = None
    max_tokens: int | None = None
    default_effort: str | None = None
    thinking: ThinkingMode | None = None
    allow_sampling: bool | None = None
    forced_tool_choice: bool | None = None
    hidden: bool = False
    owned_by: str | None = None
    # Embedding options
    dimensions: int | None = None
    input_type: str | None = None

    @property
    def public_id(self) -> str:
        return self.alias or self.id


class ModelsFile(BaseModel):
    discovery: bool = True
    allow_list: bool = False
    hide: list[str] = Field(default_factory=list)
    models: list[ModelEntry] = Field(default_factory=list)


def load_models_file(settings: Settings) -> ModelsFile:
    path: Path | None = None
    if settings.models_config:
        path = Path(settings.models_config)
        if not path.is_file():
            raise FileNotFoundError(f"MODELS_CONFIG points to a missing file: {path}")
    else:
        path = next((p for p in DEFAULT_MODELS_CONFIG_PATHS if p.is_file()), None)
    if path is None:
        log.info("No models.yaml found; using model discovery and built-in routing rules only")
        return ModelsFile()
    data = yaml.safe_load(path.read_text()) or {}
    models_file = ModelsFile.model_validate(data)
    log.info("Loaded %d model entries from %s", len(models_file.models), path)
    return models_file
