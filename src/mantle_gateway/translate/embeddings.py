"""OpenAI embeddings <-> Bedrock InvokeModel bodies for Titan and Cohere embedding models."""

from __future__ import annotations

import base64
import re
import struct
from dataclasses import dataclass
from typing import Any, Literal

Family = Literal["titan-v2", "titan-v1", "cohere-v3", "cohere-v4"]

COHERE_BATCH = 96


class EmbeddingError(ValueError):
    pass


def embedding_family(model_id: str) -> Family:
    mid = model_id.lower()
    if "titan-embed-text-v2" in mid:
        return "titan-v2"
    if "titan-embed" in mid:
        return "titan-v1"
    if re.search(r"cohere\.embed-v4", mid):
        return "cohere-v4"
    if "cohere.embed" in mid:
        return "cohere-v3"
    raise EmbeddingError(
        f"Unsupported embedding model {model_id!r}: "
        "supported families are amazon.titan-embed-* and cohere.embed-*"
    )


def normalise_input(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value) and value:
        return value
    if isinstance(value, list) and value and all(isinstance(v, (int, list)) for v in value):
        raise EmbeddingError("Token-array input is not supported; send strings")
    raise EmbeddingError("'input' must be a string or a non-empty list of strings")


@dataclass
class EmbeddingCall:
    body: dict
    count: int  # number of input texts this call covers


def build_calls(
    family: Family, texts: list[str], dimensions: int | None, input_type: str | None
) -> list[EmbeddingCall]:
    if family == "titan-v2":
        calls = []
        for t in texts:
            body: dict[str, Any] = {"inputText": t, "normalize": True}
            if dimensions:
                body["dimensions"] = dimensions
            calls.append(EmbeddingCall(body, 1))
        return calls
    if family == "titan-v1":
        return [EmbeddingCall({"inputText": t}, 1) for t in texts]
    calls = []
    for i in range(0, len(texts), COHERE_BATCH):
        batch = texts[i : i + COHERE_BATCH]
        body = {"texts": batch, "input_type": input_type or "search_document", "truncate": "END"}
        if family == "cohere-v4":
            body["embedding_types"] = ["float"]
            if dimensions:
                body["output_dimension"] = dimensions
        calls.append(EmbeddingCall(body, len(batch)))
    return calls


def parse_result(family: Family, data: dict) -> tuple[list[list[float]], int]:
    """Return (vectors, token_count) from one InvokeModel response."""
    if family.startswith("titan"):
        return [data["embedding"]], int(data.get("inputTextTokenCount") or 0)
    emb = data.get("embeddings")
    if isinstance(emb, dict):
        emb = emb.get("float")
    if not isinstance(emb, list):
        raise EmbeddingError("Unexpected Cohere embedding response shape")
    tokens = 0
    meta = data.get("meta") or {}
    billed = meta.get("billed_units") or {}
    if billed.get("input_tokens"):
        tokens = int(billed["input_tokens"])
    return emb, tokens


def encode(vector: list[float], encoding_format: str | None) -> Any:
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
    return vector


def openai_response(vectors: list[list[float]], tokens: int, model: str, encoding_format: str | None) -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": encode(v, encoding_format)}
            for i, v in enumerate(vectors)
        ],
        "model": model,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }
