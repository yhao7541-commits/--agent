"""Synchronous, bounded model client for hybrid operation decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, Field

from config.model_provider import (
    LocalRuleBasedChatModel,
    ModelConfigurationError,
    create_chat_model,
    get_model_provider,
)


class ModelCallResult(BaseModel):
    raw_text: str
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class DecisionModelClient(Protocol):
    def invoke(self, prompt: str, timeout_seconds: float) -> ModelCallResult: ...


class LangChainDecisionClient:
    """Invoke a strictly configured LangChain chat model once per decision."""

    def invoke(self, prompt: str, timeout_seconds: float) -> ModelCallResult:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number.")
        model = create_chat_model(
            temperature=0,
            request_timeout=timeout_seconds,
            require_configured=True,
        )
        if isinstance(model, LocalRuleBasedChatModel):
            raise ModelConfigurationError("Chat model configuration is required.")
        message = model.invoke(prompt)
        response_metadata = _as_mapping(getattr(message, "response_metadata", None))
        usage_metadata = _as_mapping(getattr(message, "usage_metadata", None))

        return ModelCallResult(
            raw_text=_message_text(getattr(message, "content", "")),
            provider=_metadata_text(response_metadata, "provider") or get_model_provider(),
            model=_metadata_text(response_metadata, "model_name", "model"),
            input_tokens=_metadata_token(usage_metadata, "input_tokens"),
            output_tokens=_metadata_token(usage_metadata, "output_tokens"),
        )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metadata_text(metadata: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _metadata_token(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
