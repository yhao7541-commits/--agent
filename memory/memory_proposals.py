from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from .customer_memory import MemoryProposal
from .memory_policy import (
    is_vague_memory_statement,
    memory_requires_confirmation,
    sensitivity_for_memory_type,
)


LLMMemoryExtractor = Callable[[str], dict[str, Any] | MemoryProposal | None]


def extract_memory_proposals(
    message: str,
    llm_extractor: LLMMemoryExtractor | None = None,
) -> list[MemoryProposal]:
    normalized = message.strip()
    if not normalized or is_vague_memory_statement(normalized):
        return []

    proposal = _extract_single_memory(normalized)
    if proposal:
        return [proposal]

    if llm_extractor is None:
        return []

    llm_output = llm_extractor(normalized)
    proposal = _coerce_llm_memory_proposal(llm_output)
    return [proposal] if proposal else []


def _extract_single_memory(message: str) -> MemoryProposal | None:
    if "过敏" in message:
        return _proposal(
            memory_type="service_contraindication",
            content="对精油过敏" if "精油" in message else "存在过敏约束",
            evidence=message,
            confidence=0.92,
        )
    if _is_no_marketing_request(message):
        return _proposal(
            memory_type="marketing_consent",
            content="不要营销推荐",
            evidence=message,
            confidence=0.9,
        )
    if "不喜欢大力度" in message or "不要太大力度" in message:
        return _proposal(
            memory_type="negative_preference",
            content="不喜欢大力度",
            evidence=message,
            confidence=0.88,
        )
    if "喜欢热闹" in message:
        return _proposal(
            memory_type="preference",
            content="喜欢热闹房间",
            evidence=message,
            confidence=0.86,
        )
    if "安静" in message and any(marker in message for marker in ("喜欢", "以后", "每次", "都安排", "需要")):
        return _proposal(
            memory_type="preference",
            content="喜欢安静房间",
            evidence=message,
            confidence=0.9,
        )
    return None


def _is_no_marketing_request(message: str) -> bool:
    if "营销" not in message:
        return False
    return any(marker in message for marker in ("不要", "别", "不接受", "拒绝", "别给我发", "不要给我发"))


def _proposal(
    memory_type: str,
    content: str,
    evidence: str,
    confidence: float,
) -> MemoryProposal:
    sensitivity = sensitivity_for_memory_type(memory_type)
    return MemoryProposal(
        type=memory_type,
        content=content,
        evidence=evidence,
        confidence=confidence,
        sensitivity=sensitivity,
        requires_confirmation=memory_requires_confirmation(memory_type, sensitivity),
    )


def _coerce_llm_memory_proposal(output: dict[str, Any] | MemoryProposal | None) -> MemoryProposal | None:
    if output is None:
        return None
    if isinstance(output, MemoryProposal):
        return output
    try:
        proposal = MemoryProposal.model_validate(output)
    except ValidationError:
        return None
    return proposal
