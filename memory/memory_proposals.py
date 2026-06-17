from __future__ import annotations

from .customer_memory import MemoryProposal
from .memory_policy import (
    is_vague_memory_statement,
    memory_requires_confirmation,
    sensitivity_for_memory_type,
)


def extract_memory_proposals(message: str) -> list[MemoryProposal]:
    normalized = message.strip()
    if not normalized or is_vague_memory_statement(normalized):
        return []

    proposal = _extract_single_memory(normalized)
    return [proposal] if proposal else []


def _extract_single_memory(message: str) -> MemoryProposal | None:
    if "过敏" in message:
        return _proposal(
            memory_type="constraint",
            content="对精油过敏" if "精油" in message else "存在过敏约束",
            evidence=message,
            confidence=0.92,
        )
    if "不要营销" in message or "别营销" in message:
        return _proposal(
            memory_type="policy_note",
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
