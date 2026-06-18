from __future__ import annotations

import re

from .customer_memory import CustomerMemory, MemoryProposal


def normalize_memory_content(content: str) -> str:
    return re.sub(r"\s+", "", content.strip())


def find_duplicate(
    proposal: MemoryProposal,
    existing_memories: list[CustomerMemory],
) -> CustomerMemory | None:
    target = normalize_memory_content(proposal.content)
    for memory in existing_memories:
        if memory.deleted_at is None and memory.type == proposal.type and normalize_memory_content(memory.content) == target:
            return memory
    return None


def find_conflict(
    proposal: MemoryProposal,
    existing_memories: list[CustomerMemory],
) -> CustomerMemory | None:
    if proposal.type not in {"preference", "negative_preference", "constraint", "policy_note"}:
        return None

    proposal_content = normalize_memory_content(proposal.content)
    for memory in existing_memories:
        if memory.deleted_at is not None or memory.type != proposal.type:
            continue
        memory_content = normalize_memory_content(memory.content)
        if _has_room_ambience_conflict(memory_content, proposal_content):
            return memory
    return None


def _has_room_ambience_conflict(left: str, right: str) -> bool:
    quiet_terms = ("安静", "静")
    lively_terms = ("热闹", "音乐", "聊天")
    left_quiet = any(term in left for term in quiet_terms)
    right_quiet = any(term in right for term in quiet_terms)
    left_lively = any(term in left for term in lively_terms)
    right_lively = any(term in right for term in lively_terms)
    return (left_quiet and right_lively) or (left_lively and right_quiet)
