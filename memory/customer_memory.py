from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryProposal(BaseModel):
    type: str
    content: str
    evidence: str
    confidence: float
    sensitivity: str
    requires_confirmation: bool
    expires_at: datetime | None = None


class CustomerMemory(BaseModel):
    id: str
    user_id: str
    type: str
    content: str
    evidence: str
    confidence: float
    sensitivity: str
    status: str = "active"
    review_status: str = "approved"
    source_conversation_id: str = ""
    source_trace_id: str = ""
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None
    version: int = 1


class MemoryWriteResult(BaseModel):
    action: Literal["created", "updated", "conflict"]
    memory: CustomerMemory
    conflict_with: CustomerMemory | None = None
