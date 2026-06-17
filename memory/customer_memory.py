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
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None


class MemoryWriteResult(BaseModel):
    action: Literal["created", "updated", "conflict"]
    memory: CustomerMemory
    conflict_with: CustomerMemory | None = None
