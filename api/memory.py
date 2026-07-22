from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from memory.customer_memory import CustomerMemory
from tools.customer_tools import get_customer_memory_store


router = APIRouter(prefix="/api/memory", tags=["Customer Memory"])


class MemoryListResponse(BaseModel):
    user_id: str
    memories: list[dict[str, Any]] = Field(default_factory=list)


class MemoryMutationResponse(BaseModel):
    memory: dict[str, Any]


class MemoryDeleteResponse(BaseModel):
    memory_id: str
    deleted: bool
    memory: dict[str, Any]


class MemoryEventsResponse(BaseModel):
    memory_id: str
    events: list[dict[str, Any]] = Field(default_factory=list)


class MemoryUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: str
    memory_type: str | None = Field(default=None, alias="type")
    content: str | None = None
    evidence: str | None = None
    confidence: float | None = None
    sensitivity: str | None = None
    expires_at: datetime | None = None
    actor: str = "ops"
    reason: str = ""


class MemoryReviewRequest(BaseModel):
    user_id: str
    actor: str = "ops"
    reason: str = ""


@router.get("/users/{user_id}/memories", response_model=MemoryListResponse)
async def list_user_memories(
    user_id: str,
    include_inactive: bool = Query(default=True),
    include_deleted: bool = Query(default=False),
) -> MemoryListResponse:
    memories = get_customer_memory_store().list_user_memories(
        user_id,
        include_inactive=include_inactive,
        include_deleted=include_deleted,
    )
    return MemoryListResponse(
        user_id=user_id,
        memories=[_memory_to_dict(memory) for memory in memories],
    )


@router.patch("/memories/{memory_id}", response_model=MemoryMutationResponse)
async def edit_memory(memory_id: str, request: MemoryUpdateRequest) -> MemoryMutationResponse:
    memory = get_customer_memory_store().update(
        user_id=request.user_id,
        memory_id=memory_id,
        actor=request.actor,
        reason=request.reason,
        type=request.memory_type,
        content=request.content,
        evidence=request.evidence,
        confidence=request.confidence,
        sensitivity=request.sensitivity,
        expires_at=request.expires_at,
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return MemoryMutationResponse(memory=_memory_to_dict(memory))


@router.post("/memories/{memory_id}/approve", response_model=MemoryMutationResponse)
async def approve_memory(memory_id: str, request: MemoryReviewRequest) -> MemoryMutationResponse:
    memory = get_customer_memory_store().approve(
        user_id=request.user_id,
        memory_id=memory_id,
        actor=request.actor,
        reason=request.reason,
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return MemoryMutationResponse(memory=_memory_to_dict(memory))


@router.post("/memories/{memory_id}/reject", response_model=MemoryMutationResponse)
async def reject_memory(memory_id: str, request: MemoryReviewRequest) -> MemoryMutationResponse:
    memory = get_customer_memory_store().reject(
        user_id=request.user_id,
        memory_id=memory_id,
        actor=request.actor,
        reason=request.reason,
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return MemoryMutationResponse(memory=_memory_to_dict(memory))


@router.delete("/memories/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(
    memory_id: str,
    user_id: str,
    actor: str = Query(default="ops"),
    reason: str = Query(default=""),
) -> MemoryDeleteResponse:
    store = get_customer_memory_store()
    deleted = store.delete(
        user_id=user_id,
        memory_id=memory_id,
        actor=actor,
        reason=reason,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found.")
    memory = store.get_memory(user_id, memory_id, include_deleted=True)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return MemoryDeleteResponse(
        memory_id=memory_id,
        deleted=True,
        memory=_memory_to_dict(memory),
    )


@router.get("/memories/{memory_id}/events", response_model=MemoryEventsResponse)
async def list_memory_events(memory_id: str) -> MemoryEventsResponse:
    events = get_customer_memory_store().list_memory_events(memory_id)
    return MemoryEventsResponse(memory_id=memory_id, events=events)


def _memory_to_dict(memory: CustomerMemory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "memory_id": memory.id,
        "user_id": memory.user_id,
        "type": memory.type,
        "content": memory.content,
        "evidence": memory.evidence,
        "confidence": memory.confidence,
        "sensitivity": memory.sensitivity,
        "status": memory.status,
        "review_status": memory.review_status,
        "source_conversation_id": memory.source_conversation_id,
        "source_trace_id": memory.source_trace_id,
        "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        "deleted_at": memory.deleted_at.isoformat() if memory.deleted_at else None,
        "version": memory.version,
    }
