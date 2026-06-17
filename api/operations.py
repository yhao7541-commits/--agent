from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.operations.graph import run_operations_turn


router = APIRouter(prefix="/api/operations", tags=["Operations Agent"])


class OperationsChatRequest(BaseModel):
    user_id: str = "local_user"
    conversation_id: str
    message: str
    booking_slots: dict[str, Any] = Field(default_factory=dict)
    confirmation_decision: str | None = None
    confirmed_tool_name: str | None = None
    confirmed_tool_arguments: dict[str, Any] = Field(default_factory=dict)
    confirmation_token: str | None = None


class OperationsChatResponse(BaseModel):
    reply: str
    intent: str
    confirmation_required: bool = False
    confirmation_request: dict[str, Any] = Field(default_factory=dict)
    booking_slots: dict[str, Any] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    trace_id: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    executed_tools: list[dict[str, Any]] = Field(default_factory=list)
    memory_proposals: list[dict[str, Any]] = Field(default_factory=list)
    rag_used: bool = False
    escalated: bool = False


@router.post("/chat", response_model=OperationsChatResponse)
async def chat(request: OperationsChatRequest) -> OperationsChatResponse:
    result = run_operations_turn(request.model_dump())
    return OperationsChatResponse(
        reply=result.get("reply", ""),
        intent=result.get("intent", "unknown"),
        confirmation_required=result.get("confirmation_required", False),
        confirmation_request=result.get("confirmation_request", {}),
        booking_slots=result.get("booking_slots", {}),
        missing_slots=result.get("missing_slots", []),
        trace_id=result.get("trace_id", ""),
        tool_calls=result.get("tool_plan", []),
        executed_tools=result.get("tool_results", []),
        memory_proposals=result.get("memory_proposals", []),
        rag_used=result.get("rag_used", False),
        escalated=result.get("escalated", False),
    )
