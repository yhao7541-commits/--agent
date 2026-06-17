from typing import Any, TypedDict


class OperationsAgentState(TypedDict, total=False):
    user_id: str
    conversation_id: str
    message: str
    intent: str
    confidence: float
    booking_slots: dict[str, Any]
    missing_slots: list[str]
    customer_context: dict[str, Any]
    retrieved_knowledge: list[dict[str, Any]]
    rag_citations: dict[str, Any]
    tool_plan: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    confirmed_tool_name: str
    confirmed_tool_arguments: dict[str, Any]
    confirmation_token: str
    confirmation_required: bool
    confirmation_request: dict[str, Any]
    memory_proposals: list[dict[str, Any]]
    escalation: dict[str, Any]
    reply: str
    trace_id: str
    trace_events: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    rag_used: bool
    escalated: bool
