from typing import Any

from pydantic import BaseModel, Field


class ToolExecutionContext(BaseModel):
    user_id: str
    conversation_id: str
    trace_id: str
    confirmed_tools: set[str] = Field(default_factory=set)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)


class SearchServicesInput(BaseModel):
    query: str


class SearchServicesOutput(BaseModel):
    services: list[dict[str, Any]]


class CheckScheduleInput(BaseModel):
    service_type: str
    date: str
    time_window: str


class CheckScheduleOutput(BaseModel):
    available: bool
    alternatives: list[str] = Field(default_factory=list)


class CreateBookingInput(BaseModel):
    service_type: str
    date: str
    time_window: str
    customer_name: str
    preferred_staff: str | None = None
    special_requests: str | None = None


class BookingOutput(BaseModel):
    booking_id: str
    status: str


class KnowledgeSearchInput(BaseModel):
    query: str
    top_k: int = 3


class KnowledgeSearchOutput(BaseModel):
    chunks: list[dict[str, Any]]


class CustomerPreferenceInput(BaseModel):
    user_id: str
    preference_type: str
    preference_value: str
    evidence: str


class CustomerPreferenceOutput(BaseModel):
    memory_id: str
    status: str


class HumanEscalationInput(BaseModel):
    reason: str
    summary: str


class HumanEscalationOutput(BaseModel):
    handoff_id: str
    status: str
