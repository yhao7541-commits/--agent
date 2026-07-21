import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DecisionIntent = Literal[
    "booking",
    "reschedule",
    "cancel",
    "consultation",
    "memory",
    "delete_memory",
    "greeting",
    "clarification",
    "escalation",
    "unknown",
]
DecisionSource = Literal[
    "llm",
    "rules",
    "rule_fallback",
    "forced_escalation",
    "confirmed_action",
    "confirmation_rejected",
]


class BookingSlotCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    service_type: str | None = None
    date: str | None = None
    time_window: str | None = None
    duration: str | None = None
    preferred_staff: str | None = None
    special_requests: str | None = None
    booking_id: str | None = None


class ModelDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    intent: DecisionIntent
    confidence: float = Field(ge=0, le=1)
    extracted_slots: BookingSlotCandidates = Field(default_factory=BookingSlotCandidates)
    ambiguities: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    suggested_action: str
    decision_summary: str = Field(max_length=160)


class DecisionSettings(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    mode: Literal["rules", "hybrid"] = "rules"
    max_attempts: int = Field(default=1, ge=1, le=3)
    per_call_timeout_seconds: float = Field(default=8, gt=0)
    total_deadline_seconds: float = Field(default=20, gt=0)
    minimum_confidence: float = Field(default=0.6, ge=0, le=1)

    @classmethod
    def from_env(cls) -> "DecisionSettings":
        max_attempts = int(os.getenv("LLM_DECISION_MAX_ATTEMPTS", "1"))
        return cls(
            mode=os.getenv("LLM_DECISION_MODE", "rules"),
            max_attempts=max(1, min(max_attempts, 3)),
            per_call_timeout_seconds=float(
                os.getenv("LLM_DECISION_PER_CALL_TIMEOUT_SECONDS", "8")
            ),
            total_deadline_seconds=float(
                os.getenv("LLM_DECISION_TOTAL_DEADLINE_SECONDS", "20")
            ),
            minimum_confidence=float(
                os.getenv("LLM_DECISION_MIN_CONFIDENCE", "0.6")
            ),
        )


class DecisionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: DecisionSource
    provider: str | None = None
    model: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    repair_count: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    fallback_reason: str | None = None
