from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.operations as operations_api
import api.appointment as appointment_api
import api.chat_handler as chat_handler
import api.consultation as consultation_api
import api.task as task_api
from agents.operations.decision_engine import (
    DecisionEngineResult,
    DecisionError,
)
from agents.operations.decision_models import DecisionMetadata, ModelDecision
from agents.operations.nodes import decide_request, initialize_turn


class StaticOperationsAgent:
    def __init__(self, result: dict):
        self.result = result

    async def arun_turn(self, _state: dict) -> dict:
        return self.result


class StaticDecisionEngine:
    def __init__(self, result: DecisionEngineResult):
        self.result = result

    def decide(self, _prompt: str) -> DecisionEngineResult:
        return self.result


def _decision(
    *,
    source: str,
    intent: str = "greeting",
    confidence: float = 0.9,
    provider: str | None = None,
    model: str | None = None,
    attempt_count: int = 0,
    repair_count: int = 0,
    fallback_reason: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    errors: list[DecisionError] | None = None,
) -> DecisionEngineResult:
    return DecisionEngineResult(
        decision=ModelDecision(
            intent=intent,
            confidence=confidence,
            suggested_action="respond",
            decision_summary="safe summary",
        ),
        metadata=DecisionMetadata(
            source=source,
            provider=provider,
            model=model,
            attempt_count=attempt_count,
            repair_count=repair_count,
            latency_ms=17,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            fallback_reason=fallback_reason,
        ),
        errors=errors or [],
    )


def _client(monkeypatch: pytest.MonkeyPatch, result: dict) -> TestClient:
    monkeypatch.setattr(
        operations_api,
        "operations_agent",
        StaticOperationsAgent(result),
    )
    app = FastAPI()
    app.include_router(operations_api.router)
    return TestClient(app)


@pytest.mark.parametrize(
    ("source", "intent"),
    [
        ("rules", "greeting"),
        ("llm", "booking"),
        ("rule_fallback", "consultation"),
        ("forced_escalation", "escalation"),
        ("confirmed_action", "booking"),
        ("confirmation_rejected", "confirmation_rejected"),
    ],
)
def test_operations_chat_always_returns_typed_decision_object(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    intent: str,
):
    result = {
        "reply": "ok",
        "intent": intent,
        "confidence": 0.91,
        "trace_id": "trace_decision_api",
        "decision_metadata": DecisionMetadata(source=source).model_dump(),
    }
    response = _client(monkeypatch, result).post(
        "/api/operations/chat",
        json={"conversation_id": "conv_decision_api", "message": "hello"},
    )

    assert response.status_code == 200
    decision = response.json()["decision"]
    assert decision == {
        "source": source,
        "intent": intent,
        "confidence": 0.91,
        "route": None,
        "attempt_count": 0,
        "repair_count": 0,
        "fallback_reason": None,
        "provider": None,
        "model": None,
        "latency_ms": 0,
        "input_tokens": None,
        "output_tokens": None,
    }


def test_operations_chat_maps_only_safe_system_owned_decision_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    result = {
        "reply": "ok",
        "intent": "booking",
        "confidence": 0.93,
        "trace_id": "trace_safe_metadata",
        "decision_route": "booking",
        "decision_metadata": {
            **DecisionMetadata(
                source="llm",
                provider="openai-compatible",
                model="configured-model",
                attempt_count=2,
                repair_count=1,
                latency_ms=1830,
                input_tokens=420,
                output_tokens=110,
            ).model_dump(),
            "raw_prompt": "SECRET_PROMPT",
            "raw_output": "SECRET_OUTPUT",
        },
    }
    body = _client(monkeypatch, result).post(
        "/api/operations/chat",
        json={"conversation_id": "conv_safe_metadata", "message": "hello"},
    ).json()

    assert body["decision"] == {
        "source": "llm",
        "intent": "booking",
        "confidence": 0.93,
        "route": "booking",
        "attempt_count": 2,
        "repair_count": 1,
        "fallback_reason": None,
        "provider": "openai-compatible",
        "model": "configured-model",
        "latency_ms": 1830,
        "input_tokens": 420,
        "output_tokens": 110,
    }
    serialized = json.dumps(body, ensure_ascii=False)
    assert "SECRET_PROMPT" not in serialized
    assert "SECRET_OUTPUT" not in serialized


@pytest.mark.parametrize(
    ("result", "expected_counts"),
    [
        (
            _decision(
                source="llm",
                provider="provider",
                model="model",
                attempt_count=1,
            ),
            {"llm_decision_started": 1, "llm_decision_completed": 1},
        ),
        (
            _decision(
                source="llm",
                attempt_count=2,
                repair_count=1,
                errors=[
                    DecisionError(code="invalid_json", attempt=1, retryable=True)
                ],
            ),
            {
                "llm_decision_started": 1,
                "llm_decision_repair": 1,
                "llm_decision_completed": 1,
            },
        ),
        (
            _decision(
                source="llm",
                attempt_count=2,
                errors=[
                    DecisionError(
                        code="provider_timeout", attempt=1, retryable=True
                    )
                ],
            ),
            {
                "llm_decision_started": 1,
                "llm_decision_retry": 1,
                "llm_decision_completed": 1,
            },
        ),
        (
            _decision(
                source="rule_fallback",
                attempt_count=3,
                fallback_reason="provider_timeout",
                errors=[
                    DecisionError(
                        code="provider_timeout", attempt=1, retryable=True
                    ),
                    DecisionError(
                        code="provider_timeout", attempt=2, retryable=True
                    ),
                    DecisionError(
                        code="provider_timeout", attempt=3, retryable=True
                    ),
                ],
            ),
            {
                "llm_decision_started": 1,
                "llm_decision_retry": 2,
                "llm_decision_fallback": 1,
            },
        ),
    ],
)
def test_hybrid_decision_trace_event_counts_are_exact_and_safe(
    monkeypatch: pytest.MonkeyPatch,
    result: DecisionEngineResult,
    expected_counts: dict[str, int],
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "hybrid")
    state = initialize_turn(
        {
            "user_id": "user_trace",
            "conversation_id": "conv_trace",
            "message": "SECRET_PROMPT api-key traceback hidden reasoning",
        }
    )

    decide_request(state, decision_engine=StaticDecisionEngine(result))

    event_types = [event["event_type"] for event in state["trace_events"]]
    for event_type, count in expected_counts.items():
        assert event_types.count(event_type) == count
    for event_type in {
        "llm_decision_started",
        "llm_decision_retry",
        "llm_decision_repair",
        "llm_decision_completed",
        "llm_decision_fallback",
    } - expected_counts.keys():
        assert event_types.count(event_type) == 0

    serialized = json.dumps(state["trace_events"], ensure_ascii=False)
    assert "SECRET_PROMPT" not in serialized
    assert "api-key" not in serialized
    assert "traceback" not in serialized
    assert "hidden reasoning" not in serialized


def test_deliberate_rules_mode_emits_no_llm_trace_events(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")
    state = initialize_turn(
        {
            "user_id": "user_rules_trace",
            "conversation_id": "conv_rules_trace",
            "message": "你好",
        }
    )

    decide_request(state)

    assert not any(
        event["event_type"].startswith("llm_decision_")
        for event in state["trace_events"]
    )


@pytest.mark.parametrize(
    ("module", "router", "path", "payload", "expected_data_keys"),
    [
        (
            appointment_api,
            appointment_api.router,
            "/api/appointment/create",
            {
                "user_id": "user_compat",
                "service_type": "推拿",
                "preferred_time": "明天下午3点",
            },
            None,
        ),
        (
            consultation_api,
            consultation_api.router,
            "/api/consultation/ask",
            {"user_id": "user_compat", "question": "退款政策是什么"},
            {
                "answer",
                "question",
                "intent",
                "rag_used",
                "rag_citations",
                "trace_id",
            },
        ),
        (
            task_api,
            task_api.router,
            "/api/task/classify",
            {"text": "你好"},
            {
                "intent",
                "confidence",
                "reply",
                "missing_slots",
                "tool_plan",
                "trace_id",
            },
        ),
    ],
)
def test_compatibility_api_top_level_shapes_do_not_gain_decision(
    monkeypatch: pytest.MonkeyPatch,
    module,
    router,
    path: str,
    payload: dict,
    expected_data_keys: set[str] | None,
):
    result = {
        "reply": "compat reply",
        "intent": "consultation",
        "confidence": 0.8,
        "trace_id": "trace_compat",
        "missing_slots": [],
        "tool_plan": [],
        "rag_used": True,
        "rag_citations": {"sources": []},
        "confirmation_required": False,
    }
    monkeypatch.setattr(module, "operations_agent", StaticOperationsAgent(result))
    app = FastAPI()
    app.include_router(router)

    body = TestClient(app).post(path, json=payload).json()

    assert set(body) == {"message", "timestamp", "data"}
    assert "decision" not in body
    if expected_data_keys is not None:
        assert set(body["data"]) == expected_data_keys


@pytest.mark.asyncio
async def test_legacy_stream_shape_remains_text_and_existing_context_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    result = {
        "reply": "compat stream reply",
        "booking_slots": {"service_type": "推拿"},
        "booking_slot_sources": {"service_type": "user"},
    }
    monkeypatch.setattr(
        chat_handler,
        "operations_agent",
        StaticOperationsAgent(result),
    )
    context = {"user_id": "user_stream", "conversation_id": "conv_stream"}

    chunks = [
        chunk
        async for chunk in chat_handler.ProcessUserInput_stream(
            "hello",
            context=context,
        )
    ]

    assert chunks == ["compat stream reply"]
    assert set(context) == {
        "user_id",
        "conversation_id",
        "booking_slots",
        "booking_slot_sources",
    }
