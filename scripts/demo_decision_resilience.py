"""Deterministic retry, repair, fallback, and confirmation demonstration."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.operations.agent import OperationsAgent
from agents.operations.decision_client import ModelCallResult
from agents.operations.decision_engine import HybridDecisionEngine
from agents.operations.decision_models import DecisionSettings, ModelDecision
from security.guardrails import build_confirmation_token, reset_confirmation_token_registry


WRITE_TOOLS = {
    "create_booking",
    "reschedule_booking",
    "cancel_booking",
    "write_customer_preference",
    "delete_customer_memory",
}
SAFE_TRACE_METADATA = {
    "mode",
    "source",
    "provider",
    "model",
    "attempt_count",
    "repair_count",
    "fallback_reason",
    "intent",
    "route",
    "latency_ms",
    "input_tokens",
    "output_tokens",
}


class ScriptedDecisionClient:
    def __init__(self, actions: list[str | Exception]) -> None:
        self.actions = list(actions)
        self.calls = 0

    def invoke(self, _prompt: str, timeout_seconds: float) -> ModelCallResult:
        assert timeout_seconds > 0
        action = self.actions[self.calls]
        self.calls += 1
        if isinstance(action, Exception):
            raise action
        return ModelCallResult(
            raw_text=action,
            provider="scripted-demo",
            model="deterministic-fake",
            input_tokens=10,
            output_tokens=5,
        )


def run_demo(*, print_output: bool = True) -> dict[str, int]:
    reset_confirmation_token_registry()
    repaired_client = ScriptedDecisionClient(
        ["not-json", _decision_json("greeting", "repaired JSON accepted")]
    )
    repaired = _agent_for(repaired_client).run_turn(
        {"conversation_id": "demo-repair", "message": "你好"}
    )

    timeout_client = ScriptedDecisionClient(
        [TimeoutError("scripted timeout") for _ in range(3)]
    )
    fallback = _agent_for(timeout_client).run_turn(
        {"conversation_id": "demo-timeout", "message": "你好"}
    )

    reset_confirmation_token_registry()
    confirmation_client = ScriptedDecisionClient([])
    confirmation_agent = _agent_for(confirmation_client)
    conversation_id = "demo-confirmation-accepted"
    user_id = "demo-user"
    arguments = {"booking_id": "booking_demo", "customer_name": user_id}
    token = build_confirmation_token(
        conversation_id,
        "cancel_booking",
        arguments,
        user_id=user_id,
    )
    accepted = confirmation_agent.run_turn(
        {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "message": "确认取消",
            "confirmed_tool_name": "cancel_booking",
            "confirmed_tool_arguments": arguments,
            "confirmation_token": token,
        }
    )
    rejected = confirmation_agent.run_turn(
        {
            "conversation_id": "demo-confirmation-rejected",
            "user_id": user_id,
            "message": "不确认了",
            "confirmation_decision": "rejected",
            "confirmed_tool_name": "cancel_booking",
            "confirmed_tool_arguments": arguments,
        }
    )

    scenarios = {
        "json_repair": repaired,
        "timeout_fallback": fallback,
        "confirmation_accepted": accepted,
        "confirmation_rejected": rejected,
    }
    summary = {
        "retry_count": _event_count(scenarios, "llm_decision_retry"),
        "repair_count": _event_count(scenarios, "llm_decision_repair"),
        "repair_success_count": int(
            repaired.get("decision_source") == "llm"
            and repaired.get("decision_metadata", {}).get("repair_count") == 1
        ),
        "fallback_count": sum(
            result.get("decision_source") == "rule_fallback"
            for result in scenarios.values()
        ),
        "confirmation_compliance_count": int(_accepted_confirmation_is_compliant(accepted))
        + int(_rejected_confirmation_is_compliant(rejected)),
        "unsafe_write_count": sum(_unsafe_write_count(result) for result in scenarios.values()),
    }
    if print_output:
        print(
            json.dumps(
                {
                    "trace": {
                        name: _sanitized_trace(result.get("trace_events", []))
                        for name, result in scenarios.items()
                    },
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    return summary


def main() -> int:
    with _hybrid_mode():
        summary = run_demo()
    return 0 if summary["unsafe_write_count"] == 0 else 1


def _agent_for(client: ScriptedDecisionClient) -> OperationsAgent:
    fallback = ModelDecision(
        intent="greeting",
        confidence=0.9,
        suggested_action="respond",
        decision_summary="deterministic rules fallback",
    )
    engine = HybridDecisionEngine(
        client=client,
        settings=DecisionSettings(
            mode="hybrid",
            max_attempts=3,
            per_call_timeout_seconds=1,
            total_deadline_seconds=5,
        ),
        fallback=lambda _prompt: fallback,
        sleep_fn=lambda _delay: None,
        jitter_fn=lambda: 0,
    )
    return OperationsAgent(decision_engine=engine)


def _decision_json(intent: str, summary: str) -> str:
    return json.dumps(
        {
            "intent": intent,
            "confidence": 0.95,
            "extracted_slots": {},
            "ambiguities": [],
            "risk_flags": [],
            "suggested_action": "respond",
            "decision_summary": summary,
        },
        ensure_ascii=False,
    )


def _event_count(scenarios: dict[str, dict[str, Any]], event_type: str) -> int:
    return sum(
        event.get("event_type") == event_type
        for result in scenarios.values()
        for event in result.get("trace_events", [])
    )


def _accepted_confirmation_is_compliant(result: dict[str, Any]) -> bool:
    return result.get("decision_source") == "confirmed_action" and any(
        item.get("tool_name") == "cancel_booking" and item.get("success") is True
        for item in result.get("tool_results", [])
    )


def _rejected_confirmation_is_compliant(result: dict[str, Any]) -> bool:
    return (
        result.get("decision_source") == "confirmation_rejected"
        and not result.get("tool_plan")
        and not result.get("tool_results")
    )


def _unsafe_write_count(result: dict[str, Any]) -> int:
    if result.get("decision_source") == "confirmed_action":
        return 0
    return sum(
        item.get("tool_name") in WRITE_TOOLS and item.get("success") is True
        for item in result.get("tool_results", [])
    )


def _sanitized_trace(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for event in events:
        metadata = event.get("metadata", {})
        sanitized.append(
            {
                "node": event.get("node"),
                "event_type": event.get("event_type"),
                "metadata": {
                    key: metadata[key]
                    for key in sorted(SAFE_TRACE_METADATA)
                    if key in metadata
                },
                "error_code": (event.get("error") or {}).get("code"),
            }
        )
    return sanitized


@contextmanager
def _hybrid_mode():
    previous = os.environ.get("OPERATIONS_DECISION_MODE")
    os.environ["OPERATIONS_DECISION_MODE"] = "hybrid"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPERATIONS_DECISION_MODE", None)
        else:
            os.environ["OPERATIONS_DECISION_MODE"] = previous


if __name__ == "__main__":
    raise SystemExit(main())
