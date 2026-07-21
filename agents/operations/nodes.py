from __future__ import annotations

from copy import deepcopy
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from memory.memory_proposals import extract_memory_proposals
from rag.citation import build_citation_metadata
from security.guardrails import (
    build_confirmation_token,
    consume_confirmation_token,
    detect_prompt_injection,
)
from tools.customer_tools import get_customer_memory_store
from tools.gateway import ToolGateway
from tools.registry import build_default_tool_registry
from tools.schemas import ToolExecutionContext

from .decision_client import LangChainDecisionClient
from .decision_engine import (
    DecisionEngineResult,
    DecisionError,
    DecisionErrorCode,
    HybridDecisionEngine,
)
from .decision_models import DecisionMetadata, DecisionSettings, ModelDecision
from .decision_prompt import build_initial_prompt
from .state import OperationsAgentState


BOOKING_KEYWORDS = ("约", "预约", "改约", "取消", "安排")
CONSULTATION_KEYWORDS = ("迟到", "价格", "多少钱", "政策", "服务", "项目", "适合", "注意", "员工", "技师", "手法")
MEDICAL_ESCALATION_KEYWORDS = (
    "受伤",
    "很疼",
    "疼痛",
    "医疗",
    "医生",
    "拉伤",
    "急救",
    "胸痛",
    "呼吸困难",
    "昏厥",
    "流血",
)
MEMORY_KEYWORDS = ("喜欢", "不喜欢", "过敏", "不要营销", "别营销")
MEMORY_DELETE_KEYWORDS = ("删除", "忘记", "不要记", "别记")
BOOKING_SLOT_UPDATE_KEYWORDS = ("今天", "明天", "上午", "下午", "晚上", "点", "分钟", "安静")
BOOKING_WRITE_TOOLS = {"create_booking", "reschedule_booking", "cancel_booking"}
BOOKING_READ_TOOLS = {"search_services", "check_schedule", "find_available_staff"}
BOOKING_OPERATION_TOOLS = BOOKING_READ_TOOLS | BOOKING_WRITE_TOOLS
BOOKING_TOOL_FAILURE_ESCALATION_THRESHOLD = 2
MEMORY_WRITE_TOOLS = {"write_customer_preference", "delete_customer_memory"}
LOCAL_TIMEZONE = timezone(timedelta(hours=8))
CONFIRMABLE_WRITE_TOOLS = BOOKING_WRITE_TOOLS | MEMORY_WRITE_TOOLS
CONFIRMED_TOOL_INTENTS = {
    "create_booking": "booking",
    "reschedule_booking": "reschedule",
    "cancel_booking": "cancel",
    "write_customer_preference": "memory",
    "delete_customer_memory": "delete_memory",
}
HARD_GUARDRAIL_REASONS = {
    "prompt_injection",
    "confirmation_bypass_attempt",
    "medical_concern",
    "refund_dispute",
}
MAX_RISK_FLAGS = 8
MAX_RISK_FLAG_LENGTH = 64
CONFIRMATION_INPUT_FIELDS = (
    "confirmation_decision",
    "confirmed_tool_name",
    "confirmed_tool_arguments",
    "confirmation_token",
)


def append_trace(
    state: OperationsAgentState,
    node: str,
    metadata: dict[str, Any] | None = None,
    event_type: str = "node_end",
    error: dict[str, Any] | None = None,
) -> OperationsAgentState:
    events = list(state.get("trace_events", []))
    events.append(
        {
            "trace_id": state.get("trace_id", ""),
            "conversation_id": state.get("conversation_id", ""),
            "node": node,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": 0,
            "metadata": metadata or {},
            "error": error,
        }
    )
    state["trace_events"] = events
    return state


def initialize_turn(state: OperationsAgentState) -> OperationsAgentState:
    state["trace_id"] = str(uuid.uuid4())
    state.setdefault("user_id", "local_user")
    state.setdefault("conversation_id", state["trace_id"])
    state.setdefault("message", "")
    state.setdefault("booking_slots", {})
    state.setdefault("booking_slot_sources", {})
    state["booking_issue"] = {}
    state["missing_slots"] = []
    state["customer_context"] = {}
    state["memory_used"] = False
    state["applied_customer_memories"] = []
    state["retrieved_knowledge"] = []
    state["rag_citations"] = {}
    state["tool_plan"] = []
    state["tool_results"] = []
    state["confirmation_required"] = False
    state["confirmation_request"] = {}
    state["memory_proposals"] = []
    state["errors"] = []
    state["rag_used"] = False
    state["escalated"] = False
    state["escalation"] = {}
    state["policy_violation"] = {}
    state["reply"] = ""
    state["intent"] = ""
    state["confidence"] = 0.0
    state["model_decision"] = {}
    state["decision_metadata"] = {}
    state["ambiguities"] = []
    state["decision_source"] = ""
    state["decision_errors"] = []
    state["decision_route"] = ""
    state["trace_events"] = []
    return append_trace(state, "initialize_turn", {"status": "ok"})


def validate_confirmation(state: OperationsAgentState) -> OperationsAgentState:
    if state.get("confirmation_decision") == "rejected":
        state["intent"] = "confirmation_rejected"
        state["confidence"] = 1.0
        state["decision_source"] = "confirmation_rejected"
        state["decision_metadata"] = DecisionMetadata(
            source="confirmation_rejected"
        ).model_dump()
        state["decision_errors"] = []
        state["ambiguities"] = []
        state["decision_route"] = "confirmation_rejected"
        state["model_decision"] = {}
        state["confirmation_required"] = False
        state["confirmation_request"] = {}
        state["tool_plan"] = []
        state["tool_results"] = []
        state["escalated"] = False
        state["escalation"] = {}
        state["policy_violation"] = {}
        state["booking_issue"] = {}
        return append_trace(
            state,
            "validate_confirmation",
            {"source": "confirmation_rejected"},
        )

    if not _has_confirmation_metadata(state):
        return append_trace(
            state,
            "validate_confirmation",
            {"source": "ordinary"},
        )

    if _has_unsafe_confirmed_tool_request(state):
        _force_escalation(
            state,
            reason="unsafe_tool_confirmation",
            requested_action="manual_tool_confirmation_review",
            policy_metadata={"tool_name": _bounded_text(state.get("confirmed_tool_name"))},
        )
        return append_trace(
            state,
            "validate_confirmation",
            {"source": "forced_escalation", "reason": "unsafe_tool_confirmation"},
            event_type="policy_violation",
        )

    tool_name = state["confirmed_tool_name"]
    intent = CONFIRMED_TOOL_INTENTS[tool_name]
    state["intent"] = intent
    state["confidence"] = 1.0
    state["decision_source"] = "confirmed_action"
    state["decision_metadata"] = DecisionMetadata(source="confirmed_action").model_dump()
    state["decision_errors"] = []
    state["ambiguities"] = []
    state["decision_route"] = "booking" if intent in {"booking", "reschedule", "cancel"} else "memory"
    state["model_decision"] = {}
    state["confirmation_required"] = False
    state["confirmation_request"] = {}
    state["tool_plan"] = []
    state["tool_results"] = []
    state["escalated"] = False
    state["escalation"] = {}
    state["policy_violation"] = {}
    state["booking_issue"] = {}
    state["missing_slots"] = []
    if tool_name in BOOKING_WRITE_TOOLS:
        state["booking_slots"] = dict(state["confirmed_tool_arguments"])
        state["booking_slot_sources"] = {
            field: "confirmed_tool_arguments" for field in state["booking_slots"]
        }
    return append_trace(
        state,
        "validate_confirmation",
        {"source": "confirmed_action", "intent": intent},
    )


def input_guardrail(state: OperationsAgentState) -> OperationsAgentState:
    classified = deepcopy(state)
    classify_intent(classified)
    reason = classified.get("escalation", {}).get("reason")
    if reason not in HARD_GUARDRAIL_REASONS:
        return append_trace(state, "input_guardrail", {"status": "ok"})

    _force_escalation(
        state,
        reason=reason,
        requested_action=classified["escalation"]["requested_action"],
    )
    return append_trace(
        state,
        "input_guardrail",
        {"source": "forced_escalation", "reason": reason},
        event_type="policy_violation",
    )


def decide_request(
    state: OperationsAgentState,
    decision_engine: HybridDecisionEngine | None = None,
) -> OperationsAgentState:
    try:
        settings = DecisionSettings.from_env()
    except ValueError:
        if os.getenv("OPERATIONS_DECISION_MODE", "rules").strip().lower() == "rules":
            return _apply_decision_result(
                state,
                _rule_engine_result(state, source="rules"),
            )
        return _apply_decision_result(
            state,
            _rule_engine_result(
                state,
                source="rule_fallback",
                fallback_reason="configuration_error",
            ),
        )

    if settings.mode == "rules":
        result = _rule_engine_result(state, source="rules")
    else:
        try:
            prompt = build_initial_prompt(
                message=state.get("message", ""),
                booking_slots=state.get("booking_slots", {}),
                booking_slot_sources=state.get("booking_slot_sources", {}),
                local_date=datetime.now(LOCAL_TIMEZONE).date().isoformat(),
                timezone="Asia/Shanghai",
            )
        except (TypeError, ValueError):
            result = _rule_engine_result(
                state,
                source="rule_fallback",
                fallback_reason="business_validation_error",
            )
        else:
            engine = decision_engine or HybridDecisionEngine(
                client=LangChainDecisionClient(),
                settings=settings,
                fallback=lambda _prompt: _rule_decision(state),
            )
            result = engine.decide(prompt)

    return _apply_decision_result(state, result)


def classify_intent(state: OperationsAgentState) -> OperationsAgentState:
    message = state.get("message", "")
    if state.get("confirmation_decision") == "rejected":
        intent = "confirmation_rejected"
        confidence = 1.0
        state["confirmation_required"] = False
        state["confirmation_request"] = {}
    elif _has_unsafe_confirmed_tool_request(state):
        intent = "escalation"
        confidence = 1.0
        state["escalated"] = True
        state["policy_violation"] = {
            "reason": "unsafe_tool_confirmation",
            "tool_name": state.get("confirmed_tool_name"),
        }
        state["escalation"] = _build_escalation(
            state,
            reason="unsafe_tool_confirmation",
            requested_action="manual_tool_confirmation_review",
        )
    elif _is_confirmation_bypass_attempt(message):
        intent = "escalation"
        confidence = 1.0
        state["escalated"] = True
        state["policy_violation"] = {"reason": "confirmation_bypass_attempt"}
        state["escalation"] = _build_escalation(
            state,
            reason="confirmation_bypass_attempt",
            requested_action="manual_security_review",
        )
    elif detect_prompt_injection(message):
        intent = "escalation"
        confidence = 1.0
        state["escalated"] = True
        state["policy_violation"] = {"reason": "prompt_injection"}
        state["escalation"] = _build_escalation(
            state,
            reason="prompt_injection",
            requested_action="manual_security_review",
        )
    elif any(keyword in message for keyword in MEDICAL_ESCALATION_KEYWORDS):
        intent = "escalation"
        confidence = 1.0
        state["escalated"] = True
        state["escalation"] = _build_escalation(
            state,
            reason="medical_concern",
            requested_action="medical_or_safety_follow_up",
        )
    elif _is_refund_dispute(message):
        intent = "escalation"
        confidence = 1.0
        state["escalated"] = True
        state["escalation"] = _build_escalation(
            state,
            reason="refund_dispute",
            requested_action="refund_or_complaint_review",
        )
    elif _is_policy_question(message):
        intent = "consultation"
        confidence = 0.88
    elif state.get("confirmed_tool_name") == "cancel_booking":
        intent = "cancel"
        confidence = 1.0
    elif state.get("confirmed_tool_name") == "reschedule_booking":
        intent = "reschedule"
        confidence = 1.0
    elif state.get("confirmed_tool_name") == "create_booking":
        intent = "booking"
        confidence = 1.0
    elif state.get("confirmed_tool_name") == "delete_customer_memory":
        intent = "delete_memory"
        confidence = 1.0
    elif state.get("confirmed_tool_name") in MEMORY_WRITE_TOOLS:
        intent = "memory"
        confidence = 1.0
    elif state.get("booking_slots") and any(keyword in message for keyword in BOOKING_SLOT_UPDATE_KEYWORDS):
        intent = "booking"
        confidence = 0.85
    elif "取消" in message:
        intent = "cancel"
        confidence = 0.9
    elif "改约" in message:
        intent = "reschedule"
        confidence = 0.9
    elif any(keyword in message for keyword in BOOKING_KEYWORDS):
        intent = "booking"
        confidence = 0.9
    elif any(keyword in message for keyword in CONSULTATION_KEYWORDS):
        intent = "consultation"
        confidence = 0.86
    elif _is_memory_delete_request(message):
        intent = "delete_memory"
        confidence = 0.86
    elif any(keyword in message for keyword in MEMORY_KEYWORDS):
        intent = "memory"
        confidence = 0.82
    elif message.strip() in {"你好", "您好", "hi", "hello"}:
        intent = "greeting"
        confidence = 0.8
    else:
        intent = "unknown"
        confidence = 0.4
        state["escalated"] = True
        state["escalation"] = _build_escalation(
            state,
            reason="low_confidence",
            requested_action="manual_intent_review",
        )

    state["intent"] = intent
    state["confidence"] = confidence
    state = append_trace(state, "classify_intent", {"intent": intent, "confidence": confidence})
    if state.get("policy_violation"):
        return append_trace(
            state,
            "input_guardrail",
            state["policy_violation"],
            event_type="policy_violation",
        )
    return state


def load_customer_context(state: OperationsAgentState) -> OperationsAgentState:
    fallback_context = {
        "user_id": state.get("user_id", "local_user"),
        "known_preferences": [],
    }
    context = ToolExecutionContext(
        user_id=state.get("user_id", "local_user"),
        conversation_id=state.get("conversation_id", ""),
        trace_id=state.get("trace_id", ""),
        trace_events=list(state.get("trace_events", [])),
    )
    result = ToolGateway(build_default_tool_registry()).execute(
        "lookup_customer_profile",
        {"user_id": state.get("user_id", "local_user")},
        context,
    )
    state["trace_events"] = context.trace_events
    state["customer_context"] = result.output if result.success else fallback_context
    return append_trace(
        state,
        "load_customer_context",
        {
            "loaded": result.success,
            "memory_count": len(state["customer_context"].get("memories", [])),
            "known_preference_count": len(state["customer_context"].get("known_preferences", [])),
        },
    )


def extract_booking_slots(state: OperationsAgentState) -> OperationsAgentState:
    if state.get("intent") not in {"booking", "cancel", "reschedule"}:
        return append_trace(state, "extract_booking_slots", {"skipped": True})
    if state.get("escalated"):
        return append_trace(state, "extract_booking_slots", {"skipped": True})
    if state.get("confirmed_tool_name"):
        state["booking_slots"] = dict(state.get("confirmed_tool_arguments", {}))
        state["booking_slot_sources"] = {
            field: "confirmed_tool_arguments" for field in state["booking_slots"]
        }
        state["missing_slots"] = []
        return append_trace(
            state,
            "extract_booking_slots",
            {
                "source": "confirmed_tool_arguments",
                "slot_sources": state["booking_slot_sources"],
                "missing_slots": [],
            },
        )

    message = state.get("message", "")
    slots: dict[str, Any] = dict(state.get("booking_slots", {}))
    slot_sources: dict[str, str] = dict(state.get("booking_slot_sources", {}))
    for field in slots:
        slot_sources.setdefault(field, "previous_turn")
    booking_id_match = re.search(r"\bbooking_[A-Za-z0-9_-]+\b", message)
    if booking_id_match:
        slots["booking_id"] = booking_id_match.group(0)
        _mark_slot_source(slot_sources, "booking_id", "user")

    if state.get("intent") == "cancel":
        slots["customer_name"] = state.get("user_id", "local_user")
        _mark_slot_source(slot_sources, "customer_name", "system")
        missing = ["booking_id"] if not slots.get("booking_id") else []
        state["booking_slots"] = slots
        state["booking_slot_sources"] = slot_sources
        state["missing_slots"] = missing
        return append_trace(
            state,
            "extract_booking_slots",
            {"slots": slots, "slot_sources": slot_sources, "missing_slots": missing},
        )

    if "肩颈" in message:
        slots["service_type"] = "肩颈放松"
        _mark_slot_source(slot_sources, "service_type", "user")
    elif "推拿" in message:
        slots["service_type"] = "推拿"
        _mark_slot_source(slot_sources, "service_type", "user")
    elif "按摩" in message:
        slots["service_type"] = "按摩"
        _mark_slot_source(slot_sources, "service_type", "user")

    normalized_date = _normalize_booking_date(message)
    if normalized_date:
        slots["date"] = normalized_date
        _mark_slot_source(slot_sources, "date", "user")

    time_match = _select_time_match(message)
    if time_match:
        slots["time_window"] = _normalize_time_match(time_match)
        _mark_slot_source(slot_sources, "time_window", "user")
    elif "上午" in message:
        slots["time_window"] = "09:00-12:00"
        _mark_slot_source(slot_sources, "time_window", "user")
    elif "下午" in message:
        slots["time_window"] = "12:00-18:00"
        _mark_slot_source(slot_sources, "time_window", "user")
    elif "晚上" in message:
        slots["time_window"] = "18:00-21:00"
        _mark_slot_source(slot_sources, "time_window", "user")

    duration_match = re.search(r"(\d+)\s*分钟", message)
    if duration_match:
        slots["duration"] = f"{duration_match.group(1)}分钟"
        _mark_slot_source(slot_sources, "duration", "user")

    if "安静" in message:
        slots["special_requests"] = _merge_special_request(
            slots.get("special_requests"),
            "安静一点的房间",
        )
        _mark_slot_source(slot_sources, "special_requests", "user")
    if "热闹" in message:
        slots["special_requests"] = _merge_special_request(
            slots.get("special_requests"),
            "热闹一点的房间",
        )
        _mark_slot_source(slot_sources, "special_requests", "user")
    applied_memories = list(state.get("applied_customer_memories", []))
    for memory in _customer_memory_candidates(state):
        preference = memory.get("content", "")
        if "安静" in preference:
            slots["special_requests"] = _merge_special_request(
                slots.get("special_requests"),
                "安静一点的房间",
            )
            _mark_slot_source(slot_sources, "special_requests", "memory")
            _append_applied_customer_memory(
                applied_memories,
                memory,
                "booking_slots.special_requests",
            )
        if "热闹" in preference:
            slots["special_requests"] = _merge_special_request(
                slots.get("special_requests"),
                "热闹一点的房间",
            )
            _mark_slot_source(slot_sources, "special_requests", "memory")
            _append_applied_customer_memory(
                applied_memories,
                memory,
                "booking_slots.special_requests",
            )
        if "大力度" in preference:
            slots["special_requests"] = _merge_special_request(
                slots.get("special_requests"),
                "避免大力度",
            )
            _mark_slot_source(slot_sources, "special_requests", "memory")
            _append_applied_customer_memory(
                applied_memories,
                memory,
                "booking_slots.special_requests",
            )
    if "李雷" in message:
        slots["preferred_staff"] = "李雷"
        _mark_slot_source(slot_sources, "preferred_staff", "user")

    slots["customer_name"] = state.get("user_id", "local_user")
    _mark_slot_source(slot_sources, "customer_name", "system")
    if state.get("intent") == "reschedule":
        if slots.get("date"):
            slots["new_date"] = slots["date"]
            _mark_slot_source(slot_sources, "new_date", slot_sources.get("date", "user"))
        if slots.get("time_window"):
            slots["new_time_window"] = slots["time_window"]
            _mark_slot_source(
                slot_sources,
                "new_time_window",
                slot_sources.get("time_window", "user"),
            )
        missing = [
            field
            for field in ("booking_id", "new_date", "new_time_window")
            if not slots.get(field)
        ]
    else:
        missing = [
            field
            for field in ("service_type", "date", "time_window")
            if not slots.get(field)
        ]

    state["booking_slots"] = slots
    state["booking_slot_sources"] = slot_sources
    state["applied_customer_memories"] = applied_memories
    state["memory_used"] = bool(applied_memories)
    state["missing_slots"] = missing
    return append_trace(
        state,
        "extract_booking_slots",
        {
            "slots": slots,
            "slot_sources": slot_sources,
            "missing_slots": missing,
            "memory_used": state["memory_used"],
            "applied_memory_count": len(applied_memories),
        },
    )


def propose_memory_writes(state: OperationsAgentState) -> OperationsAgentState:
    if state.get("confirmed_tool_name") or state.get("escalated"):
        return append_trace(state, "propose_memory_writes", {"skipped": True})

    proposals = extract_memory_proposals(state.get("message", ""))
    state["memory_proposals"] = [proposal.model_dump() for proposal in proposals]
    return append_trace(state, "propose_memory_writes", {"proposal_count": len(proposals)})


def plan_tool_calls(state: OperationsAgentState) -> OperationsAgentState:
    plan: list[dict[str, Any]] = []
    intent = state.get("intent")

    if intent == "confirmation_rejected":
        plan = []
    elif state.get("escalated"):
        escalation = state.get("escalation", {})
        plan.append(
            {
                "tool_name": "escalate_to_human",
                "arguments": {
                    "reason": escalation.get("reason", "manual_review"),
                    "summary": escalation.get("summary", ""),
                },
                "permission": "external",
            }
        )
    elif state.get("confirmed_tool_name"):
        plan.append(
            {
                "tool_name": state["confirmed_tool_name"],
                "arguments": state.get("confirmed_tool_arguments", {}),
                "permission": "write",
                "confirmed": True,
            }
        )
    elif intent == "delete_memory":
        memory = _match_memory_for_deletion(state)
        if memory:
            plan.append(
                {
                    "tool_name": "delete_customer_memory",
                    "arguments": {
                        "user_id": state.get("user_id", "local_user"),
                        "memory_id": memory["memory_id"],
                        "content": memory.get("content", ""),
                    },
                    "permission": "sensitive",
                }
            )
        else:
            state["missing_slots"] = ["memory_id"]
    elif intent == "memory" and state.get("memory_proposals"):
        proposal = state["memory_proposals"][0]
        plan.append(
            {
                "tool_name": "write_customer_preference",
                "arguments": {
                    "user_id": state.get("user_id", "local_user"),
                    "preference_type": proposal["type"],
                    "preference_value": proposal["content"],
                    "evidence": proposal["evidence"],
                },
                "permission": "sensitive",
            }
        )
    elif intent == "consultation":
        plan.append(
            {
                "tool_name": "search_knowledge_base",
                "arguments": {"query": state.get("message", ""), "top_k": 3},
                "permission": "read",
            }
        )
    elif intent == "booking" and not state.get("missing_slots"):
        slots = state.get("booking_slots", {})
        plan.append(
            {
                "tool_name": "lookup_customer_profile",
                "arguments": {"user_id": state.get("user_id", "local_user")},
                "permission": "read",
            }
        )
        plan.append(
            {
                "tool_name": "find_available_staff",
                "arguments": {
                    "service_type": slots.get("service_type"),
                    "date": slots.get("date"),
                    "time_window": slots.get("time_window"),
                    "preferred_staff": slots.get("preferred_staff"),
                },
                "permission": "read",
            }
        )
        plan.append(
            {
                "tool_name": "check_schedule",
                "arguments": {
                    "service_type": slots.get("service_type"),
                    "date": slots.get("date"),
                    "time_window": slots.get("time_window"),
                },
                "permission": "read",
            }
        )
        plan.append(
            {
                "tool_name": "create_booking",
                "arguments": slots,
                "permission": "write",
            }
        )
    elif intent == "cancel" and not state.get("missing_slots"):
        slots = state.get("booking_slots", {})
        plan.append(
            {
                "tool_name": "cancel_booking",
                "arguments": {
                    "booking_id": slots.get("booking_id"),
                    "customer_name": slots.get("customer_name", state.get("user_id", "local_user")),
                },
                "permission": "write",
            }
        )
    elif intent == "reschedule" and not state.get("missing_slots"):
        slots = state.get("booking_slots", {})
        plan.append(
            {
                "tool_name": "reschedule_booking",
                "arguments": {
                    "booking_id": slots.get("booking_id"),
                    "new_date": slots.get("new_date"),
                    "new_time_window": slots.get("new_time_window"),
                    "customer_name": slots.get("customer_name", state.get("user_id", "local_user")),
                },
                "permission": "write",
            }
        )

    state["tool_plan"] = plan
    return append_trace(state, "plan_tool_calls", {"tool_count": len(plan)})


def execute_tools(state: OperationsAgentState) -> OperationsAgentState:
    gateway = ToolGateway(build_default_tool_registry())
    context = ToolExecutionContext(
        user_id=state.get("user_id", "local_user"),
        conversation_id=state.get("conversation_id", ""),
        trace_id=state.get("trace_id", ""),
        confirmed_tools={state["confirmed_tool_name"]} if state.get("confirmed_tool_name") else set(),
        trace_events=list(state.get("trace_events", [])),
    )

    results = []
    retrieved_knowledge = list(state.get("retrieved_knowledge", []))
    for planned_tool in state.get("tool_plan", []):
        if state.get("booking_issue") and planned_tool["tool_name"] in BOOKING_WRITE_TOOLS:
            continue
        result = gateway.execute(
            planned_tool["tool_name"],
            planned_tool.get("arguments", {}),
            context,
        )
        result_data = result.model_dump()
        results.append(result_data)

        if result.confirmation_required:
            if state.get("booking_issue"):
                continue
            summary = _build_confirmation_summary(state, results, result.tool_name, planned_tool.get("arguments", {}))
            try:
                confirmation_token = build_confirmation_token(
                    state.get("conversation_id", ""),
                    result.tool_name,
                    planned_tool.get("arguments", {}),
                    user_id=state.get("user_id", "local_user"),
                )
            except ValueError:
                _force_escalation(
                    state,
                    reason="unsafe_tool_confirmation",
                    requested_action="manual_tool_confirmation_review",
                    policy_metadata={"source": "confirmation_signing"},
                )
                escalation_plan = {
                    "tool_name": "escalate_to_human",
                    "arguments": {
                        "reason": "unsafe_tool_confirmation",
                        "summary": state["escalation"]["summary"],
                    },
                    "permission": "external",
                }
                state["tool_plan"] = [escalation_plan]
                results.append(
                    gateway.execute(
                        escalation_plan["tool_name"],
                        escalation_plan["arguments"],
                        context,
                    ).model_dump()
                )
                break
            state["confirmation_required"] = True
            state["confirmation_request"] = {
                "tool_name": result.tool_name,
                "arguments": planned_tool.get("arguments", {}),
                "summary": summary,
                "message": _confirmation_message(result.tool_name),
                "confirmation_token": confirmation_token,
            }

        if result.success and result.tool_name == "search_knowledge_base":
            retrieved_knowledge = result.output.get("chunks", [])
            state["rag_used"] = True
            state["rag_citations"] = build_citation_metadata(
                planned_tool.get("arguments", {}).get("query", ""),
                retrieved_knowledge,
            )
            _append_rag_trace(context, state["rag_citations"])
        if result.success and result.tool_name in {"create_booking", "reschedule_booking", "cancel_booking"}:
            state["confirmation_required"] = False
            state["confirmation_request"] = {}
        if result.success and result.tool_name in MEMORY_WRITE_TOOLS:
            state["confirmation_required"] = False
            state["confirmation_request"] = {}
        if result.success and result.tool_name == "find_available_staff":
            _apply_staff_availability_issue(state, result.output)
        if result.success and result.tool_name == "check_schedule":
            _apply_schedule_issue(state, result.output)
        if len(_booking_tool_failures(results)) >= BOOKING_TOOL_FAILURE_ESCALATION_THRESHOLD:
            break

    _apply_tool_failure_escalation(state, context, gateway, results)

    state["tool_results"] = results
    state["trace_events"] = context.trace_events
    state["retrieved_knowledge"] = retrieved_knowledge
    if state.get("booking_issue"):
        state["tool_plan"] = [
            planned_tool
            for planned_tool in state.get("tool_plan", [])
            if planned_tool.get("tool_name") not in BOOKING_WRITE_TOOLS
        ]
    return append_trace(state, "execute_tools", {"tool_results": len(results)})


def generate_response(state: OperationsAgentState) -> OperationsAgentState:
    if state.get("intent") == "confirmation_rejected":
        state["reply"] = "已取消本次待确认操作，未执行任何写入。"
    elif state.get("escalated"):
        reason = state.get("escalation", {}).get("reason", "manual_review")
        state["reply"] = f"这个请求需要人工进一步确认，我已按 {reason} 整理上下文交给工作人员处理。"
    elif state.get("booking_issue"):
        state["reply"] = _booking_issue_reply(state)
    elif state.get("missing_slots"):
        labels = {
            "service_type": "服务项目",
            "date": "日期",
            "time_window": "时间",
            "booking_id": "预约编号或已有预约",
            "new_date": "新的日期",
            "new_time_window": "新的时间",
        }
        missing = "、".join(labels.get(field, field) for field in state["missing_slots"])
        state["reply"] = f"还需要补充{missing}，我才能继续为您安排预约。"
    elif state.get("confirmation_required"):
        state["reply"] = _confirmation_reply(state)
    elif state.get("intent") == "consultation":
        if state.get("rag_used") and not state.get("retrieved_knowledge"):
            state["reply"] = "当前知识库信息不足，无法可靠回答这个问题；请由工作人员进一步确认。"
        else:
            state["reply"] = "根据服务政策，相关问题需要以门店知识库为准；我已检索到可参考的服务说明。"
    elif _successful_tool_result(state, "create_booking"):
        booking_result = _successful_tool_result(state, "create_booking")
        state["reply"] = f"预约已创建，预约编号：{booking_result['output']['booking_id']}。"
    elif _successful_tool_result(state, "reschedule_booking"):
        booking_result = _successful_tool_result(state, "reschedule_booking")
        state["reply"] = f"预约已改约，预约编号：{booking_result['output']['booking_id']}。"
    elif _successful_tool_result(state, "cancel_booking"):
        booking_result = _successful_tool_result(state, "cancel_booking")
        state["reply"] = f"预约已取消，预约编号：{booking_result['output']['booking_id']}。"
    elif _successful_tool_result(state, "delete_customer_memory"):
        memory_result = _successful_tool_result(state, "delete_customer_memory")
        status = memory_result.get("output", {}).get("status")
        if status == "deleted":
            state["reply"] = "客户记忆已删除，后续服务安排不会再参考这条偏好。"
        else:
            state["reply"] = "没有找到可删除的客户记忆，未更改客户偏好。"
    elif _successful_tool_result(state, "write_customer_preference"):
        memory_result = _successful_tool_result(state, "write_customer_preference")
        status = memory_result.get("output", {}).get("status")
        if status == "conflict":
            state["reply"] = "检测到这条客户偏好与已有记忆冲突，系统未自动覆盖；请由工作人员或用户进一步确认后再更新。"
        elif status == "updated":
            state["reply"] = "客户偏好已更新，后续服务安排会参考最新偏好。"
        else:
            state["reply"] = "客户偏好已保存，后续服务安排会参考这条偏好。"
    elif state.get("intent") == "greeting":
        state["reply"] = "您好，我可以协助服务咨询、预约安排和客户偏好记录。"
    else:
        state["reply"] = "我暂时无法确定您的需求，可以请您说明是咨询服务还是预约安排吗？"

    return append_trace(state, "generate_response", {"reply_length": len(state["reply"])})


def output_policy_check(state: OperationsAgentState) -> OperationsAgentState:
    if "预约已创建" in state.get("reply", "") and not _successful_tool_result(state, "create_booking"):
        error = {
            "type": "false_booking_success",
            "message": "Reply claimed booking creation without a successful create_booking tool result.",
        }
        errors = list(state.get("errors", []))
        errors.append(error)
        state["errors"] = errors
        state["reply"] = "我还不能确认预约已经创建；请等待确认操作完成或由工作人员核实。"
        return append_trace(
            state,
            "output_policy_check",
            {"status": "blocked", "reason": error["type"]},
            event_type="policy_violation",
            error=error,
        )

    return append_trace(state, "output_policy_check", {"status": "ok"})


def finalize_turn(state: OperationsAgentState) -> OperationsAgentState:
    finalized = append_trace(
        state,
        "finalize_turn",
        {
            "intent": state.get("intent"),
            "confirmation_required": state.get("confirmation_required", False),
            "rag_used": state.get("rag_used", False),
        },
    )
    return clear_confirmation_inputs(finalized)


def clear_confirmation_inputs(
    state: OperationsAgentState,
) -> OperationsAgentState:
    for field in CONFIRMATION_INPUT_FIELDS:
        state.pop(field, None)
    return state


def _rule_decision(state: OperationsAgentState) -> ModelDecision:
    classified = deepcopy(state)
    classify_intent(classified)
    intent = classified.get("intent", "unknown")
    if intent == "confirmation_rejected":
        intent = "unknown"
    return ModelDecision(
        intent=intent,
        confidence=classified.get("confidence", 0.0),
        extracted_slots={},
        ambiguities=[],
        risk_flags=[],
        suggested_action="route_deterministically",
        decision_summary=f"Deterministic rules selected {intent}.",
    )


def _rule_engine_result(
    state: OperationsAgentState,
    *,
    source: Literal["rules", "rule_fallback"],
    fallback_reason: DecisionErrorCode | None = None,
) -> DecisionEngineResult:
    errors = []
    if fallback_reason is not None:
        errors.append(
            DecisionError(
                code=fallback_reason,
                attempt=0,
                retryable=False,
            )
        )
    return DecisionEngineResult(
        decision=_rule_decision(state),
        metadata=DecisionMetadata(
            source=source,
            fallback_reason=fallback_reason,
        ),
        errors=errors,
    )


def _apply_decision_result(
    state: OperationsAgentState,
    result: DecisionEngineResult,
) -> OperationsAgentState:
    decision = result.decision
    decision_data = decision.model_dump()
    metadata = result.metadata.model_dump()
    errors = [error.model_dump() for error in result.errors]
    sanitized_risk_flags = _sanitize_risk_flags(decision.risk_flags)
    decision_data["risk_flags"] = sanitized_risk_flags

    state["model_decision"] = decision_data
    state["decision_metadata"] = metadata
    state["decision_source"] = metadata["source"]
    state["decision_errors"] = errors
    state["intent"] = decision.intent
    state["confidence"] = decision.confidence
    state["ambiguities"] = list(decision.ambiguities)

    if sanitized_risk_flags:
        _force_escalation(
            state,
            reason="model_risk_flag",
            requested_action="manual_model_risk_review",
            policy_metadata={"risk_flags": sanitized_risk_flags},
        )
        state["model_decision"] = decision_data
        state["decision_metadata"] = {
            **metadata,
            "source": "forced_escalation",
        }
        state["decision_errors"] = errors
    elif decision.intent == "unknown" or decision.confidence < 0.5:
        _set_decision_escalation(
            state,
            reason="low_confidence",
            requested_action="manual_intent_review",
        )
    elif decision.ambiguities or decision.intent == "clarification":
        state["intent"] = "clarification"
        state["escalated"] = False
    elif decision.intent == "escalation":
        _set_decision_escalation(
            state,
            reason="model_escalation",
            requested_action="manual_model_review",
        )

    from .routers import route_after_decision

    state["decision_route"] = route_after_decision(state)
    return append_trace(
        state,
        "decide_request",
        {
            "source": state["decision_source"],
            "route": state["decision_route"],
            "error_codes": [error["code"] for error in errors],
        },
    )


def _set_decision_escalation(
    state: OperationsAgentState,
    *,
    reason: str,
    requested_action: str,
) -> None:
    state["intent"] = "escalation"
    state["escalated"] = True
    state["escalation"] = _build_escalation(
        state,
        reason=reason,
        requested_action=requested_action,
    )
    state["tool_plan"] = []


def _force_escalation(
    state: OperationsAgentState,
    *,
    reason: str,
    requested_action: str,
    policy_metadata: dict[str, Any] | None = None,
) -> None:
    state["intent"] = "escalation"
    state["confidence"] = 1.0
    state["escalated"] = True
    state["decision_source"] = "forced_escalation"
    state["decision_metadata"] = DecisionMetadata(source="forced_escalation").model_dump()
    state["decision_errors"] = []
    state["ambiguities"] = []
    state["decision_route"] = "escalation"
    state["model_decision"] = {}
    state["confirmation_required"] = False
    state["confirmation_request"] = {}
    state["tool_plan"] = []
    state["tool_results"] = []
    state["policy_violation"] = {"reason": reason, **(policy_metadata or {})}
    state["escalation"] = _build_escalation(
        state,
        reason=reason,
        requested_action=requested_action,
    )


def _sanitize_risk_flags(risk_flags: list[str]) -> list[str]:
    sanitized: list[str] = []
    for flag in risk_flags[:MAX_RISK_FLAGS]:
        bounded = re.sub(r"[^A-Za-z0-9_.:-]+", "_", flag).strip("_")
        sanitized.append((bounded or "unspecified")[:MAX_RISK_FLAG_LENGTH])
    return sanitized


def _bounded_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)[:MAX_RISK_FLAG_LENGTH]


def _normalize_booking_date(message: str) -> str | None:
    today = datetime.now(LOCAL_TIMEZONE).date()
    if "大后天" in message:
        return (today + timedelta(days=3)).strftime("%Y-%m-%d")
    if "后天" in message:
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")
    if "明天" in message:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if "今天" in message:
        return today.strftime("%Y-%m-%d")

    match = re.search(r"下(?:周|星期)([一二三四五六日天])", message)
    if not match:
        return None

    weekdays = {
        "一": 0,
        "二": 1,
        "三": 2,
        "四": 3,
        "五": 4,
        "六": 5,
        "日": 6,
        "天": 6,
    }
    start_of_next_week = today + timedelta(days=(7 - today.weekday()))
    return (start_of_next_week + timedelta(days=weekdays[match.group(1)])).strftime("%Y-%m-%d")


def _normalize_time_match(match: re.Match[str]) -> str:
    return _normalize_hour(match.group(1), match.group(2), match.group(3))


def _normalize_hour(period: str | None, raw_hour: str, raw_minute: str | None = None) -> str:
    chinese_numbers = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    hour = chinese_numbers.get(raw_hour, int(raw_hour) if raw_hour.isdigit() else 0)
    if period in {"下午", "晚上"} and hour < 12:
        hour += 12
    minute = _normalize_minute(raw_minute)
    return f"{hour:02d}:{minute:02d}"


def _normalize_minute(raw_minute: str | None) -> int:
    if not raw_minute:
        return 0
    if raw_minute == "半":
        return 30
    digits = raw_minute.removesuffix("分")
    return int(digits) if digits.isdigit() else 0


def _select_time_match(message: str) -> re.Match[str] | None:
    pattern = re.compile(
        r"(上午|下午|晚上)?\s*(\d{1,2}|一|二|两|三|四|五|六|七|八|九|十)点(半|[0-5]?\d分?)?(?![的儿也])"
    )
    for match in pattern.finditer(message):
        if _is_contextual_time_match(message, match):
            return match
    return None


def _is_contextual_time_match(message: str, match: re.Match[str]) -> bool:
    period, raw_hour = match.group(1), match.group(2)
    if period is not None or match.start() == 0:
        return True

    prefix = message[max(0, match.start() - 6) : match.start()]
    suffix = message[match.end() : match.end() + 3]
    has_time_context = any(
        marker in prefix or marker in suffix
        for marker in ("今天", "明天", "约", "预约", "改约", "到", "上午", "下午", "晚上")
    )
    if raw_hour in {"一", "1"} and not has_time_context:
        return False
    return has_time_context


def _merge_special_request(existing: str | None, addition: str) -> str:
    if not existing or existing == "无":
        return addition
    if addition in existing:
        return existing
    return f"{existing}；{addition}"


def _customer_memory_candidates(state: OperationsAgentState) -> list[dict[str, Any]]:
    context = state.get("customer_context", {})
    memories = [memory for memory in context.get("memories", []) if memory.get("content")]
    if memories:
        return memories
    return [
        {
            "memory_id": "",
            "type": "preference",
            "content": preference,
            "sensitivity": "normal",
        }
        for preference in context.get("known_preferences", [])
        if preference
    ]


def _append_applied_customer_memory(
    applied_memories: list[dict[str, Any]],
    memory: dict[str, Any],
    applied_to: str,
) -> None:
    record = {
        "memory_id": memory.get("memory_id") or memory.get("id", ""),
        "type": memory.get("type", "preference"),
        "content": memory.get("content", ""),
        "sensitivity": memory.get("sensitivity", "normal"),
        "applied_to": applied_to,
    }
    if not record["content"]:
        return

    key = (record["memory_id"], record["content"], record["applied_to"])
    if any(
        (existing.get("memory_id"), existing.get("content"), existing.get("applied_to")) == key
        for existing in applied_memories
    ):
        return
    applied_memories.append(record)


def _mark_slot_source(sources: dict[str, str], field: str, source: str) -> None:
    existing = sources.get(field)
    if not existing:
        sources[field] = source
        return
    if source not in existing.split("+"):
        sources[field] = f"{existing}+{source}"


def _successful_tool_result(state: OperationsAgentState, tool_name: str) -> dict[str, Any] | None:
    for result in state.get("tool_results", []):
        if result.get("tool_name") == tool_name and result.get("success"):
            return result
    return None


def _booking_tool_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for result in results:
        error_code = (result.get("error") or {}).get("code")
        if (
            result.get("tool_name") in BOOKING_OPERATION_TOOLS
            and not result.get("success")
            and error_code != "confirmation_required"
        ):
            failures.append(result)
    return failures


def _apply_tool_failure_escalation(
    state: OperationsAgentState,
    context: ToolExecutionContext,
    gateway: ToolGateway,
    results: list[dict[str, Any]],
) -> None:
    if state.get("escalated"):
        return

    failures = _booking_tool_failures(results)
    if len(failures) < BOOKING_TOOL_FAILURE_ESCALATION_THRESHOLD:
        return

    failure_summary = [
        {
            "tool_name": failure.get("tool_name"),
            "code": (failure.get("error") or {}).get("code", "tool_error"),
        }
        for failure in failures
    ]
    escalation = _build_escalation(
        state,
        reason="tool_failure",
        requested_action="booking_tool_failure_review",
    )
    escalation["tool_failures"] = failure_summary

    state["escalated"] = True
    state["escalation"] = escalation
    state["confirmation_required"] = False
    state["confirmation_request"] = {}

    escalation_plan = {
        "tool_name": "escalate_to_human",
        "arguments": {
            "reason": escalation["reason"],
            "summary": escalation["summary"],
        },
        "permission": "external",
    }
    state["tool_plan"] = [
        planned_tool
        for planned_tool in state.get("tool_plan", [])
        if planned_tool.get("tool_name") not in BOOKING_WRITE_TOOLS
    ]
    state["tool_plan"].append(escalation_plan)
    context.trace_events.append(
        {
            "trace_id": context.trace_id,
            "conversation_id": context.conversation_id,
            "node": "execute_tools",
            "event_type": "escalation_triggered",
            "metadata": {"reason": "tool_failure", "tool_failures": failure_summary},
            "error": None,
        }
    )
    results.append(
        gateway.execute(
            escalation_plan["tool_name"],
            escalation_plan["arguments"],
            context,
        ).model_dump()
    )


def _is_memory_delete_request(message: str) -> bool:
    return any(keyword in message for keyword in ("忘记", "删除", "删掉", "不要记"))


def _is_policy_question(message: str) -> bool:
    return "政策" in message and any(keyword in message for keyword in ("取消", "退款", "迟到", "服务"))


def _is_refund_dispute(message: str) -> bool:
    return "我要退款" in message or any(
        keyword in message for keyword in ("投诉", "服务很差", "争议")
    )


def _match_memory_for_deletion(state: OperationsAgentState) -> dict[str, Any] | None:
    message = state.get("message", "")
    memories = _memory_records_for_deletion(state)
    for memory in memories:
        content = memory.get("content", "")
        if content and content in message:
            return memory
        if any(keyword in message and keyword in content for keyword in ("安静", "热闹", "大力度", "营销", "过敏")):
            return memory
    return None


def _memory_records_for_deletion(state: OperationsAgentState) -> list[dict[str, Any]]:
    records = list(state.get("customer_context", {}).get("memories", []))
    seen_ids = {record.get("memory_id") or record.get("id") for record in records}
    store = get_customer_memory_store()
    for memory in store.list_user_memories(
        state.get("user_id", "local_user"),
        include_inactive=True,
        include_deleted=False,
    ):
        if memory.id in seen_ids:
            continue
        records.append(
            {
                "memory_id": memory.id,
                "type": memory.type,
                "content": memory.content,
                "sensitivity": memory.sensitivity,
                "status": memory.status,
                "review_status": memory.review_status,
                "version": memory.version,
            }
        )
        seen_ids.add(memory.id)
    return records


def _has_confirmation_metadata(state: OperationsAgentState) -> bool:
    return any(
        field in state
        for field in (
            "confirmation_decision",
            "confirmed_tool_name",
            "confirmed_tool_arguments",
            "confirmation_token",
        )
    )


def _has_unsafe_confirmed_tool_request(state: OperationsAgentState) -> bool:
    if not _has_confirmation_metadata(state):
        return False
    tool_name = state.get("confirmed_tool_name")
    arguments = state.get("confirmed_tool_arguments")
    token = state.get("confirmation_token")
    if state.get("confirmation_decision") not in {None, "rejected"}:
        return True
    if tool_name not in CONFIRMABLE_WRITE_TOOLS:
        return True
    if not isinstance(arguments, dict):
        return True
    if not isinstance(token, str) or not token:
        return True
    return not consume_confirmation_token(
        state.get("conversation_id", ""),
        tool_name,
        arguments,
        token,
        user_id=state.get("user_id", "local_user"),
    )


def _is_confirmation_bypass_attempt(message: str) -> bool:
    patterns = (
        r"(?:绕过|跳过)(?:本次|这个|所有)?确认",
        r"(?:不要|不需要|无需)(?:再|进行)?确认[，,\s]*(?:就)?(?:直接)?"
        r"(?:创建|调用|执行|预约|写入|删除|取消|改约)",
        r"^(?:不要|不需要|无需)(?:再|进行)?确认[。！!\s]*$",
    )
    return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in patterns)


def _apply_staff_availability_issue(state: OperationsAgentState, output: dict[str, Any]) -> None:
    preferred_staff = state.get("booking_slots", {}).get("preferred_staff")
    if not preferred_staff:
        return
    staff = output.get("staff", [])
    if any(candidate.get("name") == preferred_staff and candidate.get("available", False) for candidate in staff):
        return
    state["confirmation_required"] = False
    state["confirmation_request"] = {}
    state["booking_issue"] = {
        "type": "staff_unavailable",
        "message": f"{preferred_staff} 当前不可用。",
        "alternatives": [candidate.get("name") for candidate in staff if candidate.get("available")],
    }


def _apply_schedule_issue(state: OperationsAgentState, output: dict[str, Any]) -> None:
    if output.get("available", True):
        return
    state["confirmation_required"] = False
    state["confirmation_request"] = {}
    state["booking_issue"] = {
        "type": "time_conflict",
        "message": "该时间段暂不可预约。",
        "alternatives": output.get("alternatives", []),
    }


def _booking_issue_reply(state: OperationsAgentState) -> str:
    issue = state.get("booking_issue", {})
    alternatives = "、".join(str(item) for item in issue.get("alternatives", [])) or "暂无"
    if issue.get("type") == "staff_unavailable":
        return f"{issue.get('message', '指定员工暂不可用')} 可选替代员工：{alternatives}。"
    if issue.get("type") == "time_conflict":
        return f"{issue.get('message', '该时间不可用')} 附近可选时间：{alternatives}。"
    return "当前预约条件需要调整，请选择一个可用替代方案。"


def _append_rag_trace(context: ToolExecutionContext, metadata: dict[str, Any]) -> None:
    context.trace_events.append(
        {
            "trace_id": context.trace_id,
            "conversation_id": context.conversation_id,
            "node": "search_knowledge_base",
            "event_type": "rag_retrieval_completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": 0,
            "metadata": metadata,
            "error": None,
        }
    )


def _build_confirmation_summary(
    state: OperationsAgentState,
    tool_results: list[dict[str, Any]],
    tool_name: str = "create_booking",
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tool_name == "write_customer_preference":
        arguments = arguments or {}
        proposal = state.get("memory_proposals", [{}])[0] if state.get("memory_proposals") else {}
        return {
            "preference_type": arguments.get("preference_type", proposal.get("type", "preference")),
            "preference_value": arguments.get("preference_value", proposal.get("content", "")),
            "evidence": arguments.get("evidence", proposal.get("evidence", "")),
            "sensitivity": proposal.get("sensitivity", "normal"),
        }
    if tool_name == "delete_customer_memory":
        arguments = arguments or {}
        return {
            "memory_id": arguments.get("memory_id", "待确认"),
            "action": "删除客户记忆",
            "content": arguments.get("content", "待确认"),
        }
    if tool_name == "cancel_booking":
        arguments = arguments or {}
        return {
            "booking_id": arguments.get("booking_id", "待确认"),
            "action": "取消预约",
            "customer_name": arguments.get("customer_name", state.get("user_id", "local_user")),
        }
    if tool_name == "reschedule_booking":
        arguments = arguments or {}
        return {
            "booking_id": arguments.get("booking_id", "待确认"),
            "action": "改约",
            "new_date": arguments.get("new_date", "待确认"),
            "new_time": arguments.get("new_time_window", "待确认"),
            "customer_name": arguments.get("customer_name", state.get("user_id", "local_user")),
        }

    slots = state.get("booking_slots", {})
    staff_name = slots.get("preferred_staff") or "待确认员工"
    for result in tool_results:
        if result.get("tool_name") == "find_available_staff" and result.get("success"):
            staff = result.get("output", {}).get("staff", [])
            if staff:
                staff_name = staff[0].get("name", staff_name)
                break

    return {
        "service": slots.get("service_type", "待确认"),
        "staff": staff_name,
        "date": slots.get("date", "待确认"),
        "time": slots.get("time_window", "待确认"),
        "duration": slots.get("duration", "待确认"),
        "price": "门店价目表为准",
        "special_requests": slots.get("special_requests", "无"),
        "cancellation_policy": "如需取消或更改预约，请提前至少2小时通知。",
    }


def _confirmation_message(tool_name: str) -> str:
    if tool_name == "write_customer_preference":
        return "请确认是否保存这条客户偏好。"
    if tool_name == "delete_customer_memory":
        return "请确认是否删除这条客户记忆。"
    if tool_name == "cancel_booking":
        return "请确认是否取消这次预约。"
    if tool_name == "reschedule_booking":
        return "请确认是否改约这次预约。"
    return "请确认是否创建这次预约。"


def _confirmation_reply(state: OperationsAgentState) -> str:
    request = state.get("confirmation_request", {})
    summary = request.get("summary", {})
    if request.get("tool_name") == "write_customer_preference":
        return (
            "请确认是否保存这条客户偏好：\n"
            f"类型：{summary.get('preference_type', 'preference')}\n"
            f"内容：{summary.get('preference_value', '')}\n"
            f"依据：{summary.get('evidence', '')}\n"
            f"敏感级别：{summary.get('sensitivity', 'normal')}\n"
            "确认后我再写入客户记忆。"
        )
    if request.get("tool_name") == "delete_customer_memory":
        return (
            "请确认是否删除这条客户记忆：\n"
            f"记忆编号：{summary.get('memory_id', '待确认')}\n"
            f"内容：{summary.get('content', '待确认')}\n"
            "确认后我再删除这条客户记忆。"
        )
    if request.get("tool_name") == "cancel_booking":
        return (
            "请确认是否取消这次预约：\n"
            f"预约编号：{summary.get('booking_id', '待确认')}\n"
            f"客户：{summary.get('customer_name', '待确认')}\n"
            "确认后我再取消预约。"
        )
    if request.get("tool_name") == "reschedule_booking":
        return (
            "请确认是否改约这次预约：\n"
            f"预约编号：{summary.get('booking_id', '待确认')}\n"
            f"新日期：{summary.get('new_date', '待确认')}\n"
            f"新时间：{summary.get('new_time', '待确认')}\n"
            "确认后我再改约。"
        )

    return (
        "请确认是否创建这次预约：\n"
        f"服务：{summary.get('service', '待确认')}\n"
        f"员工：{summary.get('staff', '待确认')}\n"
        f"日期：{summary.get('date', '待确认')}\n"
        f"时间：{summary.get('time', '待确认')}\n"
        f"时长：{summary.get('duration', '待确认')}\n"
        f"价格：{summary.get('price', '门店价目表为准')}\n"
        f"特殊需求：{summary.get('special_requests', '无')}\n"
        f"取消政策：{summary.get('cancellation_policy', '请提前至少2小时通知。')}\n"
        "确认后我再创建预约。"
    )


def _build_escalation(
    state: OperationsAgentState,
    reason: str,
    requested_action: str,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "customer_id": state.get("user_id", "local_user"),
        "conversation_id": state.get("conversation_id", ""),
        "summary": state.get("message", ""),
        "requested_action": requested_action,
        "known_context": state.get("customer_context", {}),
        "recommended_next_step": "请由人工工作人员核实情况并继续处理。",
    }
