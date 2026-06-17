from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from tools.gateway import ToolGateway
from tools.registry import build_default_tool_registry
from tools.schemas import ToolExecutionContext

from .state import OperationsAgentState


BOOKING_KEYWORDS = ("约", "预约", "改约", "取消", "安排")
CONSULTATION_KEYWORDS = ("迟到", "价格", "多少钱", "政策", "服务", "项目", "适合", "注意")


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
    state.setdefault("trace_id", str(uuid.uuid4()))
    state.setdefault("user_id", "local_user")
    state.setdefault("conversation_id", state["trace_id"])
    state.setdefault("message", "")
    state.setdefault("booking_slots", {})
    state.setdefault("missing_slots", [])
    state.setdefault("customer_context", {})
    state.setdefault("retrieved_knowledge", [])
    state.setdefault("tool_plan", [])
    state.setdefault("tool_results", [])
    state.setdefault("confirmation_required", False)
    state.setdefault("confirmation_request", {})
    state.setdefault("memory_proposals", [])
    state.setdefault("errors", [])
    state.setdefault("rag_used", False)
    state.setdefault("escalated", False)
    return append_trace(state, "initialize_turn", {"status": "ok"})


def classify_intent(state: OperationsAgentState) -> OperationsAgentState:
    message = state.get("message", "")
    if state.get("confirmed_tool_name") in {"create_booking", "reschedule_booking", "cancel_booking"}:
        intent = "booking"
        confidence = 1.0
    elif any(keyword in message for keyword in BOOKING_KEYWORDS):
        intent = "booking"
        confidence = 0.9
    elif any(keyword in message for keyword in CONSULTATION_KEYWORDS):
        intent = "consultation"
        confidence = 0.86
    elif message.strip() in {"你好", "您好", "hi", "hello"}:
        intent = "greeting"
        confidence = 0.8
    else:
        intent = "unknown"
        confidence = 0.4
        state["escalated"] = True
        state["escalation"] = {
            "reason": "low_confidence",
            "summary": "Unable to classify the customer request confidently.",
        }

    state["intent"] = intent
    state["confidence"] = confidence
    return append_trace(state, "classify_intent", {"intent": intent, "confidence": confidence})


def load_customer_context(state: OperationsAgentState) -> OperationsAgentState:
    state["customer_context"] = {
        "user_id": state.get("user_id", "local_user"),
        "known_preferences": [],
    }
    return append_trace(state, "load_customer_context", {"loaded": True})


def extract_booking_slots(state: OperationsAgentState) -> OperationsAgentState:
    if state.get("intent") != "booking":
        return append_trace(state, "extract_booking_slots", {"skipped": True})
    if state.get("confirmed_tool_name"):
        state["booking_slots"] = dict(state.get("confirmed_tool_arguments", {}))
        state["missing_slots"] = []
        return append_trace(
            state,
            "extract_booking_slots",
            {"source": "confirmed_tool_arguments", "missing_slots": []},
        )

    message = state.get("message", "")
    slots: dict[str, Any] = {}

    if "肩颈" in message:
        slots["service_type"] = "肩颈放松"
    elif "推拿" in message:
        slots["service_type"] = "推拿"
    elif "按摩" in message:
        slots["service_type"] = "按摩"

    if "明天" in message:
        slots["date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "今天" in message:
        slots["date"] = datetime.now().strftime("%Y-%m-%d")

    time_match = re.search(r"(上午|下午|晚上)?\s*(\d{1,2}|一|二|两|三|四|五|六|七|八|九|十)点", message)
    if time_match:
        slots["time_window"] = _normalize_hour(time_match.group(1), time_match.group(2))

    duration_match = re.search(r"(\d+)\s*分钟", message)
    if duration_match:
        slots["duration"] = f"{duration_match.group(1)}分钟"

    if "安静" in message:
        slots["special_requests"] = "安静一点的房间"

    slots["customer_name"] = state.get("user_id", "local_user")
    missing = [
        field
        for field in ("service_type", "date", "time_window")
        if not slots.get(field)
    ]

    state["booking_slots"] = slots
    state["missing_slots"] = missing
    return append_trace(state, "extract_booking_slots", {"slots": slots, "missing_slots": missing})


def plan_tool_calls(state: OperationsAgentState) -> OperationsAgentState:
    plan: list[dict[str, Any]] = []
    intent = state.get("intent")

    if state.get("confirmed_tool_name"):
        plan.append(
            {
                "tool_name": state["confirmed_tool_name"],
                "arguments": state.get("confirmed_tool_arguments", {}),
                "permission": "write",
                "confirmed": True,
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
        result = gateway.execute(
            planned_tool["tool_name"],
            planned_tool.get("arguments", {}),
            context,
        )
        result_data = result.model_dump()
        results.append(result_data)

        if result.confirmation_required:
            summary = _build_confirmation_summary(state, results)
            state["confirmation_required"] = True
            state["confirmation_request"] = {
                "tool_name": result.tool_name,
                "arguments": planned_tool.get("arguments", {}),
                "summary": summary,
                "message": "请确认是否创建这次预约。",
            }

        if result.success and result.tool_name == "search_knowledge_base":
            retrieved_knowledge = result.output.get("chunks", [])
            state["rag_used"] = True
        if result.success and result.tool_name in {"create_booking", "reschedule_booking", "cancel_booking"}:
            state["confirmation_required"] = False
            state["confirmation_request"] = {}

    state["tool_results"] = results
    state["trace_events"] = context.trace_events
    state["retrieved_knowledge"] = retrieved_knowledge
    return append_trace(state, "execute_tools", {"tool_results": len(results)})


def generate_response(state: OperationsAgentState) -> OperationsAgentState:
    if state.get("escalated"):
        state["reply"] = "这个请求需要人工进一步确认，我会整理上下文后交给工作人员处理。"
    elif state.get("missing_slots"):
        labels = {"service_type": "服务项目", "date": "日期", "time_window": "时间"}
        missing = "、".join(labels.get(field, field) for field in state["missing_slots"])
        state["reply"] = f"还需要补充{missing}，我才能继续为您安排预约。"
    elif state.get("confirmation_required"):
        summary = state.get("confirmation_request", {}).get("summary", {})
        state["reply"] = (
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
    elif state.get("intent") == "consultation":
        state["reply"] = "根据服务政策，相关问题需要以门店知识库为准；我已检索到可参考的服务说明。"
    elif _successful_tool_result(state, "create_booking"):
        booking_result = _successful_tool_result(state, "create_booking")
        state["reply"] = f"预约已创建，预约编号：{booking_result['output']['booking_id']}。"
    elif state.get("intent") == "greeting":
        state["reply"] = "您好，我可以协助服务咨询、预约安排和客户偏好记录。"
    else:
        state["reply"] = "我暂时无法确定您的需求，可以请您说明是咨询服务还是预约安排吗？"

    return append_trace(state, "generate_response", {"reply_length": len(state["reply"])})


def finalize_turn(state: OperationsAgentState) -> OperationsAgentState:
    return append_trace(
        state,
        "finalize_turn",
        {
            "intent": state.get("intent"),
            "confirmation_required": state.get("confirmation_required", False),
            "rag_used": state.get("rag_used", False),
        },
    )


def _normalize_hour(period: str | None, raw_hour: str) -> str:
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
    return f"{hour:02d}:00"


def _successful_tool_result(state: OperationsAgentState, tool_name: str) -> dict[str, Any] | None:
    for result in state.get("tool_results", []):
        if result.get("tool_name") == tool_name and result.get("success"):
            return result
    return None


def _build_confirmation_summary(
    state: OperationsAgentState,
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
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
