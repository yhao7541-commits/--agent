from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from memory.memory_proposals import extract_memory_proposals
from rag.citation import build_citation_metadata
from security.guardrails import (
    build_confirmation_token,
    detect_prompt_injection,
    is_valid_confirmation_token,
)
from tools.gateway import ToolGateway
from tools.registry import build_default_tool_registry
from tools.schemas import ToolExecutionContext

from .state import OperationsAgentState


BOOKING_KEYWORDS = ("约", "预约", "改约", "取消", "安排")
CONSULTATION_KEYWORDS = ("迟到", "价格", "多少钱", "政策", "服务", "项目", "适合", "注意", "员工", "技师", "手法")
MEDICAL_ESCALATION_KEYWORDS = ("受伤", "很疼", "疼痛", "医疗", "医生")
REFUND_ESCALATION_KEYWORDS = ("退款", "投诉", "服务很差", "争议")
MEMORY_KEYWORDS = ("喜欢", "不喜欢", "过敏", "不要营销", "别营销")
BOOKING_SLOT_UPDATE_KEYWORDS = ("今天", "明天", "上午", "下午", "晚上", "点", "分钟", "安静")
BOOKING_WRITE_TOOLS = {"create_booking", "reschedule_booking", "cancel_booking"}
MEMORY_WRITE_TOOLS = {"write_customer_preference"}
LOCAL_TIMEZONE = timezone(timedelta(hours=8))


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
    state.setdefault("booking_issue", {})
    state.setdefault("missing_slots", [])
    state.setdefault("customer_context", {})
    state.setdefault("retrieved_knowledge", [])
    state.setdefault("rag_citations", {})
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
    elif any(keyword in message for keyword in REFUND_ESCALATION_KEYWORDS):
        intent = "escalation"
        confidence = 1.0
        state["escalated"] = True
        state["escalation"] = _build_escalation(
            state,
            reason="refund_dispute",
            requested_action="refund_or_complaint_review",
        )
    elif state.get("confirmed_tool_name") == "cancel_booking":
        intent = "cancel"
        confidence = 1.0
    elif state.get("confirmed_tool_name") == "reschedule_booking":
        intent = "reschedule"
        confidence = 1.0
    elif state.get("confirmed_tool_name") == "create_booking":
        intent = "booking"
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
    state["customer_context"] = {
        "user_id": state.get("user_id", "local_user"),
        "known_preferences": [],
    }
    return append_trace(state, "load_customer_context", {"loaded": True})


def extract_booking_slots(state: OperationsAgentState) -> OperationsAgentState:
    if state.get("intent") not in {"booking", "cancel", "reschedule"}:
        return append_trace(state, "extract_booking_slots", {"skipped": True})
    if state.get("escalated"):
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
    slots: dict[str, Any] = dict(state.get("booking_slots", {}))
    booking_id_match = re.search(r"\bbooking_[A-Za-z0-9_-]+\b", message)
    if booking_id_match:
        slots["booking_id"] = booking_id_match.group(0)

    if state.get("intent") == "cancel":
        slots["customer_name"] = state.get("user_id", "local_user")
        missing = ["booking_id"] if not slots.get("booking_id") else []
        state["booking_slots"] = slots
        state["missing_slots"] = missing
        return append_trace(state, "extract_booking_slots", {"slots": slots, "missing_slots": missing})

    if "肩颈" in message:
        slots["service_type"] = "肩颈放松"
    elif "推拿" in message:
        slots["service_type"] = "推拿"
    elif "按摩" in message:
        slots["service_type"] = "按摩"

    if "明天" in message:
        slots["date"] = (datetime.now(LOCAL_TIMEZONE) + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "今天" in message:
        slots["date"] = datetime.now(LOCAL_TIMEZONE).strftime("%Y-%m-%d")

    time_match = re.search(
        r"(上午|下午|晚上)?\s*(\d{1,2}|一|二|两|三|四|五|六|七|八|九|十)点(?![的儿也])",
        message,
    )
    if time_match:
        slots["time_window"] = _normalize_hour(time_match.group(1), time_match.group(2))
    elif "上午" in message:
        slots["time_window"] = "09:00-12:00"
    elif "下午" in message:
        slots["time_window"] = "12:00-18:00"
    elif "晚上" in message:
        slots["time_window"] = "18:00-21:00"

    duration_match = re.search(r"(\d+)\s*分钟", message)
    if duration_match:
        slots["duration"] = f"{duration_match.group(1)}分钟"

    if "安静" in message:
        slots["special_requests"] = "安静一点的房间"
    if "李雷" in message:
        slots["preferred_staff"] = "李雷"

    slots["customer_name"] = state.get("user_id", "local_user")
    if state.get("intent") == "reschedule":
        if slots.get("date"):
            slots["new_date"] = slots["date"]
        if slots.get("time_window"):
            slots["new_time_window"] = slots["time_window"]
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
    state["missing_slots"] = missing
    return append_trace(state, "extract_booking_slots", {"slots": slots, "missing_slots": missing})


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
            state["confirmation_required"] = True
            state["confirmation_request"] = {
                "tool_name": result.tool_name,
                "arguments": planned_tool.get("arguments", {}),
                "summary": summary,
                "message": _confirmation_message(result.tool_name),
                "confirmation_token": build_confirmation_token(
                    state.get("conversation_id", ""),
                    result.tool_name,
                    planned_tool.get("arguments", {}),
                ),
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
        if result.success and result.tool_name == "write_customer_preference":
            state["confirmation_required"] = False
            state["confirmation_request"] = {}
        if result.success and result.tool_name == "find_available_staff":
            _apply_staff_availability_issue(state, result.output)
        if result.success and result.tool_name == "check_schedule":
            _apply_schedule_issue(state, result.output)

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
    elif _successful_tool_result(state, "write_customer_preference"):
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


def _has_unsafe_confirmed_tool_request(state: OperationsAgentState) -> bool:
    tool_name = state.get("confirmed_tool_name")
    if tool_name not in BOOKING_WRITE_TOOLS | MEMORY_WRITE_TOOLS:
        return False
    return not is_valid_confirmation_token(
        state.get("conversation_id", ""),
        tool_name,
        state.get("confirmed_tool_arguments", {}),
        state.get("confirmation_token"),
    )


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
