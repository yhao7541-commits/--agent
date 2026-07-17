import agents.operations.nodes as operation_nodes
from agents.operations.graph import build_operations_graph, run_operations_turn
from agents.operations.nodes import output_policy_check
from datetime import datetime, timedelta, timezone
from tools.base import ToolDefinition, ToolPermission
from tools.customer_tools import reset_customer_memory_store
from tools.registry import build_default_tool_registry
from tools.schemas import (
    CheckScheduleInput,
    CheckScheduleOutput,
    FindAvailableStaffInput,
    FindAvailableStaffOutput,
)

LOCAL_TIMEZONE = timezone(timedelta(hours=8))


def _next_weekday_date(weekday: int) -> str:
    today = datetime.now(LOCAL_TIMEZONE).date()
    start_of_next_week = today + timedelta(days=(7 - today.weekday()))
    return (start_of_next_week + timedelta(days=weekday)).strftime("%Y-%m-%d")


def test_operations_graph_compiles():
    graph = build_operations_graph()

    assert graph is not None


def test_output_policy_check_runs_before_finalize():
    result = run_operations_turn(
        {
            "user_id": "user_policy_001",
            "conversation_id": "conv_policy_001",
            "message": "我想约一个肩颈放松",
        }
    )

    nodes = [event["node"] for event in result["trace_events"]]

    assert "output_policy_check" in nodes
    assert nodes.index("generate_response") < nodes.index("output_policy_check")
    assert nodes.index("output_policy_check") < nodes.index("finalize_turn")


def test_output_policy_check_blocks_false_booking_success_reply():
    result = output_policy_check(
        {
            "trace_id": "trace_policy_001",
            "conversation_id": "conv_policy_002",
            "reply": "预约已创建，预约编号：booking_fake。",
            "tool_results": [],
            "errors": [],
            "trace_events": [],
        }
    )

    assert "预约已创建" not in result["reply"]
    assert result["errors"][-1]["type"] == "false_booking_success"
    assert result["trace_events"][-1]["node"] == "output_policy_check"
    assert result["trace_events"][-1]["event_type"] == "policy_violation"


def test_incomplete_booking_asks_follow_up_without_create_booking():
    result = run_operations_turn(
        {
            "user_id": "user_001",
            "conversation_id": "conv_001",
            "message": "我想约一个肩颈放松",
        }
    )

    assert result["intent"] == "booking"
    assert "date" in result["missing_slots"]
    assert "time_window" in result["missing_slots"]
    assert result["confirmation_required"] is False
    assert all(plan["tool_name"] != "create_booking" for plan in result["tool_plan"])
    assert "日期" in result["reply"] or "时间" in result["reply"]


def test_booking_slot_follow_up_merges_existing_slots():
    first_turn = run_operations_turn(
        {
            "user_id": "user_010",
            "conversation_id": "conv_010",
            "message": "我想约一个肩颈放松",
        }
    )

    result = run_operations_turn(
        {
            "user_id": "user_010",
            "conversation_id": "conv_010",
            "message": "明天下午3点",
            "booking_slots": first_turn["booking_slots"],
        }
    )

    assert result["intent"] == "booking"
    assert result["missing_slots"] == []
    assert result["booking_slots"]["service_type"] == "肩颈放松"
    assert result["booking_slots"]["time_window"] == "15:00"
    assert result["confirmation_required"] is True
    assert result["confirmation_request"]["tool_name"] == "create_booking"


def test_fuzzy_booking_time_is_preserved_as_range():
    result = run_operations_turn(
        {
            "user_id": "user_011",
            "conversation_id": "conv_011",
            "message": "我想明天下午约肩颈放松",
        }
    )

    assert result["intent"] == "booking"
    assert result["missing_slots"] == []
    assert result["booking_slots"]["time_window"] == "12:00-18:00"
    assert result["confirmation_required"] is True


def test_half_hour_booking_time_is_normalized():
    result = run_operations_turn(
        {
            "user_id": "user_half_hour",
            "conversation_id": "conv_half_hour",
            "message": "我想明天下午3点半约肩颈放松",
        }
    )

    assert result["intent"] == "booking"
    assert result["missing_slots"] == []
    assert result["booking_slots"]["time_window"] == "15:30"
    assert result["booking_slot_sources"]["time_window"] == "user"
    assert result["confirmation_required"] is True


def test_day_after_tomorrow_booking_date_is_normalized():
    result = run_operations_turn(
        {
            "user_id": "user_day_after_tomorrow",
            "conversation_id": "conv_day_after_tomorrow",
            "message": "我想后天上午10点约推拿",
        }
    )

    assert result["intent"] == "booking"
    assert result["missing_slots"] == []
    assert result["booking_slots"]["date"] == (
        datetime.now(LOCAL_TIMEZONE) + timedelta(days=2)
    ).strftime("%Y-%m-%d")
    assert result["booking_slots"]["time_window"] == "10:00"
    assert result["confirmation_required"] is True


def test_next_weekday_booking_date_is_normalized():
    result = run_operations_turn(
        {
            "user_id": "user_next_weekday",
            "conversation_id": "conv_next_weekday",
            "message": "我想下周五晚上7点约按摩",
        }
    )

    assert result["intent"] == "booking"
    assert result["missing_slots"] == []
    assert result["booking_slots"]["date"] == _next_weekday_date(4)
    assert result["booking_slots"]["time_window"] == "19:00"
    assert result["confirmation_required"] is True


def test_complete_booking_requires_confirmation_before_create_booking():
    result = run_operations_turn(
        {
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "我想明天下午3点约肩颈放松",
        }
    )

    assert result["intent"] == "booking"
    assert result["missing_slots"] == []
    assert result["confirmation_required"] is True
    assert result["confirmation_request"]["tool_name"] == "create_booking"
    assert any(plan["tool_name"] == "create_booking" for plan in result["tool_plan"])
    assert not any(
        tool_result.get("tool_name") == "create_booking" and tool_result.get("success")
        for tool_result in result["tool_results"]
    )


def test_complete_booking_confirmation_contract_has_business_summary():
    result = run_operations_turn(
        {
            "user_id": "user_004",
            "conversation_id": "conv_004",
            "message": "我想明天下午3点约肩颈放松60分钟，需要安静一点的房间",
        }
    )

    request = result["confirmation_request"]
    summary = request["summary"]

    assert request["tool_name"] == "create_booking"
    assert summary["service"] == "肩颈放松"
    assert summary["staff"] != ""
    assert summary["date"] == request["arguments"]["date"]
    assert summary["time"] == "15:00"
    assert summary["duration"] == "60分钟"
    assert summary["price"] != ""
    assert "安静" in summary["special_requests"]
    assert "取消" in summary["cancellation_policy"]

    for label in ("服务：", "员工：", "日期：", "时间：", "时长：", "价格：", "特殊需求：", "取消政策："):
        assert label in result["reply"]


def test_booking_slot_sources_mark_user_and_system_values():
    result = run_operations_turn(
        {
            "user_id": "user_sources",
            "conversation_id": "conv_sources",
            "message": "我想明天下午3点约肩颈放松60分钟，需要安静一点的房间",
        }
    )

    sources = result["booking_slot_sources"]
    assert sources["service_type"] == "user"
    assert sources["date"] == "user"
    assert sources["time_window"] == "user"
    assert sources["duration"] == "user"
    assert sources["special_requests"] == "user"
    assert sources["customer_name"] == "system"
    assert any(
        event["node"] == "extract_booking_slots"
        and event["metadata"].get("slot_sources", {}).get("service_type") == "user"
        for event in result["trace_events"]
    )


def test_unavailable_preferred_staff_suggests_alternative_without_booking_confirmation():
    result = run_operations_turn(
        {
            "user_id": "user_012",
            "conversation_id": "conv_012",
            "message": "我想明天下午3点约肩颈放松，指定李雷",
        }
    )

    assert result["intent"] == "booking"
    assert result["confirmation_required"] is False
    assert result["booking_issue"]["type"] == "staff_unavailable"
    assert "李雷" in result["booking_issue"]["message"]
    assert result["booking_issue"]["alternatives"]
    assert all(plan["tool_name"] != "create_booking" for plan in result["tool_plan"])
    assert "可选" in result["reply"] or "替代" in result["reply"]


def test_schedule_conflict_suggests_nearby_times_without_booking_confirmation():
    result = run_operations_turn(
        {
            "user_id": "user_013",
            "conversation_id": "conv_013",
            "message": "我想明天下午5点约肩颈放松",
        }
    )

    assert result["intent"] == "booking"
    assert result["confirmation_required"] is False
    assert result["booking_issue"]["type"] == "time_conflict"
    assert result["booking_issue"]["alternatives"]
    assert all(plan["tool_name"] != "create_booking" for plan in result["tool_plan"])
    assert "可选" in result["reply"] or "附近" in result["reply"]


def test_cancel_booking_without_booking_id_asks_for_existing_booking():
    result = run_operations_turn(
        {
            "user_id": "user_014",
            "conversation_id": "conv_014",
            "message": "我要取消预约",
        }
    )

    assert result["intent"] == "cancel"
    assert "booking_id" in result["missing_slots"]
    assert result["confirmation_required"] is False
    assert all(plan["tool_name"] != "cancel_booking" for plan in result["tool_plan"])
    assert "预约编号" in result["reply"] or "已有预约" in result["reply"]


def test_cancel_booking_with_booking_id_requires_confirmation():
    result = run_operations_turn(
        {
            "user_id": "user_015",
            "conversation_id": "conv_015",
            "message": "取消预约 booking_1234",
        }
    )

    assert result["intent"] == "cancel"
    assert result["missing_slots"] == []
    assert result["confirmation_required"] is True
    assert result["confirmation_request"]["tool_name"] == "cancel_booking"
    assert result["confirmation_request"]["arguments"]["booking_id"] == "booking_1234"
    assert not any(
        tool_result.get("tool_name") == "cancel_booking" and tool_result.get("success")
        for tool_result in result["tool_results"]
    )


def test_reschedule_without_existing_booking_id_asks_for_old_booking():
    result = run_operations_turn(
        {
            "user_id": "user_016",
            "conversation_id": "conv_016",
            "message": "我要改约到明天下午3点",
        }
    )

    assert result["intent"] == "reschedule"
    assert "booking_id" in result["missing_slots"]
    assert result["confirmation_required"] is False
    assert all(plan["tool_name"] != "reschedule_booking" for plan in result["tool_plan"])


def test_reschedule_with_booking_id_and_new_time_requires_confirmation():
    result = run_operations_turn(
        {
            "user_id": "user_017",
            "conversation_id": "conv_017",
            "message": "把 booking_5678 改约到明天下午3点",
        }
    )

    assert result["intent"] == "reschedule"
    assert result["missing_slots"] == []
    assert result["confirmation_required"] is True
    assert result["confirmation_request"]["tool_name"] == "reschedule_booking"
    assert result["confirmation_request"]["arguments"]["booking_id"] == "booking_5678"
    assert result["confirmation_request"]["arguments"]["new_time_window"] == "15:00"
    assert not any(
        tool_result.get("tool_name") == "reschedule_booking" and tool_result.get("success")
        for tool_result in result["tool_results"]
    )


def test_confirmed_booking_executes_create_booking():
    pending = run_operations_turn(
        {
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "我想明天下午3点约肩颈放松",
        }
    )

    result = run_operations_turn(
        {
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "确认",
            "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
            "confirmation_token": pending["confirmation_request"]["confirmation_token"],
        }
    )

    assert result["intent"] == "booking"
    assert result["confirmation_required"] is False
    assert any(
        tool_result.get("tool_name") == "create_booking" and tool_result.get("success")
        for tool_result in result["tool_results"]
    )
    assert "预约已创建" in result["reply"]


def test_policy_question_uses_knowledge_tool_path():
    result = run_operations_turn(
        {
            "user_id": "user_003",
            "conversation_id": "conv_003",
            "message": "如果我迟到20分钟会怎么样？",
        }
    )

    assert result["intent"] == "consultation"
    assert any(plan["tool_name"] == "search_knowledge_base" for plan in result["tool_plan"])
    assert result["retrieved_knowledge"]
    assert result["rag_used"] is True


def test_memory_preference_proposes_confirmed_write_tool():
    result = run_operations_turn(
        {
            "user_id": "user_008",
            "conversation_id": "conv_008",
            "message": "我以后都喜欢安静一点的房间",
        }
    )

    assert result["intent"] == "memory"
    assert result["escalated"] is False
    assert result["memory_proposals"][0]["content"] == "喜欢安静房间"
    assert result["confirmation_required"] is True
    assert result["confirmation_request"]["tool_name"] == "write_customer_preference"
    assert any(plan["tool_name"] == "write_customer_preference" for plan in result["tool_plan"])


def test_confirmed_memory_write_executes_tool():
    pending = run_operations_turn(
        {
            "user_id": "user_008",
            "conversation_id": "conv_008",
            "message": "我以后都喜欢安静一点的房间",
        }
    )

    result = run_operations_turn(
        {
            "user_id": "user_008",
            "conversation_id": "conv_008",
            "message": "确认",
            "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
            "confirmation_token": pending["confirmation_request"]["confirmation_token"],
        }
    )

    assert result["intent"] == "memory"
    assert result["confirmation_required"] is False
    assert any(
        tool_result.get("tool_name") == "write_customer_preference" and tool_result.get("success")
        for tool_result in result["tool_results"]
    )
    assert "偏好已保存" in result["reply"]


def test_booking_uses_stored_customer_preference_in_confirmation_summary():
    reset_customer_memory_store()
    pending_memory = run_operations_turn(
        {
            "user_id": "user_memory_recall",
            "conversation_id": "conv_memory_recall",
            "message": "我以后都喜欢安静一点的房间",
        }
    )
    run_operations_turn(
        {
            "user_id": "user_memory_recall",
            "conversation_id": "conv_memory_recall",
            "message": "确认",
            "confirmed_tool_name": pending_memory["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending_memory["confirmation_request"]["arguments"],
            "confirmation_token": pending_memory["confirmation_request"]["confirmation_token"],
        }
    )

    result = run_operations_turn(
        {
            "user_id": "user_memory_recall",
            "conversation_id": "conv_memory_recall_booking",
            "message": "我想明天下午3点约肩颈放松",
        }
    )

    assert "喜欢安静房间" in result["customer_context"]["known_preferences"]
    assert "安静" in result["confirmation_request"]["summary"]["special_requests"]
    assert "memory" in result["booking_slot_sources"]["special_requests"]
    assert result["memory_used"] is True
    assert any(
        memory["content"] == "喜欢安静房间"
        and memory["applied_to"] == "booking_slots.special_requests"
        for memory in result["applied_customer_memories"]
    )
    assert any(
        event["node"] == "load_customer_context"
        and event["metadata"].get("memory_count") == 1
        for event in result["trace_events"]
    )
    assert any(
        event["node"] == "extract_booking_slots"
        and event["metadata"].get("memory_used") is True
        for event in result["trace_events"]
    )


def test_confirmed_memory_delete_removes_preference_from_future_context():
    reset_customer_memory_store()
    pending_memory = run_operations_turn(
        {
            "user_id": "user_memory_delete",
            "conversation_id": "conv_memory_delete",
            "message": "我以后都喜欢安静一点的房间",
        }
    )
    run_operations_turn(
        {
            "user_id": "user_memory_delete",
            "conversation_id": "conv_memory_delete",
            "message": "确认",
            "confirmed_tool_name": pending_memory["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending_memory["confirmation_request"]["arguments"],
            "confirmation_token": pending_memory["confirmation_request"]["confirmation_token"],
        }
    )

    pending_delete = run_operations_turn(
        {
            "user_id": "user_memory_delete",
            "conversation_id": "conv_memory_delete",
            "message": "请忘记安静房间这个偏好",
        }
    )
    result = run_operations_turn(
        {
            "user_id": "user_memory_delete",
            "conversation_id": "conv_memory_delete",
            "message": "确认",
            "confirmed_tool_name": pending_delete["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending_delete["confirmation_request"]["arguments"],
            "confirmation_token": pending_delete["confirmation_request"]["confirmation_token"],
        }
    )
    follow_up = run_operations_turn(
        {
            "user_id": "user_memory_delete",
            "conversation_id": "conv_memory_delete_follow_up",
            "message": "我想明天下午3点约肩颈放松",
        }
    )

    assert pending_delete["confirmation_required"] is True
    assert pending_delete["confirmation_request"]["tool_name"] == "delete_customer_memory"
    assert result["confirmation_required"] is False
    assert any(
        tool_result.get("tool_name") == "delete_customer_memory" and tool_result.get("success")
        for tool_result in result["tool_results"]
    )
    assert follow_up["customer_context"]["known_preferences"] == []
    assert follow_up["confirmation_request"]["summary"]["special_requests"] == "无"


def test_conflicting_memory_write_does_not_claim_saved():
    reset_customer_memory_store()
    pending_quiet = run_operations_turn(
        {
            "user_id": "user_memory_conflict",
            "conversation_id": "conv_memory_conflict",
            "message": "我以后都喜欢安静一点的房间",
        }
    )
    run_operations_turn(
        {
            "user_id": "user_memory_conflict",
            "conversation_id": "conv_memory_conflict",
            "message": "确认",
            "confirmed_tool_name": pending_quiet["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending_quiet["confirmation_request"]["arguments"],
            "confirmation_token": pending_quiet["confirmation_request"]["confirmation_token"],
        }
    )
    pending_lively = run_operations_turn(
        {
            "user_id": "user_memory_conflict",
            "conversation_id": "conv_memory_conflict",
            "message": "我喜欢热闹一点的房间",
        }
    )

    result = run_operations_turn(
        {
            "user_id": "user_memory_conflict",
            "conversation_id": "conv_memory_conflict",
            "message": "确认",
            "confirmed_tool_name": pending_lively["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending_lively["confirmation_request"]["arguments"],
            "confirmation_token": pending_lively["confirmation_request"]["confirmation_token"],
        }
    )

    memory_result = next(
        tool_result
        for tool_result in result["tool_results"]
        if tool_result["tool_name"] == "write_customer_preference"
    )
    assert memory_result["output"]["status"] == "conflict"
    assert "冲突" in result["reply"]
    assert "已保存" not in result["reply"]


def test_sensitive_memory_proposal_requires_confirmation():
    result = run_operations_turn(
        {
            "user_id": "user_009",
            "conversation_id": "conv_009",
            "message": "我对精油过敏，请以后不要用",
        }
    )

    proposal = result["memory_proposals"][0]

    assert result["intent"] == "memory"
    assert proposal["sensitivity"] == "sensitive"
    assert proposal["requires_confirmation"] is True
    assert result["confirmation_required"] is True


def test_medical_concern_escalates_to_human_with_summary():
    result = run_operations_turn(
        {
            "user_id": "user_005",
            "conversation_id": "conv_005",
            "message": "按摩后肩膀受伤了，现在很疼怎么办？",
        }
    )

    assert result["escalated"] is True
    assert result["escalation"]["reason"] == "medical_concern"
    assert any(plan["tool_name"] == "escalate_to_human" for plan in result["tool_plan"])
    assert any(
        tool_result.get("tool_name") == "escalate_to_human" and tool_result.get("success")
        for tool_result in result["tool_results"]
    )
    assert result["escalation"]["summary"]
    assert "人工" in result["reply"]


def test_refund_dispute_escalates_to_human():
    result = run_operations_turn(
        {
            "user_id": "user_006",
            "conversation_id": "conv_006",
            "message": "我要退款，昨天的服务很差，我要投诉",
        }
    )

    assert result["escalated"] is True
    assert result["escalation"]["reason"] == "refund_dispute"
    assert any(plan["tool_name"] == "escalate_to_human" for plan in result["tool_plan"])


def test_low_confidence_unknown_request_escalates_to_human():
    result = run_operations_turn(
        {
            "user_id": "user_007",
            "conversation_id": "conv_007",
            "message": "随便吧那个什么处理一下",
        }
    )

    assert result["escalated"] is True
    assert result["escalation"]["reason"] == "low_confidence"
    assert any(plan["tool_name"] == "escalate_to_human" for plan in result["tool_plan"])


def test_repeated_booking_tool_failures_escalate_to_human(monkeypatch):
    registry = build_default_tool_registry()

    def fail_handler(arguments, context):
        raise RuntimeError("booking dependency unavailable")

    registry.register(
        ToolDefinition(
            name="find_available_staff",
            description="Failing staff lookup",
            permission=ToolPermission.READ,
            requires_confirmation=False,
            input_schema=FindAvailableStaffInput,
            output_schema=FindAvailableStaffOutput,
            handler=fail_handler,
            max_retries=0,
        )
    )
    registry.register(
        ToolDefinition(
            name="check_schedule",
            description="Failing schedule lookup",
            permission=ToolPermission.READ,
            requires_confirmation=False,
            input_schema=CheckScheduleInput,
            output_schema=CheckScheduleOutput,
            handler=fail_handler,
            max_retries=0,
        )
    )
    monkeypatch.setattr(operation_nodes, "build_default_tool_registry", lambda: registry)

    result = run_operations_turn(
        {
            "user_id": "user_tool_failure",
            "conversation_id": "conv_tool_failure",
            "message": "我想明天下午3点约肩颈放松",
        }
    )

    assert result["escalated"] is True
    assert result["escalation"]["reason"] == "tool_failure"
    assert result["confirmation_required"] is False
    assert all(plan["tool_name"] != "create_booking" for plan in result["tool_plan"])
    assert all(tool_result.get("tool_name") != "create_booking" for tool_result in result["tool_results"])
    assert any(
        tool_result.get("tool_name") == "escalate_to_human" and tool_result.get("success")
        for tool_result in result["tool_results"]
    )
    assert "人工" in result["reply"]
