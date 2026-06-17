from agents.operations.graph import build_operations_graph, run_operations_turn


def test_operations_graph_compiles():
    graph = build_operations_graph()

    assert graph is not None


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
