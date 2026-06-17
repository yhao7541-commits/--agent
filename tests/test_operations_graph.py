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
