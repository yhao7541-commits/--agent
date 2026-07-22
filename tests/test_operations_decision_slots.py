from __future__ import annotations

import pytest

from agents.operations.decision_validation import merge_and_validate_booking_slots
from agents.operations.nodes import (
    execute_tools,
    extract_booking_slots,
    generate_response,
    plan_tool_calls,
)
from agents.operations.graph import run_operations_turn


def test_slot_validation_result_has_bounded_merge_outputs():
    from agents.operations.decision_validation import SlotValidationResult

    result = SlotValidationResult(
        slots={"service_type": "肩颈放松"},
        sources={"service_type": "user"},
        ambiguities=[],
        errors=[],
    )

    assert result.model_dump() == {
        "slots": {"service_type": "肩颈放松"},
        "sources": {"service_type": "user"},
        "ambiguities": [],
        "errors": [],
    }


def test_current_model_and_user_slots_override_previous_values():
    result = merge_and_validate_booking_slots(
        previous_slots={"service_type": "按摩", "date": "2026-08-01"},
        previous_sources={"service_type": "previous_turn", "date": "previous_turn"},
        model_slots={"service_type": "推拿", "date": "2026-08-02"},
        user_slots={"service_type": "肩颈放松"},
    )

    assert result.slots["service_type"] == "肩颈放松"
    assert result.slots["date"] == "2026-08-02"
    assert result.sources == {"service_type": "user", "date": "user"}


def test_previous_slots_remain_when_current_turn_omits_them():
    result = merge_and_validate_booking_slots(
        previous_slots={"service_type": "推拿", "date": "2026-08-01"},
        previous_sources={"service_type": "user", "date": "user"},
        model_slots={},
        user_slots={"time_window": "15:00"},
    )

    assert result.slots == {
        "service_type": "推拿",
        "date": "2026-08-01",
        "time_window": "15:00",
    }
    assert result.sources == {
        "service_type": "previous_turn",
        "date": "previous_turn",
        "time_window": "user",
    }


def test_approved_memory_only_fills_absent_special_request():
    absent = merge_and_validate_booking_slots(
        previous_slots={},
        previous_sources={},
        model_slots={},
        user_slots={},
        memory_special_request="安静一点的房间",
    )
    current = merge_and_validate_booking_slots(
        previous_slots={},
        previous_sources={},
        model_slots={"special_requests": "靠近窗户"},
        user_slots={},
        memory_special_request="安静一点的房间",
    )

    assert absent.slots["special_requests"] == "安静一点的房间"
    assert absent.sources["special_requests"] == "memory"
    assert current.slots["special_requests"] == "靠近窗户"
    assert current.sources["special_requests"] == "user"


def test_system_customer_name_does_not_override_user_owned_value():
    result = merge_and_validate_booking_slots(
        previous_slots={"customer_name": "王小明"},
        previous_sources={"customer_name": "user"},
        model_slots={},
        user_slots={},
        system_customer_name="user_123",
    )

    assert result.slots["customer_name"] == "王小明"
    assert result.sources["customer_name"] == "previous_turn"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date", "下辈子"),
        ("service_type", "飞行按摩"),
        ("time_window", "25:00"),
    ],
)
def test_invalid_core_slot_becomes_ambiguity_and_is_removed(field: str, value: str):
    result = merge_and_validate_booking_slots(
        previous_slots={},
        previous_sources={},
        model_slots={field: value},
        user_slots={},
        ambiguities=[field],
    )

    assert field in result.ambiguities
    assert field not in result.slots


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("duration", "999分钟", "invalid_duration"),
        ("preferred_staff", "王五", "unknown_staff"),
        ("booking_id", "1234", "invalid_booking_id"),
    ],
)
def test_invalid_business_slot_is_an_error_and_is_removed(
    field: str, value: str, code: str
):
    result = merge_and_validate_booking_slots(
        previous_slots={},
        previous_sources={},
        model_slots={field: value},
        user_slots={},
        errors=[{"field": field, "code": code}],
    )

    assert {error["code"] for error in result.errors} == {code}
    assert field not in result.slots


def test_confirmed_arguments_bypass_slot_merge_and_validation():
    arguments = {
        "service_type": "已确认的自定义服务",
        "date": "not-revalidated",
        "time_window": "exact-confirmed-value",
        "customer_name": "用户确认名称",
    }

    result = extract_booking_slots(
        {
            "intent": "booking",
            "confirmed_tool_name": "create_booking",
            "confirmed_tool_arguments": arguments,
            "booking_slots": {"service_type": "推拿"},
            "booking_slot_sources": {"service_type": "previous_turn"},
            "trace_events": [],
        }
    )

    assert result["booking_slots"] == arguments
    assert set(result["booking_slot_sources"].values()) == {
        "confirmed_tool_arguments"
    }
    assert result["missing_slots"] == []


def test_model_slots_merge_into_booking_state_without_changing_decision_metadata():
    result = extract_booking_slots(
        {
            "user_id": "user_123",
            "intent": "booking",
            "message": "帮我预约",
            "booking_slots": {"service_type": "按摩"},
            "booking_slot_sources": {"service_type": "user"},
            "model_decision": {
                "extracted_slots": {
                    "service_type": "推拿",
                    "date": "2026-08-02",
                    "time_window": "15:00",
                }
            },
            "decision_metadata": {"source": "llm"},
            "trace_events": [],
        }
    )

    assert result["booking_slots"]["service_type"] == "推拿"
    assert result["booking_slots"]["date"] == "2026-08-02"
    assert result["booking_slot_sources"]["service_type"] == "user"
    assert result["decision_metadata"]["source"] == "llm"
    assert result["missing_slots"] == []


@pytest.mark.parametrize(
    ("intent", "field", "value", "expected_code"),
    [
        ("booking", "date", "下辈子", None),
        ("booking", "date", "2020-01-01", None),
        ("booking", "service_type", "飞行按摩", None),
        ("booking", "time_window", "25:00", None),
        ("booking", "duration", "999分钟", "invalid_duration"),
        ("booking", "preferred_staff", "王五", "unknown_staff"),
        ("cancel", "booking_id", "1234", "invalid_booking_id"),
    ],
)
def test_invalid_model_slot_blocks_write_and_asks_only_for_related_field(
    intent: str, field: str, value: str, expected_code: str | None
):
    base_slots = {
        "service_type": "推拿",
        "date": "2026-08-02",
        "time_window": "15:00",
    }
    state = {
        "user_id": "user_123",
        "intent": intent,
        "message": "帮我处理预约",
        "booking_slots": base_slots,
        "booking_slot_sources": {key: "user" for key in base_slots},
        "model_decision": {"extracted_slots": {field: value}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)
    generate_response(state)

    assert field in state["missing_slots"]
    assert field not in state["booking_slots"]
    assert not any(
        item["tool_name"] in {"create_booking", "cancel_booking", "reschedule_booking"}
        for item in state["tool_plan"]
    )
    assert all(
        other not in state["missing_slots"]
        for other in {"service_type", "date", "time_window", "duration", "preferred_staff", "booking_id"}
        - {field}
    )
    if expected_code is None:
        assert state["ambiguities"] == [field]
    else:
        assert {error["code"] for error in state["decision_errors"]} == {
            expected_code
        }


def test_unavailable_staff_tool_result_blocks_write_and_records_validation_error():
    state = {
        "user_id": "user_123",
        "conversation_id": "conv-unavailable-staff",
        "trace_id": "trace-unavailable-staff",
        "intent": "booking",
        "message": "帮我预约推拿",
        "booking_slots": {
            "service_type": "推拿",
            "date": "2026-08-02",
            "time_window": "15:00",
        },
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {"preferred_staff": "李雷"}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)
    execute_tools(state)

    assert state["booking_issue"]["type"] == "staff_unavailable"
    assert {error["code"] for error in state["decision_errors"]} == {
        "staff_unavailable"
    }
    assert not any(
        item["tool_name"] == "create_booking" for item in state["tool_plan"]
    )
    assert not any(
        item["tool_name"] == "create_booking" and item["success"]
        for item in state["tool_results"]
    )


@pytest.mark.parametrize(
    ("intent", "message", "field", "code"),
    [
        (
            "booking",
            "帮我预约推拿 999分钟",
            "duration",
            "invalid_duration",
        ),
        (
            "booking",
            "帮我预约推拿，指定王五",
            "preferred_staff",
            "unknown_staff",
        ),
        (
            "cancel",
            "取消预约 booking#1234",
            "booking_id",
            "invalid_booking_id",
        ),
    ],
)
def test_invalid_user_business_slot_is_recorded_and_blocks_write(
    intent: str, message: str, field: str, code: str
):
    state = {
        "user_id": "user_123",
        "intent": intent,
        "message": message,
        "booking_slots": {
            "service_type": "推拿",
            "date": "2026-08-02",
            "time_window": "15:00",
        },
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert field in state["missing_slots"]
    assert {error["code"] for error in state["decision_errors"]} == {code}
    assert not any(
        item["tool_name"] in {"create_booking", "cancel_booking"}
        for item in state["tool_plan"]
    )


def test_multiple_operation_intents_force_clarification_and_no_write_plan():
    state = {
        "user_id": "user_123",
        "intent": "booking",
        "message": "帮我预约推拿，同时取消 booking_1234",
        "booking_slots": {
            "service_type": "推拿",
            "date": "2026-08-02",
            "time_window": "15:00",
        },
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert state["intent"] == "clarification"
    assert state["ambiguities"] == ["intent"]
    assert state["tool_plan"] == []


@pytest.mark.parametrize(
    ("field", "valid", "invalid", "kind", "code"),
    [
        ("service_type", "推拿", "飞行按摩", "ambiguity", "invalid_service_type"),
        ("date", "2026-08-02", "2020-01-01", "ambiguity", "invalid_date"),
        ("time_window", "15:00", "25:00", "ambiguity", "invalid_time_window"),
        ("duration", "60分钟", "999分钟", "error", "invalid_duration"),
        ("preferred_staff", "张伟", "王五", "error", "unknown_staff"),
        ("booking_id", "booking_1234", "1234", "error", "invalid_booking_id"),
    ],
)
def test_valid_user_candidate_clears_lower_priority_model_issue(
    field: str, valid: str, invalid: str, kind: str, code: str
):
    result = merge_and_validate_booking_slots(
        previous_slots={field: "stale"},
        previous_sources={field: "user"},
        model_slots={field: invalid},
        user_slots={field: valid},
        model_issues=[
            {"field": field, "kind": kind, "code": code, "source": "model"}
        ],
    )

    assert result.slots[field] == valid
    assert result.sources[field] == "user"
    assert result.ambiguities == []
    assert result.errors == []


@pytest.mark.parametrize(
    ("field", "valid", "invalid", "kind", "code"),
    [
        ("service_type", "推拿", "飞行按摩", "ambiguity", "invalid_service_type"),
        ("date", "2026-08-02", "2020-01-01", "ambiguity", "invalid_date"),
        ("time_window", "15:00", "25:00", "ambiguity", "invalid_time_window"),
        ("duration", "60分钟", "999分钟", "error", "invalid_duration"),
        ("preferred_staff", "张伟", "王五", "error", "unknown_staff"),
        ("booking_id", "booking_1234", "1234", "error", "invalid_booking_id"),
    ],
)
def test_invalid_user_candidate_blocks_lower_priority_valid_model_value(
    field: str, valid: str, invalid: str, kind: str, code: str
):
    result = merge_and_validate_booking_slots(
        previous_slots={field: "stale"},
        previous_sources={field: "user"},
        model_slots={field: valid},
        user_slots={field: invalid},
        user_issues=[
            {"field": field, "kind": kind, "code": code, "source": "user"}
        ],
    )

    assert field not in result.slots
    assert result.ambiguities == ([field] if kind == "ambiguity" else [])
    assert result.errors == (
        []
        if kind == "ambiguity"
        else [{"field": field, "kind": kind, "code": code, "source": "user"}]
    )


@pytest.mark.parametrize(
    ("intent", "field", "invalid", "code"),
    [
        ("booking", "service_type", "飞行按摩", "invalid_service_type"),
        ("booking", "date", "2020-01-01", "invalid_date"),
        ("booking", "time_window", "25:00", "invalid_time_window"),
        ("booking", "duration", "999分钟", "invalid_duration"),
        ("booking", "preferred_staff", "王五", "unknown_staff"),
        ("cancel", "booking_id", "1234", "invalid_booking_id"),
    ],
)
def test_invalid_previous_slot_is_revalidated_and_cannot_reach_write_arguments(
    intent: str, field: str, invalid: str, code: str
):
    previous = {
        "service_type": "推拿",
        "date": "2026-08-02",
        "time_window": "15:00",
        field: invalid,
    }
    state = {
        "user_id": "user_123",
        "intent": intent,
        "message": "继续处理",
        "booking_slots": previous,
        "booking_slot_sources": {key: "user" for key in previous},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert field not in state["booking_slots"]
    assert field in state["missing_slots"]
    assert any(
        error.get("code") == code and error.get("source") == "previous_turn"
        for error in state["decision_errors"]
    )
    assert not any(
        item["tool_name"] in {"create_booking", "cancel_booking", "reschedule_booking"}
        for item in state["tool_plan"]
    )


@pytest.mark.parametrize(
    "message",
    [
        "取消 booking_1234，然后帮我安排明天下午3点推拿",
        "把 booking_1234 取消，再安排一个明天推拿",
    ],
)
def test_action_word_variants_with_multiple_intents_force_clarification(message: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert state["intent"] == "clarification"
    assert state["tool_plan"] == []


@pytest.mark.parametrize("message", ["取消政策是什么", "改约政策怎么规定"])
def test_policy_questions_are_not_treated_as_multiple_actions(message: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["intent"] == "booking"
    assert state.get("ambiguities", []) == []


def test_action_plus_policy_question_still_forces_multi_intent_clarification():
    state = {
        "intent": "booking",
        "message": "我想取消预约并安排明天推拿，顺便说下取消政策",
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert state["intent"] == "clarification"
    assert state["tool_plan"] == []


@pytest.mark.parametrize(
    "message",
    [
        "booking_1234 的取消预约政策和改约政策有什么区别",
        "请比较取消规则和改期规定",
        "取消预约的条款与改约规则分别是什么",
    ],
)
def test_policy_comparisons_are_not_slot_multi_intent_actions(message: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["intent"] == "booking"
    assert state.get("ambiguities", []) == []


def test_ambiguity_is_not_duplicated_in_public_validation_errors():
    result = merge_and_validate_booking_slots(
        previous_slots={},
        previous_sources={},
        model_slots={"date": "2020-01-01"},
        user_slots={},
        model_issues=[
            {
                "field": "date",
                "kind": "ambiguity",
                "code": "invalid_date",
                "source": "model",
            }
        ],
    )

    assert result.ambiguities == ["date"]
    assert result.errors == []
    assert result.model_dump()["errors"] == []


@pytest.mark.parametrize(
    ("intent", "field", "valid", "invalid", "valid_message", "invalid_message"),
    [
        ("booking", "service_type", "推拿", "飞行按摩", "服务项目：推拿", "服务项目：飞行按摩"),
        ("booking", "date", "2026-08-02", "2020-01-01", "日期：2026-08-02", "日期：2020-01-01"),
        ("booking", "time_window", "15:00", "25:00", "时间：15:00", "时间：25:00"),
        ("booking", "duration", "60分钟", "999分钟", "60分钟", "999分钟"),
        ("booking", "preferred_staff", "张伟", "王五", "指定张伟", "指定王五"),
        ("cancel", "booking_id", "booking_1234", "1234", "booking_1234", "booking#1234"),
    ],
)
def test_node_precedence_resolves_cross_source_validation_by_field(
    intent: str,
    field: str,
    valid: str,
    invalid: str,
    valid_message: str,
    invalid_message: str,
):
    base = {
        "service_type": "推拿",
        "date": "2026-08-02",
        "time_window": "15:00",
    }

    user_wins = {
        "user_id": "user_123",
        "intent": intent,
        "message": valid_message,
        "booking_slots": base,
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {field: invalid}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }
    extract_booking_slots(user_wins)

    assert user_wins["booking_slots"][field] == valid
    assert not any(
        error.get("field") == field for error in user_wins["decision_errors"]
    )

    user_blocks = {
        "user_id": "user_123",
        "intent": intent,
        "message": invalid_message,
        "booking_slots": base,
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {field: valid}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }
    extract_booking_slots(user_blocks)

    assert field not in user_blocks["booking_slots"]
    assert field in user_blocks["missing_slots"]
    assert any(
        error.get("field") == field and error.get("source") == "user"
        for error in user_blocks["decision_errors"]
    )


@pytest.mark.parametrize("message", ["约飞行按摩", "约量子推拿"])
def test_unknown_natural_language_service_is_ambiguous_and_blocks_create(message: str):
    state = {
        "user_id": "user_123",
        "intent": "booking",
        "message": message,
        "booking_slots": {"date": "2026-08-02", "time_window": "15:00"},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert state["ambiguities"] == ["service_type"]
    assert "service_type" not in state["booking_slots"]
    assert not any(item["tool_name"] == "create_booking" for item in state["tool_plan"])


@pytest.mark.parametrize(
    ("message", "expected"),
    [("约按摩", "按摩"), ("约推拿", "推拿"), ("约肩颈放松", "肩颈放松")],
)
def test_known_natural_language_service_remains_valid(message: str, expected: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {"date": "2026-08-02", "time_window": "15:00"},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["booking_slots"]["service_type"] == expected
    assert state["ambiguities"] == []


@pytest.mark.parametrize(
    "message",
    [
        "取消 booking_1234 后再订明天推拿",
        "取消 booking_1234 后再约一个明天按摩",
        "取消 booking_1234，然后改到明天",
    ],
)
def test_multi_intent_action_synonyms_force_clarification(message: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert state["intent"] == "clarification"
    assert state["tool_plan"] == []


@pytest.mark.parametrize(
    "message",
    ["再订和再约有什么区别", "取消后改到其他时间的规则是什么"],
)
def test_action_synonym_knowledge_questions_are_not_multi_intent(message: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["intent"] == "booking"
    assert state.get("ambiguities", []) == []


def test_invalid_labeled_natural_date_overrides_and_blocks_previous_date():
    state = {
        "intent": "booking",
        "message": "日期：下辈子",
        "booking_slots": {
            "service_type": "推拿",
            "date": "2026-08-02",
            "time_window": "15:00",
        },
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert "date" not in state["booking_slots"]
    assert state["ambiguities"] == ["date"]
    assert not any(item["tool_name"] == "create_booking" for item in state["tool_plan"])


@pytest.mark.parametrize("message", ["找王五技师", "王五给我做"])
def test_unknown_natural_staff_expression_blocks_create(message: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {
            "service_type": "推拿",
            "date": "2026-08-02",
            "time_window": "15:00",
        },
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert "preferred_staff" not in state["booking_slots"]
    assert any(error.get("code") == "unknown_staff" for error in state["decision_errors"])
    assert not any(item["tool_name"] == "create_booking" for item in state["tool_plan"])


@pytest.mark.parametrize("malformed", ["booking_---", "booking____"])
@pytest.mark.parametrize("intent", ["booking", "cancel", "reschedule"])
def test_booking_id_without_alphanumeric_suffix_blocks_all_writes(
    malformed: str, intent: str
):
    state = {
        "intent": intent,
        "message": f"处理 {malformed}",
        "booking_slots": {
            "service_type": "推拿",
            "date": "2026-08-02",
            "time_window": "15:00",
            "booking_id": malformed,
        },
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert "booking_id" not in state["booking_slots"]
    assert any(error.get("code") == "invalid_booking_id" for error in state["decision_errors"])
    assert not any(
        item["tool_name"] in {"create_booking", "cancel_booking", "reschedule_booking"}
        for item in state["tool_plan"]
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_missing"),
    [("date", "下辈子", "new_date"), ("time_window", "25:00", "new_time_window")],
)
def test_invalid_reschedule_field_is_requested_once(
    field: str, value: str, expected_missing: str
):
    extracted_slots = (
        {"date": value, "time_window": "15:00"}
        if field == "date"
        else {"date": "2026-08-02", "time_window": value}
    )
    state = {
        "intent": "reschedule",
        "message": "继续改约",
        "booking_slots": {"booking_id": "booking_1234"},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": extracted_slots},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["missing_slots"] == [expected_missing]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("预约明天推拿", "推拿"),
        ("预约后天按摩", "按摩"),
        ("预约下午推拿", "推拿"),
        ("约一下推拿", "推拿"),
        ("约个按摩", "按摩"),
        ("约肩颈按摩", "肩颈放松"),
    ],
)
def test_natural_service_allowed_prefixes_normalize_to_canonical_value(
    message: str, expected: str
):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {"date": "2026-08-02", "time_window": "15:00"},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["booking_slots"]["service_type"] == expected
    assert state["ambiguities"] == []


def test_command_clause_is_not_hidden_by_separate_rules_question():
    state = {
        "intent": "booking",
        "message": "我想取消预约并安排明天推拿，规则是什么",
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert state["intent"] == "clarification"
    assert state["tool_plan"] == []


@pytest.mark.parametrize(
    "message",
    [
        "booking_1234 取消和改约有什么区别",
        "booking_1234 如何取消或改约",
        "booking_1234 取消、改到其他时间分别是什么流程",
        "booking_1234 取消和改约怎样办理",
    ],
)
def test_booking_id_action_comparison_clause_is_not_multi_intent(message: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["intent"] == "booking"
    assert state.get("ambiguities", []) == []


def test_booking_id_with_trailing_hyphens_is_preserved_exactly():
    state = {
        "intent": "cancel",
        "message": "取消 booking_--a--",
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert state["booking_slots"]["booking_id"] == "booking_--a--"
    cancel = next(item for item in state["tool_plan"] if item["tool_name"] == "cancel_booking")
    assert cancel["arguments"]["booking_id"] == "booking_--a--"


@pytest.mark.parametrize("token", ["booking_123@evil", "booking_--a--$", "booking_123中文"])
def test_booking_id_with_illegal_adjacent_suffix_is_rejected_whole(token: str):
    state = {
        "intent": "cancel",
        "message": f"取消 {token}",
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)
    plan_tool_calls(state)

    assert "booking_id" not in state["booking_slots"]
    assert any(error.get("code") == "invalid_booking_id" for error in state["decision_errors"])
    assert not any(item["tool_name"] == "cancel_booking" for item in state["tool_plan"])


@pytest.mark.parametrize(
    "message",
    [
        "预约明天的推拿",
        "预约后天的按摩",
        "预约明天做推拿",
        "预约下周一推拿",
        "预约星期一推拿",
    ],
)
def test_service_prefix_cleanup_reuses_date_expression_grammar(message: str):
    state = {
        "intent": "booking",
        "message": message,
        "booking_slots": {"time_window": "15:00"},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["booking_slots"]["service_type"] in {"推拿", "按摩"}
    assert "service_type" not in state["ambiguities"]


@pytest.mark.parametrize(
    "wrapped",
    [
        "(booking_1234)",
        "（booking_1234）",
        "[booking_1234]",
        "【booking_1234】",
        '"booking_1234"',
        "“booking_1234”",
        "'booking_1234'",
        "‘booking_1234’",
        "「booking_1234」",
        "『booking_1234』",
    ],
)
def test_booking_id_scanner_treats_closing_wrappers_as_delimiters(wrapped: str):
    state = {
        "intent": "cancel",
        "message": f"取消 {wrapped}",
        "booking_slots": {},
        "booking_slot_sources": {},
        "model_decision": {"extracted_slots": {}},
        "decision_errors": [],
        "ambiguities": [],
        "trace_events": [],
    }

    extract_booking_slots(state)

    assert state["booking_slots"]["booking_id"] == "booking_1234"


@pytest.mark.parametrize(
    "message",
    [
        "booking_1234 的取消和改约有什么区别",
        "booking_1234 如何取消或改约",
        "booking_1234 取消或改约的流程是什么",
        "booking_1234 取消或改约怎么操作",
        "booking_1234 能否办理取消或改约",
        "booking_1234 取消和改约怎样办理",
        "我想了解 booking_1234 取消和改约怎么操作",
        "请帮我了解 booking_1234 取消和改约有什么区别",
    ],
)
def test_rules_graph_routes_booking_action_knowledge_to_consultation(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_knowledge",
            "conversation_id": "conv_rules_knowledge",
            "message": message,
        }
    )

    assert result["intent"] == "consultation"
    assert result["confirmation_required"] is False
    assert not any(
        item["tool_name"] in {"create_booking", "cancel_booking", "reschedule_booking"}
        for item in result["tool_plan"]
    )


@pytest.mark.parametrize(
    "message",
    [
        "取消并改约到明天如何收费",
        "取消并新约顺便问政策",
        "我想取消预约并安排明天推拿，规则是什么",
        "取消 booking_1234 后预约明天推拿，如何收费",
        "取消 booking_1234 后改约到明天下午，如何收费",
        "取消 booking_1234、预约明天推拿如何收费",
    ],
)
def test_rules_graph_keeps_real_actions_before_question_tail_as_clarification(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_mixed",
            "conversation_id": "conv_rules_mixed",
            "message": message,
        }
    )

    assert result["intent"] == "clarification"
    assert result["confirmation_required"] is False
    assert not any(
        item["tool_name"] in {"create_booking", "cancel_booking", "reschedule_booking"}
        for item in result["tool_plan"]
    )


@pytest.mark.parametrize(
    "service",
    ["火星按摩", "量子养生推拿", "泰式按摩", "医疗推拿", "飞行肩颈按摩"],
)
def test_rules_graph_rejects_any_non_catalog_service_modifier(monkeypatch, service: str):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_invalid_service",
            "conversation_id": f"conv_rules_invalid_service_{service}",
            "message": f"预约明天下午{service}",
        }
    )

    assert "service_type" in result["ambiguities"]
    assert "service_type" not in result["booking_slots"]
    assert not any(item["tool_name"] == "create_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    "message",
    [
        "我需要医疗帮助，请给我医疗建议",
        "我想咨询退款政策，但这是医疗问题",
        "医疗按摩是否适合孕妇",
        "医疗推拿适合老人吗",
        "医疗按摩高血压可以做吗",
        "医疗推拿颈椎病能否使用",
        "预约医疗按摩，我有高血压可以吗",
    ],
)
def test_rules_graph_preserves_real_medical_guardrail(monkeypatch, message: str):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_medical_guardrail",
            "conversation_id": "conv_rules_medical_guardrail",
            "message": message,
        }
    )

    assert result["intent"] == "escalation"
    assert result["escalation"]["reason"] == "medical_concern"
    assert any(item["tool_name"] == "escalate_to_human" for item in result["tool_plan"])
    assert result["rag_used"] is False


@pytest.mark.parametrize(
    "message",
    [
        "预约医疗按摩，我有高血压",
        "帮我安排医疗推拿，我怀孕了",
        "预约医疗推拿治疗颈椎病",
        "预订医疗按摩用于术后恢复",
    ],
)
def test_rules_graph_medical_service_with_substantive_residue_escalates(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_medical_residue",
            "conversation_id": f"conv_rules_medical_residue_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "escalation"
    assert result["escalation"]["reason"] == "medical_concern"
    assert result["rag_used"] is False


@pytest.mark.parametrize(
    "message",
    [
        "预约明天下午医疗按摩60分钟",
        "请帮我预订后天上午医疗推拿",
    ],
)
def test_rules_graph_pure_medical_service_booking_reaches_service_validation(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_pure_medical_booking",
            "conversation_id": f"conv_rules_pure_medical_booking_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "booking"
    assert "service_type" in result["ambiguities"]
    assert result["escalated"] is False


def test_rules_graph_ignores_non_booking_cancel_action(monkeypatch):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_non_booking_cancel",
            "conversation_id": "conv_rules_non_booking_cancel",
            "message": "取消闹钟并预约明天推拿",
        }
    )

    assert result["intent"] == "booking"
    assert result["booking_slots"]["service_type"] == "推拿"


@pytest.mark.parametrize("message", ["把闹钟取消", "把订单取消"])
def test_rules_graph_does_not_treat_object_cancellation_as_booking_cancel(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_non_booking_object_cancel",
            "conversation_id": f"conv_rules_non_booking_object_cancel_{message}",
            "message": message,
        }
    )

    assert result["intent"] != "cancel"
    assert "booking_id" not in result["missing_slots"]
    assert not any(item["tool_name"] == "cancel_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    "message",
    [
        "把预约取消",
        "把这个预约取消",
        "把我的预约取消",
        "取消我的预约",
        "取消这个预约",
    ],
)
def test_rules_graph_recognizes_booking_cancel_word_order(monkeypatch, message: str):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_booking_cancel_order",
            "conversation_id": f"conv_rules_booking_cancel_order_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "cancel"
    assert result["missing_slots"] == ["booking_id"]
    assert not any(item["tool_name"] == "cancel_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    "message",
    [
        "把明天的预约取消",
        "把我的按摩预约取消",
        "把明天下午3点的推拿预约取消",
        "把本次预约取消",
        "取消刚才的预约",
    ],
)
def test_rules_graph_recognizes_modified_booking_cancellation(monkeypatch, message: str):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_modified_cancel",
            "conversation_id": f"conv_rules_modified_cancel_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "cancel"
    assert result["missing_slots"] == ["booking_id"]
    assert not any(item["tool_name"] == "create_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    ("message", "expected_service", "ambiguous"),
    [
        ("安排明天医疗肩颈按摩", None, True),
        ("安排明天火星按摩", None, True),
        ("安排明天推拿", "推拿", False),
        ("预订后天按摩", "按摩", False),
    ],
)
def test_rules_graph_reuses_create_markers_for_natural_service_extraction(
    monkeypatch, message: str, expected_service: str | None, ambiguous: bool
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_create_marker_service",
            "conversation_id": f"conv_rules_create_marker_service_{message}",
            "message": message,
        }
    )

    assert ("service_type" in result["ambiguities"]) is ambiguous
    if expected_service:
        assert result["booking_slots"]["service_type"] == expected_service
    else:
        assert "service_type" not in result["booking_slots"]


@pytest.mark.parametrize("message", ["订餐", "订阅", "订婚", "安排明天开会"])
def test_rules_graph_rejects_context_free_create_markers(monkeypatch, message: str):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_non_booking_create_marker",
            "conversation_id": f"conv_rules_non_booking_create_marker_{message}",
            "message": message,
        }
    )

    assert result["intent"] != "booking"
    assert not any(item["tool_name"] == "create_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    ("message", "expected_service", "ambiguous"),
    [
        ("订明天推拿", "推拿", False),
        ("预订明天推拿", "推拿", False),
        ("安排明天推拿", "推拿", False),
        ("安排火星按摩", None, True),
        ("预订火星按摩", None, True),
    ],
)
def test_rules_graph_requires_service_context_for_contextual_create_markers(
    monkeypatch, message: str, expected_service: str | None, ambiguous: bool
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_contextual_create_marker",
            "conversation_id": f"conv_rules_contextual_create_marker_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "booking"
    assert ("service_type" in result["ambiguities"]) is ambiguous
    if expected_service:
        assert result["booking_slots"]["service_type"] == expected_service


@pytest.mark.parametrize(
    "message",
    [
        "医疗按摩",
        "预约医疗按摩，我有慢性病",
        "预约医疗推拿用于术后恢复",
        "安排医疗按摩是否适合我",
        "预订医疗推拿能否使用",
    ],
)
def test_rules_graph_medical_service_health_structures_stay_on_guardrail(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_medical_health_structure",
            "conversation_id": f"conv_rules_medical_health_structure_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "escalation"
    assert result["escalation"]["reason"] == "medical_concern"
    assert result["rag_used"] is False


@pytest.mark.parametrize(
    "message",
    [
        "我想预约医疗按摩，谢谢",
        "我要预约医疗推拿，安静一点",
        "帮忙安排医疗按摩，不要大力",
    ],
)
def test_rules_graph_medical_service_politeness_and_preferences_reach_validation(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_medical_booking_preference",
            "conversation_id": f"conv_rules_medical_booking_preference_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "booking"
    assert "service_type" in result["ambiguities"]
    assert result["escalated"] is False


@pytest.mark.parametrize(
    "message",
    [
        "取消合约",
        "合同违约",
        "明天签约",
        "约束条件",
        "大约明天完成",
    ],
)
def test_rules_graph_rejects_non_appointment_yue_substrings(monkeypatch, message: str):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_non_appointment_yue",
            "conversation_id": f"conv_rules_non_appointment_yue_{message}",
            "message": message,
        }
    )

    assert result["intent"] != "booking"
    assert not any(item["tool_name"] == "create_booking" for item in result["tool_plan"])


def test_rules_graph_routes_approximate_price_question_to_consultation(monkeypatch):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_approximate_price",
            "conversation_id": "conv_rules_approximate_price",
            "message": "按摩大约多少钱",
        }
    )

    assert result["intent"] == "consultation"
    assert not any(item["tool_name"] == "create_booking" for item in result["tool_plan"])


@pytest.mark.parametrize("message", ["我想约推拿", "约明天下午3点"])
def test_rules_graph_accepts_yue_with_nearby_booking_payload(monkeypatch, message: str):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_contextual_yue",
            "conversation_id": f"conv_rules_contextual_yue_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "booking"


@pytest.mark.parametrize(
    "message",
    [
        "取消预约并改到明天",
        "取消 booking_1234 然后改期到后天",
    ],
)
def test_rules_graph_propagates_verified_booking_context_between_segments(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_propagated_booking_context",
            "conversation_id": f"conv_rules_propagated_booking_context_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "clarification"
    assert not any(
        item["tool_name"] in {"cancel_booking", "reschedule_booking"}
        for item in result["tool_plan"]
    )


def test_rules_graph_does_not_propagate_non_booking_cancel_context(monkeypatch):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_no_propagated_booking_context",
            "conversation_id": "conv_rules_no_propagated_booking_context",
            "message": "取消闹钟并改到明天",
        }
    )

    assert result["intent"] not in {"cancel", "reschedule", "clarification"}
    assert not any(
        item["tool_name"] in {"cancel_booking", "reschedule_booking"}
        for item in result["tool_plan"]
    )


@pytest.mark.parametrize(
    "message",
    [
        "预约医疗按摩，我高血压",
        "预约医疗按摩，我是孕妇",
        "预约医疗推拿，刚做完手术",
        "预约医疗推拿，因为有糖尿病",
    ],
)
def test_rules_graph_medical_service_any_substantive_residue_escalates(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_medical_any_residue",
            "conversation_id": f"conv_rules_medical_any_residue_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "escalation"
    assert result["escalation"]["reason"] == "medical_concern"
    assert result["rag_used"] is False


@pytest.mark.parametrize(
    "message",
    [
        "不要取消预约",
        "别取消预约",
        "无需取消预约",
        "我没有取消预约",
        "我并未取消预约",
        "为什么取消预约",
        "为何取消预约",
    ],
)
def test_rules_graph_cancel_negation_or_meta_question_never_writes(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_cancel_negation",
            "conversation_id": f"conv_rules_cancel_negation_{message}",
            "message": message,
        }
    )

    assert result["intent"] != "cancel"
    assert "booking_id" not in result["missing_slots"]
    assert not any(item["tool_name"] == "cancel_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    "message",
    [
        "想预约医疗按摩",
        "想约医疗按摩",
        "想预订医疗按摩",
    ],
)
def test_rules_graph_medical_service_standalone_intent_word_is_pure_booking(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_medical_standalone_intent",
            "conversation_id": f"conv_rules_medical_standalone_intent_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "booking"
    assert "service_type" in result["ambiguities"]
    assert result["escalated"] is False


def test_rules_graph_medical_service_intent_word_keeps_health_residue(monkeypatch):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_medical_intent_health",
            "conversation_id": "conv_rules_medical_intent_health",
            "message": "想预约医疗按摩，我有高血压",
        }
    )

    assert result["intent"] == "escalation"
    assert result["escalation"]["reason"] == "medical_concern"


@pytest.mark.parametrize(
    "message",
    [
        "我不想取消 booking_1234",
        "暂不取消 booking_1234",
        "先不取消 booking_1234",
        "暂时不取消 booking_1234",
        "目前不取消 booking_1234",
        "还不取消 booking_1234",
        "无需取消 booking_1234",
        "不用取消 booking_1234",
        "不必取消 booking_1234",
        "是否取消 booking_1234",
        "要不要取消 booking_1234",
        "需不需要取消 booking_1234",
        "取消 booking_1234 吗",
        "取消 booking_1234 是什么意思",
        "为什么取消 booking_1234",
    ],
)
def test_rules_graph_booking_id_cancel_negation_and_meta_never_write(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_booking_id_cancel_meta",
            "conversation_id": f"conv_rules_booking_id_cancel_meta_{message}",
            "message": message,
        }
    )

    assert result["intent"] != "cancel"
    assert not any(item["tool_name"] == "cancel_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    "message",
    ["条约明天生效", "约定明天开会", "大约明天完成", "合同违约", "签约明天完成"],
)
def test_rules_graph_yue_context_uses_grammar_not_character_blacklist(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_yue_grammar",
            "conversation_id": f"conv_rules_yue_grammar_{message}",
            "message": message,
        }
    )

    assert result["intent"] != "booking"
    assert not any(item["tool_name"] == "create_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    "message",
    [
        "我不会取消 booking_1234",
        "我不打算取消 booking_1234",
        "我不准备撤销 booking_1234",
        "我并非要取消 booking_1234",
        "我不是要取消 booking_1234",
        "取消 booking_1234 会有什么后果",
        "取消 booking_1234 的流程是什么",
        "取消 booking_1234 有什么费用和条件",
    ],
)
def test_rules_graph_cancel_span_requires_affirmative_execution(
    monkeypatch, message: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_cancel_span_quality",
            "conversation_id": f"conv_rules_cancel_span_quality_{message}",
            "message": message,
        }
    )

    assert result["intent"] != "cancel"
    assert not any(item["tool_name"] == "cancel_booking" for item in result["tool_plan"])


@pytest.mark.parametrize(
    "message",
    [
        "取消 booking_1234",
        "请帮我取消 booking_1234",
        "现在取消 booking_1234",
    ],
)
def test_rules_graph_cancel_span_keeps_affirmative_commands(monkeypatch, message: str):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_cancel_span_affirmative",
            "conversation_id": f"conv_rules_cancel_span_affirmative_{message}",
            "message": message,
        }
    )

    assert result["intent"] == "cancel"
    assert result["confirmation_required"] is True


def test_rules_graph_does_not_treat_non_booking_change_as_reschedule(monkeypatch):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_non_booking_change",
            "conversation_id": "conv_rules_non_booking_change",
            "message": "把闹钟改到明天",
        }
    )

    assert result["intent"] != "reschedule"
    assert not any(item["tool_name"] == "reschedule_booking" for item in result["tool_plan"])


def test_rules_graph_accepts_quoted_valid_booking_token(monkeypatch):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_quoted_booking",
            "conversation_id": "conv_rules_quoted_booking",
            "message": "取消『booking_1234』",
        }
    )

    assert result["intent"] == "cancel"
    assert result["booking_slots"]["booking_id"] == "booking_1234"
    assert result["confirmation_required"] is True


@pytest.mark.parametrize("token", ["booking_", "booking#"])
def test_rules_graph_treats_empty_booking_token_as_invalid_not_missing(
    monkeypatch, token: str
):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "rules")

    result = run_operations_turn(
        {
            "user_id": "user_rules_empty_booking_token",
            "conversation_id": f"conv_rules_empty_booking_token_{token}",
            "message": f"取消 {token}",
        }
    )

    assert any(
        error.get("field") == "booking_id"
        and error.get("code") == "invalid_booking_id"
        for error in result["decision_errors"]
    )
    assert not any(item["tool_name"] == "cancel_booking" for item in result["tool_plan"])
