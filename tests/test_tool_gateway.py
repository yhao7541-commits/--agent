from pydantic import BaseModel

from tools.base import ToolDefinition, ToolPermission
from tools.gateway import ToolGateway
from tools.registry import ToolRegistry, build_default_tool_registry
from tools.schemas import ToolExecutionContext


class CustomInput(BaseModel):
    value: str


class CustomOutput(BaseModel):
    accepted: bool


def test_unknown_tool_returns_controlled_error():
    gateway = ToolGateway(build_default_tool_registry())
    context = ToolExecutionContext(user_id="user_001", conversation_id="conv_001", trace_id="trace_001")

    result = gateway.execute("missing_tool", {}, context)

    assert result.success is False
    assert result.error["code"] == "unknown_tool"
    assert context.trace_events[-1]["event_type"] == "tool_error"


def test_invalid_arguments_do_not_call_handler():
    calls = []

    def handler(arguments, context):
        calls.append(arguments)
        return {"accepted": True}

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="custom_tool",
            description="Custom test tool",
            permission=ToolPermission.READ,
            requires_confirmation=False,
            input_schema=CustomInput,
            output_schema=CustomOutput,
            handler=handler,
        )
    )
    gateway = ToolGateway(registry)
    context = ToolExecutionContext(user_id="user_001", conversation_id="conv_001", trace_id="trace_001")

    result = gateway.execute("custom_tool", {"wrong": "field"}, context)

    assert result.success is False
    assert result.error["code"] == "validation_error"
    assert calls == []


def test_unconfirmed_create_booking_is_rejected():
    gateway = ToolGateway(build_default_tool_registry())
    context = ToolExecutionContext(user_id="user_001", conversation_id="conv_001", trace_id="trace_001")

    result = gateway.execute(
        "create_booking",
        {
            "service_type": "肩颈放松",
            "date": "2026-06-18",
            "time_window": "15:00",
            "customer_name": "user_001",
        },
        context,
    )

    assert result.success is False
    assert result.confirmation_required is True
    assert result.error["code"] == "confirmation_required"


def test_unconfirmed_customer_preference_write_is_rejected():
    gateway = ToolGateway(build_default_tool_registry())
    context = ToolExecutionContext(user_id="user_001", conversation_id="conv_001", trace_id="trace_001")

    result = gateway.execute(
        "write_customer_preference",
        {
            "user_id": "user_001",
            "preference_type": "preference",
            "preference_value": "喜欢安静房间",
            "evidence": "我以后都喜欢安静一点的房间",
        },
        context,
    )

    assert result.success is False
    assert result.confirmation_required is True
    assert result.error["code"] == "confirmation_required"


def test_confirmed_customer_preference_write_executes():
    gateway = ToolGateway(build_default_tool_registry())
    context = ToolExecutionContext(
        user_id="user_001",
        conversation_id="conv_001",
        trace_id="trace_001",
        confirmed_tools={"write_customer_preference"},
    )

    result = gateway.execute(
        "write_customer_preference",
        {
            "user_id": "user_001",
            "preference_type": "preference",
            "preference_value": "喜欢安静房间",
            "evidence": "我以后都喜欢安静一点的房间",
        },
        context,
    )

    assert result.success is True
    assert result.output["status"] == "stored"


def test_read_tool_executes_without_confirmation_and_traces():
    gateway = ToolGateway(build_default_tool_registry())
    context = ToolExecutionContext(user_id="user_001", conversation_id="conv_001", trace_id="trace_001")

    result = gateway.execute("search_services", {"query": "肩颈"}, context)

    assert result.success is True
    assert result.output["services"]
    assert context.trace_events[-1]["event_type"] == "tool_executed"
    assert context.trace_events[-1]["metadata"]["tool_name"] == "search_services"


def test_default_registry_contains_staff_and_customer_read_tools():
    registry = build_default_tool_registry()

    assert registry.get("find_available_staff") is not None
    assert registry.get("lookup_customer_profile") is not None


def test_staff_and_customer_read_tools_execute_without_confirmation():
    gateway = ToolGateway(build_default_tool_registry())
    context = ToolExecutionContext(user_id="user_001", conversation_id="conv_001", trace_id="trace_001")

    staff_result = gateway.execute(
        "find_available_staff",
        {"service_type": "肩颈放松", "date": "2026-06-18", "time_window": "15:00"},
        context,
    )
    profile_result = gateway.execute("lookup_customer_profile", {"user_id": "user_001"}, context)

    assert staff_result.success is True
    assert staff_result.output["staff"]
    assert profile_result.success is True
    assert profile_result.output["user_id"] == "user_001"
