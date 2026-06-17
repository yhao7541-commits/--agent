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


def test_read_tool_executes_without_confirmation_and_traces():
    gateway = ToolGateway(build_default_tool_registry())
    context = ToolExecutionContext(user_id="user_001", conversation_id="conv_001", trace_id="trace_001")

    result = gateway.execute("search_services", {"query": "肩颈"}, context)

    assert result.success is True
    assert result.output["services"]
    assert context.trace_events[-1]["event_type"] == "tool_executed"
    assert context.trace_events[-1]["metadata"]["tool_name"] == "search_services"
