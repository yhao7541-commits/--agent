from __future__ import annotations

from .base import ToolDefinition, ToolPermission
from .booking_tools import (
    cancel_booking,
    check_schedule,
    create_booking,
    find_available_staff,
    reschedule_booking,
    search_services,
)
from .customer_tools import lookup_customer_profile, write_customer_preference
from .escalation_tools import escalate_to_human
from .knowledge_tools import search_knowledge_base
from .schemas import (
    BookingOutput,
    CheckScheduleInput,
    CheckScheduleOutput,
    CreateBookingInput,
    CustomerPreferenceInput,
    CustomerPreferenceOutput,
    FindAvailableStaffInput,
    FindAvailableStaffOutput,
    HumanEscalationInput,
    HumanEscalationOutput,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
    LookupCustomerProfileInput,
    LookupCustomerProfileOutput,
    SearchServicesInput,
    SearchServicesOutput,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="search_services",
            description="Search service offerings by name or need.",
            permission=ToolPermission.READ,
            requires_confirmation=False,
            input_schema=SearchServicesInput,
            output_schema=SearchServicesOutput,
            handler=search_services,
        )
    )
    registry.register(
        ToolDefinition(
            name="check_schedule",
            description="Check whether a requested service time is available.",
            permission=ToolPermission.READ,
            requires_confirmation=False,
            input_schema=CheckScheduleInput,
            output_schema=CheckScheduleOutput,
            handler=check_schedule,
        )
    )
    registry.register(
        ToolDefinition(
            name="find_available_staff",
            description="Find available staff for a service and time.",
            permission=ToolPermission.READ,
            requires_confirmation=False,
            input_schema=FindAvailableStaffInput,
            output_schema=FindAvailableStaffOutput,
            handler=find_available_staff,
        )
    )
    registry.register(
        ToolDefinition(
            name="lookup_customer_profile",
            description="Load customer context and known preferences.",
            permission=ToolPermission.READ,
            requires_confirmation=False,
            input_schema=LookupCustomerProfileInput,
            output_schema=LookupCustomerProfileOutput,
            handler=lookup_customer_profile,
        )
    )
    for name, handler, status_description in (
        ("create_booking", create_booking, "Create a confirmed booking."),
        ("reschedule_booking", reschedule_booking, "Reschedule an existing booking."),
        ("cancel_booking", cancel_booking, "Cancel an existing booking."),
    ):
        registry.register(
            ToolDefinition(
                name=name,
                description=status_description,
                permission=ToolPermission.WRITE,
                requires_confirmation=True,
                input_schema=CreateBookingInput,
                output_schema=BookingOutput,
                handler=handler,
            )
        )
    registry.register(
        ToolDefinition(
            name="write_customer_preference",
            description="Store a long-term customer preference.",
            permission=ToolPermission.SENSITIVE,
            requires_confirmation=True,
            input_schema=CustomerPreferenceInput,
            output_schema=CustomerPreferenceOutput,
            handler=write_customer_preference,
        )
    )
    registry.register(
        ToolDefinition(
            name="search_knowledge_base",
            description="Search service and policy knowledge.",
            permission=ToolPermission.READ,
            requires_confirmation=False,
            input_schema=KnowledgeSearchInput,
            output_schema=KnowledgeSearchOutput,
            handler=search_knowledge_base,
        )
    )
    registry.register(
        ToolDefinition(
            name="escalate_to_human",
            description="Create a structured human handoff summary.",
            permission=ToolPermission.EXTERNAL,
            requires_confirmation=False,
            input_schema=HumanEscalationInput,
            output_schema=HumanEscalationOutput,
            handler=escalate_to_human,
        )
    )
    return registry
