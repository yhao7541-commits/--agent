from functools import partial

from langgraph.graph import END, StateGraph

from .decision_engine import HybridDecisionEngine
from .nodes import (
    clear_confirmation_inputs,
    decide_request,
    execute_tools,
    extract_booking_slots,
    finalize_turn,
    generate_response,
    initialize_turn,
    input_guardrail,
    load_customer_context,
    output_policy_check,
    plan_tool_calls,
    propose_memory_writes,
    validate_confirmation,
)
from .routers import (
    route_after_confirmation,
    route_after_decision,
    route_after_guardrail,
)
from .state import OperationsAgentState


def build_operations_graph(decision_engine: HybridDecisionEngine | None = None):
    workflow = StateGraph(OperationsAgentState)
    workflow.add_node("initialize_turn", initialize_turn)
    workflow.add_node("validate_confirmation", validate_confirmation)
    workflow.add_node("input_guardrail", input_guardrail)
    workflow.add_node("load_customer_context", load_customer_context)
    workflow.add_node(
        "decide_request",
        partial(decide_request, decision_engine=decision_engine),
    )
    workflow.add_node("extract_booking_slots", extract_booking_slots)
    workflow.add_node("propose_memory_writes", propose_memory_writes)
    workflow.add_node("plan_tool_calls", plan_tool_calls)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("output_policy_check", output_policy_check)
    workflow.add_node("finalize_turn", finalize_turn)

    workflow.set_entry_point("initialize_turn")
    workflow.add_edge("initialize_turn", "validate_confirmation")
    workflow.add_conditional_edges(
        "validate_confirmation",
        route_after_confirmation,
        {
            "confirmed": "plan_tool_calls",
            "rejected": "generate_response",
            "escalated": "plan_tool_calls",
            "ordinary": "input_guardrail",
        },
    )
    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {
            "escalated": "plan_tool_calls",
            "decide": "load_customer_context",
        },
    )
    workflow.add_edge("load_customer_context", "decide_request")
    workflow.add_conditional_edges(
        "decide_request",
        route_after_decision,
        {
            "booking": "extract_booking_slots",
            "consultation": "plan_tool_calls",
            "memory": "propose_memory_writes",
            "greeting": "generate_response",
            "clarification": "generate_response",
            "escalation": "plan_tool_calls",
        },
    )
    workflow.add_edge("extract_booking_slots", "plan_tool_calls")
    workflow.add_edge("propose_memory_writes", "plan_tool_calls")
    workflow.add_edge("plan_tool_calls", "execute_tools")
    workflow.add_edge("execute_tools", "generate_response")
    workflow.add_edge("generate_response", "output_policy_check")
    workflow.add_edge("output_policy_check", "finalize_turn")
    workflow.add_edge("finalize_turn", END)
    return workflow.compile()


def run_operations_turn(
    state: OperationsAgentState,
    decision_engine: HybridDecisionEngine | None = None,
) -> OperationsAgentState:
    graph = build_operations_graph(decision_engine=decision_engine)
    return clear_confirmation_inputs(graph.invoke(state))
