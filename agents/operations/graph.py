from langgraph.graph import END, StateGraph

from .nodes import (
    classify_intent,
    execute_tools,
    extract_booking_slots,
    finalize_turn,
    generate_response,
    initialize_turn,
    load_customer_context,
    plan_tool_calls,
)
from .state import OperationsAgentState


def build_operations_graph():
    workflow = StateGraph(OperationsAgentState)
    workflow.add_node("initialize_turn", initialize_turn)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("load_customer_context", load_customer_context)
    workflow.add_node("extract_booking_slots", extract_booking_slots)
    workflow.add_node("plan_tool_calls", plan_tool_calls)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("finalize_turn", finalize_turn)

    workflow.set_entry_point("initialize_turn")
    workflow.add_edge("initialize_turn", "classify_intent")
    workflow.add_edge("classify_intent", "load_customer_context")
    workflow.add_edge("load_customer_context", "extract_booking_slots")
    workflow.add_edge("extract_booking_slots", "plan_tool_calls")
    workflow.add_edge("plan_tool_calls", "execute_tools")
    workflow.add_edge("execute_tools", "generate_response")
    workflow.add_edge("generate_response", "finalize_turn")
    workflow.add_edge("finalize_turn", END)
    return workflow.compile()


def run_operations_turn(state: OperationsAgentState) -> OperationsAgentState:
    graph = build_operations_graph()
    return graph.invoke(state)
