from .state import OperationsAgentState


def route_after_confirmation(state: OperationsAgentState) -> str:
    source = state.get("decision_source")
    if source == "confirmed_action":
        return "confirmed"
    if source == "confirmation_rejected":
        return "rejected"
    if source == "forced_escalation" or state.get("escalated"):
        return "escalated"
    return "ordinary"


def route_after_guardrail(state: OperationsAgentState) -> str:
    return "escalated" if state.get("escalated") else "decide"


def route_after_decision(state: OperationsAgentState) -> str:
    intent = state.get("intent")
    if state.get("escalated") or state.get("decision_source") == "forced_escalation":
        return "escalation"
    if intent == "unknown" or state.get("confidence", 1.0) < 0.5:
        return "escalation"
    if state.get("ambiguities"):
        return "clarification"
    if intent in {"booking", "reschedule", "cancel"}:
        return "booking"
    if intent == "consultation":
        return "consultation"
    if intent in {"memory", "delete_memory"}:
        return "memory"
    if intent == "greeting":
        return "greeting"
    if intent == "clarification":
        return "clarification"
    return "escalation"


def is_booking_intent(state: OperationsAgentState) -> bool:
    return state.get("intent") in {"booking", "reschedule", "cancel"}


def is_consultation_intent(state: OperationsAgentState) -> bool:
    return state.get("intent") == "consultation"


def needs_escalation(state: OperationsAgentState) -> bool:
    return bool(state.get("escalated")) or state.get("confidence", 1.0) < 0.5
