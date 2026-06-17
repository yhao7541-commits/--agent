from .state import OperationsAgentState


def is_booking_intent(state: OperationsAgentState) -> bool:
    return state.get("intent") in {"booking", "reschedule", "cancel"}


def is_consultation_intent(state: OperationsAgentState) -> bool:
    return state.get("intent") == "consultation"


def needs_escalation(state: OperationsAgentState) -> bool:
    return bool(state.get("escalated")) or state.get("confidence", 1.0) < 0.5
