from __future__ import annotations

from .graph import run_operations_turn
from .state import OperationsAgentState


class OperationsAgent:
    """Single operations agent facade for LangGraph orchestration."""

    def run_turn(self, state: OperationsAgentState) -> OperationsAgentState:
        return run_operations_turn(state)
