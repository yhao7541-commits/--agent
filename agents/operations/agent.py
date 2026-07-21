from __future__ import annotations

from starlette.concurrency import run_in_threadpool

from .decision_engine import HybridDecisionEngine
from .graph import build_operations_graph
from .nodes import clear_confirmation_inputs
from .state import OperationsAgentState


class OperationsAgent:
    """Single operations agent facade for LangGraph orchestration."""

    def __init__(self, decision_engine: HybridDecisionEngine | None = None) -> None:
        self._decision_engine = decision_engine
        self._graph = build_operations_graph(decision_engine=decision_engine)

    def run_turn(self, state: OperationsAgentState) -> OperationsAgentState:
        return clear_confirmation_inputs(self._graph.invoke(state))

    async def arun_turn(self, state: OperationsAgentState) -> OperationsAgentState:
        return await run_in_threadpool(self.run_turn, state)
