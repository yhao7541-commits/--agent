from __future__ import annotations

import asyncio
from pathlib import Path
import time

from agents.operations import agent as agent_module
from agents.operations.agent import OperationsAgent


def test_operations_agent_builds_one_graph_per_instance(monkeypatch):
    built_graphs = []

    class FakeGraph:
        def __init__(self):
            self.invocations = []

        def invoke(self, state):
            self.invocations.append(state)
            return state

    def fake_build_operations_graph(decision_engine=None):
        graph = FakeGraph()
        built_graphs.append((decision_engine, graph))
        return graph

    monkeypatch.setattr(
        agent_module,
        "build_operations_graph",
        fake_build_operations_graph,
    )

    first = OperationsAgent()
    first.run_turn({"message": "first"})
    first.run_turn({"message": "second"})
    second = OperationsAgent()
    second.run_turn({"message": "third"})

    assert len(built_graphs) == 2
    assert len(built_graphs[0][1].invocations) == 2
    assert len(built_graphs[1][1].invocations) == 1


def test_arun_turn_allows_two_blocking_turns_to_overlap(monkeypatch):
    agent = OperationsAgent()

    def blocking_run_turn(state):
        time.sleep(0.15)
        return state

    monkeypatch.setattr(agent, "run_turn", blocking_run_turn)

    async def run_concurrently():
        started = time.perf_counter()
        results = await asyncio.gather(
            agent.arun_turn({"message": "first"}),
            agent.arun_turn({"message": "second"}),
        )
        return results, time.perf_counter() - started

    results, elapsed = asyncio.run(run_concurrently())

    assert [result["message"] for result in results] == ["first", "second"]
    assert elapsed < 0.27


def test_async_api_paths_do_not_call_sync_run_turn_directly():
    project_root = Path(__file__).resolve().parents[1]
    async_paths = (
        "api/operations.py",
        "api/appointment.py",
        "api/consultation.py",
        "api/task.py",
        "api/chat_handler.py",
    )

    direct_callers = [
        relative_path
        for relative_path in async_paths
        if ".run_turn(" in (project_root / relative_path).read_text(encoding="utf-8")
    ]

    assert direct_callers == []
