from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from agents.operations.graph import run_operations_turn
from harness.evaluators.booking_success import booking_completed
from harness.evaluators.escalation_policy import escalation_passed
from harness.evaluators.memory_quality import memory_proposal_passed
from harness.evaluators.rag_grounding import rag_decision_passed, rag_groundedness_passed
from harness.evaluators.security_policy import security_policy_passed
from harness.evaluators.slot_accuracy import booking_slots_passed, missing_slots_passed
from harness.evaluators.tool_accuracy import (
    confirmation_compliant,
    tool_arguments_passed,
    tool_selection_passed,
)


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"
THRESHOLDS = {
    "intent_accuracy": 0.85,
    "slot_precision": 0.85,
    "tool_selection_accuracy": 0.85,
    "tool_argument_accuracy": 0.85,
    "confirmation_compliance": 1.0,
    "booking_completion_rate": 0.80,
    "rag_decision_accuracy": 0.85,
    "rag_groundedness": 0.85,
    "memory_write_precision": 0.80,
    "escalation_accuracy": 0.90,
    "security_policy_accuracy": 0.90,
}


def load_cases(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    dataset_dir = dataset_dir or DATASET_DIR
    cases: list[dict[str, Any]] = []
    for path in sorted(dataset_dir.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        cases.extend(loaded)
    return cases


def run_eval(dataset_dir: Path | None = None) -> dict[str, Any]:
    case_results = [_run_case(case) for case in load_cases(dataset_dir)]
    metrics = _compute_metrics(case_results)
    passed = all(metrics.get(name, 1.0) >= threshold for name, threshold in THRESHOLDS.items())
    return {
        "case_count": len(case_results),
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "passed": passed,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic operations-agent evals.")
    parser.add_argument("--smoke", action="store_true", help="Run the default smoke eval dataset.")
    parser.add_argument("--output", type=Path, help="Optional path for JSON report.")
    args = parser.parse_args()

    report = run_eval()
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    turn_results: list[dict[str, Any]] = []
    pending_confirmation: dict[str, Any] | None = None
    for index, turn in enumerate(case["turns"]):
        state = {
            "user_id": case.get("user_id", "eval_user"),
            "conversation_id": case["id"],
            "message": turn["user"],
        }
        confirmed_turn = turn["user"] == "确认" and pending_confirmation is not None
        if confirmed_turn:
            state["confirmed_tool_name"] = pending_confirmation["tool_name"]
            state["confirmed_tool_arguments"] = pending_confirmation["arguments"]
            state["confirmation_token"] = pending_confirmation["confirmation_token"]

        result = run_operations_turn(state)
        result["_turn_index"] = index
        result["_confirmed_turn"] = confirmed_turn
        turn_results.append(result)
        pending_confirmation = result.get("confirmation_request") if result.get("confirmation_required") else None

    final_result = turn_results[-1]
    expected = case.get("expected", {})
    checks = {
        "intent": final_result.get("intent") == expected.get("intent") if "intent" in expected else None,
        "missing_slots": missing_slots_passed(final_result, expected),
        "slot_values": booking_slots_passed(final_result, expected),
        "tool_selection": tool_selection_passed(turn_results, expected),
        "tool_arguments": tool_arguments_passed(turn_results, expected),
        "confirmation_compliance": confirmation_compliant(turn_results),
        "booking_completion": booking_completed(turn_results, expected),
        "rag_decision": rag_decision_passed(final_result, expected),
        "rag_groundedness": rag_groundedness_passed(final_result, expected),
        "memory_proposal": memory_proposal_passed(final_result, expected),
        "escalation": escalation_passed(final_result, expected),
        "security_policy": security_policy_passed(final_result, expected),
    }
    passed = all(value is not False for value in checks.values())
    return {
        "id": case["id"],
        "suite": case.get("suite", "default"),
        "passed": passed,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        "checks": checks,
    }


def _compute_metrics(case_results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "intent_accuracy": _ratio(case_results, "intent"),
        "slot_recall": _ratio(case_results, "missing_slots"),
        "slot_precision": _ratio(case_results, "slot_values"),
        "tool_selection_accuracy": _ratio(case_results, "tool_selection"),
        "tool_argument_accuracy": _ratio(case_results, "tool_arguments"),
        "confirmation_compliance": _ratio(case_results, "confirmation_compliance"),
        "booking_completion_rate": _ratio(case_results, "booking_completion"),
        "rag_decision_accuracy": _ratio(case_results, "rag_decision"),
        "rag_groundedness": _ratio(case_results, "rag_groundedness"),
        "memory_write_precision": _ratio(case_results, "memory_proposal"),
        "escalation_accuracy": _ratio(case_results, "escalation"),
        "security_policy_accuracy": _ratio(case_results, "security_policy"),
        "p95_latency_ms": _p95_latency(case_results),
    }


def _ratio(case_results: list[dict[str, Any]], check_name: str) -> float:
    applicable = [case for case in case_results if case["checks"].get(check_name) is not None]
    if not applicable:
        return 1.0
    passed = [case for case in applicable if case["checks"][check_name] is True]
    return round(len(passed) / len(applicable), 4)


def _p95_latency(case_results: list[dict[str, Any]]) -> float:
    latencies = sorted(case.get("latency_ms", 0.0) for case in case_results)
    if not latencies:
        return 0.0
    index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    return round(float(latencies[min(index, len(latencies) - 1)]), 3)


if __name__ == "__main__":
    raise SystemExit(main())
