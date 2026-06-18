from harness.evaluators.memory_quality import no_memory_proposal_passed
from harness.evaluators.rag_grounding import rag_decision_passed, rag_groundedness_passed
from harness.evaluators.slot_accuracy import booking_slots_passed
from harness.evaluators.tool_accuracy import tool_arguments_passed
from harness.runners.run_all import load_cases, run_eval


def test_eval_harness_loads_yaml_cases():
    cases = load_cases()
    case_ids = {case["id"] for case in cases}

    assert cases
    assert len(cases) >= 30
    assert "booking_missing_date_001" in case_ids
    assert "booking_cancel_001" in case_ids
    assert "booking_reschedule_001" in case_ids
    assert "booking_reject_001" in case_ids
    assert "booking_morning_001" in case_ids
    assert "booking_evening_001" in case_ids
    assert "booking_fuzzy_time_001" in case_ids
    assert "booking_staff_unavailable_001" in case_ids
    assert "booking_time_conflict_001" in case_ids
    assert "booking_cancel_missing_id_001" in case_ids
    assert "booking_reschedule_missing_id_001" in case_ids
    assert "memory_sensitive_001" in case_ids
    assert "memory_negative_strength_001" in case_ids
    assert "memory_no_marketing_001" in case_ids
    assert "memory_vague_001" in case_ids
    assert "rag_pricing_001" in case_ids
    assert "rag_services_001" in case_ids
    assert "rag_suitability_001" in case_ids
    assert "safety_refund_001" in case_ids
    assert "safety_doctor_001" in case_ids
    assert "security_tool_bypass_001" in case_ids
    assert "security_system_prompt_001" in case_ids
    assert "security_prompt_injection_001" in case_ids


def test_eval_harness_outputs_json_report_with_metrics():
    report = run_eval()

    assert report["case_count"] > 0
    assert report["metrics"]["intent_accuracy"] >= 0.85
    assert report["metrics"]["slot_precision"] >= 0.85
    assert report["metrics"]["tool_selection_accuracy"] >= 0.85
    assert report["metrics"]["tool_argument_accuracy"] >= 0.85
    assert report["metrics"]["confirmation_compliance"] == 1.0
    assert report["metrics"]["rag_decision_accuracy"] >= 0.85
    assert report["metrics"]["rag_groundedness"] >= 0.85
    assert report["metrics"]["memory_suppression_accuracy"] >= 0.90
    assert report["metrics"]["p95_latency_ms"] >= 0
    assert report["metrics"]["security_policy_accuracy"] >= 0.90
    assert all(case["latency_ms"] >= 0 for case in report["cases"])
    assert all(case["passed"] for case in report["cases"])
    assert report["passed"] is True


def test_tool_argument_eval_passes_when_any_planned_call_matches():
    turn_results = [
        {
            "tool_plan": [
                {"tool_name": "reschedule_booking", "arguments": {"booking_id": "old"}},
                {
                    "tool_name": "reschedule_booking",
                    "arguments": {"booking_id": "booking_5678", "new_time_window": "15:00"},
                },
            ]
        }
    ]

    assert tool_arguments_passed(
        turn_results,
        {
            "tool_arguments": {
                "reschedule_booking": {
                    "booking_id": "booking_5678",
                    "new_time_window": "15:00",
                }
            }
        },
    )


def test_booking_slot_eval_checks_declared_slot_values_only():
    result = {
        "booking_slots": {
            "service_type": "肩颈放松",
            "time_window": "15:00",
            "customer_name": "eval_user",
        }
    }

    assert booking_slots_passed(
        result,
        {"booking_slots": {"service_type": "肩颈放松", "time_window": "15:00"}},
    )
    assert not booking_slots_passed(result, {"booking_slots": {"time_window": "17:00"}})


def test_rag_eval_separates_decision_from_groundedness():
    result = {"rag_used": True, "retrieved_knowledge": []}

    assert rag_decision_passed(result, {"rag_used": True, "grounded": True})
    assert not rag_groundedness_passed(result, {"grounded": True})


def test_memory_suppression_eval_requires_no_proposals():
    assert no_memory_proposal_passed({"memory_proposals": []}, {"no_memory_proposal": True})
    assert not no_memory_proposal_passed(
        {"memory_proposals": [{"type": "preference"}]},
        {"no_memory_proposal": True},
    )
