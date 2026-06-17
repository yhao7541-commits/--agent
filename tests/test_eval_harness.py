from harness.evaluators.tool_accuracy import tool_arguments_passed
from harness.runners.run_all import load_cases, run_eval


def test_eval_harness_loads_yaml_cases():
    cases = load_cases()

    assert cases
    assert any(case["id"] == "booking_missing_date_001" for case in cases)
    assert any(case["id"] == "booking_cancel_001" for case in cases)
    assert any(case["id"] == "booking_reschedule_001" for case in cases)
    assert any(case["id"] == "security_prompt_injection_001" for case in cases)


def test_eval_harness_outputs_json_report_with_metrics():
    report = run_eval()

    assert report["case_count"] > 0
    assert report["metrics"]["intent_accuracy"] >= 0.85
    assert report["metrics"]["tool_selection_accuracy"] >= 0.85
    assert report["metrics"]["tool_argument_accuracy"] >= 0.85
    assert report["metrics"]["confirmation_compliance"] == 1.0
    assert report["metrics"]["security_policy_accuracy"] >= 0.90
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
