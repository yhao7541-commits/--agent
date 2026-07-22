from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.evaluators.decision_comparison import (
    compute_decision_metrics,
    estimate_usage_cost,
    nearest_rank_percentile,
    summarize_usage,
)
from harness.runners.run_decision_comparison import (
    DatasetValidationError,
    LiveModelUnavailable,
    ensure_model_attempts,
    load_dataset,
    provider_completion_summary,
    resolve_output_path,
    run_comparison,
    validate_dataset,
    _normalized_case_result,
    _run_case,
)
from harness.runners.run_all import load_cases as load_general_harness_cases
from scripts.demo_decision_resilience import run_demo
from tools.customer_tools import get_customer_memory_store, reset_customer_memory_store


DATASET_PATH = Path("harness/datasets/decision_long_tail_cases.yaml")


def _case(
    *,
    expected_intent: str = "booking",
    predicted_intent: str = "booking",
    expected_slots: dict[str, str] | None = None,
    predicted_slots: dict[str, str] | None = None,
    expected_clarification: bool = False,
    predicted_clarification: bool = False,
    source: str = "llm",
    attempt_count: int = 1,
    latency_ms: int = 10,
    valid_structured_output: bool = True,
    expect_model_call: bool = True,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict:
    return {
        "expected": {
            "intent": expected_intent,
            "slots": expected_slots or {},
            "requires_clarification": expected_clarification,
        },
        "prediction": {
            "intent": predicted_intent,
            "slots": predicted_slots or {},
            "requires_clarification": predicted_clarification,
            "valid_structured_output": valid_structured_output,
        },
        "decision_metadata": {
            "source": source,
            "attempt_count": attempt_count,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "expect_model_call": expect_model_call,
    }


def test_intent_accuracy_uses_exact_normalized_enum():
    metrics = compute_decision_metrics(
        [
            _case(expected_intent="booking", predicted_intent=" BOOKING "),
            _case(expected_intent="cancel", predicted_intent="reschedule"),
            _case(expected_intent="greeting", predicted_intent="hello"),
        ]
    )

    assert metrics["intent_accuracy"] == pytest.approx(1 / 3)


def test_slot_precision_recall_micro_average_field_value_pairs():
    metrics = compute_decision_metrics(
        [
            _case(
                expected_slots={"service_type": "肩颈放松", "time_window": "15:00"},
                predicted_slots={
                    "service_type": " 肩颈放松 ",
                    "time_window": "16:00",
                    "preferred_staff": "未指定字段应排除",
                },
            ),
            _case(
                expected_slots={"date": "2026-07-23"},
                predicted_slots={},
            ),
        ]
    )

    assert metrics["slot_precision"] == pytest.approx(1 / 3)
    assert metrics["slot_recall"] == pytest.approx(1 / 3)


def test_unexpected_predicted_slot_pair_is_a_false_positive():
    metrics = compute_decision_metrics(
        [
            _case(
                expected_slots={"service_type": "肩颈放松"},
                predicted_slots={
                    "service_type": "肩颈放松",
                    "preferred_staff": "张伟",
                },
            )
        ]
    )

    assert metrics["slot_precision"] == 0.5
    assert metrics["slot_recall"] == 1.0


def test_ambiguity_accuracy_uses_expected_boolean():
    metrics = compute_decision_metrics(
        [
            _case(expected_clarification=True, predicted_clarification=True),
            _case(expected_clarification=False, predicted_clarification=True),
        ]
    )

    assert metrics["ambiguity_accuracy"] == 0.5


def test_fallback_rate_uses_all_hybrid_cases_not_only_model_required_cases():
    metrics = compute_decision_metrics(
        [
            _case(source="rule_fallback", expect_model_call=True),
            _case(source="forced_escalation", attempt_count=0, expect_model_call=False),
        ]
    )

    assert metrics["fallback_rate"] == 0.5


def test_nearest_rank_percentiles():
    values = [1, 2, 3, 4, 100]

    assert nearest_rank_percentile(values, 50) == 3
    assert nearest_rank_percentile(values, 95) == 100
    assert nearest_rank_percentile([], 95) == 0


def test_unavailable_usage_is_reported_not_invented():
    usage = summarize_usage([_case(input_tokens=None, output_tokens=None)])

    assert usage == {
        "available": False,
        "reason": "provider_usage_unavailable",
        "case_count": 1,
        "input_tokens": None,
        "output_tokens": None,
        "average_input_tokens": None,
        "average_output_tokens": None,
    }


def test_usage_summary_excludes_cases_that_should_not_call_the_model():
    usage = summarize_usage(
        [
            _case(input_tokens=100, output_tokens=50, expect_model_call=True),
            _case(input_tokens=None, output_tokens=None, expect_model_call=False),
        ]
    )

    assert usage["available"] is True
    assert usage["case_count"] == 1
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50


def test_live_runner_rejects_missing_provider_configuration_before_cases(tmp_path):
    calls: list[str] = []

    def unavailable():
        raise LiveModelUnavailable("provider configuration unavailable")

    def must_not_run(*_args, **_kwargs):
        calls.append("case")
        pytest.fail("cases must not run before provider preflight succeeds")

    with pytest.raises(LiveModelUnavailable):
        run_comparison(
            DATASET_PATH,
            output=tmp_path / "report.json",
            preflight_fn=unavailable,
            case_runner=must_not_run,
        )

    assert calls == []


def test_live_runner_reports_all_fallback_and_low_provider_success_as_bad_cases():
    summary = provider_completion_summary(
        [
            _case(source="rule_fallback", attempt_count=3),
            _case(source="rule_fallback", attempt_count=3),
        ]
    )

    assert summary["provider_success_rate"] == 0.0
    assert summary["fallback_rate"] == 1.0
    assert summary["bad_case_evidence"] is True
    assert summary["uplift_claim_allowed"] is False


def test_provider_success_counts_valid_model_output_before_forced_escalation():
    forced = _case(source="forced_escalation", attempt_count=1)

    summary = provider_completion_summary([forced])

    assert summary["provider_success_count"] == 1
    assert summary["provider_success_rate"] == 1.0
    assert summary["fallback_count"] == 0


def test_explicit_rule_fallback_is_not_provider_success():
    fallback = _case(source="rule_fallback", attempt_count=1)

    summary = provider_completion_summary([fallback])

    assert summary["provider_success_count"] == 0
    assert summary["fallback_count"] == 1


def test_model_required_cases_must_record_provider_attempt():
    with pytest.raises(ValueError, match="attempt_count"):
        ensure_model_attempts([_case(attempt_count=0, expect_model_call=True)])

    ensure_model_attempts([_case(attempt_count=0, expect_model_call=False)])


def test_default_output_is_timestamped_and_explicit_output_is_not_overwritten(tmp_path):
    now = datetime(2026, 7, 22, 12, 34, 56, tzinfo=timezone.utc)
    default_path = resolve_output_path(None, now=now)
    explicit = tmp_path / "fixed.json"
    explicit.write_text("existing", encoding="utf-8")

    assert default_path.as_posix().endswith(
        "data/evaluation/decision-comparison-20260722T123456Z.json"
    )
    with pytest.raises(FileExistsError):
        resolve_output_path(explicit, now=now)
    assert resolve_output_path(explicit, now=now, force=True) == explicit


def test_existing_output_is_rejected_before_preflight_or_case_execution(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("do not overwrite", encoding="utf-8")
    calls: list[str] = []

    def preflight():
        calls.append("preflight")
        return {}

    def case_runner(*_args):
        calls.append("case")
        return {}

    with pytest.raises(FileExistsError):
        run_comparison(
            DATASET_PATH,
            output=output,
            preflight_fn=preflight,
            case_runner=case_runner,
        )

    assert calls == []
    assert output.read_text(encoding="utf-8") == "do not overwrite"


def test_written_predictions_are_normalized_and_include_the_output_path(tmp_path):
    output = tmp_path / "normalized.json"

    def case_runner(case, _mode, _namespace):
        return {
            "id": case["id"],
            "family": case["family"],
            "expect_model_call": case["expect_model_call"],
            "expected": case["expected"],
            "prediction": {
                "intent": " BOOKING ",
                "slots": {
                    "service_type": " 肩颈   放松 ",
                    "preferred_staff": " ",
                    "date": None,
                },
                "requires_clarification": False,
                "valid_structured_output": True,
            },
            "decision_metadata": {
                "source": "llm",
                "attempt_count": 1,
                "latency_ms": 1,
                "input_tokens": 1,
                "output_tokens": 1,
            },
        }

    returned = run_comparison(
        DATASET_PATH,
        output=output,
        preflight_fn=lambda: {"provider": "fake", "model": "fake"},
        case_runner=case_runner,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    prediction = persisted["modes"]["hybrid"]["cases"][0]["prediction"]

    assert prediction["intent"] == "booking"
    assert prediction["slots"] == {"service_type": "肩颈 放松"}
    assert persisted["output_path"] == returned["output_path"] == str(output)


def test_cost_requires_usage_and_both_pricing_inputs():
    usage = summarize_usage([_case(input_tokens=1_000, output_tokens=500)])

    assert estimate_usage_cost(usage, input_cost_per_million=None, output_cost_per_million=2) == {
        "available": False,
        "reason": "pricing_unavailable",
        "estimated_cost": None,
    }
    assert estimate_usage_cost(usage, input_cost_per_million=1, output_cost_per_million=2) == {
        "available": True,
        "reason": None,
        "estimated_cost": 0.002,
    }
    unavailable = summarize_usage([_case(input_tokens=None, output_tokens=500)])
    assert estimate_usage_cost(
        unavailable, input_cost_per_million=1, output_cost_per_million=2
    )["reason"] == "provider_usage_unavailable"


def test_dataset_is_versioned_isolated_and_covers_approved_families():
    dataset = load_dataset(DATASET_PATH)

    validate_dataset(dataset)
    cases = dataset["cases"]
    assert 28 <= len(cases) <= 35
    assert len({case["id"] for case in cases}) == len(cases)
    assert dataset["version"]
    assert dataset["date_anchor"]
    assert dataset["timezone"] == "Asia/Shanghai"
    assert all("intent" in case["expected"] for case in cases)
    assert all("requires_clarification" in case["expected"] for case in cases)
    assert all(isinstance(case["expect_model_call"], bool) for case in cases)
    assert all("prior_turns" in case for case in cases)
    assert {
        "colloquial_typo",
        "correction",
        "vague_time",
        "contradictory_slots",
        "negation",
        "combined_intent",
        "safety_recognition",
    } <= {case["family"] for case in cases}


def test_dataset_validation_rejects_secret_shaped_content():
    dataset = {
        "version": "v1",
        "date_anchor": "2026-07-22",
        "timezone": "Asia/Shanghai",
        "cases": [
            {
                "id": "bad-secret",
                "family": "colloquial_typo",
                "message": "sk-test-secret-value-that-must-not-be-stored",
                "prior_turns": [],
                "expect_model_call": True,
                "expected": {
                    "intent": "greeting",
                    "requires_clarification": False,
                    "slots": {},
                },
            }
        ],
    }

    with pytest.raises(DatasetValidationError, match="secret"):
        validate_dataset(dataset)


def test_general_harness_ignores_mapping_dataset_and_keeps_list_datasets(tmp_path):
    (tmp_path / "legacy.yaml").write_text(
        "- id: legacy_001\n  turns:\n    - user: hello\n",
        encoding="utf-8",
    )
    (tmp_path / "comparison.yaml").write_text(
        "version: v1\ndate_anchor: '2026-07-22'\ncases: []\n",
        encoding="utf-8",
    )

    assert load_general_harness_cases(tmp_path) == [
        {"id": "legacy_001", "turns": [{"user": "hello"}]}
    ]


def test_resilience_demo_recovers_and_never_performs_an_unsafe_write(monkeypatch):
    monkeypatch.setenv("OPERATIONS_DECISION_MODE", "hybrid")
    result = run_demo(print_output=False)

    assert result["repair_success_count"] == 1
    assert result["retry_count"] >= 2
    assert result["fallback_count"] == 1
    assert result["confirmation_compliance_count"] == 2
    assert result["unsafe_write_count"] == 0


def test_forced_escalation_preserves_valid_provider_output_for_metrics():
    model_decision = {
        "intent": "booking",
        "confidence": 0.95,
        "extracted_slots": {},
        "ambiguities": [],
        "risk_flags": ["medical"],
        "suggested_action": "escalate",
        "decision_summary": "requires deterministic safety escalation",
    }
    normalized = _normalized_case_result(
        {
            "id": "forced-valid",
            "family": "safety_recognition",
            "expect_model_call": True,
            "expected": {
                "intent": "escalation",
                "requires_clarification": False,
                "slots": {},
            },
        },
        {
            "intent": "escalation",
            "decision_route": "escalation",
            "model_decision": model_decision,
            "decision_metadata": {
                "source": "forced_escalation",
                "attempt_count": 1,
                "fallback_reason": None,
            },
        },
    )

    assert normalized["prediction"]["valid_structured_output"] is True


def test_real_case_runner_closes_temp_database_and_restores_store(tmp_path, monkeypatch):
    reset_customer_memory_store(":memory:")
    original_store = get_customer_memory_store()
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    result = _run_case(
        {
            "id": "real-rules-case",
            "family": "colloquial_typo",
            "message": "你好",
            "prior_turns": [],
            "expect_model_call": False,
            "expected": {
                "intent": "greeting",
                "requires_clarification": False,
                "slots": {},
            },
        },
        "rules",
        "eval-real-cleanup",
    )

    assert result["id"] == "real-rules-case"
    assert get_customer_memory_store() is original_store
    assert list(tmp_path.iterdir()) == []
