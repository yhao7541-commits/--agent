"""Pure, JSON-serializable metrics for decision-layer comparisons."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from agents.operations.decision_models import DecisionIntent


_INTENTS = set(DecisionIntent.__args__)


def compute_decision_metrics(cases: list[dict[str, Any]]) -> dict[str, float]:
    """Compute exact semantic metrics over normalized per-case dictionaries."""
    intent_matches = 0
    ambiguity_matches = 0
    slot_true_positive = 0
    slot_false_positive = 0
    slot_false_negative = 0

    for case in cases:
        expected = case.get("expected", {})
        prediction = case.get("prediction", {})
        expected_intent = normalize_intent(expected.get("intent"))
        intent_matches += expected_intent is not None and normalize_intent(
            prediction.get("intent")
        ) == expected_intent
        ambiguity_matches += bool(prediction.get("requires_clarification")) == bool(
            expected.get("requires_clarification")
        )

        expected_pairs = set(normalize_slots(expected.get("slots", {})).items())
        predicted_pairs = set(normalize_slots(prediction.get("slots", {})).items())
        slot_true_positive += len(expected_pairs & predicted_pairs)
        slot_false_positive += len(predicted_pairs - expected_pairs)
        slot_false_negative += len(expected_pairs - predicted_pairs)

    case_count = len(cases)
    precision_denominator = slot_true_positive + slot_false_positive
    recall_denominator = slot_true_positive + slot_false_negative
    latencies = [
        _nonnegative_number(case.get("decision_metadata", {}).get("latency_ms"))
        for case in cases
    ]
    model_cases = [case for case in cases if case.get("expect_model_call") is True]
    valid_count = sum(
        bool(case.get("prediction", {}).get("valid_structured_output"))
        for case in model_cases
    )
    fallback_count = sum(
        case.get("decision_metadata", {}).get("source") == "rule_fallback"
        for case in cases
    )

    return {
        "intent_accuracy": _safe_ratio(intent_matches, case_count),
        "slot_precision": _safe_ratio(slot_true_positive, precision_denominator),
        "slot_recall": _safe_ratio(slot_true_positive, recall_denominator),
        "ambiguity_accuracy": _safe_ratio(ambiguity_matches, case_count),
        "valid_structured_output_rate": _safe_ratio(valid_count, len(model_cases)),
        "fallback_rate": _safe_ratio(fallback_count, case_count),
        "p50_latency_ms": nearest_rank_percentile(latencies, 50),
        "p95_latency_ms": nearest_rank_percentile(latencies, 95),
    }


def nearest_rank_percentile(values: Iterable[float], percentile: int) -> float:
    """Return the nearest-rank percentile (1-based ceil rank)."""
    if percentile < 1 or percentile > 100:
        raise ValueError("percentile must be between 1 and 100")
    ordered = sorted(_nonnegative_number(value) for value in values)
    if not ordered:
        return 0
    rank = math.ceil((percentile / 100) * len(ordered))
    return ordered[rank - 1]


def summarize_usage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate provider usage only when every included case reported both counts."""
    model_cases = [case for case in cases if case.get("expect_model_call") is not False]
    token_pairs = [
        (
            case.get("decision_metadata", {}).get("input_tokens"),
            case.get("decision_metadata", {}).get("output_tokens"),
        )
        for case in model_cases
    ]
    complete = bool(token_pairs) and all(
        _is_token_count(input_tokens) and _is_token_count(output_tokens)
        for input_tokens, output_tokens in token_pairs
    )
    if not complete:
        return {
            "available": False,
            "reason": "provider_usage_unavailable",
            "case_count": len(model_cases),
            "input_tokens": None,
            "output_tokens": None,
            "average_input_tokens": None,
            "average_output_tokens": None,
        }

    total_input = sum(pair[0] for pair in token_pairs)
    total_output = sum(pair[1] for pair in token_pairs)
    return {
        "available": True,
        "reason": None,
        "case_count": len(model_cases),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "average_input_tokens": total_input / len(model_cases),
        "average_output_tokens": total_output / len(model_cases),
    }


def estimate_usage_cost(
    usage: Mapping[str, Any],
    *,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> dict[str, Any]:
    """Estimate cost only from complete usage and two nonnegative price inputs."""
    if not usage.get("available"):
        return {
            "available": False,
            "reason": "provider_usage_unavailable",
            "estimated_cost": None,
        }
    if not _is_price(input_cost_per_million) or not _is_price(output_cost_per_million):
        return {
            "available": False,
            "reason": "pricing_unavailable",
            "estimated_cost": None,
        }
    cost = (
        usage["input_tokens"] * input_cost_per_million
        + usage["output_tokens"] * output_cost_per_million
    ) / 1_000_000
    return {
        "available": True,
        "reason": None,
        "estimated_cost": round(cost, 10),
    }


def normalize_intent(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _INTENTS else None


def normalize_slots(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        field.strip().lower(): " ".join(slot_value.strip().casefold().split())
        for field, slot_value in value.items()
        if isinstance(field, str)
        and field.strip()
        and isinstance(slot_value, str)
        and slot_value.strip()
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _nonnegative_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, value)


def _is_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_price(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )
