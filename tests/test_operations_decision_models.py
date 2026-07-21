import math
import os

import pytest
from pydantic import ValidationError

from agents.operations.decision_models import (
    DecisionMetadata,
    DecisionSettings,
    ModelDecision,
)
from agents.operations.state import OperationsAgentState


@pytest.fixture(autouse=True)
def clear_decision_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("LLM_DECISION_"):
            monkeypatch.delenv(name, raising=False)


def test_model_decision_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        ModelDecision(
            intent="unsupported",
            confidence=0.8,
            suggested_action="ask_follow_up",
            decision_summary="Need more information.",
        )


def test_model_decision_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ModelDecision(
            intent="booking",
            confidence=0.8,
            suggested_action="collect_booking_slots",
            decision_summary="Collect the remaining booking details.",
            provider_debug_id="not part of the contract",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", "0.8"),
        ("extracted_slots", {"duration": 60}),
    ],
)
def test_model_decision_rejects_coercive_inputs(field, value):
    decision = {
        "intent": "booking",
        "confidence": 0.8,
        "suggested_action": "collect_booking_slots",
        "decision_summary": "Collect the remaining booking details.",
    }
    decision[field] = value

    with pytest.raises(ValidationError):
        ModelDecision(**decision)


def test_decision_settings_caps_attempts_at_three(monkeypatch):
    monkeypatch.setenv("LLM_DECISION_MAX_ATTEMPTS", "9")

    settings = DecisionSettings.from_env()

    assert settings.max_attempts == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("per_call_timeout_seconds", math.inf),
        ("total_deadline_seconds", math.nan),
        ("minimum_confidence", math.inf),
    ],
)
def test_decision_settings_reject_non_finite_values(field, value):
    with pytest.raises(ValidationError):
        DecisionSettings(**{field: value})


def test_decision_settings_is_immutable():
    settings = DecisionSettings()

    with pytest.raises(ValidationError):
        settings.mode = "hybrid"


def test_decision_metadata_keeps_rule_sources_distinct():
    rules = DecisionMetadata(source="rules")
    fallback = DecisionMetadata(source="rule_fallback")

    assert rules.source == "rules"
    assert fallback.source == "rule_fallback"
    assert rules.source != fallback.source
    assert rules.latency_ms == 0
    assert fallback.fallback_reason is None


def test_decision_metadata_uses_approved_diagnostic_defaults():
    metadata = DecisionMetadata(source="llm")

    assert metadata.provider is None
    assert metadata.model is None
    assert metadata.attempt_count == 0
    assert metadata.repair_count == 0
    assert metadata.latency_ms == 0
    assert metadata.input_tokens is None
    assert metadata.output_tokens is None
    assert metadata.fallback_reason is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt_count", -1),
        ("repair_count", -1),
        ("latency_ms", -1),
        ("input_tokens", -1),
        ("output_tokens", -1),
    ],
)
def test_decision_metadata_rejects_negative_diagnostics(field, value):
    with pytest.raises(ValidationError):
        DecisionMetadata(source="llm", **{field: value})


def test_decision_metadata_rejects_extra_fields():
    with pytest.raises(ValidationError):
        DecisionMetadata(source="llm", lateny_ms=10)


def test_operations_state_includes_all_decision_fields():
    annotations = OperationsAgentState.__annotations__

    assert {
        "model_decision",
        "decision_metadata",
        "ambiguities",
        "decision_source",
        "decision_errors",
        "decision_route",
    } <= annotations.keys()
