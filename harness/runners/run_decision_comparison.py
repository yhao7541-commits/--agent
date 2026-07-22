"""Run an opt-in, frozen rules-versus-live-hybrid decision comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agents.operations.agent import OperationsAgent
from agents.operations.decision_models import DecisionIntent, DecisionSettings, ModelDecision
from config.model_provider import (
    LocalRuleBasedChatModel,
    ModelConfigurationError,
    create_chat_model,
    get_model_provider,
)
from harness.evaluators.decision_comparison import (
    compute_decision_metrics,
    estimate_usage_cost,
    normalize_intent,
    normalize_slots,
    summarize_usage,
)
from security.guardrails import reset_confirmation_token_registry
from tools.customer_tools import isolated_customer_memory_store


DEFAULT_DATASET = Path("harness/datasets/decision_long_tail_cases.yaml")
DEFAULT_OUTPUT_DIR = Path("data/evaluation")
PROMPT_VERSION = "operations-decision-v1"
_INTENTS = set(DecisionIntent.__args__)
_REQUIRED_FAMILIES = {
    "colloquial_typo",
    "correction",
    "vague_time",
    "contradictory_slots",
    "negation",
    "combined_intent",
    "safety_recognition",
}
_SECRET_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|api[_-]?key\s*[:=]|bearer\s+[A-Za-z0-9._-]{12,})",
    re.IGNORECASE,
)
_CASE_STATE_LOCK = threading.Lock()


class DatasetValidationError(ValueError):
    """Raised when the frozen semantic dataset violates its governance contract."""


class LiveModelUnavailable(RuntimeError):
    """Raised before case execution when the configured live provider is unusable."""


def load_dataset(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise DatasetValidationError("dataset root must be a mapping")
    return loaded


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    version = dataset.get("version")
    date_anchor = dataset.get("date_anchor")
    timezone_name = dataset.get("timezone")
    cases = dataset.get("cases")
    if not isinstance(version, str) or not version.strip():
        raise DatasetValidationError("dataset version is required")
    if not isinstance(date_anchor, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_anchor):
        raise DatasetValidationError("date_anchor must use YYYY-MM-DD")
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise DatasetValidationError("timezone is required")
    if _SECRET_PATTERN.search(json.dumps(dataset, ensure_ascii=False)):
        raise DatasetValidationError("dataset contains secret-shaped content")
    if not isinstance(cases, list) or not 25 <= len(cases) <= 40:
        raise DatasetValidationError("dataset must contain approximately 30 cases")

    ids: set[str] = set()
    families: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise DatasetValidationError("each case must be a mapping")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise DatasetValidationError("case IDs must be nonempty and unique")
        ids.add(case_id)
        family = case.get("family")
        if not isinstance(family, str) or not family:
            raise DatasetValidationError(f"case {case_id} requires a family")
        families.add(family)
        if not isinstance(case.get("message"), str):
            raise DatasetValidationError(f"case {case_id} requires a message")
        if not isinstance(case.get("prior_turns"), list):
            raise DatasetValidationError(f"case {case_id} requires explicit prior_turns")
        if not isinstance(case.get("expect_model_call"), bool):
            raise DatasetValidationError(f"case {case_id} requires expect_model_call")
        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise DatasetValidationError(f"case {case_id} requires expected labels")
        if expected.get("intent") not in _INTENTS:
            raise DatasetValidationError(f"case {case_id} has an invalid expected intent")
        if not isinstance(expected.get("requires_clarification"), bool):
            raise DatasetValidationError(
                f"case {case_id} requires expected requires_clarification"
            )
        if not isinstance(expected.get("slots", {}), dict):
            raise DatasetValidationError(f"case {case_id} expected slots must be a mapping")
        for prior_turn in case["prior_turns"]:
            if not isinstance(prior_turn, dict) or not isinstance(prior_turn.get("user"), str):
                raise DatasetValidationError(
                    f"case {case_id} prior turns require explicit user text"
                )
            if "accepted_slots" not in prior_turn or not isinstance(
                prior_turn["accepted_slots"], dict
            ):
                raise DatasetValidationError(
                    f"case {case_id} prior turns require explicit accepted_slots"
                )

    if not _REQUIRED_FAMILIES <= families:
        missing = sorted(_REQUIRED_FAMILIES - families)
        raise DatasetValidationError(f"dataset is missing approved families: {missing}")


def preflight_live_model() -> dict[str, Any]:
    """Validate live configuration and native timeout construction without calling it."""
    settings = DecisionSettings.from_env()
    try:
        model = create_chat_model(
            temperature=0,
            request_timeout=settings.per_call_timeout_seconds,
            require_configured=True,
        )
    except (ModelConfigurationError, TypeError, ValueError) as exc:
        raise LiveModelUnavailable(
            "live provider configuration or native timeout support is unavailable"
        ) from exc
    if isinstance(model, LocalRuleBasedChatModel):
        raise LiveModelUnavailable("a live provider is required")
    return {
        "provider": get_model_provider(),
        "model": _configured_model_name(),
        "temperature": 0,
        "native_timeout": True,
    }


def ensure_model_attempts(cases: list[dict[str, Any]]) -> None:
    missing = [
        case.get("id", "unknown")
        for case in cases
        if case.get("expect_model_call") is True
        and case.get("decision_metadata", {}).get("attempt_count", 0) <= 0
    ]
    if missing:
        raise ValueError(f"model-required cases must record attempt_count > 0: {missing}")


def provider_completion_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    required = [case for case in cases if case.get("expect_model_call") is True]
    provider_successes = sum(
        case.get("decision_metadata", {}).get("attempt_count", 0) > 0
        and case.get("decision_metadata", {}).get("source") != "rule_fallback"
        and case.get("prediction", {}).get("valid_structured_output") is True
        for case in required
    )
    fallbacks = sum(
        case.get("decision_metadata", {}).get("source") == "rule_fallback"
        for case in required
    )
    provider_success_rate = provider_successes / len(required) if required else 1.0
    fallback_rate = fallbacks / len(required) if required else 0.0
    return {
        "model_required_case_count": len(required),
        "provider_success_count": provider_successes,
        "provider_success_rate": provider_success_rate,
        "fallback_count": fallbacks,
        "fallback_rate": fallback_rate,
        "bad_case_evidence": bool(required) and provider_success_rate < 0.8,
        "uplift_claim_allowed": False,
    }


def resolve_output_path(
    output: Path | None,
    *,
    force: bool = False,
    now: datetime | None = None,
) -> Path:
    current = now or datetime.now(timezone.utc)
    path = output or DEFAULT_OUTPUT_DIR / (
        f"decision-comparison-{current.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path}")
    return path


def run_comparison(
    dataset_path: Path,
    *,
    output: Path | None = None,
    force: bool = False,
    preflight_fn: Callable[[], Mapping[str, Any] | None] = preflight_live_model,
    case_runner: Callable[[dict[str, Any], str, str], dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run both modes only after live-provider preflight has succeeded."""
    output_path = resolve_output_path(output, force=force, now=now)
    provider_config = dict(preflight_fn() or {})
    dataset = load_dataset(dataset_path)
    validate_dataset(dataset)
    run_case = case_runner or _run_case
    case_results: dict[str, list[dict[str, Any]]] = {}
    run_id = uuid.uuid4().hex
    for mode in ("rules", "hybrid"):
        mode_results = [
            _normalize_result_record(
                run_case(case, mode, f"eval-{run_id}-{mode}-{index:03d}")
            )
            for index, case in enumerate(dataset["cases"])
        ]
        if mode == "hybrid":
            ensure_model_attempts(mode_results)
        case_results[mode] = mode_results

    prices = _pricing_from_env()
    hybrid_usage = summarize_usage(case_results["hybrid"])
    report = {
        "report_version": "decision-comparison-v1",
        "output_path": str(output_path),
        "generated_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "configuration": {
            **provider_config,
            "temperature": 0,
            "prompt_version": PROMPT_VERSION,
            "dataset_version": dataset["version"],
            "timezone": dataset["timezone"],
            "date_anchor": dataset["date_anchor"],
        },
        "case_count": len(dataset["cases"]),
        "modes": {
            mode: {
                "metrics": compute_decision_metrics(results),
                "cases": results,
            }
            for mode, results in case_results.items()
        },
        "provider_completion": provider_completion_summary(case_results["hybrid"]),
        "usage": hybrid_usage,
        "cost": estimate_usage_cost(
            hybrid_usage,
            input_cost_per_million=prices[0],
            output_cost_per_million=prices[1],
        ),
    }
    _write_json_report(output_path, report, force=force)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare deterministic rules with a configured live hybrid model."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--require-live-model", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.require_live_model:
        print("Refusing comparison without --require-live-model.")
        return 2
    try:
        report = run_comparison(
            args.dataset,
            output=args.output,
            force=args.force,
        )
    except (LiveModelUnavailable, ModelConfigurationError, ValueError) as exc:
        print(f"Live comparison unavailable: {exc}")
        return 2
    except FileExistsError as exc:
        print(str(exc))
        return 1
    print(
        json.dumps(
            {
                "output_path": report["output_path"],
                "case_count": report["case_count"],
                "provider_completion": report["provider_completion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_case(case: dict[str, Any], mode: str, namespace: str) -> dict[str, Any]:
    """Run one isolated case; serialized because the application store is process-global."""
    with _CASE_STATE_LOCK:
        return _run_case_sequential(case, mode, namespace)


def _run_case_sequential(
    case: dict[str, Any], mode: str, namespace: str
) -> dict[str, Any]:
    reset_confirmation_token_registry()
    temp_dir = Path(tempfile.mkdtemp(prefix="decision-eval-"))
    try:
        with isolated_customer_memory_store(temp_dir / "memory.sqlite3"):
            state: dict[str, Any] = {
                "user_id": f"{namespace}-user",
                "conversation_id": f"{namespace}-conversation",
                "message": "",
            }
            with _decision_mode(mode):
                agent = OperationsAgent()
                for prior_turn in case["prior_turns"]:
                    state["message"] = prior_turn["user"]
                    state = agent.run_turn(state)
                    state["booking_slots"] = dict(prior_turn["accepted_slots"])
                    state["booking_slot_sources"] = {
                        field: "previous_turn" for field in state["booking_slots"]
                    }
                state["message"] = case["message"]
                result = agent.run_turn(state)
    finally:
        shutil.rmtree(temp_dir)
    return _normalized_case_result(case, result)


def _normalized_case_result(case: dict[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = result.get("decision_metadata", {})
    model_decision = result.get("model_decision", {})
    extracted_slots = model_decision.get("extracted_slots", {}) if isinstance(model_decision, dict) else {}
    if not isinstance(extracted_slots, dict):
        extracted_slots = {}
    valid_structured_output = False
    if (
        metadata.get("attempt_count", 0) > 0
        and metadata.get("source") != "rule_fallback"
        and metadata.get("fallback_reason") is None
        and isinstance(model_decision, dict)
    ):
        try:
            ModelDecision.model_validate(model_decision)
        except ValueError:
            pass
        else:
            valid_structured_output = True
    return {
        "id": case["id"],
        "family": case["family"],
        "expect_model_call": case["expect_model_call"],
        "expected": {
            "intent": case["expected"]["intent"],
            "slots": case["expected"].get("slots", {}),
            "requires_clarification": case["expected"]["requires_clarification"],
        },
        "prediction": {
            "intent": normalize_intent(result.get("intent")),
            "slots": normalize_slots(extracted_slots),
            "requires_clarification": bool(result.get("ambiguities"))
            or result.get("decision_route") == "clarification",
            "valid_structured_output": valid_structured_output,
        },
        "decision_metadata": {
            "source": metadata.get("source"),
            "provider": metadata.get("provider"),
            "model": metadata.get("model"),
            "attempt_count": metadata.get("attempt_count", 0),
            "repair_count": metadata.get("repair_count", 0),
            "fallback_reason": metadata.get("fallback_reason"),
            "latency_ms": metadata.get("latency_ms", 0),
            "input_tokens": metadata.get("input_tokens"),
            "output_tokens": metadata.get("output_tokens"),
        },
    }


def _normalize_result_record(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    prediction = dict(normalized.get("prediction", {}))
    prediction["intent"] = normalize_intent(prediction.get("intent"))
    prediction["slots"] = normalize_slots(prediction.get("slots", {}))
    normalized["prediction"] = prediction
    return normalized


@contextmanager
def _decision_mode(mode: str):
    previous = os.environ.get("OPERATIONS_DECISION_MODE")
    os.environ["OPERATIONS_DECISION_MODE"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPERATIONS_DECISION_MODE", None)
        else:
            os.environ["OPERATIONS_DECISION_MODE"] = previous


def _configured_model_name() -> str | None:
    provider = get_model_provider()
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT")
    return os.getenv("LLM_MODEL")


def _pricing_from_env() -> tuple[float | None, float | None]:
    return (
        _optional_nonnegative_float(os.getenv("LLM_INPUT_COST_PER_1M_TOKENS")),
        _optional_nonnegative_float(os.getenv("LLM_OUTPUT_COST_PER_1M_TOKENS")),
    )


def _optional_nonnegative_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _write_json_report(path: Path, report: Mapping[str, Any], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
