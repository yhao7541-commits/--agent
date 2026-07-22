"""Synchronous hybrid model decision orchestration with deterministic fallback."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from httpx import TimeoutException as HttpxTimeoutError
from httpx import TransportError as HttpxTransportError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from pydantic import BaseModel, ConfigDict, ValidationError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeoutError

from agents.operations.decision_client import DecisionModelClient
from agents.operations.decision_models import (
    DecisionMetadata,
    DecisionSettings,
    ModelDecision,
)
from agents.operations.decision_prompt import build_repair_prompt
from config.model_provider import ModelConfigurationError


DecisionErrorCode = Literal[
    "provider_timeout",
    "rate_limited",
    "provider_5xx",
    "transport_error",
    "authentication_error",
    "configuration_error",
    "invalid_json",
    "schema_validation_error",
    "low_confidence",
    "business_validation_error",
    "total_deadline_exceeded",
]


class DecisionError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: DecisionErrorCode
    attempt: int
    retryable: bool


class DecisionEngineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ModelDecision
    metadata: DecisionMetadata
    errors: list[DecisionError]


class LocalDecisionRejection(RuntimeError):
    """Raised when local policy rejects a model decision before a provider call."""


@dataclass(frozen=True)
class _ClassifiedError:
    code: DecisionErrorCode
    retryable: bool
    provider_attempted: bool = True


class _TokenTotals:
    def __init__(self) -> None:
        self._response_count = 0
        self._input_total = 0
        self._output_total = 0
        self._input_complete = True
        self._output_complete = True

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self._response_count += 1
        if input_tokens is None:
            self._input_complete = False
        else:
            self._input_total += input_tokens
        if output_tokens is None:
            self._output_complete = False
        else:
            self._output_total += output_tokens

    @property
    def input_tokens(self) -> int | None:
        if not self._response_count or not self._input_complete:
            return None
        return self._input_total

    @property
    def output_tokens(self) -> int | None:
        if not self._response_count or not self._output_complete:
            return None
        return self._output_total


class HybridDecisionEngine:
    _BACKOFF_BASE_SECONDS = 0.05
    _BACKOFF_MAX_SECONDS = 0.2
    _JITTER_MAX_SECONDS = 0.1

    def __init__(
        self,
        *,
        client: DecisionModelClient,
        settings: DecisionSettings,
        fallback: Callable[[str], ModelDecision | dict[str, Any]],
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        jitter_fn: Callable[[], Any] | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._fallback = fallback
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self._jitter = jitter_fn if jitter_fn is not None else self._default_jitter

    def decide(self, original_task: str) -> DecisionEngineResult:
        started_at = self._monotonic()
        deadline = started_at + self._settings.total_deadline_seconds
        max_calls = min(3, self._settings.max_attempts)
        call_count = 0
        attempt_count = 0
        repair_count = 0
        prompt = original_task
        prompt_is_repair = False
        errors: list[DecisionError] = []
        tokens = _TokenTotals()
        provider: str | None = None
        model: str | None = None

        def fallback_result(reason: DecisionErrorCode) -> DecisionEngineResult:
            return self._fallback_result(
                original_task=original_task,
                reason=reason,
                started_at=started_at,
                provider=provider,
                model=model,
                attempt_count=attempt_count,
                repair_count=repair_count,
                tokens=tokens,
                errors=errors,
            )

        def deadline_result() -> DecisionEngineResult:
            self._append_deadline_error(errors, attempt_count)
            return fallback_result("total_deadline_exceeded")

        while call_count < max_calls:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return deadline_result()

            call_count += 1
            if prompt_is_repair:
                repair_count += 1
            try:
                response = self._client.invoke(
                    prompt,
                    timeout_seconds=min(
                        self._settings.per_call_timeout_seconds,
                        remaining,
                    ),
                )
            except Exception as exc:
                classified = _classify_exception(exc)
                if classified is None:
                    raise
                if classified.provider_attempted:
                    attempt_count += 1
                error_attempt = attempt_count if classified.provider_attempted else 0
                errors.append(
                    DecisionError(
                        code=classified.code,
                        attempt=error_attempt,
                        retryable=classified.retryable,
                    )
                )
                if self._monotonic() >= deadline:
                    return deadline_result()
                if not classified.retryable or call_count >= max_calls:
                    return fallback_result(classified.code)
                if not self._backoff_before_retry(call_count, deadline):
                    return deadline_result()
                continue

            attempt_count += 1
            tokens.add(response.input_tokens, response.output_tokens)
            provider = response.provider or provider
            model = response.model or model
            if self._monotonic() >= deadline:
                return deadline_result()

            repair_errors: Any | None = None
            try:
                payload = json.loads(response.raw_text)
            except json.JSONDecodeError:
                repair_code: DecisionErrorCode = "invalid_json"
                repair_errors = [{"loc": ("root",), "type": "json_invalid"}]
            else:
                try:
                    decision = ModelDecision.model_validate(payload)
                except ValidationError as exc:
                    repair_code = "schema_validation_error"
                    repair_errors = exc

            if repair_errors is not None:
                errors.append(
                    DecisionError(
                        code=repair_code,
                        attempt=attempt_count,
                        retryable=True,
                    )
                )
                if call_count >= max_calls:
                    return fallback_result(repair_code)
                prompt = build_repair_prompt(original_task, repair_errors)
                prompt_is_repair = True
                if not self._backoff_before_retry(call_count, deadline):
                    return deadline_result()
                continue

            if self._monotonic() >= deadline:
                return deadline_result()

            if (
                decision.confidence < self._settings.minimum_confidence
                or decision.intent == "unknown"
            ):
                errors.append(
                    DecisionError(
                        code="low_confidence",
                        attempt=attempt_count,
                        retryable=False,
                    )
                )
                return fallback_result("low_confidence")

            return DecisionEngineResult(
                decision=decision,
                metadata=self._metadata(
                    source="llm",
                    started_at=started_at,
                    provider=provider,
                    model=model,
                    attempt_count=attempt_count,
                    repair_count=repair_count,
                    tokens=tokens,
                ),
                errors=errors,
            )

        raise AssertionError("decision call loop terminated unexpectedly")

    def _backoff_before_retry(self, call_count: int, deadline: float) -> bool:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            return False
        base = min(
            self._BACKOFF_BASE_SECONDS * (2 ** (call_count - 1)),
            self._BACKOFF_MAX_SECONDS,
        )
        raw_jitter = self._jitter()
        if isinstance(raw_jitter, bool):
            jitter = 0.0
        else:
            try:
                jitter = float(raw_jitter)
            except (TypeError, ValueError, OverflowError):
                jitter = 0.0
        if not math.isfinite(jitter):
            jitter = 0.0
        jitter = min(max(jitter, 0.0), self._JITTER_MAX_SECONDS)
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            return False
        delay = min(base + jitter, remaining)
        if delay > 0:
            self._sleep(delay)
        return deadline - self._monotonic() > 0

    def _default_jitter(self) -> float:
        return random.random() * self._JITTER_MAX_SECONDS

    @staticmethod
    def _append_deadline_error(
        errors: list[DecisionError], attempt_count: int
    ) -> None:
        if errors and errors[-1].code == "total_deadline_exceeded":
            return
        errors.append(
            DecisionError(
                code="total_deadline_exceeded",
                attempt=attempt_count,
                retryable=False,
            )
        )

    def _fallback_result(
        self,
        *,
        original_task: str,
        reason: DecisionErrorCode,
        started_at: float,
        provider: str | None,
        model: str | None,
        attempt_count: int,
        repair_count: int,
        tokens: _TokenTotals,
        errors: list[DecisionError],
    ) -> DecisionEngineResult:
        decision = ModelDecision.model_validate(self._fallback(original_task))
        return DecisionEngineResult(
            decision=decision,
            metadata=self._metadata(
                source="rule_fallback",
                started_at=started_at,
                provider=provider,
                model=model,
                attempt_count=attempt_count,
                repair_count=repair_count,
                tokens=tokens,
                fallback_reason=reason,
            ),
            errors=errors,
        )

    def _metadata(
        self,
        *,
        source: Literal["llm", "rule_fallback"],
        started_at: float,
        provider: str | None,
        model: str | None,
        attempt_count: int,
        repair_count: int,
        tokens: _TokenTotals,
        fallback_reason: DecisionErrorCode | None = None,
    ) -> DecisionMetadata:
        return DecisionMetadata(
            source=source,
            provider=provider,
            model=model,
            attempt_count=attempt_count,
            repair_count=repair_count,
            latency_ms=_latency_ms(started_at, self._monotonic()),
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            fallback_reason=fallback_reason,
        )


def _classify_exception(exc: Exception) -> _ClassifiedError | None:
    if isinstance(exc, ModelConfigurationError):
        return _ClassifiedError(
            code="configuration_error",
            retryable=False,
            provider_attempted=False,
        )
    if isinstance(exc, LocalDecisionRejection):
        return _ClassifiedError(
            code="business_validation_error",
            retryable=False,
            provider_attempted=False,
        )
    status_code = _status_code(exc)
    if status_code == 408:
        return _ClassifiedError("provider_timeout", retryable=True)
    if status_code in {401, 403}:
        return _ClassifiedError("authentication_error", retryable=False)
    if status_code == 429:
        return _ClassifiedError("rate_limited", retryable=True)
    if status_code is not None and 500 <= status_code <= 599:
        return _ClassifiedError("provider_5xx", retryable=True)
    if status_code is not None and 400 <= status_code <= 499:
        return _ClassifiedError("configuration_error", retryable=False)
    if isinstance(
        exc,
        (TimeoutError, HttpxTimeoutError, OpenAITimeoutError, RequestsTimeoutError),
    ):
        return _ClassifiedError("provider_timeout", retryable=True)
    if isinstance(
        exc,
        (
            ConnectionError,
            HttpxTransportError,
            OpenAIConnectionError,
            RequestsConnectionError,
        ),
    ):
        return _ClassifiedError("transport_error", retryable=True)
    return None


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _latency_ms(started_at: float, finished_at: float) -> int:
    return max(0, round((finished_at - started_at) * 1_000))
