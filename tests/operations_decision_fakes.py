from __future__ import annotations

import math
from dataclasses import dataclass

from agents.operations.decision_client import ModelCallResult


@dataclass(frozen=True)
class FakeCall:
    prompt: str
    timeout_seconds: float


@dataclass(frozen=True)
class FakeOutcome:
    value: ModelCallResult | BaseException
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        _validate_duration(self.elapsed_seconds, "elapsed_seconds")


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        _validate_duration(seconds, "seconds")
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        _validate_duration(seconds, "seconds")
        self.sleep_calls.append(seconds)
        self.advance(seconds)


class FakeHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider secret response for HTTP {status_code}")
        self.status_code = status_code


class FakeRetryableTransportError(ConnectionError):
    pass


class ProgrammableDecisionClient:
    def __init__(self, clock: FakeClock, outcomes: list[FakeOutcome]) -> None:
        self._clock = clock
        self._outcomes = list(outcomes)
        self.calls: list[FakeCall] = []
        self.active_calls = 0
        self.max_active_calls = 0

    def invoke(self, prompt: str, timeout_seconds: float) -> ModelCallResult:
        if self.active_calls:
            raise AssertionError("model calls must not overlap")
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            self.calls.append(FakeCall(prompt, timeout_seconds))
            if not self._outcomes:
                raise AssertionError("unexpected model call")
            outcome = self._outcomes.pop(0)
            self._clock.advance(outcome.elapsed_seconds)
            if isinstance(outcome.value, BaseException):
                raise outcome.value
            return outcome.value
        finally:
            self.active_calls -= 1


def model_result(
    raw_text: str,
    *,
    provider: str | None = "fake-provider",
    model: str | None = "fake-model",
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
) -> ModelCallResult:
    return ModelCallResult(
        raw_text=raw_text,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _validate_duration(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
