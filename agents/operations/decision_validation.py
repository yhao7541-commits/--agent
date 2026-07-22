"""Deterministic merge and business validation for model-proposed booking slots."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict


BOOKING_ID_PATTERN = re.compile(
    r"booking_(?=[A-Za-z0-9_-]*[A-Za-z0-9])[A-Za-z0-9_-]+"
)
BOOKING_ID_TOKEN_PATTERN = re.compile(
    r"\bbooking(?:_|#)[^\s，。；;！？?,.()（）\[\]【】\"'“”‘’「」『』]*"
)


class SlotValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slots: dict[str, Any]
    sources: dict[str, str]
    ambiguities: list[str]
    errors: list[dict[str, Any]]


def resolve_booking_slots(
    *,
    previous_slots: dict[str, Any],
    previous_sources: dict[str, str],
    model_slots: dict[str, Any],
    user_slots: dict[str, Any],
    memory_special_request: str | None = None,
    system_customer_name: str | None = None,
    ambiguities: list[str] | None = None,
    errors: list[dict[str, Any]] | None = None,
    previous_issues: list[dict[str, Any]] | None = None,
    model_issues: list[dict[str, Any]] | None = None,
    user_issues: list[dict[str, Any]] | None = None,
) -> tuple[SlotValidationResult, list[dict[str, Any]]]:
    """Merge already-normalized candidates with their deterministic validation."""

    del previous_sources  # Previous provenance is deliberately collapsed at the turn boundary.
    slots: dict[str, Any] = {}
    sources: dict[str, str] = {}
    active_issues: dict[str, dict[str, Any]] = {}
    _apply_candidates(
        slots,
        sources,
        active_issues,
        previous_slots,
        previous_issues or [],
        slot_source="previous_turn",
    )
    _apply_candidates(
        slots,
        sources,
        active_issues,
        model_slots,
        model_issues or [],
        slot_source="user",
    )
    _apply_candidates(
        slots,
        sources,
        active_issues,
        user_slots,
        user_issues or [],
        slot_source="user",
    )

    if memory_special_request and not slots.get("special_requests"):
        slots["special_requests"] = memory_special_request
        sources["special_requests"] = "memory"
    if system_customer_name and not slots.get("customer_name"):
        slots["customer_name"] = system_customer_name
        sources["customer_name"] = "system"

    for field in ambiguities or []:
        active_issues[field] = {
            "field": field,
            "kind": "ambiguity",
            "code": f"invalid_{field}",
            "source": "current",
        }
    for error in errors or []:
        field = error["field"]
        active_issues[field] = {
            **error,
            "kind": error.get("kind", "error"),
            "source": error.get("source", "current"),
        }
    for field in active_issues:
        slots.pop(field, None)
        sources.pop(field, None)

    final_issues = list(active_issues.values())

    result = SlotValidationResult(
        slots=slots,
        sources=sources,
        ambiguities=[
            issue["field"] for issue in final_issues if issue["kind"] == "ambiguity"
        ],
        errors=[issue for issue in final_issues if issue["kind"] == "error"],
    )
    return result, final_issues


def merge_and_validate_booking_slots(
    *,
    previous_slots: dict[str, Any],
    previous_sources: dict[str, str],
    model_slots: dict[str, Any],
    user_slots: dict[str, Any],
    memory_special_request: str | None = None,
    system_customer_name: str | None = None,
    ambiguities: list[str] | None = None,
    errors: list[dict[str, Any]] | None = None,
    previous_issues: list[dict[str, Any]] | None = None,
    model_issues: list[dict[str, Any]] | None = None,
    user_issues: list[dict[str, Any]] | None = None,
) -> SlotValidationResult:
    """Return the stable public result without internal precedence bookkeeping."""

    result, _ = resolve_booking_slots(
        previous_slots=previous_slots,
        previous_sources=previous_sources,
        model_slots=model_slots,
        user_slots=user_slots,
        memory_special_request=memory_special_request,
        system_customer_name=system_customer_name,
        ambiguities=ambiguities,
        errors=errors,
        previous_issues=previous_issues,
        model_issues=model_issues,
        user_issues=user_issues,
    )
    return result


def _apply_candidates(
    slots: dict[str, Any],
    sources: dict[str, str],
    active_issues: dict[str, dict[str, Any]],
    candidates: dict[str, Any],
    issues: list[dict[str, Any]],
    *,
    slot_source: str,
) -> None:
    issues_by_field = {issue["field"]: issue for issue in issues}
    for field, value in candidates.items():
        if value in (None, ""):
            continue
        slots.pop(field, None)
        sources.pop(field, None)
        active_issues.pop(field, None)
        issue = issues_by_field.get(field)
        if issue:
            active_issues[field] = issue
        else:
            slots[field] = value
            sources[field] = slot_source
