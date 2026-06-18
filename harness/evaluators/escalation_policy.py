def escalation_passed(final_result: dict, expected: dict) -> bool | None:
    if "escalated" not in expected:
        return None
    return final_result.get("escalated", False) is expected["escalated"]


def escalation_reason_passed(final_result: dict, expected: dict) -> bool | None:
    expected_reason = expected.get("escalation_reason")
    if not expected_reason:
        return None
    return final_result.get("escalation", {}).get("reason") == expected_reason
