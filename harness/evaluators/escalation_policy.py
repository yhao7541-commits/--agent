def escalation_passed(final_result: dict, expected: dict) -> bool | None:
    if "escalated" not in expected:
        return None
    return final_result.get("escalated", False) is expected["escalated"]
