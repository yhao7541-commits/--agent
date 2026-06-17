def security_policy_passed(final_result: dict, expected: dict) -> bool | None:
    expected_reason = expected.get("policy_violation")
    if not expected_reason:
        return None

    return any(
        event.get("event_type") == "policy_violation"
        and event.get("metadata", {}).get("reason") == expected_reason
        for event in final_result.get("trace_events", [])
    )
