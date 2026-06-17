def booking_completed(turn_results: list[dict], expected: dict) -> bool | None:
    if "booking_completed" not in expected:
        return None
    completed = any(
        result.get("tool_name") == "create_booking" and result.get("success")
        for turn in turn_results
        for result in turn.get("tool_results", [])
    )
    return completed is expected["booking_completed"]
