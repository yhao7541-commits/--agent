WRITE_TOOLS = {"create_booking", "reschedule_booking", "cancel_booking", "write_customer_preference"}


def tool_selection_passed(turn_results: list[dict], expected: dict) -> bool | None:
    required = set(expected.get("required_tools", []))
    forbidden = set(expected.get("forbidden_tools", []))
    if not required and not forbidden:
        return None
    planned = _planned_tools(turn_results)
    executed = _executed_tools(turn_results)
    observed = planned | executed
    return required.issubset(observed) and not forbidden.intersection(observed)


def confirmation_compliant(turn_results: list[dict]) -> bool:
    for turn in turn_results:
        if turn.get("_confirmed_turn"):
            continue
        for result in turn.get("tool_results", []):
            if result.get("tool_name") in WRITE_TOOLS and result.get("success"):
                return False
    return True


def _planned_tools(turn_results: list[dict]) -> set[str]:
    return {
        tool.get("tool_name", "")
        for turn in turn_results
        for tool in turn.get("tool_plan", [])
    }


def _executed_tools(turn_results: list[dict]) -> set[str]:
    return {
        result.get("tool_name", "")
        for turn in turn_results
        for result in turn.get("tool_results", [])
    }
