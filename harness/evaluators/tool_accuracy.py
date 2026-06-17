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


def tool_arguments_passed(turn_results: list[dict], expected: dict) -> bool | None:
    expected_arguments = expected.get("tool_arguments")
    if not expected_arguments:
        return None

    for tool_name, required_arguments in expected_arguments.items():
        observed_calls = [
            tool.get("arguments", {})
            for turn in turn_results
            for tool in turn.get("tool_plan", [])
            if tool.get("tool_name") == tool_name
        ]
        if not any(_arguments_match(arguments, required_arguments) for arguments in observed_calls):
            return False
    return True


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


def _arguments_match(observed: dict, expected: dict) -> bool:
    return all(observed.get(key) == value for key, value in expected.items())
