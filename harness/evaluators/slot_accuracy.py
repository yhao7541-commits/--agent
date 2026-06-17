def missing_slots_passed(final_result: dict, expected: dict) -> bool | None:
    if "missing_slots" not in expected:
        return None
    return set(final_result.get("missing_slots", [])) == set(expected["missing_slots"])
