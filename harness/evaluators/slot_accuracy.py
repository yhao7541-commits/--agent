def missing_slots_passed(final_result: dict, expected: dict) -> bool | None:
    if "missing_slots" not in expected:
        return None
    return set(final_result.get("missing_slots", [])) == set(expected["missing_slots"])


def booking_slots_passed(final_result: dict, expected: dict) -> bool | None:
    expected_slots = expected.get("booking_slots")
    if not expected_slots:
        return None

    observed_slots = final_result.get("booking_slots", {})
    return all(observed_slots.get(field) == value for field, value in expected_slots.items())
