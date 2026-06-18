def memory_proposal_passed(final_result: dict, expected: dict) -> bool | None:
    expected_proposal = expected.get("memory_proposal")
    if not expected_proposal:
        return None
    proposals = final_result.get("memory_proposals", [])
    if not proposals:
        return False
    proposal = proposals[0]
    if proposal.get("type") != expected_proposal.get("type"):
        return False
    return all(fragment in proposal.get("content", "") for fragment in expected_proposal.get("content_contains", []))


def no_memory_proposal_passed(final_result: dict, expected: dict) -> bool | None:
    if "no_memory_proposal" not in expected:
        return None
    has_no_proposals = not final_result.get("memory_proposals", [])
    return has_no_proposals is expected["no_memory_proposal"]


def memory_recall_passed(final_result: dict, expected: dict) -> bool | None:
    expected_recall = expected.get("memory_recall")
    if not expected_recall:
        return None

    known_preferences = " ".join(final_result.get("customer_context", {}).get("known_preferences", []))
    summary = final_result.get("confirmation_request", {}).get("summary", {})
    summary_text = " ".join(str(value) for value in summary.values())

    return all(
        fragment in known_preferences
        for fragment in expected_recall.get("known_preference_contains", [])
    ) and all(
        fragment in summary_text
        for fragment in expected_recall.get("confirmation_summary_contains", [])
    )


def memory_deletion_passed(final_result: dict, expected: dict) -> bool | None:
    expected_deletion = expected.get("memory_deleted")
    if not expected_deletion:
        return None

    known_preferences = " ".join(final_result.get("customer_context", {}).get("known_preferences", []))
    summary = final_result.get("confirmation_request", {}).get("summary", {})
    summary_text = " ".join(str(value) for value in summary.values())

    return all(
        fragment not in known_preferences
        for fragment in expected_deletion.get("known_preference_not_contains", [])
    ) and all(
        fragment not in summary_text
        for fragment in expected_deletion.get("confirmation_summary_not_contains", [])
    )
