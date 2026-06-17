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
