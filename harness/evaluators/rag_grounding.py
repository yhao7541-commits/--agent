def rag_decision_passed(final_result: dict, expected: dict) -> bool | None:
    if "rag_used" not in expected:
        return None
    if final_result.get("rag_used", False) is not expected["rag_used"]:
        return False
    if expected.get("grounded"):
        return bool(final_result.get("retrieved_knowledge"))
    return True
