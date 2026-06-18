def rag_decision_passed(final_result: dict, expected: dict) -> bool | None:
    if "rag_used" not in expected:
        return None
    return final_result.get("rag_used", False) is expected["rag_used"]


def rag_groundedness_passed(final_result: dict, expected: dict) -> bool | None:
    if "grounded" not in expected:
        return None
    has_sources = bool(final_result.get("retrieved_knowledge"))
    return has_sources is expected["grounded"]
