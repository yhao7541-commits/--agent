SENSITIVE_MEMORY_TYPES = {
    "constraint",
    "policy_note",
    "service_contraindication",
    "marketing_consent",
}
VAGUE_MARKERS = ("随便", "可能", "也许", "看情况", "都行")


def is_vague_memory_statement(message: str) -> bool:
    return any(marker in message for marker in VAGUE_MARKERS)


def sensitivity_for_memory_type(memory_type: str) -> str:
    if memory_type in SENSITIVE_MEMORY_TYPES:
        return "sensitive"
    return "normal"


def memory_requires_confirmation(memory_type: str, sensitivity: str) -> bool:
    return sensitivity == "sensitive" or memory_type in SENSITIVE_MEMORY_TYPES
