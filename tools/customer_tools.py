from pydantic import BaseModel


def write_customer_preference(arguments: BaseModel, context) -> dict:
    return {
        "memory_id": f"memory_{context.trace_id[:8]}",
        "status": "stored",
    }


def lookup_customer_profile(arguments: BaseModel, context) -> dict:
    return {
        "user_id": getattr(arguments, "user_id", context.user_id),
        "known_preferences": [],
    }
