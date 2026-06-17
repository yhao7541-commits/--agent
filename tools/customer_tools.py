from pydantic import BaseModel


def write_customer_preference(arguments: BaseModel, context) -> dict:
    return {
        "memory_id": f"memory_{context.trace_id[:8]}",
        "status": "stored",
    }
