from pydantic import BaseModel


def escalate_to_human(arguments: BaseModel, context) -> dict:
    return {
        "handoff_id": f"handoff_{context.trace_id[:8]}",
        "status": "queued",
    }
