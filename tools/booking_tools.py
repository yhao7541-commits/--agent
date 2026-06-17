from __future__ import annotations

from pydantic import BaseModel


def search_services(arguments: BaseModel, context) -> dict:
    query = getattr(arguments, "query", "")
    services = [
        {
            "name": "肩颈放松",
            "duration_minutes": 60,
            "price": "门店价目表为准",
        },
        {
            "name": "推拿",
            "duration_minutes": 90,
            "price": "门店价目表为准",
        },
    ]
    if query:
        services = [service for service in services if query in service["name"]] or services
    return {"services": services}


def check_schedule(arguments: BaseModel, context) -> dict:
    return {
        "available": True,
        "alternatives": [],
    }


def create_booking(arguments: BaseModel, context) -> dict:
    booking_id = f"booking_{context.trace_id[:8]}"
    return {"booking_id": booking_id, "status": "confirmed"}


def reschedule_booking(arguments: BaseModel, context) -> dict:
    booking_id = f"booking_{context.trace_id[:8]}"
    return {"booking_id": booking_id, "status": "rescheduled"}


def cancel_booking(arguments: BaseModel, context) -> dict:
    booking_id = f"booking_{context.trace_id[:8]}"
    return {"booking_id": booking_id, "status": "cancelled"}
