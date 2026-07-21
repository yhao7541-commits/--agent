"""
预约兼容 API

旧 URL 保留，内部统一交给 OperationsAgent 编排和 Tool Gateway 写确认。
"""
import uuid

from fastapi import APIRouter, HTTPException

from agents.operations import OperationsAgent
from .core.response_models import AppointmentRequest, DataResponse


router = APIRouter(prefix="/api/appointment", tags=["预约管理"])
operations_agent = OperationsAgent()


@router.post("/create", response_model=DataResponse)
async def create_appointment(request: AppointmentRequest):
    """提交预约请求，返回 Operations 写操作确认结果。"""
    try:
        message_parts = [f"我想预约{request.preferred_time}的{request.service_type}"]
        if request.notes:
            message_parts.append(request.notes)

        result = operations_agent.run_turn(
            {
                "user_id": request.user_id,
                "conversation_id": f"appointment_{uuid.uuid4().hex[:8]}",
                "message": "，".join(message_parts),
                "booking_slots": {
                    "service_type": request.service_type,
                    "date": request.preferred_time,
                    "time_window": request.preferred_time,
                },
                "booking_slot_sources": {
                    "service_type": "api_request",
                    "date": "api_request",
                    "time_window": "api_request",
                },
            }
        )

        return DataResponse(
            message="预约请求已进入确认流程"
            if result.get("confirmation_required")
            else "预约请求已处理",
            data=result,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
