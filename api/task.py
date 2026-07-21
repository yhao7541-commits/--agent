"""
任务分类兼容 API

分类结果来自单一 OperationsAgent 的 classify_intent 节点，不再维护独立分类 Agent。
"""
import uuid

from fastapi import APIRouter, HTTPException

from agents.operations import OperationsAgent
from .core.response_models import DataResponse, TaskClassificationRequest


router = APIRouter(prefix="/api/task", tags=["任务分类"])
operations_agent = OperationsAgent()


@router.post("/classify", response_model=DataResponse)
async def classify_task(request: TaskClassificationRequest):
    """分类任务并返回本轮 Operations 状态摘要。"""
    try:
        context = request.context or {}
        result = await operations_agent.arun_turn(
            {
                "user_id": context.get("user_id", "local_user"),
                "conversation_id": context.get("conversation_id", f"task_{uuid.uuid4().hex[:8]}"),
                "message": request.text,
                "booking_slots": context.get("booking_slots", {}),
                "booking_slot_sources": context.get("booking_slot_sources", {}),
            }
        )

        return DataResponse(
            message="任务分类成功",
            data={
                "intent": result.get("intent", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "reply": result.get("reply", ""),
                "missing_slots": result.get("missing_slots", []),
                "tool_plan": result.get("tool_plan", []),
                "trace_id": result.get("trace_id", ""),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
