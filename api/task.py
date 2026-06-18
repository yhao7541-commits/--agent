"""
简化的任务分类API

只保留第一版核心功能
"""
from fastapi import APIRouter, HTTPException
from .core.response_models import (
    TaskClassificationRequest,
    DataResponse
)

router = APIRouter(prefix="/api/task", tags=["任务分类"])
task_agent = None


@router.post("/classify", response_model=DataResponse)
async def classify_task(request: TaskClassificationRequest):
    """分类任务"""
    try:
        result = await _get_task_agent().classify_task(request.text)
        
        return DataResponse(
            message="任务分类成功",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _get_task_agent():
    global task_agent
    if task_agent is None:
        from agents.appointment_agent import AppointmentAgent
        from agents.consultant_agent import ConsultantAgent
        from agents.task_classification_agent import TaskClassificationAgent

        task_agent = TaskClassificationAgent(AppointmentAgent(), ConsultantAgent())
    return task_agent
