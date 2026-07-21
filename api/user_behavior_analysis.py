"""
用户行为分析 API

该模块只做 Service 层聚合，不再维护独立用户行为 Agent。
"""

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from services.user_behavior_service import UserBehaviorService


router = APIRouter(prefix="/api/user-behavior", tags=["用户行为分析"])
router_underscore = APIRouter(prefix="/api/user_behavior", tags=["用户行为分析"])


class UserAnalysisResponse(BaseModel):
    """用户分析响应"""

    favorite_technician_id: Optional[int] = None
    favorite_technician_name: Optional[str] = None
    favorite_service: Optional[str] = None
    favorite_duration: Optional[int] = None
    total_appointments: int = 0
    days_since_last_appointment: Optional[int] = None
    should_send_reminder: bool = False


async def get_user_analysis(user_id: str = "default_user") -> UserAnalysisResponse:
    """获取用户行为分析数据。"""
    try:
        analysis = UserBehaviorService().analyze_user_patterns(user_id)
        technician_id = _parse_technician_id(analysis.get("preferred_technician"))

        return UserAnalysisResponse(
            favorite_technician_id=technician_id,
            favorite_technician_name=_lookup_technician_name(technician_id),
            total_appointments=analysis.get("total_appointments", 0),
            should_send_reminder=analysis.get("pattern") in {"occasional_user", "active_user"},
        )
    except Exception as exc:
        logging.error("获取用户分析数据失败: %s", exc)
        return UserAnalysisResponse()


@router.get("/analysis", response_model=UserAnalysisResponse)
async def get_default_user_analysis():
    """获取默认用户的行为分析数据。"""
    return await get_user_analysis("default_user")


@router.get("/dashboard_data", response_model=UserAnalysisResponse)
async def get_dashboard_data():
    """获取用户行为仪表板数据。"""
    return await get_user_analysis("default_user")


@router_underscore.get("/dashboard_data", response_model=UserAnalysisResponse)
async def get_dashboard_data_underscore():
    """获取用户行为仪表板数据（下划线版本）。"""
    return await get_user_analysis("default_user")


class ReminderRequest(BaseModel):
    """发送提醒请求"""

    user_id: str = "default_user"


class ReminderResponse(BaseModel):
    """提醒消息响应"""

    message: str
    technician_available_times: Optional[list] = None


@router.post("/send-reminder", response_model=ReminderResponse, summary="发送回访提醒")
async def send_reminder(request: ReminderRequest):
    """生成并返回回访提醒消息。"""
    try:
        analysis = await get_user_analysis(request.user_id)
        if analysis.total_appointments:
            message = "您好，系统已根据您的历史预约生成回访提醒；如需再次预约，可以直接告诉我服务项目和时间。"
        else:
            message = "您好，当前还没有足够的历史预约记录；如需预约，可以直接告诉我服务项目和时间。"
        return ReminderResponse(message=message, technician_available_times=[])
    except Exception as exc:
        logging.error("生成回访提醒失败: %s", exc)
        return ReminderResponse(
            message="系统暂时无法生成回访提醒，请稍后再试或直接联系我们预约。",
            technician_available_times=[],
        )


def _parse_technician_id(raw_value: object) -> Optional[int]:
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _lookup_technician_name(technician_id: Optional[int]) -> Optional[str]:
    if technician_id is None:
        return None
    try:
        from db import TechnicianDBRouter

        db = TechnicianDBRouter()
        tech_info = db.get_technician_by_id(technician_id)
        return tech_info.get("name") if tech_info else None
    except Exception as exc:
        logging.error("查询技师名称失败: %s", exc)
        return None
