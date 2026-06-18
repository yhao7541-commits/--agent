"""
API核心组件初始化

导出核心的响应模型、异常处理等组件
"""
from .exceptions import BusinessException, api_exception_handler, general_exception_handler
from .response_models import (
    AppointmentRequest,
    AppointmentResponse,
    BaseResponse,
    ConsultationRequest,
    ConsultationResponse,
    DataResponse,
    TaskClassificationRequest,
    TaskClassificationResponse,
    UserBehaviorRequest,
    UserBehaviorResponse,
)

__all__ = [
    "BaseResponse",
    "DataResponse",
    "AppointmentRequest",
    "AppointmentResponse",
    "ConsultationRequest",
    "ConsultationResponse",
    "UserBehaviorRequest",
    "UserBehaviorResponse",
    "TaskClassificationRequest",
    "TaskClassificationResponse",
    "BusinessException",
    "api_exception_handler",
    "general_exception_handler",
]
