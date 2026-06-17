"""
业务服务层模块

包含：
- 知识库服务
- 技师服务  
- 预约服务
- 用户行为服务
- 推荐调度服务
- 文本嵌入工具
"""

from .text_embedding import (
    embed_input,
    find_best_match_indices,
    save_technician_embeddings,
    load_technician_embeddings
)

__all__ = [
    'embed_input',
    'find_best_match_indices',
    'save_technician_embeddings',
    'load_technician_embeddings',
    'KnowledgeService',
    'TechnicianService',
    'AppointmentService',
    'UserBehaviorService',
    'RecommendationService'
]


def __getattr__(name):
    if name == 'KnowledgeService':
        from .knowledge_service import KnowledgeService
        return KnowledgeService
    if name == 'TechnicianService':
        from .technician_service import TechnicianService
        return TechnicianService
    if name == 'AppointmentService':
        from .appointment_service import AppointmentService
        return AppointmentService
    if name == 'UserBehaviorService':
        from .user_behavior_service import UserBehaviorService
        return UserBehaviorService
    if name == 'RecommendationService':
        from .recommendation_service import RecommendationService
        return RecommendationService
    raise AttributeError(f"module 'services' has no attribute {name!r}")
