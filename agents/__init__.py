from config.constants import SharedState, StateEnum

__all__ = [
    'AppointmentAgent',
    'ConsultantAgent', 
    'TaskClassificationAgent',
    'UserBehaviorAgent',
    'SharedState',
    'StateEnum'
]


def __getattr__(name):
    if name == 'AppointmentAgent':
        from .appointment_agent import AppointmentAgent
        return AppointmentAgent
    if name == 'ConsultantAgent':
        from .consultant_agent import ConsultantAgent
        return ConsultantAgent
    if name == 'TaskClassificationAgent':
        from .task_classification_agent import TaskClassificationAgent
        return TaskClassificationAgent
    if name == 'UserBehaviorAgent':
        from .user_behavior_agent import UserBehaviorAgent
        return UserBehaviorAgent
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
