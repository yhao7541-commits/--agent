from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_agent_modules_are_removed():
    obsolete_paths = [
        "agents/appointment_agent.py",
        "agents/consultant_agent.py",
        "agents/task_classification_agent.py",
        "agents/user_behavior_agent.py",
        "agents/appointment",
        "agents/consultant",
        "agents/task_classification",
        "agents/user_behavior",
        "tests/test_appointment_agent.py",
        "tests/test_consultant_agent.py",
        "tests/test_task_classification_agent.py",
        "tests/test_user_behavior_agent.py",
    ]

    leftovers = [path for path in obsolete_paths if (ROOT / path).exists()]

    assert leftovers == []


def test_compatibility_apis_do_not_import_legacy_agents():
    checked_files = [
        "agents/__init__.py",
        "api/appointment.py",
        "api/consultation.py",
        "api/task.py",
        "api/chat_handler.py",
        "api/user_behavior_analysis.py",
    ]
    forbidden_terms = [
        "AppointmentAgent",
        "ConsultantAgent",
        "TaskClassificationAgent",
        "UserBehaviorAgent",
        "appointment_agent",
        "consultant_agent",
        "task_classification_agent",
        "user_behavior_agent",
    ]

    matches = []
    for relative_path in checked_files:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in text:
                matches.append(f"{relative_path}: {term}")

    assert matches == []


def test_operations_api_uses_operations_agent_facade():
    text = (ROOT / "api/operations.py").read_text(encoding="utf-8")

    assert "OperationsAgent" in text
    assert "from agents.operations.graph import run_operations_turn" not in text


def test_main_docs_describe_single_agent_tool_orchestration():
    checked_files = [
        "README.md",
        "docs/architecture.md",
        "PROJECT_LEARNING_NOTES.md",
        ".github/skills/interview-prep/SKILL.md",
        ".github/skills/resume-writer/SKILL.md",
    ]
    forbidden_phrases = [
        "多 Agent 协作架构",
        "多智能体协作",
        "TaskClassificationAgent -> AppointmentAgent",
        "任务分类智能体、咨询智能体、预约智能体和用户行为智能体",
    ]

    missing = []
    stale = []
    for relative_path in checked_files:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        if "单一 Operations Agent" not in text:
            missing.append(relative_path)
        if "Tool Gateway" not in text:
            missing.append(f"{relative_path}: Tool Gateway")
        for phrase in forbidden_phrases:
            if phrase in text:
                stale.append(f"{relative_path}: {phrase}")

    assert missing == []
    assert stale == []
