from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_eval_and_operations_sections():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for text in (
        "评估结果",
        "工具治理",
        "记忆生命周期",
        "RAG grounding",
        "Trace 回放",
        "Docker 设置",
        "已知限制",
    ):
        assert text in readme


def test_engineering_docs_exist():
    for relative_path in (
        "docs/architecture.md",
        "docs/tool-governance.md",
        "docs/memory-lifecycle.md",
        "docs/evaluation.md",
        "docs/demo-script.md",
        "docs/security-policy.md",
    ):
        path = ROOT / relative_path
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("# ")
