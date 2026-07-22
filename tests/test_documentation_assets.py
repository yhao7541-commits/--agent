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


def test_hybrid_decision_docs_describe_verified_runtime_without_uplift_claims():
    documentation_paths = (
        "README.md",
        "docs/architecture.md",
        "docs/evaluation.md",
        "docs/demo-script.md",
    )
    documents = {
        relative_path: (ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in documentation_paths
    }

    for relative_path, text in documents.items():
        assert "混合 LLM 决策" in text, relative_path
        assert "确定性 Tool Gateway" in text, relative_path

    combined = "\n".join(documents.values())
    for required_text in (
        "共享三次调用预算",
        "规则回退",
        "LangGraph 条件路由",
        "确定性韧性演示",
        "可选的真实模型语义对比",
        "不声明准确率提升",
        "模型输出具有非确定性",
        "真实模型路径依赖凭据",
        "没有分布式熔断器",
        "没有线上业务结果证据",
    ):
        assert required_text in combined

    assert "186 条" in documents["README.md"]
    assert "186 条" in documents["docs/evaluation.md"]
