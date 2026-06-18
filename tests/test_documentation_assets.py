from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_eval_and_operations_sections():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for text in (
        "Evaluation Results",
        "Tool governance",
        "Memory lifecycle",
        "RAG grounding",
        "Trace replay",
        "Docker setup",
        "Known limitations",
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
