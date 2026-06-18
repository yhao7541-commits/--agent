from pathlib import Path

from scripts.seed_demo_data import build_demo_seed


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_fastapi_on_port_8000():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "python:" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "uvicorn" in dockerfile
    assert "app:app" in dockerfile


def test_docker_compose_uses_env_example_and_data_volume():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert ".env.example" in compose
    assert "8000:8000" in compose
    assert "wellness-data" in compose


def test_seed_demo_data_shape():
    seed = build_demo_seed()

    assert seed["services"]
    assert seed["staff"]
    assert seed["knowledge_documents"]
