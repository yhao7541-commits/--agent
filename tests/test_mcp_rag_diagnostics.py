import pytest
import subprocess
import sys

from scripts.check_mcp_rag import run_check


class FakeMcpClient:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or {}
        self.error = error
        self.calls = []

    def call_tool(self, tool_name: str, arguments: dict):
        self.calls.append({"tool_name": tool_name, "arguments": arguments})
        if self.error:
            raise self.error
        return self.results.get(tool_name, {})


def _query_result():
    return {
        "isError": False,
        "content": [
            {
                "type": "text",
                "text": (
                    "**References (JSON):**\n```json\n"
                    '{"citations":[{"chunk_id":"wellness:001","source":"booking_policy.md",'
                    '"score":0.92,"text_snippet":"late arrival policy"}]}\n```'
                ),
            }
        ],
    }


def test_run_check_lists_collections_and_reports_chunks():
    client = FakeMcpClient(
        {
            "list_collections": {
                "isError": False,
                "content": [
                    {
                        "type": "text",
                        "text": "## Available Collections\n\n1. **knowledge_hub** - 46 documents",
                    }
                ],
            },
            "query_knowledge_hub": _query_result(),
        }
    )

    report = run_check(
        query="late arrival policy",
        top_k=2,
        collection="knowledge_hub",
        min_chunks=1,
        include_collections=True,
        client=client,
    )

    assert report["ok"] is True
    assert report["collection"] == "knowledge_hub"
    assert report["chunk_count"] == 1
    assert report["chunks"][0]["source"] == "booking_policy.md"
    assert report["collections_text"] == "## Available Collections\n\n1. **knowledge_hub** - 46 documents"
    assert client.calls == [
        {"tool_name": "list_collections", "arguments": {"include_stats": True}},
        {
            "tool_name": "query_knowledge_hub",
            "arguments": {
                "query": "late arrival policy",
                "top_k": 2,
                "collection": "knowledge_hub",
            },
        },
    ]


def test_run_check_fails_when_min_chunks_not_met():
    client = FakeMcpClient({"query_knowledge_hub": {"isError": False, "content": []}})

    report = run_check(
        query="late arrival policy",
        top_k=2,
        collection="empty",
        min_chunks=1,
        include_collections=False,
        client=client,
    )

    assert report["ok"] is False
    assert report["chunk_count"] == 0
    assert report["errors"] == ["Expected at least 1 chunks but received 0."]


def test_run_check_can_require_source_substring():
    client = FakeMcpClient({"query_knowledge_hub": _query_result()})

    report = run_check(
        query="late arrival policy",
        top_k=2,
        collection="knowledge_hub",
        min_chunks=1,
        include_collections=False,
        required_source_substring="docs/knowledge",
        client=client,
    )

    assert report["ok"] is False
    assert report["errors"] == ["No chunk source contained required substring: docs/knowledge."]


def test_run_check_returns_controlled_error_when_mcp_client_fails():
    client = FakeMcpClient(error=RuntimeError("server unavailable"))

    report = run_check(
        query="late arrival policy",
        top_k=2,
        collection=None,
        min_chunks=1,
        include_collections=True,
        required_source_substring=None,
        client=client,
    )

    assert report["ok"] is False
    assert report["chunk_count"] == 0
    assert report["errors"] == ["MCP diagnostic failed: RuntimeError"]
    assert "server unavailable" not in str(report)


@pytest.mark.parametrize("min_chunks", [0, 1])
def test_run_check_accepts_optional_collection(min_chunks):
    client = FakeMcpClient({"query_knowledge_hub": _query_result()})

    report = run_check(
        query="pricing",
        top_k=1,
        collection=None,
        min_chunks=min_chunks,
        include_collections=False,
        required_source_substring=None,
        client=client,
    )

    assert report["ok"] is True
    assert "collection" not in client.calls[0]["arguments"]


def test_script_entrypoint_runs_from_project_root(tmp_path):
    server_script = tmp_path / "fake_mcp_server.py"
    server_script.write_text(
        """
import json
import sys

for line in sys.stdin:
    payload = json.loads(line)
    method = payload.get("method")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": {}}), flush=True)
    elif method == "tools/call":
        result = {
            "content": [
                {
                    "type": "text",
                    "text": "**References (JSON):**\\n```json\\n"
                    "{\\"citations\\":[{\\"chunk_id\\":\\"entry:001\\","
                    "\\"source\\":\\"entry_policy.md\\",\\"score\\":0.8,"
                    "\\"text_snippet\\":\\"entrypoint policy\\"}]}\\n```",
                }
            ],
            "isError": False,
        }
        print(json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": result}), flush=True)
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_mcp_rag.py",
            "--query",
            "entrypoint policy",
            "--top-k",
            "1",
            "--min-chunks",
            "1",
            "--skip-collections",
        ],
        check=False,
        cwd=".",
        env={
            "RAG_MCP_COMMAND": sys.executable,
            "RAG_MCP_ARGS": str(server_script),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert "entry_policy.md" in completed.stdout
