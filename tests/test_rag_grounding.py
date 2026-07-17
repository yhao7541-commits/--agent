import sys

from agents.operations.graph import run_operations_turn
from rag.citation import build_citation_metadata
from rag.local_faiss_adapter import LocalKnowledgeAdapter
from rag.mcp_rag_adapter import McpRagAdapter, StdioMcpToolClient
from tools import knowledge_tools


class FakeMcpClient:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or {}
        self.error = error
        self.calls = []

    def call_tool(self, tool_name: str, arguments: dict):
        self.calls.append({"tool_name": tool_name, "arguments": arguments})
        if self.error:
            raise self.error
        return self.result


def test_local_adapter_returns_source_metadata_for_policy_query():
    adapter = LocalKnowledgeAdapter()

    chunks = adapter.search("如果我迟到20分钟会怎么样？")

    assert chunks
    assert chunks[0]["source"] == "booking_policy.md"
    assert chunks[0]["chunk_id"]
    assert chunks[0]["score"] > 0
    assert chunks[0]["text_preview"]


def test_mcp_adapter_normalizes_structured_citations_without_default_collection():
    client = FakeMcpClient(
        {
            "isError": False,
            "content": [
                {"type": "text", "text": "retrieval text"},
                {
                    "type": "text",
                    "text": (
                        "**References (JSON):**\n```json\n"
                        '{"citations":[{"chunk_id":"policy:001","source":"policy.md",'
                        '"score":0.91,"text_snippet":"late arrival policy"}],'
                        '"metadata":{"query":"late policy","result_count":1}}\n```'
                    ),
                },
            ],
        }
    )
    adapter = McpRagAdapter(client=client, tool_name="query_knowledge_hub")

    chunks = adapter.search("late policy", top_k=2)

    assert client.calls == [
        {
            "tool_name": "query_knowledge_hub",
            "arguments": {"query": "late policy", "top_k": 2},
        }
    ]
    assert chunks == [
        {
            "source": "policy.md",
            "chunk_id": "policy:001",
            "score": 0.91,
            "text_preview": "late arrival policy",
        }
    ]


def test_mcp_adapter_passes_collection_only_when_configured():
    client = FakeMcpClient({"isError": False, "content": []})
    adapter = McpRagAdapter(
        client=client,
        tool_name="query_knowledge_hub",
        collection="wellness_ops",
    )

    adapter.search("pricing", top_k=3)

    assert client.calls[0]["arguments"] == {
        "query": "pricing",
        "top_k": 3,
        "collection": "wellness_ops",
    }


def test_mcp_adapter_returns_empty_chunks_on_mcp_error():
    adapter = McpRagAdapter(
        client=FakeMcpClient(error=RuntimeError("mcp server unavailable")),
        tool_name="query_knowledge_hub",
    )

    assert adapter.search("pricing", top_k=3) == []


def test_stdio_mcp_client_calls_query_tool_and_normalizes_response(tmp_path):
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
                    "{\\"citations\\":[{\\"chunk_id\\":\\"mcp:001\\","
                    "\\"source\\":\\"mcp_policy.md\\",\\"score\\":0.87,"
                    "\\"text_snippet\\":\\"mcp returned policy\\"}],"
                    "\\"metadata\\":{\\"result_count\\":1}}\\n```",
                }
            ],
            "isError": False,
        }
        print(json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    client = StdioMcpToolClient(
        command=sys.executable,
        args=[str(server_script)],
        timeout_seconds=3,
    )
    adapter = McpRagAdapter(client=client)

    chunks = adapter.search("late policy", top_k=1)

    assert chunks == [
        {
            "source": "mcp_policy.md",
            "chunk_id": "mcp:001",
            "score": 0.87,
            "text_preview": "mcp returned policy",
        }
    ]


def test_knowledge_tool_uses_mcp_adapter_when_backend_is_mcp(monkeypatch):
    class FakeMcpRagAdapter:
        def search(self, query: str, top_k: int = 3):
            return [
                {
                    "source": "mcp_policy.md",
                    "chunk_id": "mcp_policy:001",
                    "score": 0.88,
                    "text_preview": query,
                }
            ][:top_k]

    monkeypatch.setenv("RAG_BACKEND", "mcp")
    monkeypatch.setattr(knowledge_tools, "McpRagAdapter", FakeMcpRagAdapter)

    arguments = type("Arguments", (), {"query": "late policy", "top_k": 1})()
    result = knowledge_tools.search_knowledge_base(arguments, context=None)

    assert result["chunks"][0]["source"] == "mcp_policy.md"


def test_citation_metadata_keeps_query_and_chunks():
    chunks = [
        {
            "source": "booking_policy.md",
            "chunk_id": "booking_policy:001",
            "score": 0.9,
            "text_preview": "迟到超过15分钟可能需要改约。",
        }
    ]

    metadata = build_citation_metadata("迟到怎么办", chunks)

    assert metadata["rag_used"] is True
    assert metadata["query"] == "迟到怎么办"
    assert metadata["chunks"] == chunks


def test_policy_question_records_rag_citations_in_trace():
    result = run_operations_turn(
        {
            "user_id": "user_001",
            "conversation_id": "conv_001",
            "message": "如果我迟到20分钟会怎么样？",
        }
    )

    rag_events = [
        event
        for event in result["trace_events"]
        if event["event_type"] == "rag_retrieval_completed"
    ]

    assert result["rag_used"] is True
    assert result["retrieved_knowledge"]
    assert rag_events
    assert rag_events[-1]["metadata"]["rag_used"] is True
    assert rag_events[-1]["metadata"]["chunks"][0]["source"] == "booking_policy.md"


def test_confirmed_booking_does_not_trigger_rag():
    pending = run_operations_turn(
        {
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "我想明天下午3点约肩颈放松",
        }
    )

    result = run_operations_turn(
        {
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "确认",
            "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
        }
    )

    assert result["rag_used"] is False
    assert all(event["event_type"] != "rag_retrieval_completed" for event in result["trace_events"])


def test_memory_preference_does_not_trigger_rag():
    result = run_operations_turn(
        {
            "user_id": "user_003",
            "conversation_id": "conv_003",
            "message": "我以后都喜欢安静一点的房间",
        }
    )

    assert result["intent"] == "memory"
    assert result["rag_used"] is False
    assert all(event["event_type"] != "rag_retrieval_completed" for event in result["trace_events"])


def test_knowledge_gap_returns_insufficient_information_reply():
    result = run_operations_turn(
        {
            "user_id": "user_004",
            "conversation_id": "conv_004",
            "message": "你们有没有火星移民按摩套餐政策？",
        }
    )

    assert result["intent"] == "consultation"
    assert result["rag_used"] is True
    assert result["retrieved_knowledge"] == []
    assert "知识库" in result["reply"]
    assert "不足" in result["reply"]
