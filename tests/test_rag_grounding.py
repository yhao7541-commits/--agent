from agents.operations.graph import run_operations_turn
from rag.citation import build_citation_metadata
from rag.local_faiss_adapter import LocalKnowledgeAdapter


def test_local_adapter_returns_source_metadata_for_policy_query():
    adapter = LocalKnowledgeAdapter()

    chunks = adapter.search("如果我迟到20分钟会怎么样？")

    assert chunks
    assert chunks[0]["source"] == "booking_policy.md"
    assert chunks[0]["chunk_id"]
    assert chunks[0]["score"] > 0
    assert chunks[0]["text_preview"]


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
