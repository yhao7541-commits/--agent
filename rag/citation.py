from __future__ import annotations

from security.guardrails import sanitize_tool_output


def build_citation_metadata(query: str, chunks: list[dict]) -> dict:
    return {
        "rag_used": True,
        "query": query,
        "chunks": [
            {
                "source": chunk.get("source", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": chunk.get("score", 0),
                "text_preview": sanitize_tool_output(chunk.get("text_preview", "")),
            }
            for chunk in chunks
        ],
    }
