from __future__ import annotations


def build_citation_metadata(query: str, chunks: list[dict]) -> dict:
    return {
        "rag_used": True,
        "query": query,
        "chunks": [
            {
                "source": chunk.get("source", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": chunk.get("score", 0),
                "text_preview": chunk.get("text_preview", ""),
            }
            for chunk in chunks
        ],
    }
