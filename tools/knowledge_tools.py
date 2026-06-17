from pydantic import BaseModel


def search_knowledge_base(arguments: BaseModel, context) -> dict:
    query = getattr(arguments, "query", "")
    return {
        "chunks": [
            {
                "source": "booking_policy.md",
                "chunk_id": "booking_policy:001",
                "score": 0.82,
                "text_preview": f"与 '{query}' 相关的门店预约和迟到政策说明。",
            }
        ]
    }
