from pydantic import BaseModel

from rag.local_faiss_adapter import LocalKnowledgeAdapter


def search_knowledge_base(arguments: BaseModel, context) -> dict:
    query = getattr(arguments, "query", "")
    top_k = getattr(arguments, "top_k", 3)
    chunks = LocalKnowledgeAdapter().search(query, top_k=top_k)
    return {
        "chunks": chunks,
    }
