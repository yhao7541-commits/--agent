import os

from pydantic import BaseModel

from rag.local_faiss_adapter import LocalKnowledgeAdapter
from rag.mcp_rag_adapter import McpRagAdapter


def search_knowledge_base(arguments: BaseModel, context) -> dict:
    query = getattr(arguments, "query", "")
    top_k = getattr(arguments, "top_k", 3)
    chunks = _build_rag_adapter().search(query, top_k=top_k)
    return {
        "chunks": chunks,
    }


def _build_rag_adapter():
    if os.getenv("RAG_BACKEND", "local").strip().lower() == "mcp":
        return McpRagAdapter()
    return LocalKnowledgeAdapter()
