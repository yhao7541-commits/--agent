from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.mcp_rag_adapter import McpRagAdapter, McpToolClient, StdioMcpToolClient  # noqa: E402


def run_check(
    *,
    query: str,
    top_k: int,
    collection: str | None,
    min_chunks: int,
    include_collections: bool,
    required_source_substring: str | None = None,
    client: McpToolClient | None = None,
) -> dict[str, Any]:
    effective_collection = collection if collection is not None else os.getenv("RAG_MCP_COLLECTION") or None
    tool_name = os.getenv("RAG_MCP_TOOL", "query_knowledge_hub")
    client = client or StdioMcpToolClient.from_env()
    report: dict[str, Any] = {
        "ok": False,
        "backend": "mcp",
        "tool": tool_name,
        "collection": effective_collection,
        "query": query,
        "top_k": top_k,
        "min_chunks": min_chunks,
        "required_source_substring": required_source_substring,
        "chunk_count": 0,
        "chunks": [],
        "errors": [],
    }

    try:
        if include_collections:
            collections_result = client.call_tool("list_collections", {"include_stats": True})
            report["collections_text"] = _content_text(collections_result)
            if collections_result.get("isError"):
                report["errors"].append("MCP list_collections returned an error.")

        adapter = McpRagAdapter(
            client=client,
            tool_name=tool_name,
            collection=effective_collection,
        )
        chunks = adapter.search(query, top_k=top_k)
        report["chunks"] = chunks
        report["chunk_count"] = len(chunks)

        if len(chunks) < min_chunks:
            report["errors"].append(
                f"Expected at least {min_chunks} chunks but received {len(chunks)}."
            )
        if required_source_substring and not any(
            required_source_substring in str(chunk.get("source", "")) for chunk in chunks
        ):
            report["errors"].append(
                f"No chunk source contained required substring: {required_source_substring}."
            )

        report["ok"] = not report["errors"]
        return report
    except Exception as exc:
        return {
            **report,
            "errors": [f"MCP diagnostic failed: {type(exc).__name__}"],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the MCP-backed RAG connection.")
    parser.add_argument("--query", default="late arrival policy")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--collection")
    parser.add_argument("--min-chunks", type=int, default=1)
    parser.add_argument(
        "--require-source",
        help="Require at least one returned chunk source to contain this substring.",
    )
    parser.add_argument(
        "--skip-collections",
        action="store_true",
        help="Skip calling the MCP list_collections tool.",
    )
    args = parser.parse_args(argv)

    report = run_check(
        query=args.query,
        top_k=args.top_k,
        collection=args.collection,
        min_chunks=args.min_chunks,
        include_collections=not args.skip_collections,
        required_source_substring=args.require_source,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def _content_text(result: dict[str, Any]) -> str:
    parts = []
    for block in result.get("content", []):
        if isinstance(block, dict):
            text = block.get("text", "")
        else:
            text = getattr(block, "text", "")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


if __name__ == "__main__":
    sys.exit(main())
