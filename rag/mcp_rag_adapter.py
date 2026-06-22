from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from typing import Any, Protocol


class McpToolClient(Protocol):
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool and return the raw JSON-RPC result payload."""


class StdioMcpToolClient:
    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "StdioMcpToolClient":
        command = os.getenv("RAG_MCP_COMMAND")
        if not command:
            raise RuntimeError("RAG_MCP_COMMAND is required when RAG_BACKEND=mcp.")
        return cls(
            command=command,
            args=_split_args(os.getenv("RAG_MCP_ARGS", "")),
            cwd=os.getenv("RAG_MCP_CWD") or None,
            timeout_seconds=float(os.getenv("RAG_MCP_TIMEOUT_SECONDS", "10")),
        )

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        proc = subprocess.Popen(
            [self.command, *self.args],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        try:
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "clientInfo": {
                            "name": "wellness-operations-agent",
                            "version": "0.1.0",
                        },
                        "capabilities": {},
                    },
                },
            )
            self._read_response(proc, 1)
            self._send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                },
            )
            response = self._read_response(proc, 2)
            if "error" in response:
                raise RuntimeError("MCP tool call failed.")
            return response.get("result", {})
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    def _send(self, proc: subprocess.Popen, payload: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise RuntimeError("MCP process stdin is unavailable.")
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    def _read_response(self, proc: subprocess.Popen, request_id: int) -> dict[str, Any]:
        if proc.stdout is None:
            raise RuntimeError("MCP process stdout is unavailable.")
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") == request_id:
                return payload
        raise TimeoutError(f"MCP response {request_id} timed out.")


class McpRagAdapter:
    def __init__(
        self,
        client: McpToolClient | None = None,
        tool_name: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.client = client
        self.tool_name = tool_name or os.getenv("RAG_MCP_TOOL", "query_knowledge_hub")
        self.collection = collection if collection is not None else os.getenv("RAG_MCP_COLLECTION")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        arguments: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
        }
        if self.collection:
            arguments["collection"] = self.collection

        try:
            result = self._client().call_tool(self.tool_name, arguments)
        except Exception:
            return []
        if result.get("isError"):
            return []
        return _chunks_from_mcp_result(result)

    def _client(self) -> McpToolClient:
        if self.client is None:
            self.client = StdioMcpToolClient.from_env()
        return self.client


def _chunks_from_mcp_result(result: dict[str, Any]) -> list[dict]:
    structured = _extract_structured_payload(result.get("content", []))
    chunks = []
    for index, citation in enumerate(structured.get("citations", []), 1):
        chunks.append(
            {
                "source": citation.get("source", "unknown"),
                "chunk_id": citation.get("chunk_id") or f"mcp:{index}",
                "score": float(citation.get("score", 0.0) or 0.0),
                "text_preview": citation.get("text_snippet", ""),
            }
        )
    return chunks


def _extract_structured_payload(content_blocks: list[Any]) -> dict[str, Any]:
    for block in content_blocks:
        text = _block_text(block)
        if not text:
            continue
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    return {}


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("text", ""))
    return str(getattr(block, "text", ""))


def _split_args(value: str) -> list[str]:
    if not value.strip():
        return []
    return shlex.split(value, posix=os.name != "nt")
