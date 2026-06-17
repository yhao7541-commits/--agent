from __future__ import annotations

import hmac
import json
import re
from hashlib import sha256
from typing import Any


CONFIRMATION_TOKEN_SECRET = b"wellness-operations-confirmation-v1"
PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+the\s+system\s+prompt",
    r"system\s+prompt",
    r"developer\s+message",
    r"忽略.*(规则|指令|系统)",
    r"覆盖.*(规则|策略|系统)",
    r"不需要确认",
    r"绕过确认",
    r"直接(创建|调用|执行)",
)


def detect_prompt_injection(message: str) -> bool:
    return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS)


def build_confirmation_token(
    conversation_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    payload = _confirmation_payload(conversation_id, tool_name, arguments)
    return hmac.new(CONFIRMATION_TOKEN_SECRET, payload.encode("utf-8"), sha256).hexdigest()


def is_valid_confirmation_token(
    conversation_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    token: str | None,
) -> bool:
    if not token:
        return False
    expected = build_confirmation_token(conversation_id, tool_name, arguments)
    return hmac.compare_digest(expected, token)


def sanitize_tool_output(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_tool_output(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_tool_output(item) for key, item in value.items() if not _is_sensitive_key(key)}
    return value


def _confirmation_payload(
    conversation_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "conversation_id": conversation_id,
            "tool_name": tool_name,
            "arguments": arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sanitize_text(value: str) -> str:
    if detect_prompt_injection(value):
        return "[redacted_tool_instruction]"
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in {"system_prompt", "developer_message", "hidden_prompt", "raw_prompt"}
