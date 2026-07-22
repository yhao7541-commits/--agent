"""Security helpers, including a deliberately process-local confirmation registry.

The confirmation registry is safe only inside one Python process. Multi-process
deployments must replace it with shared storage that supports atomic consumption.
"""

from __future__ import annotations

import base64
import hmac
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable


CONFIRMATION_TOKEN_SECRET = b"wellness-operations-confirmation-v2"
CONFIRMATION_TOKEN_TTL_SECONDS = 300
MAX_CONFIRMATION_DRAFTS = 10_000
MAX_CONFIRMATION_ID_BYTES = 256
MAX_CONFIRMATION_TOOL_NAME_BYTES = 128
MAX_CONFIRMATION_NONCE_LENGTH = 128
MAX_CONFIRMATION_TOKEN_LENGTH = 2_048
CONFIRMATION_PAYLOAD_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
CONFIRMATION_SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
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


@dataclass(frozen=True)
class _ConfirmationDraft:
    nonce: str
    issued_at: float
    expires_at: float
    conversation_id: str
    user_id: str
    tool_name: str
    arguments_hash: str


# Process-local by design for this milestone. A multi-process deployment must
# replace this registry with shared atomic storage before enabling confirmations.
_confirmation_drafts: dict[str, _ConfirmationDraft] = {}
_active_confirmation_nonces: dict[tuple[str, str, str, str], str] = {}
_confirmation_lock = threading.RLock()
_confirmation_clock: Callable[[], float] = time.time


def _default_confirmation_nonce() -> str:
    return secrets.token_urlsafe(24)


_confirmation_nonce_factory: Callable[[], str] = _default_confirmation_nonce


def detect_prompt_injection(message: str) -> bool:
    return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS)


def build_confirmation_token(
    conversation_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    user_id: str = "local_user",
) -> str:
    with _confirmation_lock:
        _validate_signing_field(
            "conversation_id",
            conversation_id,
            MAX_CONFIRMATION_ID_BYTES,
        )
        _validate_signing_field("user_id", user_id, MAX_CONFIRMATION_ID_BYTES)
        _validate_signing_field(
            "tool_name",
            tool_name,
            MAX_CONFIRMATION_TOOL_NAME_BYTES,
        )
        now = _confirmation_clock()
        _cleanup_confirmation_drafts(now)
        nonce = _confirmation_nonce_factory()
        _validate_signing_field("nonce", nonce, MAX_CONFIRMATION_NONCE_LENGTH)
        if nonce in _confirmation_drafts:
            raise RuntimeError("Confirmation nonce factory returned a duplicate nonce.")
        arguments_hash = _arguments_hash(arguments)
        binding = (user_id, conversation_id, tool_name, arguments_hash)
        previous_nonce = _active_confirmation_nonces.get(binding)
        if previous_nonce is not None:
            _remove_confirmation_draft(previous_nonce)
        draft = _ConfirmationDraft(
            nonce=nonce,
            issued_at=now,
            expires_at=now + CONFIRMATION_TOKEN_TTL_SECONDS,
            conversation_id=conversation_id,
            user_id=user_id,
            tool_name=tool_name,
            arguments_hash=arguments_hash,
        )
        if len(_confirmation_drafts) >= MAX_CONFIRMATION_DRAFTS:
            oldest_nonce = min(
                _confirmation_drafts,
                key=lambda key: _confirmation_drafts[key].issued_at,
            )
            _remove_confirmation_draft(oldest_nonce)
        _confirmation_drafts[nonce] = draft
        _active_confirmation_nonces[binding] = nonce
        return _encode_confirmation_token(draft)


def is_valid_confirmation_token(
    conversation_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    token: str | None,
    *,
    user_id: str = "local_user",
) -> bool:
    with _confirmation_lock:
        now = _confirmation_clock()
        _cleanup_confirmation_drafts(now)
        draft = _validated_confirmation_draft(
            conversation_id,
            user_id,
            tool_name,
            arguments,
            token,
            now,
        )
        return draft is not None


def consume_confirmation_token(
    conversation_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    token: str | None,
    *,
    user_id: str = "local_user",
) -> bool:
    """Atomically validate and consume one process-local confirmation draft."""
    with _confirmation_lock:
        now = _confirmation_clock()
        _cleanup_confirmation_drafts(now)
        draft = _validated_confirmation_draft(
            conversation_id,
            user_id,
            tool_name,
            arguments,
            token,
            now,
        )
        if draft is None:
            return False
        _remove_confirmation_draft(draft.nonce)
        return True


def reset_confirmation_token_registry(
    *,
    clock: Callable[[], float] | None = None,
    nonce_factory: Callable[[], str] | None = None,
) -> None:
    """Reset process-local confirmation drafts and deterministic test seams."""
    global _confirmation_clock, _confirmation_nonce_factory
    with _confirmation_lock:
        _confirmation_drafts.clear()
        _active_confirmation_nonces.clear()
        _confirmation_clock = clock or time.time
        _confirmation_nonce_factory = nonce_factory or _default_confirmation_nonce


def sanitize_tool_output(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_tool_output(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_tool_output(item) for key, item in value.items() if not _is_sensitive_key(key)}
    return value


def _encode_confirmation_token(draft: _ConfirmationDraft) -> str:
    payload = _canonical_json(
        {
            "ah": draft.arguments_hash,
            "cid": draft.conversation_id,
            "exp": draft.expires_at,
            "iat": draft.issued_at,
            "nonce": draft.nonce,
            "tool": draft.tool_name,
            "uid": draft.user_id,
            "v": 2,
        }
    )
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    signed = f"v2.{encoded}"
    signature = hmac.new(
        CONFIRMATION_TOKEN_SECRET,
        signed.encode("ascii"),
        sha256,
    ).hexdigest()
    return f"{signed}.{signature}"


def _validated_confirmation_draft(
    conversation_id: str,
    user_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    token: str | None,
    now: float,
) -> _ConfirmationDraft | None:
    if not _is_valid_runtime_field(conversation_id, MAX_CONFIRMATION_ID_BYTES):
        return None
    if not _is_valid_runtime_field(user_id, MAX_CONFIRMATION_ID_BYTES):
        return None
    if not _is_valid_runtime_field(tool_name, MAX_CONFIRMATION_TOOL_NAME_BYTES):
        return None
    payload = _decode_confirmation_token(token)
    if payload is None:
        return None
    nonce = payload.get("nonce")
    if not isinstance(nonce, str):
        return None
    draft = _confirmation_drafts.get(nonce)
    if draft is None or now >= draft.expires_at:
        return None
    expected = {
        "ah": draft.arguments_hash,
        "cid": draft.conversation_id,
        "exp": draft.expires_at,
        "iat": draft.issued_at,
        "nonce": draft.nonce,
        "tool": draft.tool_name,
        "uid": draft.user_id,
        "v": 2,
    }
    if payload != expected:
        return None
    return draft if (
        conversation_id == draft.conversation_id
        and user_id == draft.user_id
        and tool_name == draft.tool_name
        and _arguments_hash(arguments) == draft.arguments_hash
    ) else None


def _decode_confirmation_token(token: str | None) -> dict[str, Any] | None:
    if (
        not isinstance(token, str)
        or len(token) > MAX_CONFIRMATION_TOKEN_LENGTH
        or not token.isascii()
    ):
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v2":
        return None
    if not CONFIRMATION_PAYLOAD_SEGMENT_PATTERN.fullmatch(parts[1]):
        return None
    if not CONFIRMATION_SIGNATURE_PATTERN.fullmatch(parts[2]):
        return None
    signed = f"{parts[0]}.{parts[1]}"
    expected_signature = hmac.new(
        CONFIRMATION_TOKEN_SECRET,
        signed.encode("ascii"),
        sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_signature, parts[2]):
        return None
    try:
        padding = "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(parts[1] + padding).decode("utf-8")
        payload = json.loads(decoded)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cleanup_confirmation_drafts(now: float) -> None:
    expired = [
        nonce
        for nonce, draft in _confirmation_drafts.items()
        if now >= draft.expires_at
    ]
    for nonce in expired:
        _remove_confirmation_draft(nonce)


def _remove_confirmation_draft(nonce: str) -> None:
    draft = _confirmation_drafts.pop(nonce, None)
    if draft is None:
        return
    binding = (
        draft.user_id,
        draft.conversation_id,
        draft.tool_name,
        draft.arguments_hash,
    )
    if _active_confirmation_nonces.get(binding) == nonce:
        _active_confirmation_nonces.pop(binding, None)


def _validate_signing_field(name: str, value: str, max_bytes: int) -> None:
    if not _is_valid_runtime_field(value, max_bytes):
        raise ValueError(f"{name} exceeds maximum length or has an invalid type.")


def _is_valid_runtime_field(value: Any, max_bytes: int) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return len(value.encode("utf-8")) <= max_bytes
    except UnicodeEncodeError:
        return False


def _arguments_hash(arguments: dict[str, Any]) -> str:
    return sha256(_canonical_json(arguments).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
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
