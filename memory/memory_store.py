from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .customer_memory import CustomerMemory, MemoryProposal, MemoryWriteResult, utc_now
from .memory_deduper import find_conflict, find_duplicate


MEMORY_DB_PATH_ENV = "CUSTOMER_MEMORY_DB_PATH"
DEFAULT_MEMORY_DB_PATH = Path("data/customer_memory.sqlite3")
ACTIVE_STATUSES = {"active"}


class MemoryStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path) if db_path is not None else ":memory:"
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize_schema()

    @classmethod
    def from_env(cls) -> "MemoryStore":
        return cls(os.getenv(MEMORY_DB_PATH_ENV, str(DEFAULT_MEMORY_DB_PATH)))

    def list_user_memories(
        self,
        user_id: str,
        *,
        include_inactive: bool = False,
        include_deleted: bool = False,
    ) -> list[CustomerMemory]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM customer_memories
                WHERE user_id = ?
                ORDER BY created_at ASC
                """,
                (user_id,),
            ).fetchall()

        memories = [self._memory_from_row(row) for row in rows]
        visible_memories: list[CustomerMemory] = []
        for memory in memories:
            memory = self._with_expiration_status(memory)
            if memory.deleted_at is not None and not include_deleted:
                continue
            if not include_inactive and not self._is_active(memory):
                continue
            visible_memories.append(memory)
        return visible_memories

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._connection.close()

    def get_memory(
        self,
        user_id: str,
        memory_id: str,
        *,
        include_deleted: bool = False,
    ) -> CustomerMemory | None:
        with self._lock:
            memory = self._get_memory(user_id, memory_id)
        if memory is None:
            return None
        memory = self._with_expiration_status(memory)
        if memory.deleted_at is not None and not include_deleted:
            return None
        return memory

    def upsert(
        self,
        user_id: str,
        proposal: MemoryProposal,
        trace_id: str = "",
        conversation_id: str = "",
        trace_events: list[dict[str, Any]] | None = None,
    ) -> MemoryWriteResult:
        with self._lock:
            existing_memories = self.list_user_memories(
                user_id,
                include_inactive=True,
                include_deleted=False,
            )
            duplicate = find_duplicate(proposal, existing_memories)
            if duplicate:
                previous = duplicate.model_dump(mode="json")
                duplicate.evidence = proposal.evidence
                duplicate.confidence = proposal.confidence
                duplicate.source_trace_id = trace_id
                duplicate.source_conversation_id = conversation_id
                duplicate.expires_at = proposal.expires_at
                duplicate.updated_at = utc_now()
                duplicate.version += 1
                self._save_memory(duplicate)
                self._record_event(
                    duplicate,
                    "memory_updated",
                    previous_value=previous,
                    new_value=duplicate.model_dump(mode="json"),
                )
                _append_trace(trace_events, trace_id, conversation_id, "memory_updated", duplicate)
                return MemoryWriteResult(action="updated", memory=duplicate)

            candidate = CustomerMemory(
                id=f"memory_{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                type=proposal.type,
                content=proposal.content,
                evidence=proposal.evidence,
                confidence=proposal.confidence,
                sensitivity=proposal.sensitivity,
                status=_initial_status(proposal),
                review_status=_initial_review_status(proposal),
                source_conversation_id=conversation_id,
                source_trace_id=trace_id,
                expires_at=proposal.expires_at,
            )
            conflict = find_conflict(proposal, existing_memories)
            if conflict:
                self._record_event(
                    candidate,
                    "memory_conflict",
                    previous_value=conflict.model_dump(mode="json"),
                    new_value=candidate.model_dump(mode="json"),
                )
                _append_trace(trace_events, trace_id, conversation_id, "memory_conflict", candidate)
                return MemoryWriteResult(action="conflict", memory=candidate, conflict_with=conflict)

            self._insert_memory(candidate)
            event_type = "memory_pending_review" if candidate.status == "pending_review" else "memory_written"
            self._record_event(candidate, event_type, new_value=candidate.model_dump(mode="json"))
            _append_trace(trace_events, trace_id, conversation_id, event_type, candidate)
            return MemoryWriteResult(action="created", memory=candidate)

    def update(
        self,
        user_id: str,
        memory_id: str,
        *,
        actor: str = "system",
        reason: str = "",
        type: str | None = None,
        content: str | None = None,
        evidence: str | None = None,
        confidence: float | None = None,
        sensitivity: str | None = None,
        expires_at: datetime | None = None,
    ) -> CustomerMemory | None:
        with self._lock:
            memory = self._get_memory(user_id, memory_id)
            if memory is None or memory.deleted_at is not None:
                return None
            previous = memory.model_dump(mode="json")
            if type is not None:
                memory.type = type
            if content is not None:
                memory.content = content
            if evidence is not None:
                memory.evidence = evidence
            if confidence is not None:
                memory.confidence = confidence
            if sensitivity is not None:
                memory.sensitivity = sensitivity
            if expires_at is not None:
                memory.expires_at = expires_at
            memory.version += 1
            memory.updated_at = utc_now()
            self._save_memory(memory)
            self._record_event(
                memory,
                "memory_updated",
                previous_value=previous,
                new_value=memory.model_dump(mode="json"),
                actor=actor,
                reason=reason,
            )
            return memory

    def approve(
        self,
        user_id: str,
        memory_id: str,
        *,
        actor: str = "system",
        reason: str = "",
    ) -> CustomerMemory | None:
        return self._set_review_state(
            user_id=user_id,
            memory_id=memory_id,
            status="active",
            review_status="approved",
            event_type="memory_approved",
            actor=actor,
            reason=reason,
        )

    def reject(
        self,
        user_id: str,
        memory_id: str,
        *,
        actor: str = "system",
        reason: str = "",
    ) -> CustomerMemory | None:
        return self._set_review_state(
            user_id=user_id,
            memory_id=memory_id,
            status="rejected",
            review_status="rejected",
            event_type="memory_rejected",
            actor=actor,
            reason=reason,
        )

    def delete(
        self,
        user_id: str,
        memory_id: str,
        trace_id: str = "",
        conversation_id: str = "",
        trace_events: list[dict[str, Any]] | None = None,
        *,
        actor: str = "system",
        reason: str = "",
    ) -> bool:
        with self._lock:
            memory = self._get_memory(user_id, memory_id)
            if memory is None or memory.deleted_at is not None:
                return False
            previous = memory.model_dump(mode="json")
            memory.deleted_at = utc_now()
            memory.updated_at = memory.deleted_at
            memory.status = "deleted"
            memory.version += 1
            self._save_memory(memory)
            self._record_event(
                memory,
                "memory_deleted",
                previous_value=previous,
                new_value=memory.model_dump(mode="json"),
                actor=actor,
                reason=reason,
            )
            _append_trace(trace_events, trace_id, conversation_id, "memory_deleted", memory)
            return True

    def list_memory_events(self, memory_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM customer_memory_events
                WHERE memory_id = ?
                ORDER BY created_at ASC
                """,
                (memory_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def _set_review_state(
        self,
        *,
        user_id: str,
        memory_id: str,
        status: str,
        review_status: str,
        event_type: str,
        actor: str,
        reason: str,
    ) -> CustomerMemory | None:
        with self._lock:
            memory = self._get_memory(user_id, memory_id)
            if memory is None or memory.deleted_at is not None:
                return None
            previous = memory.model_dump(mode="json")
            memory.status = status
            memory.review_status = review_status
            memory.updated_at = utc_now()
            memory.version += 1
            self._save_memory(memory)
            self._record_event(
                memory,
                event_type,
                previous_value=previous,
                new_value=memory.model_dump(mode="json"),
                actor=actor,
                reason=reason,
            )
            return memory

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sensitivity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    source_conversation_id TEXT NOT NULL,
                    source_trace_id TEXT NOT NULL,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    version INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_customer_memories_user_id
                ON customer_memories(user_id)
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_memory_events (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    previous_value TEXT,
                    new_value TEXT,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _insert_memory(self, memory: CustomerMemory) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO customer_memories (
                    id, user_id, type, content, evidence, confidence, sensitivity,
                    status, review_status, source_conversation_id, source_trace_id,
                    expires_at, created_at, updated_at, deleted_at, version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _memory_values(memory),
            )

    def _save_memory(self, memory: CustomerMemory) -> None:
        with self._connection:
            self._connection.execute(
                """
                UPDATE customer_memories
                SET user_id = ?, type = ?, content = ?, evidence = ?, confidence = ?,
                    sensitivity = ?, status = ?, review_status = ?,
                    source_conversation_id = ?, source_trace_id = ?, expires_at = ?,
                    created_at = ?, updated_at = ?, deleted_at = ?, version = ?
                WHERE id = ?
                """,
                (
                    memory.user_id,
                    memory.type,
                    memory.content,
                    memory.evidence,
                    memory.confidence,
                    memory.sensitivity,
                    memory.status,
                    memory.review_status,
                    memory.source_conversation_id,
                    memory.source_trace_id,
                    _datetime_to_str(memory.expires_at),
                    _datetime_to_str(memory.created_at),
                    _datetime_to_str(memory.updated_at),
                    _datetime_to_str(memory.deleted_at),
                    memory.version,
                    memory.id,
                ),
            )

    def _get_memory(self, user_id: str, memory_id: str) -> CustomerMemory | None:
        row = self._connection.execute(
            """
            SELECT * FROM customer_memories
            WHERE user_id = ? AND id = ?
            """,
            (user_id, memory_id),
        ).fetchone()
        return self._memory_from_row(row) if row else None

    def _record_event(
        self,
        memory: CustomerMemory,
        event_type: str,
        *,
        previous_value: dict[str, Any] | None = None,
        new_value: dict[str, Any] | None = None,
        actor: str = "system",
        reason: str = "",
    ) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO customer_memory_events (
                    id, memory_id, user_id, event_type, previous_value,
                    new_value, actor, reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"memory_event_{uuid.uuid4().hex[:10]}",
                    memory.id,
                    memory.user_id,
                    event_type,
                    _json_or_none(previous_value),
                    _json_or_none(new_value),
                    actor,
                    reason,
                    _datetime_to_str(utc_now()),
                ),
            )

    def _memory_from_row(self, row: sqlite3.Row) -> CustomerMemory:
        return CustomerMemory(
            id=row["id"],
            user_id=row["user_id"],
            type=row["type"],
            content=row["content"],
            evidence=row["evidence"],
            confidence=float(row["confidence"]),
            sensitivity=row["sensitivity"],
            status=row["status"],
            review_status=row["review_status"],
            source_conversation_id=row["source_conversation_id"],
            source_trace_id=row["source_trace_id"],
            expires_at=_datetime_from_str(row["expires_at"]),
            created_at=_datetime_from_str(row["created_at"]) or utc_now(),
            updated_at=_datetime_from_str(row["updated_at"]) or utc_now(),
            deleted_at=_datetime_from_str(row["deleted_at"]),
            version=int(row["version"]),
        )

    def _event_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "memory_id": row["memory_id"],
            "user_id": row["user_id"],
            "event_type": row["event_type"],
            "previous_value": _json_from_str(row["previous_value"]),
            "new_value": _json_from_str(row["new_value"]),
            "actor": row["actor"],
            "reason": row["reason"],
            "created_at": row["created_at"],
        }

    def _is_active(self, memory: CustomerMemory) -> bool:
        if memory.deleted_at is not None:
            return False
        if self._is_expired(memory):
            return False
        return memory.status in ACTIVE_STATUSES and memory.review_status == "approved"

    def _with_expiration_status(self, memory: CustomerMemory) -> CustomerMemory:
        if self._is_expired(memory) and memory.status == "active":
            memory.status = "expired"
        return memory

    def _is_expired(self, memory: CustomerMemory) -> bool:
        return memory.expires_at is not None and memory.expires_at <= utc_now()


def _initial_status(proposal: MemoryProposal) -> str:
    if proposal.sensitivity == "sensitive":
        return "pending_review"
    return "active"


def _initial_review_status(proposal: MemoryProposal) -> str:
    if proposal.sensitivity == "sensitive":
        return "pending"
    return "approved"


def _memory_values(memory: CustomerMemory) -> tuple[Any, ...]:
    return (
        memory.id,
        memory.user_id,
        memory.type,
        memory.content,
        memory.evidence,
        memory.confidence,
        memory.sensitivity,
        memory.status,
        memory.review_status,
        memory.source_conversation_id,
        memory.source_trace_id,
        _datetime_to_str(memory.expires_at),
        _datetime_to_str(memory.created_at),
        _datetime_to_str(memory.updated_at),
        _datetime_to_str(memory.deleted_at),
        memory.version,
    )


def _datetime_to_str(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _datetime_from_str(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _json_or_none(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_from_str(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return json.loads(value)


def _append_trace(
    trace_events: list[dict[str, Any]] | None,
    trace_id: str,
    conversation_id: str,
    event_type: str,
    memory: CustomerMemory,
) -> None:
    if trace_events is None:
        return
    trace_events.append(
        {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "node": "memory_store",
            "event_type": event_type,
            "timestamp": utc_now().astimezone(timezone.utc).isoformat(),
            "metadata": {
                "memory_id": memory.id,
                "memory_type": memory.type,
                "sensitivity": memory.sensitivity,
                "status": memory.status,
                "review_status": memory.review_status,
                "version": memory.version,
            },
            "error": None,
        }
    )
