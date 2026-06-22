from __future__ import annotations

import argparse
from collections.abc import Sequence

from .trace_schema import TraceEvent
from .trace_store import JsonlTraceStore


def format_replay(events: list[TraceEvent]) -> str:
    if not events:
        return "Trace: not found"

    lines = [
        f"Trace: {events[0].trace_id}",
        f"Conversation: {events[0].conversation_id}",
        _format_trace_summary(events),
    ]
    for index, event in enumerate(events, 1):
        status = _event_status(event)
        lines.append(f"{index}. {event.node}: {status}, {event.latency_ms}ms")
    return "\n".join(lines)


def replay_trace(trace_id: str, store: JsonlTraceStore) -> str:
    return format_replay(store.read_trace(trace_id))


def _event_status(event: TraceEvent) -> str:
    if event.event_type.startswith("tool_"):
        tool_name = event.metadata.get("tool_name", "unknown_tool")
        if event.error:
            return f"{event.event_type} {tool_name} failed {event.error.get('code', 'error')}"
        return f"{event.event_type} {tool_name}"
    if event.error:
        return event.error.get("code", "error")
    if "intent" in event.metadata:
        return str(event.metadata["intent"])
    return str(event.metadata.get("status", "ok"))


def _format_trace_summary(events: list[TraceEvent]) -> str:
    error_count = sum(1 for event in events if event.error)
    tool_event_count = sum(1 for event in events if event.event_type.startswith("tool_"))
    rag_used = any(
        event.event_type == "rag_retrieval_completed" or event.metadata.get("rag_used") is True
        for event in events
    )
    escalated = any(
        event.event_type == "escalation_triggered"
        or event.metadata.get("escalated") is True
        or (
            event.event_type == "tool_executed"
            and event.metadata.get("tool_name") == "escalate_to_human"
        )
        for event in events
    )
    return (
        f"Summary: events={len(events)}, errors={error_count}, "
        f"tool_events={tool_event_count}, rag_used={str(rag_used).lower()}, "
        f"escalated={str(escalated).lower()}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay an operations agent trace.")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--path", default="data/traces.jsonl")
    args = parser.parse_args(argv)
    print(replay_trace(args.trace_id, JsonlTraceStore(args.path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
