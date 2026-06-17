from __future__ import annotations

import argparse

from .trace_schema import TraceEvent
from .trace_store import JsonlTraceStore


def format_replay(events: list[TraceEvent]) -> str:
    if not events:
        return "Trace: not found"

    lines = [f"Trace: {events[0].trace_id}"]
    for index, event in enumerate(events, 1):
        status = _event_status(event)
        lines.append(f"{index}. {event.node}: {status}, {event.latency_ms}ms")
    return "\n".join(lines)


def replay_trace(trace_id: str, store: JsonlTraceStore) -> str:
    return format_replay(store.read_trace(trace_id))


def _event_status(event: TraceEvent) -> str:
    if event.error:
        return event.error.get("code", "error")
    if "intent" in event.metadata:
        return str(event.metadata["intent"])
    return str(event.metadata.get("status", "ok"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an operations agent trace.")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--path", default="data/traces.jsonl")
    args = parser.parse_args()
    print(replay_trace(args.trace_id, JsonlTraceStore(args.path)))


if __name__ == "__main__":
    main()
