from observability.replay import format_replay
from observability.trace_schema import TraceEvent
from observability.trace_store import JsonlTraceStore


def test_trace_event_schema_accepts_required_fields():
    event = TraceEvent(
        trace_id="trace_001",
        conversation_id="conv_001",
        node="initialize_turn",
        event_type="node_end",
        metadata={"status": "ok"},
    )

    assert event.trace_id == "trace_001"
    assert event.node == "initialize_turn"
    assert event.error is None


def test_jsonl_trace_store_writes_and_reads_trace(tmp_path):
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    event = TraceEvent(
        trace_id="trace_001",
        conversation_id="conv_001",
        node="classify_intent",
        event_type="node_end",
        metadata={"intent": "booking"},
    )

    store.append(event)
    events = store.read_trace("trace_001")

    assert len(events) == 1
    assert events[0].metadata["intent"] == "booking"


def test_format_replay_prints_ordered_node_summary():
    events = [
        TraceEvent(
            trace_id="trace_001",
            conversation_id="conv_001",
            node="initialize_turn",
            event_type="node_end",
            latency_ms=3,
            metadata={"status": "ok"},
        ),
        TraceEvent(
            trace_id="trace_001",
            conversation_id="conv_001",
            node="classify_intent",
            event_type="node_end",
            latency_ms=12,
            metadata={"intent": "booking"},
        ),
    ]

    output = format_replay(events)

    assert "Trace: trace_001" in output
    assert "1. initialize_turn: ok, 3ms" in output
    assert "2. classify_intent: booking, 12ms" in output
