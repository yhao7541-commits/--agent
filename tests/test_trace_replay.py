from observability.replay import format_replay, main
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


def test_format_replay_includes_tool_failure_details():
    events = [
        TraceEvent(
            trace_id="trace_001",
            conversation_id="conv_001",
            node="tool_gateway",
            event_type="tool_error",
            metadata={"tool_name": "create_booking"},
            error={"code": "validation_error", "message": "Tool arguments failed schema validation."},
        )
    ]

    output = format_replay(events)

    assert "tool_gateway: tool_error create_booking failed validation_error" in output


def test_replay_cli_reads_jsonl_store(tmp_path, capsys):
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    store.append(
        TraceEvent(
            trace_id="trace_cli_001",
            conversation_id="conv_cli_001",
            node="classify_intent",
            event_type="node_end",
            metadata={"intent": "booking"},
        )
    )

    exit_code = main(["--trace-id", "trace_cli_001", "--path", str(store.path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Trace: trace_cli_001" in captured.out
    assert "classify_intent: booking" in captured.out
