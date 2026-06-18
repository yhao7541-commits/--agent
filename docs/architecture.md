# Architecture

The project now has two runtime surfaces. The original web/chat flow remains available, while `/api/operations/chat` exposes a stateful LangGraph operations runtime.

```text
Client
  -> FastAPI operations endpoint
  -> LangGraph state machine
  -> Tool Governance Gateway
  -> Booking, Knowledge, Memory, Escalation tools
  -> Trace events and eval records
```

The important engineering boundary is that the graph plans work, the gateway governs tool execution, and observability records what happened. Existing service and database layers remain in place for the legacy application surface.

Set `OPERATIONS_TRACE_STORE_PATH` to persist `/api/operations/chat` trace events to JSONL. The replay CLI can then inspect a historical run:

```bash
python -m observability.replay --trace-id <trace_id> --path data/traces.jsonl
```

## Runtime Nodes

- `initialize_turn`
- `classify_intent`
- `load_customer_context`
- `extract_booking_slots`
- `propose_memory_writes`
- `plan_tool_calls`
- `execute_tools`
- `generate_response`
- `output_policy_check`
- `finalize_turn`

This keeps intent, slot filling, memory proposals, tool planning, confirmation, response policy checks, and trace finalization visible and testable. Booking slot provenance is carried in state and trace metadata so user-provided, memory-derived, and system-filled values can be audited.
