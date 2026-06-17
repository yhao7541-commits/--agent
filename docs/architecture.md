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

## Runtime Nodes

- `initialize_turn`
- `classify_intent`
- `load_customer_context`
- `extract_booking_slots`
- `propose_memory_writes`
- `plan_tool_calls`
- `execute_tools`
- `generate_response`
- `finalize_turn`

This keeps intent, slot filling, memory proposals, tool planning, confirmation, response generation, and trace finalization visible and testable.
