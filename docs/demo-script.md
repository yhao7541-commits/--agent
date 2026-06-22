# Demo Script

Use `/api/operations/chat` or the `/operations` console for these demos.

## 1. Incomplete Booking

User: `我想约一个肩颈放松`

Expected: intent is booking, missing date and time, no `create_booking` execution.

## 2. Complete Booking Requires Confirmation

User: `我想明天下午3点约肩颈放松`

Expected: the agent plans read tools and `create_booking`, but the gateway returns a confirmation request before any write succeeds.

More realistic time expressions:

- `我想明天下午3点半约肩颈放松` -> `time_window=15:30`
- `我想后天上午10点约推拿` -> date is normalized to the day after tomorrow
- `我想下周五晚上7点约按摩` -> date is normalized to next week's Friday

## 3. Confirmed Booking Executes Write

Send the previous `confirmation_request.tool_name` and `confirmation_request.arguments` back with message `确认`.

Expected: `create_booking` succeeds and the reply includes a booking id.

## 4. Policy Question Uses RAG

User: `如果我迟到20分钟会怎么样？`

Expected: `search_knowledge_base` runs and trace metadata includes source chunks from `booking_policy.md`.

Local deterministic mode:

```powershell
$env:RAG_BACKEND="local"
```

MCP-backed mode:

```powershell
$env:RAG_BACKEND="mcp"
$env:RAG_MCP_COMMAND="python"
$env:RAG_MCP_ARGS="-m src.mcp_server.server"
$env:RAG_MCP_CWD="D:\Dev\RAG\MODULAR-RAG-MCP-SERVER"
$env:RAG_MCP_COLLECTION="knowledge_hub"
python scripts/check_mcp_rag.py --collection knowledge_hub --query "late arrival policy" --min-chunks 1
```

Expected for the diagnostic: `ok=true`, `chunk_count >= 1`, and `chunks[].source` shows which external documents were used. If the source is not a wellness document, the MCP link is healthy but the collection needs domain-aligned content before a polished demo.

Optional domain gate:

```powershell
python scripts/check_mcp_rag.py --collection knowledge_hub --query "late arrival policy" --min-chunks 1 --require-source docs/knowledge
```

Expected: this command exits non-zero until the external MCP collection contains wellness knowledge sources.

## 5. Preference Creates Memory Proposal

User: `我以后都喜欢安静一点的房间`

Expected: a memory proposal is produced and `write_customer_preference` requires confirmation.

## 6. Trace Replay

Set a trace path before running the operations request:

```powershell
$env:OPERATIONS_TRACE_STORE_PATH="data/traces.jsonl"
```

After the API response returns a `trace_id`, run:

```powershell
python -m observability.replay --trace-id <trace_id> --path data/traces.jsonl
```

Expected: replay shows the node sequence, confirmation interception, tool calls, RAG retrieval events, and final response summary in execution order.
