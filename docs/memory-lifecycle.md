# Memory Lifecycle

Customer memory is modeled as proposals before it becomes a stored preference.

```text
conversation turn
  -> extract candidate memory
  -> classify memory type
  -> check sensitivity
  -> dedupe or detect conflict
  -> produce proposal
  -> write through governed tool after confirmation
```

The proposal schema includes type, content, evidence, confidence, sensitivity, confirmation requirement, and optional expiration.

`write_customer_preference` is the governed write boundary for memory. After Tool Gateway confirmation, it builds a `MemoryProposal`, calls `MemoryStore.upsert()`, and records memory lifecycle trace events such as `memory_written` and `memory_updated`. `delete_customer_memory` is the governed delete boundary and also requires confirmation before calling `MemoryStore.delete()`. `lookup_customer_profile` reads the same store and returns active preference content plus memory IDs as customer context.

During booking recall, `load_customer_context` records the number of loaded memories in trace metadata. If a stored preference is applied to booking slots, the graph sets `memory_used=true`, records each applied memory in `applied_customer_memories`, marks the slot source as `memory`, and returns the same fields from `/api/operations/chat`.

## Current Rules

- Clear preferences such as a quiet-room preference create proposals.
- Vague statements such as "maybe" or "whatever" do not create memory.
- Allergy and privacy-like constraints are sensitive and require confirmation.
- Duplicate memory updates the existing record.
- Conflicting memory returns a conflict result rather than silently overwriting.
- Deletes are routed through `delete_customer_memory`, require confirmation, and emit `memory_deleted` trace events.
