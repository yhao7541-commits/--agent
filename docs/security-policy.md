# Security Policy

The current operations runtime follows conservative guardrails:

- Write tools cannot execute without explicit confirmation.
- Sensitive memory writes and deletes require confirmation.
- Unknown tools return structured errors.
- Invalid tool arguments do not call handlers.
- RAG chunks are treated as source content, not instructions.
- API responses do not expose raw prompts.
- Trace events should be reviewed before sharing externally because they can contain user-provided text.

## Escalation Triggers

The runtime escalates low-confidence requests, medical or safety concerns, refund disputes, complaints, repeated booking tool failures, and other cases where automation should not decide alone.
