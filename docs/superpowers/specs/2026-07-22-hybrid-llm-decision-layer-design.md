# Hybrid LLM Decision Layer Design

**Date:** 2026-07-22  
**Status:** User-approved design, pending written-spec review  
**Target delivery window:** Three days  
**Primary audience:** AI product manager and AI application / Agent engineering interviews

## 1. Summary

The current Operations Agent has strong deterministic governance boundaries, but its main graph is linear and its intent, slot extraction, and response decisions are mostly rule-based. This upgrade adds a real-model decision layer without weakening the existing Tool Gateway, confirmation token, booking conflict, memory review, RAG citation, trace, or evaluation controls.

The result is a hybrid runtime:

```text
deterministic input safety
  -> structured LLM decision
  -> schema and business validation
  -> LangGraph conditional routing
  -> deterministic tool planning and execution
  -> output policy
  -> trace and comparative evaluation
```

The LLM may interpret and recommend. It may not directly execute tools, create confirmation tokens, override business validators, or claim that a side effect succeeded.

## 2. Goals

1. Add real LLM-based intent and slot understanding for long-tail language.
2. Replace the current all-linear graph with explicit conditional routing.
3. Keep side effects governed by deterministic code.
4. Recover from transient provider failures and malformed model output within a bounded call and latency budget.
5. Expose enough decision metadata to diagnose model behavior, latency, token usage, repair, and fallback.
6. Compare the hybrid runtime against the existing rule baseline on the same long-tail dataset.
7. Preserve existing API and page contracts while adding optional diagnostic fields.
8. Produce a visibly deeper but defensible interview story within three days.

## 3. Non-goals

The first milestone will not:

- split the runtime into multiple agents;
- allow the LLM to call tools directly;
- implement distributed queues or Redis-backed circuit breakers;
- build production RBAC, approval assignment, or compliance reporting;
- create a prompt-management platform;
- replace the current database or vector store;
- refactor unrelated legacy modules;
- claim online accuracy, conversion, labor savings, or other unmeasured business outcomes.

## 4. Current-state constraints

- `OperationsAgent` is a thin facade over `run_operations_turn`.
- `graph.py` currently executes ten nodes in one fixed sequence.
- `classify_intent` and `extract_booking_slots` are deterministic keyword / regex implementations.
- `ToolGateway` already provides schema validation, confirmation checks, bounded read retries, timeout handling, output sanitization, and trace events.
- Confirmation tokens bind conversation, tool name, and exact arguments.
- Sensitive memory remains pending until operational approval.
- The smoke harness currently covers 184 deterministic cases.
- Existing compatibility APIs and pages must continue working.
- The working tree contains a large in-progress single-agent migration; implementation must be surgical and must not overwrite unrelated changes.

## 5. Proposed architecture

```text
FastAPI request
  -> initialize_turn
  -> validate_confirmation
       | valid confirmed action -> confirmed execution path
       | invalid confirmation -> forced escalation
       | ordinary request -> input_guardrail
  -> input_guardrail
       | hard safety match -> escalation path
       | ordinary request -> decide_request
  -> decide_request (HybridDecisionEngine)
       | structured LLM result
       | rule fallback result
       | forced escalation result
  -> route_decision (conditional edges)
       | booking / reschedule / cancel
       | consultation
       | memory / delete_memory
       | greeting
       | clarification / escalation
  -> deterministic business nodes
  -> deterministic Tool Planner
  -> Tool Gateway
  -> output policy check
  -> finalize_turn
```

### 5.1 Responsibility boundaries

| Component | Owns | Must not own |
| --- | --- | --- |
| Input guardrail | hard security and safety rules | free-form business interpretation |
| HybridDecisionEngine | model call, structured parse, retry, repair, fallback | direct tool execution |
| Decision validators | schema, enum, date/time, service and ambiguity validation | provider transport |
| LangGraph router | selecting the next business branch | tool permissions |
| Tool Planner | mapping validated state to allowlisted tools | model transport or confirmation token creation |
| Tool Gateway | schema, permission, confirmation, timeout, retry, output sanitization, trace | intent interpretation |
| Output policy | preventing false success and unsafe replies | hidden model reasoning |

## 6. Decision contracts

### 6.1 Model-owned structured output

The model returns a schema equivalent to:

```python
class BookingSlotCandidates(BaseModel):
    service_type: str | None = None
    date: str | None = None
    time_window: str | None = None
    duration: str | None = None
    preferred_staff: str | None = None
    special_requests: str | None = None
    booking_id: str | None = None


class ModelDecision(BaseModel):
    intent: Literal[
        "booking",
        "reschedule",
        "cancel",
        "consultation",
        "memory",
        "delete_memory",
        "greeting",
        "clarification",
        "escalation",
        "unknown",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_slots: BookingSlotCandidates
    ambiguities: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    suggested_action: str
    decision_summary: str
```

`decision_summary` is a short operational explanation, not hidden chain-of-thought. It must be length-limited and must not request or reveal system prompts.

The model's confidence is self-reported and is not treated as calibrated probability. It is a routing signal that must be evaluated empirically.

### 6.2 System-owned decision metadata

The runtime, not the model, adds:

```python
class DecisionMetadata(BaseModel):
    source: Literal["llm", "rule_fallback", "forced_escalation"]
    provider: str | None = None
    model: str | None = None
    attempt_count: int = 0
    repair_count: int = 0
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    fallback_reason: str | None = None
```

Provider and token fields are best effort. A provider that does not expose usage metadata returns `null`; the runtime must not invent values.

### 6.3 State additions

`OperationsAgentState` gains optional fields:

- `model_decision`
- `decision_metadata`
- `ambiguities`
- `decision_source`
- `decision_errors`
- `decision_route`

All additions are optional to preserve compatibility with existing callers and tests.

## 7. Model prompt and data minimization

The decision prompt includes only:

- the current user message;
- previously accepted booking slots and their provenance;
- allowed intent values and route descriptions;
- the current local date and timezone;
- the structured output schema;
- concise safety and ambiguity instructions.

The prompt excludes:

- API keys and secrets;
- raw database rows;
- unrelated conversation history;
- internal exception stacks;
- confirmation-token secrets;
- hidden prompts from other tools or systems.

The production adapter uses the existing model-provider configuration. Tests use a programmable fake client and never require a live API key.

## 8. Retry, repair, and deadline policy

### 8.1 Unified call budget

Each decision has a maximum of three model calls total. Timeout retries and malformed-output repair share this budget.

Example:

```text
call 1 -> timeout
call 2 -> invalid JSON
call 3 -> repaired valid decision
```

No fourth decision call occurs.

Default configuration:

```env
LLM_DECISION_MAX_ATTEMPTS=3
LLM_DECISION_TIMEOUT_SECONDS=10
LLM_DECISION_TOTAL_DEADLINE_SECONDS=25
LLM_DECISION_MIN_CONFIDENCE=0.65
```

### 8.2 Retryable failures

Retry with short exponential backoff and jitter for:

- provider timeout;
- HTTP 429;
- provider 5xx;
- retryable transport failure.

Do not retry:

- invalid API key;
- authorization / permission failure;
- unsupported model or deterministic configuration error;
- a request rejected by local safety policy.

### 8.3 JSON and schema repair

When parsing or Pydantic validation fails, the next call receives:

- the original decision task;
- the target schema;
- a sanitized, field-level validation-error summary;
- an instruction to regenerate the complete JSON object.

The repair prompt must not contain secrets, a full Python stack, or hidden runtime instructions. Repair consumes the same three-call budget.

### 8.4 Budget exhaustion

When attempts or the overall deadline are exhausted:

1. record a structured fallback reason;
2. run the existing rule classifier and slot extractor through a stable fallback adapter;
3. mark `source=rule_fallback`;
4. route to clarification or escalation if fallback confidence remains insufficient.

## 9. Error taxonomy

Decision errors use stable codes:

- `provider_timeout`
- `rate_limited`
- `provider_5xx`
- `transport_error`
- `authentication_error`
- `configuration_error`
- `invalid_json`
- `schema_validation_error`
- `low_confidence`
- `business_validation_error`
- `total_deadline_exceeded`

User-facing replies remain concise and do not expose provider internals.

## 10. Confirmation and safety fast paths

### 10.1 Confirmed action

A request containing confirmation metadata is validated before the LLM is called.

- Valid token: execute only the exact confirmed tool and arguments.
- Invalid token: do not call the write tool; create a security escalation.
- Rejected confirmation: clear the pending request and report that no write occurred.

The model never reinterprets or changes an already-confirmed action.

### 10.2 Hard input guardrail

Deterministic checks run before model interpretation for:

- prompt-injection patterns;
- instructions to bypass confirmation;
- medical injury or urgent safety concerns;
- serious refund disputes or complaints;
- malformed confirmed-action fields.

These conditions may force escalation. The LLM cannot lower their severity.

## 11. Slot merge and business validation

Slot precedence is:

```text
confirmed tool arguments
  > current user message extraction
  > accepted previous-turn slots
  > active and approved customer memory
  > system defaults
```

Rules:

- Current explicit user input may override previous state and memory.
- Memory fills missing values only.
- System defaults may not overwrite a user value.
- Every accepted slot retains provenance.
- Date, time, duration, service, staff, and booking identifiers are normalized and validated after model output.
- Unsupported or contradictory values become `ambiguities`; they do not silently become write arguments.
- Multiple simultaneous business intents route to clarification in the first milestone unless a deterministic priority rule can safely handle them.

## 12. Conditional graph behavior

### 12.1 Booking branch

```text
load customer context
  -> merge and validate slots
  -> missing or ambiguous fields? ask clarification
  -> plan customer lookup / staff availability / schedule check
  -> booking issue? return alternatives
  -> create confirmation request
  -> confirmed request on later turn executes exact write
```

### 12.2 Consultation branch

```text
plan knowledge search
  -> execute RAG tool
  -> validate non-empty source metadata
  -> produce the existing grounded response contract
```

Free-form LLM answer generation is not added to transactional responses in this milestone. A later milestone may add grounded consultation generation behind citations and an independent output policy.

### 12.3 Memory branch

```text
create MemoryProposal
  -> ask user confirmation
  -> Tool Gateway write
  -> sensitive memory remains pending review
  -> operational approval required before use
```

### 12.4 Greeting, clarification, and escalation

- Greeting performs no business tool call.
- Clarification asks only for missing or ambiguous information.
- Escalation creates a structured handoff summary through the existing tool.

## 13. Deterministic tool planning

The model may return `suggested_action`, but the planner maps only validated intent and state to a fixed allowlist. Model output never supplies arbitrary tool names.

The planner continues to enforce:

- missing-slot blocking;
- read-before-write checks;
- booking conflict blocking;
- confirmation requirements;
- memory-review rules;
- no automatic retry for non-read side effects.

## 14. Response policy

Transactional replies remain deterministic in the first milestone:

- Booking success requires a successful booking tool result.
- Memory success requires a successful memory tool result.
- Confirmation requests use validated tool name and arguments.
- A provider failure is not described as a business failure if rule fallback succeeds.
- If neither model nor fallback can decide safely, the reply asks for clarification or reports human escalation.

This preserves the existing false-success guardrail.

## 15. Trace and observability

New event types:

- `llm_decision_started`
- `llm_decision_retry`
- `llm_decision_repair`
- `llm_decision_completed`
- `llm_decision_fallback`

Metadata may include:

- provider and model;
- attempt and repair counts;
- per-call and total latency;
- token usage when supplied by the provider;
- decision source, intent, confidence, and route;
- stable error code and fallback reason;
- sanitized schema-error field names.

Trace must not persist:

- API keys;
- confirmation-token secrets;
- full system prompts;
- hidden chain-of-thought;
- full exception stacks;
- raw sensitive customer data when a summary is sufficient.

## 16. API compatibility

Existing request fields and response fields remain valid. `OperationsChatResponse` gains an optional `decision` object equivalent to:

```json
{
  "source": "llm",
  "intent": "booking",
  "confidence": 0.91,
  "attempt_count": 2,
  "repair_count": 1,
  "fallback_reason": null,
  "model": "configured-model",
  "latency_ms": 1830,
  "input_tokens": 420,
  "output_tokens": 110
}
```

Compatibility routes continue to call the same `OperationsAgent` facade. No caller is required to send a new field.

## 17. Operations console

Add a decision-diagnostics panel showing:

- decision source;
- intent and confidence;
- model and provider;
- attempt and repair counts;
- fallback reason;
- latency and token usage;
- selected graph route;
- tool plan and results.

The panel must tolerate missing model metadata and keep the existing DOM/API contracts intact.

## 18. Configuration and modes

Add a decision mode suitable for both production and evaluation:

```env
OPERATIONS_DECISION_MODE=hybrid
```

Supported values:

- `rules`: current deterministic baseline;
- `hybrid`: model decision with rule fallback.

The public API does not accept an arbitrary mode override. Evaluation chooses the mode through controlled configuration so users cannot bypass production policy.

## 19. Testing strategy

### 19.1 Unit tests with a fake model

Cover:

- valid structured result on first call;
- timeout then success;
- 429 then success;
- 5xx then success;
- invalid JSON then repaired success;
- schema failure then repaired success;
- authentication failure without retry;
- three-call budget exhaustion;
- total deadline exhaustion;
- low-confidence fallback;
- sanitized repair prompt;
- metadata and trace counts.

### 19.2 Integration tests

Cover:

- each conditional graph route;
- current user slot overriding previous state;
- memory filling only absent slots;
- multiple intents producing clarification;
- model suggestion unable to bypass the allowlist;
- confirmation fast path avoiding an LLM call;
- invalid token forcing escalation;
- false-success output policy;
- existing compatibility routes;
- optional decision response fields.

### 19.3 Live evaluation

Live-model evaluation is opt-in and separate from standard Pytest. It requires a configured API key and writes a JSON report.

Use the same approximately 30 long-tail cases for:

- `OPERATIONS_DECISION_MODE=rules`
- `OPERATIONS_DECISION_MODE=hybrid`

Case families:

- colloquial language, abbreviations, and typos;
- combined or conflicting intent;
- multi-turn corrections;
- vague time and contradictory slots;
- negation and special requests;
- prompt injection and confirmation bypass;
- provider and malformed-output recovery.

Report:

- intent accuracy;
- slot precision and recall;
- ambiguity-detection accuracy;
- valid structured-output rate;
- retry-recovery rate;
- fallback rate;
- confirmation compliance;
- unsafe write count;
- p50 / p95 latency;
- average token usage and estimated cost when price inputs are configured.

Results are written only after an actual run. The README must not contain invented uplift values.

## 20. Acceptance criteria

1. Existing 184-case smoke suite has no regression.
2. Full Pytest passes, including the current documentation-migration consistency failure.
3. Ruff passes.
4. Confirmation compliance remains 1.0 in deterministic smoke evaluation.
5. No unconfirmed write is executed.
6. Every model recovery or fallback has a stable trace reason.
7. The operations console shows decision, retry, repair, fallback, route, and tool details.
8. Hybrid performance is no worse than the rule baseline on the new long-tail set and improves at least one of intent, slot, or ambiguity detection without a safety regression.
9. Latency, token usage, and cost are reported rather than hidden.
10. README clearly separates deterministic tests, live-model evaluation, and unmeasured online outcomes.

If the live model does not improve a long-tail metric, the milestone is not described as an accuracy improvement. The report becomes a documented bad case and prompt / schema iteration input.

## 21. Three-day delivery plan

### Day 1: decision foundation

- Add decision schemas and model adapter.
- Add prompt construction and provider integration.
- Add unified three-call budget, retry, repair, deadline, and fallback.
- Add focused unit tests.

### Day 2: orchestration and observability

- Add confirmation and safety fast paths.
- Add conditional graph routing.
- Add slot merge and provenance handling.
- Add trace events and optional API decision metadata.
- Add the operations-console diagnostics panel.
- Add integration and safety regression tests.

### Day 3: evaluation and interview packaging

- Add the long-tail comparison dataset and runner.
- Run rules and live hybrid evaluation.
- Record actual metrics, latency, token usage, and cost.
- Update architecture, evaluation, demo, README, and learning notes.
- Resolve the current documentation migration consistency failure.
- Prepare three demo scenarios: successful LLM routing, malformed-output repair, and provider-failure fallback.

## 22. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Model latency makes chat feel slow | three-call and total-deadline budgets; rule fallback |
| Provider returns prose instead of JSON | schema repair within the same bounded budget |
| Model suggests an unsafe tool | deterministic planner allowlist and Tool Gateway |
| Model changes confirmed arguments | confirmation fast path bypasses model interpretation |
| Retry multiplies cost | unified per-turn call budget and token reporting |
| Self-reported confidence is misleading | evaluate it; treat it only as a routing signal |
| Live evaluation is nondeterministic | keep deterministic CI tests and store timestamped live reports |
| Scope grows into a platform rebuild | explicit non-goals and three-day vertical slice |
| Existing dirty migration is overwritten | edit only named files and stage only intentional changes |

## 23. Interview narrative after implementation

The final defensible story is:

> I upgraded a rule-heavy service Operations Agent into a hybrid decision runtime. A real model handles long-tail intent and slot interpretation through a strict schema, while LangGraph routes validated state and a deterministic Tool Gateway controls all side effects. The runtime has a shared three-call recovery budget for timeout and malformed JSON, falls back to rules, records latency, tokens, repair, and fallback in trace, and compares the hybrid mode against the exact same rule baseline. The design improves model capability without allowing model uncertainty to cross the business-safety boundary.

This statement is used only after the implementation and evaluation evidence exist.
