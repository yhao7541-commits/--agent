# Hybrid LLM Decision Layer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the rule-heavy Operations Agent into a hybrid runtime that uses a real LLM for structured intent and slot decisions while preserving deterministic safety, tool governance, fallback, traceability, and comparative evaluation.

**Architecture:** Add a focused decision-model contract, LangChain-backed provider adapter, and retry/repair engine with a hard three-call budget. Inject that engine into the existing OperationsAgent graph, add explicit confirmation/safety fast paths and conditional routing, then expose decision diagnostics through the operations API, console, and evaluation harness. Tool selection and all side effects remain deterministic.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, LangChain chat models, LangGraph, Pytest, PyYAML, Jinja2, vanilla JavaScript, Ruff.

---

## Execution workspace and safety constraint

The current repository has 75 pre-existing dirty entries that constitute the in-progress single-agent migration. Several implementation targets (`nodes.py`, `api/operations.py`, and `operations_console.html`) are already dirty. A normal worktree created from `HEAD` would omit that baseline, while staging in the current worktree could mix earlier user changes into feature commits.

### Day 0A: create an isolated, reproducible migration baseline

This step requires explicit user approval before creating the local snapshot commit. Use these fixed paths and the existing environment; do not install a second environment:

```powershell
$sourceRepo = 'D:\Dev\按摩房预约-agent\smart-appointment-ai-agent'
$worktreePath = 'D:\Dev\按摩房预约-agent\.worktrees\hybrid-llm-decision'
$transferDir = 'D:\Dev\按摩房预约-agent\.worktree-transfer\hybrid-llm-decision'
$python = 'D:\Dev\按摩房预约-agent\smart-appointment-ai-agent\.venv\Scripts\python.exe'

if (Test-Path -LiteralPath $worktreePath) { throw "Worktree path already exists: $worktreePath" }
if (Test-Path -LiteralPath $transferDir) { throw "Transfer path already exists: $transferDir" }
if (git -C $sourceRepo branch --list codex/hybrid-llm-decision) { throw 'Branch already exists' }

New-Item -ItemType Directory -Path $transferDir | Out-Null
git -C $sourceRepo diff --binary HEAD --output="$transferDir\tracked.patch"
git -C $sourceRepo ls-files --others --exclude-standard
```

Review the untracked manifest before copying anything. The pre-approved migration allowlist is:

- `agents/operations/agent.py`
- `api/memory.py`
- `tests/test_memory_api.py`
- `tests/test_memory_llm_extraction.py`
- `tests/test_memory_management_page.py`
- `tests/test_memory_store_sqlite.py`
- `tests/test_single_agent_migration.py`
- `web/templates/memory_management.html`

Do not copy caches, `.env`, runtime data, or unreviewed files. If another untracked source file is required, stop and obtain approval for the expanded allowlist. Then create the worktree, apply the tracked patch, and copy only the approved untracked paths while preserving their relative paths:

```powershell
git -C $sourceRepo worktree add -b codex/hybrid-llm-decision $worktreePath HEAD
git -C $worktreePath apply "$transferDir\tracked.patch"

$approvedUntracked = @(
  'agents/operations/agent.py',
  'api/memory.py',
  'tests/test_memory_api.py',
  'tests/test_memory_llm_extraction.py',
  'tests/test_memory_management_page.py',
  'tests/test_memory_store_sqlite.py',
  'tests/test_single_agent_migration.py',
  'web/templates/memory_management.html'
)
foreach ($relativePath in $approvedUntracked) {
  $sourcePath = Join-Path $sourceRepo $relativePath
  if (-not (Test-Path -LiteralPath $sourcePath)) { throw "Missing approved source: $relativePath" }
  $destinationPath = Join-Path $worktreePath $relativePath
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationPath) | Out-Null
  Copy-Item -LiteralPath $sourcePath -Destination $destinationPath
}

Push-Location $worktreePath
& $python -m pytest -q
& $python -m harness.runners.run_all --smoke
& $python -m ruff check .
Pop-Location
```

The worktree must reproduce the baseline evidence below before the snapshot. After explicit approval, create a local-only boundary commit and tag:

```powershell
git -C $worktreePath add -A
git -C $worktreePath diff --cached --name-status
git -C $worktreePath commit -m "chore: snapshot in-progress operations migration"
git -C $worktreePath tag codex/hybrid-llm-baseline
```

Never push or merge this branch or tag automatically. At completion, present two choices: (a) explicitly approve publishing the migration baseline plus feature commits, or (b) export `git diff --binary codex/hybrid-llm-baseline..codex/hybrid-llm-decision` and apply only that feature delta back to the dirty source tree for review. If baseline-snapshot approval is not granted, pause execution; do not silently implement in the dirty source worktree.

### Day 0B: verify live-provider readiness

Without printing secret values, verify the configured provider, model, base URL when applicable, and API-key presence. Construct the strict production client with a one-second native timeout, but do not make a semantic model call yet. Missing configuration or inability to forward the native timeout is a blocking `configuration_error` for the live comparison; deterministic implementation and fake-client resilience work may continue.

## Baseline evidence

Run from `D:\Dev\按摩房预约-agent\smart-appointment-ai-agent` before implementation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m harness.runners.run_all --smoke
.\.venv\Scripts\python.exe -m ruff check .
```

Expected current baseline:

- Pytest: `138 passed, 1 failed`; the one known failure is `tests/test_single_agent_migration.py::test_main_docs_describe_single_agent_tool_orchestration`.
- Smoke: `case_count=184`, `failed_case_count=0`, `failed_thresholds={}`.
- Ruff: `All checks passed!`.

Stop and investigate if the baseline differs before feature changes begin.

## File map

### New runtime files

- `agents/operations/decision_models.py`: Pydantic contracts, decision source/error enums, and environment-backed settings.
- `agents/operations/decision_prompt.py`: initial and repair prompt construction with data minimization.
- `agents/operations/decision_client.py`: provider-neutral client protocol and LangChain production adapter.
- `agents/operations/decision_engine.py`: structured parse, hard three-call budget, retry, repair, deadline, metadata aggregation, and rule fallback orchestration.
- `agents/operations/decision_validation.py`: normalized slot merge and business-validation results.

### Existing runtime files to modify

- `config/model_provider.py`: strict configured-model creation and native timeout forwarding.
- `agents/operations/state.py`: optional decision and ambiguity state.
- `agents/operations/nodes.py`: confirmation/safety fast paths, decision node, slot merge hook, and trace emission.
- `agents/operations/routers.py`: complete conditional-route functions.
- `agents/operations/graph.py`: dependency-injected decision engine and conditional edges.
- `agents/operations/agent.py`: optional engine injection while retaining the facade.
- `agents/operations/__init__.py`: public decision contracts needed by API/tests.
- `api/operations.py`: always-present `/api/operations/chat` decision object.
- `.env.example`: hybrid mode, timeout, call-budget, deadline, and confidence settings.

### UI files

- `web/templates/operations_console.html`: decision-diagnostics panel and safe nullable rendering.

### Test files

- `tests/operations_decision_fakes.py`: programmable fake client, fake clock, and scripted outcomes.
- `tests/test_operations_decision_models.py`: schemas and settings.
- `tests/test_operations_decision_client.py`: configured-provider and timeout adapter.
- `tests/test_operations_decision_engine.py`: success, retry, repair, deadline, aggregation, and fallback.
- `tests/test_operations_decision_routing.py`: confirmation/safety fast paths and graph branches.
- `tests/test_operations_decision_slots.py`: merge precedence and business validation.
- `tests/test_operations_decision_api.py`: decision response contract and trace metadata.
- `tests/test_operations_console_template.py`: diagnostics panel contract.
- Existing operations, migration, API, graph, and gateway tests remain regression coverage.

### Evaluation and documentation files

- `harness/datasets/decision_long_tail_cases.yaml`: approximately 30 versioned semantic cases.
- `harness/evaluators/decision_comparison.py`: normalization and exact metric definitions.
- `harness/runners/run_decision_comparison.py`: isolated rules-vs-hybrid runner and JSON report writer.
- `scripts/demo_decision_resilience.py`: deterministic retry/repair/fallback demonstration without an application backdoor.
- `tests/test_decision_comparison.py`: dataset isolation and metric math.
- `README.md`, `docs/architecture.md`, `docs/evaluation.md`, `docs/demo-script.md`, `PROJECT_LEARNING_NOTES.md`: verified implementation, results, limits, and interview explanation.

---

### Task 1: Add strict decision contracts and bounded settings

**Files:**
- Create: `agents/operations/decision_models.py`
- Modify: `agents/operations/state.py:4-34`
- Modify: `.env.example:7-27`
- Test: `tests/test_operations_decision_models.py`

- [ ] **Step 1: Write failing schema and settings tests**

```python
import pytest
from pydantic import ValidationError

from agents.operations.decision_models import (
    DecisionMetadata,
    DecisionSettings,
    ModelDecision,
)


def test_model_decision_rejects_unknown_intent_and_extra_fields():
    with pytest.raises(ValidationError):
        ModelDecision.model_validate(
            {
                "intent": "invented_intent",
                "confidence": 0.9,
                "extracted_slots": {},
                "ambiguities": [],
                "risk_flags": [],
                "suggested_action": "none",
                "decision_summary": "unknown",
                "unexpected": "value",
            }
        )


def test_settings_hard_cap_attempts_at_three(monkeypatch):
    monkeypatch.setenv("LLM_DECISION_MAX_ATTEMPTS", "9")
    settings = DecisionSettings.from_env()
    assert settings.max_attempts == 3


def test_rules_source_is_distinct_from_rule_fallback():
    assert DecisionMetadata(source="rules").source == "rules"
    assert DecisionMetadata(source="rule_fallback").source == "rule_fallback"
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_models.py -q
```

Expected: FAIL because `agents.operations.decision_models` does not exist.

- [ ] **Step 3: Implement the strict contracts**

Use `ConfigDict(extra="forbid")` for model-owned structured output. Define:

```python
DecisionIntent = Literal[
    "booking", "reschedule", "cancel", "consultation", "memory",
    "delete_memory", "greeting", "clarification", "escalation", "unknown",
]
DecisionSource = Literal[
    "llm", "rules", "rule_fallback", "forced_escalation",
    "confirmed_action", "confirmation_rejected",
]

class BookingSlotCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service_type: str | None = None
    date: str | None = None
    time_window: str | None = None
    duration: str | None = None
    preferred_staff: str | None = None
    special_requests: str | None = None
    booking_id: str | None = None

class ModelDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: DecisionIntent
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_slots: BookingSlotCandidates = Field(default_factory=BookingSlotCandidates)
    ambiguities: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    suggested_action: str
    decision_summary: str = Field(max_length=160)
```

Implement `DecisionSettings.from_env()` with these invariants:

- mode is `rules` or `hybrid`;
- attempts are clamped to `1..3`;
- per-call timeout and total deadline are positive;
- minimum confidence is within `0..1`.

Add optional state keys from the spec and document the new environment values in `.env.example` without adding a real key.

- [ ] **Step 4: Run focused tests and existing state tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_models.py tests/test_operations_graph.py::test_operations_graph_compiles -q
```

Expected: PASS.

- [ ] **Step 5: Commit only Task 1 files**

```powershell
git add agents/operations/decision_models.py agents/operations/state.py .env.example tests/test_operations_decision_models.py
git diff --cached --name-status
git commit -m "feat: add operations decision contracts"
```

Expected staged scope: exactly the four named files.

---

### Task 2: Add a real-model client with native timeout enforcement

**Files:**
- Create: `agents/operations/decision_client.py`
- Modify: `config/model_provider.py:47-90`
- Test: `tests/test_operations_decision_client.py`

- [ ] **Step 1: Write failing provider-adapter tests**

Cover these behaviors:

```python
def test_strict_chat_model_rejects_placeholder_configuration(monkeypatch): ...
def test_decision_client_passes_remaining_timeout_to_model_factory(monkeypatch): ...
def test_decision_client_extracts_text_and_usage_metadata(monkeypatch): ...
def test_local_rule_model_is_not_accepted_as_hybrid_provider(monkeypatch): ...
```

The fake model should capture the `request_timeout` value and return an `AIMessage` with deterministic `usage_metadata`.

- [ ] **Step 2: Run and verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_client.py -q
```

Expected: FAIL because strict creation and `LangChainDecisionClient` are absent.

- [ ] **Step 3: Extend the model factory without breaking legacy callers**

Change the public signature to:

```python
def create_chat_model(
    temperature: float = 0,
    *,
    request_timeout: float | None = None,
    require_configured: bool = False,
):
```

Existing calls keep `require_configured=False`. Hybrid decision calls use `True`. When strict mode is requested, missing credentials or unsupported provider configuration raises a stable configuration exception instead of returning `LocalRuleBasedChatModel`.

Pass `timeout=request_timeout` to `ChatOpenAI` and `AzureChatOpenAI`. Confirm installed versions accept that parameter with a focused construction test; do not guess a provider-specific keyword.

- [ ] **Step 4: Implement the client boundary**

```python
class ModelCallResult(BaseModel):
    raw_text: str
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

class DecisionModelClient(Protocol):
    def invoke(self, prompt: str, timeout_seconds: float) -> ModelCallResult: ...
```

`LangChainDecisionClient.invoke()` creates a strict model with the remaining timeout, calls it synchronously, extracts message text and best-effort usage, and never logs the prompt or credentials.

- [ ] **Step 5: Run client and provider regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_client.py tests/test_mcp_rag_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```powershell
git add agents/operations/decision_client.py config/model_provider.py tests/test_operations_decision_client.py
git diff --cached --name-status
git commit -m "feat: add bounded LLM decision client"
```

---

### Task 3: Build minimized initial and repair prompts

**Files:**
- Create: `agents/operations/decision_prompt.py`
- Test: `tests/test_operations_decision_engine.py`

- [ ] **Step 1: Write failing prompt tests**

```python
def test_initial_prompt_contains_only_allowed_context():
    prompt = build_initial_prompt(
        message="明天下午想做肩颈放松",
        booking_slots={"service_type": "肩颈放松"},
        booking_slot_sources={"service_type": "previous_turn"},
        local_date="2026-07-22",
    )
    assert "明天下午想做肩颈放松" in prompt
    assert "DecisionResult" in prompt
    assert "API_KEY" not in prompt

def test_repair_prompt_contains_sanitized_field_errors_not_stack():
    prompt = build_repair_prompt(original_task="...", errors=[{"loc": ["intent"], "msg": "bad"}])
    assert "intent" in prompt
    assert "Traceback" not in prompt
```

- [ ] **Step 2: Run and verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_engine.py -q
```

Expected: FAIL because prompt builders do not exist.

- [ ] **Step 3: Implement prompt builders**

Use deterministic JSON serialization for allowed context and `ModelDecision.model_json_schema()`. Limit validation-error content to field locations, stable error types, and short sanitized messages. Do not include raw exceptions, confirmation secrets, customer database rows, or unrelated history.

- [ ] **Step 4: Run prompt tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_engine.py -q
```

Expected: prompt tests PASS; later engine tests may still be absent, not failing.

- [ ] **Step 5: Commit Task 3**

```powershell
git add agents/operations/decision_prompt.py tests/test_operations_decision_engine.py
git diff --cached --name-status
git commit -m "feat: add structured decision prompts"
```

---

### Task 4: Implement the three-call retry, repair, deadline, and fallback engine

**Files:**
- Create: `agents/operations/decision_engine.py`
- Create: `tests/operations_decision_fakes.py`
- Modify: `tests/test_operations_decision_engine.py`

- [ ] **Step 1: Add failing first-call success test**

The scripted fake returns a valid JSON object and usage metadata. Assert:

- source is `llm`;
- attempt count is 1;
- repair count is 0;
- input/output tokens equal the provider values;
- final structured decision is validated.

- [ ] **Step 2: Implement the minimal one-call engine and make it pass**

The engine constructor accepts the model client, settings, fallback callable, `monotonic_fn`, `sleep_fn`, and jitter function. Dependency injection keeps unit tests deterministic.

- [ ] **Step 3: Add failing recovery-budget tests**

```python
def test_timeout_then_success_records_provider_timeout_and_two_attempts(): ...
def test_rate_limit_then_success_records_stable_rate_limited_code(): ...
def test_provider_5xx_then_success_records_stable_provider_5xx_code(): ...
def test_schema_validation_failure_builds_sanitized_repair_prompt(): ...
def test_timeout_and_invalid_json_share_three_call_budget(): ...
def test_fourth_call_is_never_started(): ...
def test_timeout_for_each_attempt_is_clamped_to_remaining_deadline(): ...
def test_total_deadline_exhaustion_returns_stable_deadline_exceeded_code(): ...
def test_attempts_never_overlap(): ...
def test_authentication_error_does_not_retry(): ...
def test_configuration_error_uses_zero_provider_calls(): ...
def test_token_totals_include_malformed_attempt_usage(): ...
def test_low_confidence_returns_rule_fallback_with_reason(): ...
def test_budget_exhaustion_returns_rule_fallback(): ...
```

- [ ] **Step 4: Implement stable error classification and recovery**

Use a loop bounded by `settings.max_attempts` and total monotonic deadline. Each iteration:

1. computes remaining time;
2. invokes the client with `min(per_call_timeout, remaining)`;
3. parses JSON and validates `ModelDecision`;
4. replaces the next prompt with a sanitized repair prompt after parse/schema failure;
5. retries only timeout, 429, 5xx, or classified transport failures;
6. aggregates latency and tokens across all attempted calls;
7. calls the rule fallback on terminal failure or low confidence.

Do not retry local configuration errors or provider authentication/permission errors.

- [ ] **Step 5: Run engine tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_engine.py -q
```

Expected: PASS.

- [ ] **Step 6: Run model and engine group**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_models.py tests/test_operations_decision_client.py tests/test_operations_decision_engine.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```powershell
git add agents/operations/decision_engine.py tests/operations_decision_fakes.py tests/test_operations_decision_engine.py
git diff --cached --name-status
git commit -m "feat: add resilient hybrid decision engine"
```

---

### Task 5: Add confirmation and hard-safety fast paths plus conditional graph routing

**Files:**
- Modify: `agents/operations/nodes.py:62-196`
- Modify: `agents/operations/routers.py:1-13`
- Modify: `agents/operations/graph.py:1-47`
- Modify: `agents/operations/agent.py:1-11`
- Modify: `agents/operations/__init__.py:1-5`
- Test: `tests/test_operations_decision_routing.py`
- Regression: `tests/test_operations_graph.py`

- [ ] **Step 1: Write failing fast-path tests with a counting fake client**

Cover:

- valid confirmation executes exact confirmed path and model call count remains 0;
- rejection sets `source=confirmation_rejected`, creates no tool plan, and calls no model;
- invalid confirmation escalates with `unsafe_tool_confirmation` and calls no model;
- injection, confirmation bypass, medical concern, and refund dispute each take their fixed escalation reason and call no model.
- any non-empty model `risk_flags`, including unknown future values, forces `source=forced_escalation`, records reason `model_risk_flag`, and permits only the human-handoff path.

- [ ] **Step 2: Run routing tests and verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_routing.py -q
```

Expected: FAIL because fast-path nodes and injected engine are absent.

- [ ] **Step 3: Add focused nodes**

Add or extract:

```python
def validate_confirmation(state): ...
def input_guardrail(state): ...
def decide_request(state, decision_engine): ...
```

Keep existing `classify_intent` available as the deterministic rule-mode/fallback implementation. Avoid duplicating its keyword policy. The fallback adapter operates on a copy of state and returns the shared decision contract without leaking its internal trace events.

- [ ] **Step 4: Add explicit route functions**

```python
def route_after_confirmation(state) -> Literal["confirmed", "rejected", "escalated", "ordinary"]: ...
def route_after_guardrail(state) -> Literal["escalated", "decide"]: ...
def route_after_decision(state) -> Literal[
    "booking", "consultation", "memory", "greeting", "clarification", "escalation"
]: ...
```

All multi-intent decisions route to clarification. Unknown/low fallback decisions route to escalation.

Add explicit model-driven tests for all six route outcomes: booking, consultation, memory, greeting, clarification, and escalation. A malicious `suggested_action="drop_database"` must be ignored rather than becoming a tool name. Run the existing false-success regression `test_output_policy_check_blocks_false_booking_success_reply` to prove that model text cannot claim a successful write without a successful deterministic tool result.

- [ ] **Step 5: Rebuild graph with dependency injection and conditional edges**

Change signatures without breaking defaults:

```python
def build_operations_graph(decision_engine: HybridDecisionEngine | None = None): ...
def run_operations_turn(state, decision_engine: HybridDecisionEngine | None = None): ...

class OperationsAgent:
    def __init__(self, decision_engine=None):
        self.decision_engine = decision_engine
    def run_turn(self, state):
        return run_operations_turn(state, decision_engine=self.decision_engine)
```

Default construction reads `OPERATIONS_DECISION_MODE`; rules mode must remain deterministic and require no API key.

- [ ] **Step 6: Run routing and graph regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_routing.py tests/test_operations_graph.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

Stage only the named files and inspect the staged diff before committing:

```powershell
git add agents/operations/nodes.py agents/operations/routers.py agents/operations/graph.py agents/operations/agent.py agents/operations/__init__.py tests/test_operations_decision_routing.py tests/test_operations_graph.py
git diff --cached --name-status
git commit -m "feat: route operations decisions conditionally"
```

---

### Task 6: Merge model slots with deterministic provenance and business validation

**Files:**
- Create: `agents/operations/decision_validation.py`
- Modify: `agents/operations/nodes.py:228-398`
- Test: `tests/test_operations_decision_slots.py`
- Regression: `tests/test_operations_graph.py`

- [ ] **Step 1: Write failing precedence tests**

Cover:

- confirmed arguments bypass merge;
- current model/user slots override previous values;
- previous values remain when current input omits them;
- approved memory fills only an absent special request;
- system customer name never overrides user-owned fields;
- invalid date/service/time becomes an ambiguity and cannot enter write arguments;
- invalid duration, unknown or unavailable staff, and malformed booking ID become validation errors and cannot enter write arguments;
- multiple intents always produce clarification.

- [ ] **Step 2: Run and verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_slots.py -q
```

- [ ] **Step 3: Implement normalized merge and validation results**

```python
class SlotValidationResult(BaseModel):
    slots: dict[str, Any]
    sources: dict[str, str]
    ambiguities: list[str]
    errors: list[dict[str, Any]]
```

Keep the externally visible source value `user` for accepted current-turn values so existing contracts remain stable. Decision metadata identifies whether the current-turn interpretation came from LLM or rules. Reuse existing normalization helpers instead of creating a second date/time grammar.

- [ ] **Step 4: Wire the result into booking state**

Writes are blocked whenever ambiguity or a correctable business-validation error remains. The response asks only for the ambiguous/missing fields.

- [ ] **Step 5: Run slot and booking regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_slots.py tests/test_operations_graph.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```powershell
git add agents/operations/decision_validation.py agents/operations/nodes.py tests/test_operations_decision_slots.py tests/test_operations_graph.py
git diff --cached --name-status
git commit -m "feat: validate hybrid booking slots"
```

---

### Task 7: Persist decision trace events and expose the operations API contract

**Files:**
- Modify: `agents/operations/nodes.py:38-83`
- Modify: `api/operations.py:18-79`
- Test: `tests/test_operations_decision_api.py`
- Regression: `tests/test_operations_api.py`

- [ ] **Step 1: Write failing API and trace tests**

Assert that `/api/operations/chat` always returns a `decision` object for:

- deliberate rules mode;
- LLM success;
- rule fallback;
- forced escalation;
- confirmed action;
- confirmation rejection.

Assert nullable model/token fields are present as `null`. Assert compatibility endpoints retain their existing top-level shapes.

- [ ] **Step 2: Add failing trace-event tests**

Check exact counts and safe metadata for:

- `llm_decision_started`;
- `llm_decision_retry`;
- `llm_decision_repair`;
- `llm_decision_completed`;
- `llm_decision_fallback`.

Assert no key, raw prompt, hidden reasoning, or traceback is persisted.

- [ ] **Step 3: Implement trace emission and API response model**

Add a typed decision response model whose inner fields are nullable but whose object is always emitted by the operations endpoint. Map only system-owned metadata and validated intent/confidence. Do not expose raw model output.

- [ ] **Step 4: Run focused API tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_api.py tests/test_operations_api.py tests/test_trace_replay.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

```powershell
git add agents/operations/nodes.py api/operations.py tests/test_operations_decision_api.py tests/test_operations_api.py
git diff --cached --name-status
git commit -m "feat: expose operations decision diagnostics"
```

---

### Task 8: Add the operations-console decision diagnostics panel

**Files:**
- Modify: `web/templates/operations_console.html:299-383`
- Modify: `tests/test_operations_console.py:13-36`
- Modify: `tests/test_operations_console_template.py:1-18`

- [ ] **Step 1: Write failing HTML contract tests**

Require stable IDs:

```python
assert 'id="decision-panel"' in html
assert 'id="decision-source-value"' in html
assert 'id="decision-confidence-value"' in html
assert 'id="decision-json"' in html
assert "data.decision || {}" in template
```

- [ ] **Step 2: Run and verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_console.py tests/test_operations_console_template.py -q
```

- [ ] **Step 3: Add a minimal Chinese diagnostics panel**

Show source, intent/confidence, provider/model, attempts/repairs, fallback reason, latency, token usage, and route. Render `null` as `-`; use `textContent`, never `innerHTML`, for model-originated values. Preserve all existing IDs and fetch paths.

- [ ] **Step 4: Run console tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_console.py tests/test_operations_console_template.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 8**

```powershell
git add web/templates/operations_console.html tests/test_operations_console.py tests/test_operations_console_template.py
git diff --cached --name-status
git commit -m "feat: show LLM decision diagnostics"
```

---

### Task 9: Add deterministic resilience evaluation and frozen live comparison

**Files:**
- Create: `harness/datasets/decision_long_tail_cases.yaml`
- Create: `harness/evaluators/decision_comparison.py`
- Create: `harness/runners/run_decision_comparison.py`
- Create: `scripts/demo_decision_resilience.py`
- Create: `tests/test_decision_comparison.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing metric tests**

Test exact normalization and aggregation:

```python
def test_intent_accuracy_uses_exact_normalized_enum(): ...
def test_slot_precision_recall_micro_average_field_value_pairs(): ...
def test_ambiguity_accuracy_uses_expected_boolean(): ...
def test_nearest_rank_percentiles(): ...
def test_unavailable_usage_is_reported_not_invented(): ...
def test_live_runner_rejects_missing_provider_configuration_before_cases(): ...
def test_live_runner_reports_all_fallback_and_low_provider_success_as_bad_cases(): ...
def test_model_required_cases_must_record_provider_attempt(): ...
def test_default_output_is_timestamped_and_explicit_output_is_not_overwritten(): ...
def test_cost_requires_usage_and_both_pricing_inputs(): ...
```

- [ ] **Step 2: Implement pure metric functions and make tests pass**

Metric functions accept per-case dictionaries and return JSON-serializable output. They do not call the model or read global state.

- [ ] **Step 3: Add dataset-validation tests before the dataset**

Require:

- unique case IDs;
- explicit dataset version and date anchor;
- expected intent and `requires_clarification` for every case;
- explicit `expect_model_call` for every case;
- explicit prior turns for multi-turn cases;
- no secrets or live customer identifiers;
- approximately 30 cases across the approved families.

- [ ] **Step 4: Add the long-tail dataset**

Include colloquial/typo, corrections, vague time, contradictory slots, negation, combined intent, and safety recognition. Provider faults and malformed JSON stay in fake-client resilience tests, not in the live semantic dataset.

- [ ] **Step 5: Implement the comparison runner**

Required CLI shape:

```powershell
.\.venv\Scripts\python.exe -m harness.runners.run_decision_comparison `
  --dataset harness/datasets/decision_long_tail_cases.yaml `
  --require-live-model
```

The runner:

- runs `rules` and `hybrid` against the same frozen cases;
- uses unique evaluation namespaces and resets booking/memory state per case;
- exits with code 2 before running cases if credentials, provider configuration, or native timeout support is unavailable;
- requires every hybrid case with `expect_model_call=true` to report `attempt_count > 0`;
- reports provider-success completion and fallback rate among `expect_model_call=true` cases; all-fallback or low-success output remains auditable bad-case evidence and must never be described as uplift;
- records provider, model, temperature 0, prompt version, dataset version, timezone, and date anchor;
- persists expected labels and normalized per-case predictions;
- computes exact metrics from the spec;
- reads optional `LLM_INPUT_COST_PER_1M_TOKENS` and `LLM_OUTPUT_COST_PER_1M_TOKENS`; computes cost only when provider usage and both prices exist, otherwise reports cost as unavailable with a reason;
- writes by default to the timestamped path `data/evaluation/decision-comparison-YYYYMMDDTHHMMSSZ.json`;
- fails if an explicit `--output` already exists unless `--force` is supplied;
- never prints or writes API keys.

Add `data/evaluation/*.json` to `.gitignore`; retain reports locally and copy only verified aggregate numbers into docs.

- [ ] **Step 6: Run evaluation unit tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_decision_comparison.py tests/test_eval_harness.py -q
```

Expected: PASS.

- [ ] **Step 7: Add and run a deterministic resilience demonstration**

`scripts/demo_decision_resilience.py` injects an in-process scripted fake client into `OperationsAgent`. It runs four scenarios: invalid JSON followed by repaired JSON, repeated provider timeout followed by rule fallback, valid confirmation acceptance, and confirmation rejection. It prints a sanitized trace plus a JSON summary containing retry, repair, fallback, confirmation-compliance, and unsafe-write counts. It must not add an HTTP development mode, public failure switch, or require a real API key.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_decision_engine.py tests/test_operations_decision_routing.py -q
.\.venv\Scripts\python.exe scripts/demo_decision_resilience.py
```

Expected: tests PASS; the script exits 0, demonstrates both recovery paths, and reports `unsafe_write_count=0` with no live key.

- [ ] **Step 8: Commit Task 9**

```powershell
git add harness/datasets/decision_long_tail_cases.yaml harness/evaluators/decision_comparison.py harness/runners/run_decision_comparison.py scripts/demo_decision_resilience.py tests/test_decision_comparison.py .gitignore
git diff --cached --name-status
git commit -m "feat: compare rule and hybrid decisions"
```

---

### Task 10: Restore documentation consistency and document only verified behavior

**Files:**
- Modify: `PROJECT_LEARNING_NOTES.md:1-153`
- Modify: `README.md:1-154,466-525`
- Modify: `docs/architecture.md`
- Modify: `docs/evaluation.md`
- Modify: `docs/demo-script.md`
- Test: `tests/test_single_agent_migration.py`
- Test: `tests/test_documentation_assets.py`

- [ ] **Step 1: Fix the one known baseline documentation test narrowly**

Add the exact UTF-8 phrase `单一 Operations Agent` to the current-architecture introduction in `PROJECT_LEARNING_NOTES.md`. Do not rewrite the 100-question body as part of this fix.

- [ ] **Step 2: Run the named migration test**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_single_agent_migration.py::test_main_docs_describe_single_agent_tool_orchestration -q
```

Expected: PASS.

- [ ] **Step 3: Add failing documentation assertions for the new runtime**

Require README/architecture/evaluation/demo docs to mention:

- hybrid LLM decision plus deterministic Tool Gateway;
- hard three-call shared budget;
- rules fallback;
- conditional LangGraph routes;
- deterministic resilience vs opt-in live semantic comparison;
- no invented improvement claim.

- [ ] **Step 4: Update docs with actual implementation facts**

Do not add live metric values until Task 11 produces a report. Keep current limitations explicit: model nondeterminism, credential dependency, no distributed breaker, and no online business outcome evidence.

- [ ] **Step 5: Run documentation tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_single_agent_migration.py tests/test_documentation_assets.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 10**

```powershell
git add PROJECT_LEARNING_NOTES.md README.md docs/architecture.md docs/evaluation.md docs/demo-script.md tests/test_single_agent_migration.py tests/test_documentation_assets.py
git diff --cached --name-status
git commit -m "docs: explain hybrid operations decisions"
```

---

### Task 11: Run live comparison, record defensible results, and perform full verification

**Files:**
- Modify only after real output: `README.md`
- Modify only after real output: `docs/evaluation.md`
- Local ignored artifact: `data/evaluation/decision-comparison-*.json`

- [ ] **Step 1: Verify live provider configuration without printing secrets**

Check only whether required variables are configured. Never echo API keys. Confirm the selected model supports the bounded timeout path. Missing credentials, provider configuration, or native-timeout support blocks the live comparison and must be reported as such; deterministic feature verification may still complete, but it is not live-model evidence.

- [ ] **Step 2: Run the live comparison**

```powershell
.\.venv\Scripts\python.exe -m harness.runners.run_decision_comparison `
  --dataset harness/datasets/decision_long_tail_cases.yaml `
  --require-live-model
```

Expected: exit 0 and a report containing both `rules` and `hybrid` results, per-case predictions, frozen configuration, latency, and usage availability.

- [ ] **Step 3: Validate the report before using its numbers**

Check:

- case count equals the dataset;
- both modes used identical case IDs;
- no API key or secret appears;
- no case failed because state leaked from another case;
- token/cost is numeric only when supported/configured;
- every `expect_model_call=true` hybrid case has `attempt_count > 0`;
- provider-success completion and fallback rate are reported for model-required cases, with all-fallback or low-success runs treated as bad-case evidence rather than uplift;
- any improvement statement matches the actual aggregate values.

- [ ] **Step 4: Update docs with actual results or an explicit no-uplift result**

If hybrid improves a metric, state the dataset, baseline, hybrid value, and date. If it does not, document the bad cases and do not claim accuracy improvement.

- [ ] **Step 5: Run focused feature tests**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_operations_decision_models.py `
  tests/test_operations_decision_client.py `
  tests/test_operations_decision_engine.py `
  tests/test_operations_decision_routing.py `
  tests/test_operations_decision_slots.py `
  tests/test_operations_decision_api.py `
  tests/test_decision_comparison.py -q
```

Expected: PASS.

- [ ] **Step 6: Run complete repository verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m harness.runners.run_all --smoke
```

Expected:

- Pytest: all tests pass, no remaining documentation-migration failure.
- Ruff: `All checks passed!`.
- Smoke: at least the original 184 cases, `failed_case_count=0`, `failed_thresholds={}`.

- [ ] **Step 7: Manually verify the operations console**

Start the app, open `/operations`, and exercise the real configured-model path:

1. valid LLM booking route;
2. confirmed write fast path with no second LLM call.

Verify the diagnostics panel, confirmation flow, trace timeline, and existing panels remain usable.

Demonstrate malformed output repair and provider-failure fallback separately through the in-process test seam:

```powershell
.\.venv\Scripts\python.exe scripts/demo_decision_resilience.py
```

Do not expose fake failures through an HTTP query parameter, environment toggle, or public development endpoint.

- [ ] **Step 8: Commit only verified result documentation**

```powershell
git add README.md docs/evaluation.md
git diff --cached --name-status
git commit -m "docs: record hybrid decision evaluation"
```

Skip this commit if no documentation changed. Do not commit the ignored raw report or any key.

---

## Final completion checklist

- [ ] Hybrid mode uses a real configured model and never silently substitutes the local rule model.
- [ ] Rules mode requires no API key and preserves deterministic baseline behavior.
- [ ] Timeout retry and JSON repair share a hard maximum of three calls.
- [ ] Total deadline includes calls, backoff, and repair work.
- [ ] Confirmation, rejection, invalid token, and hard guardrails call no LLM.
- [ ] Multiple intents and unresolved ambiguity cannot plan a write.
- [ ] Tool names remain allowlisted and all writes still pass Tool Gateway confirmation.
- [ ] `/api/operations/chat` always emits the decision object; compatibility APIs keep existing shapes.
- [ ] Trace contains safe decision diagnostics but no secrets, raw prompts, or hidden reasoning.
- [ ] Resilience tests are deterministic; live evaluation is isolated and reproducible enough to audit.
- [ ] Existing smoke suite has no regression.
- [ ] Full Pytest and Ruff pass.
- [ ] README claims match actual stored evaluation evidence.
