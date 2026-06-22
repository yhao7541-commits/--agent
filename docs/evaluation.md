# Evaluation

The deterministic eval harness lives in `harness/`.

Run locally:

```bash
python -m harness.runners.run_all --smoke
```

The runner loads YAML cases, executes the operations graph, applies evaluator checks, and emits a JSON report.

The JSON report includes a `governance` block for release review:

- `failed_thresholds` lists any metric below its gate.
- `failed_case_ids` lists individual cases that failed evaluator checks.
- `suite_counts` shows coverage distribution across booking, RAG, memory, security, escalation, and tool-routing suites.
- `p95_latency_ms` carries the latency guardrail used by the smoke report.

## Current Metrics

- `intent_accuracy`
- `slot_recall`
- `slot_precision`
- `tool_selection_accuracy`
- `tool_argument_accuracy`
- `confirmation_compliance`
- `booking_completion_rate`
- `rag_decision_accuracy`
- `rag_groundedness`
- `memory_write_precision`
- `memory_suppression_accuracy`
- `memory_recall_accuracy`
- `memory_delete_accuracy`
- `escalation_accuracy`
- `escalation_reason_accuracy`
- `security_policy_accuracy`
- `p95_latency_ms`

Latest local smoke result:

| Metric | Result | Threshold |
| --- | ---: | ---: |
| `intent_accuracy` | 1.00 | 0.85 |
| `slot_recall` | 1.00 | n/a |
| `slot_precision` | 1.00 | 0.85 |
| `tool_selection_accuracy` | 1.00 | 0.85 |
| `tool_argument_accuracy` | 1.00 | 0.85 |
| `confirmation_compliance` | 1.00 | 1.00 |
| `booking_completion_rate` | 1.00 | 0.80 |
| `rag_decision_accuracy` | 1.00 | 0.85 |
| `rag_groundedness` | 1.00 | 0.85 |
| `memory_write_precision` | 1.00 | 0.80 |
| `memory_suppression_accuracy` | 1.00 | 0.90 |
| `memory_recall_accuracy` | 1.00 | 0.80 |
| `memory_delete_accuracy` | 1.00 | 0.80 |
| `escalation_accuracy` | 1.00 | 0.90 |
| `escalation_reason_accuracy` | 1.00 | 0.90 |
| `security_policy_accuracy` | 1.00 | 0.90 |
| `p95_latency_ms` | reported | n/a |

The current smoke suite has 184 cases and all metrics pass their initial thresholds. The next scale-up step is adding harder multi-turn and edge-case scenarios beyond the current smoke coverage.
