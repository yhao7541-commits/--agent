# Evaluation

The deterministic eval harness lives in `harness/`.

Run locally:

```bash
python -m harness.runners.run_all --smoke
```

The runner loads YAML cases, executes the operations graph, applies evaluator checks, and emits a JSON report.

## Current Metrics

- `intent_accuracy`
- `slot_recall`
- `tool_selection_accuracy`
- `confirmation_compliance`
- `booking_completion_rate`
- `rag_decision_accuracy`
- `memory_write_precision`
- `escalation_accuracy`

The current smoke suite has 9 cases and all metrics pass their initial thresholds. The next scale-up step is expanding datasets toward the planned 150+ cases.
