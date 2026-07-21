# 评估

确定性评估框架位于 `harness/`。

本地运行：

```bash
python -m harness.runners.run_all --smoke
```

运行器会加载 YAML 用例、执行运营 graph、应用 evaluator 检查，并输出 JSON 报告。

JSON 报告包含用于发布审查的 `governance` 区块：

- `failed_thresholds`：列出低于门槛的指标。
- `failed_case_ids`：列出 evaluator 检查失败的具体用例。
- `suite_counts`：展示 booking、RAG、memory、security、escalation、tool-routing 等套件的覆盖分布。
- `p95_latency_ms`：记录 smoke 报告使用的延迟护栏。

## 当前指标

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

最新本地 smoke 结果：

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| `intent_accuracy` | 1.00 | 0.85 |
| `slot_recall` | 1.00 | 不适用 |
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
| `p95_latency_ms` | 已报告 | 不适用 |

当前 smoke 套件包含 184 条用例，所有指标均通过初始门槛。下一步扩展方向是加入更难的多轮和边界场景，而不是只扩大当前 smoke 覆盖。
