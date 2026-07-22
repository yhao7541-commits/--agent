# 评估

确定性评估框架位于 `harness/`。

本地运行：

```bash
python -m harness.runners.run_all --smoke
```

运行器会加载 YAML 用例、执行运营 graph、应用 evaluator 检查，并输出 JSON 报告。

## 混合决策的两类验证

混合 LLM 决策与确定性 Tool Gateway 分开验证：前者检查结构化意图/槽位、重试、修复和回退，后者继续检查工具白名单、确认与副作用。超时重试和 JSON/Schema 修复使用同一个硬性的共享三次调用预算；有效决策再由 LangGraph 条件路由分发，失败时进入规则回退。

不依赖凭据的确定性韧性演示使用脚本化 fake client，可重复覆盖 JSON 修复、超时耗尽回退、确认接受和确认拒绝：

```bash
python scripts/demo_decision_resilience.py
```

可选的真实模型语义对比使用冻结的 long-tail 数据集，并且必须显式提供可构造原生超时客户端的 provider 配置：

```bash
python -m harness.runners.run_decision_comparison --dataset harness/datasets/decision_long_tail_cases.yaml --require-live-model
```

这两类证据不能混用：确定性韧性演示证明控制流和安全边界，不证明语义效果；真实模型对比才可用于比较 rules 与 hybrid。当前没有生成可审计的真实模型报告，因此不声明准确率提升，也不提供 live 指标。模型输出具有非确定性，真实模型路径依赖凭据，当前进程内实现没有分布式熔断器，并且没有线上业务结果证据。

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

当前 smoke 套件包含 186 条用例，所有指标均通过初始门槛。这些数字来自确定性 smoke，不代表混合 LLM 决策相对规则方案的提升。下一步扩展方向是加入更难的多轮和边界场景，而不是只扩大当前 smoke 覆盖。
