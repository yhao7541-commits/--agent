# 演示脚本

以下演示可以使用 `/api/operations/chat`，也可以使用 `/operations` 控制台。

## 1. 预约信息不完整

用户：`我想约一个肩颈放松`

预期结果：意图识别为预约，但缺少日期和时间，不会执行 `create_booking`。

## 2. 信息完整的预约需要确认

用户：`我想明天下午3点约肩颈放松`

预期结果：Agent 会规划读取工具和 `create_booking`，但 gateway 会在任何写操作成功前返回确认请求。

更贴近真实用户的时间表达：

- `我想明天下午3点半约肩颈放松` -> `time_window=15:30`
- `我想后天上午10点约推拿` -> 日期会标准化为后天
- `我想下周五晚上7点约按摩` -> 日期会标准化为下周五

## 3. 确认后才执行写入

把上一步返回的 `confirmation_request.tool_name` 和 `confirmation_request.arguments` 随消息 `确认` 一起发回。

预期结果：`create_booking` 执行成功，回复中包含预约 ID。

## 4. 政策问题使用 RAG

用户：`如果我迟到20分钟会怎么样？`

预期结果：系统执行 `search_knowledge_base`，trace metadata 中包含来自 `booking_policy.md` 的来源 chunk。

本地确定性模式：

```powershell
$env:RAG_BACKEND="local"
```

MCP 接入模式：

```powershell
$env:RAG_BACKEND="mcp"
$env:RAG_MCP_COMMAND="python"
$env:RAG_MCP_ARGS="-m src.mcp_server.server"
$env:RAG_MCP_CWD="D:\Dev\RAG\MODULAR-RAG-MCP-SERVER"
$env:RAG_MCP_COLLECTION="wellness_service_ops"
$env:RAG_MCP_TIMEOUT_SECONDS="45"
python scripts/check_mcp_rag.py --collection wellness_service_ops --query "late arrival policy" --min-chunks 1
```

诊断预期：`ok=true`、`chunk_count >= 1`，并且 `chunks[].source` 包含 `booking_policy.md` 等 wellness 政策文档。

可选的知识域校验：

```powershell
python scripts/check_mcp_rag.py --collection wellness_service_ops --query "late arrival policy" --min-chunks 1 --require-source booking_policy.md
```

预期结果：只有当外部 MCP collection 包含 wellness 预约政策来源时，这条命令才会以 0 退出码结束。

## 5. 偏好表达生成记忆 proposal

用户：`我以后都喜欢安静一点的房间`

预期结果：系统生成 memory proposal，并且 `write_customer_preference` 需要用户确认。

## 6. 已存偏好会应用到预约

先确认 memory proposal，然后发送：

用户：`我想明天下午3点约肩颈放松`

预期结果：预约确认摘要包含安静房间偏好，`memory_used=true`，`applied_customer_memories[]` 列出已应用的客户偏好，trace metadata 显示加载和应用的记忆数量。

## 7. Trace 回放

执行 operations 请求前先设置 trace 路径：

```powershell
$env:OPERATIONS_TRACE_STORE_PATH="data/traces.jsonl"
```

API 响应返回 `trace_id` 后，运行：

```powershell
python -m observability.replay --trace-id <trace_id> --path data/traces.jsonl
```

预期结果：replay 会按执行顺序展示节点序列、确认拦截、工具调用、RAG 检索事件和最终回复摘要。

## 8. 混合决策韧性与真实模型对比

混合 LLM 决策只解释用户意图和槽位，确定性 Tool Gateway 继续控制工具白名单、参数、确认和写操作。超时重试与 JSON/Schema 修复共用硬性的共享三次调用预算，成功结果经 LangGraph 条件路由进入业务分支，预算耗尽则规则回退。

先运行不需要凭据的确定性韧性演示：

```powershell
python scripts/demo_decision_resilience.py
```

预期结果：依次展示非法 JSON 后修复、三次超时后回退、合法确认和拒绝确认，并在汇总中报告 `unsafe_write_count=0`。这是脚本化 fake client 的控制流证据，不是模型准确率证据。

配置真实 provider 后，才运行可选的真实模型语义对比：

```powershell
python -m harness.runners.run_decision_comparison --dataset harness/datasets/decision_long_tail_cases.yaml --require-live-model
```

当前没有可审计的 live 报告，因此不声明准确率提升。演示时还应明确：模型输出具有非确定性，真实模型路径依赖凭据，当前没有分布式熔断器，也没有线上业务结果证据。
