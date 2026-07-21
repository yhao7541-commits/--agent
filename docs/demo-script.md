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
