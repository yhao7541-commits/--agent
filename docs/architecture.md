# 架构说明

项目当前采用单一 Operations Agent 编排、多工具执行：原有 Web/chat URL 作为兼容入口继续可用，内部和新的 `/api/operations/chat` 一样进入状态化 LangGraph 运营运行时。

```text
客户端
  -> FastAPI 运营接口
  -> LangGraph 状态机
  -> Tool Gateway 工具治理网关
  -> 预约、知识库、记忆、人工介入工具
  -> Trace 事件和评估记录
```

```text
咨询消息
  -> classify_intent
  -> plan_tool_calls
  -> search_knowledge_base
  -> RAG_BACKEND=local：使用 docs/knowledge 的确定性适配器
  -> RAG_BACKEND=mcp：通过 stdio MCP 调用 query_knowledge_hub
  -> 带引用来源的 rag_retrieval_completed trace 事件
  -> generate_response
```

关键工程边界是：graph 负责任务规划，gateway 负责工具执行治理，observability 负责记录实际发生的过程。原有 service 和 database 层继续服务旧应用入口。

## 混合决策边界

混合 LLM 决策只负责生成结构化的意图、槽位、置信度和歧义；确定性 Tool Gateway 掌握工具白名单、schema、权限、确认与副作用。模型建议不能变成任意工具名，也不能绕过写操作确认。

决策引擎把超时重试与 JSON/Schema 修复计入同一个硬性的共享三次调用预算。有效模型结果进入 LangGraph 条件路由；预算耗尽、总截止时间超限、配置不可用或低置信度则进入规则回退。确认接受、确认拒绝和硬安全命中是无模型快速路径，因此不会为已经确定的动作再调用模型。

该边界已经由 fake client 的确定性韧性演示和回归测试覆盖；可选的真实模型语义对比需要单独提供 provider 凭据。当前没有真实模型报告，所以不声明准确率提升。运行时仍受模型输出具有非确定性、真实模型路径依赖凭据以及没有分布式熔断器的限制，也没有线上业务结果证据。

RAG grounding 由 `RAG_BACKEND` 选择。默认 `local` 后端使用仓库内确定性知识文件，保证 CI 和评估稳定；设置 `RAG_BACKEND=mcp` 后，`search_knowledge_base` 会通过外部 stdio MCP 服务执行，服务命令由 `RAG_MCP_COMMAND`、`RAG_MCP_ARGS` 和 `RAG_MCP_CWD` 配置。`RAG_MCP_COLLECTION` 是可选项，只有配置后才会传给外部服务。可以用 `python scripts/check_mcp_rag.py --collection wellness_service_ops --query "late arrival policy" --min-chunks 1` 验证 MCP 服务可达，并确认目标 collection 能返回带来源的 chunk。需要强制校验来源时，加上 `--require-source booking_policy.md`，这样没有命中预期 wellness 知识域时诊断会失败。

本地 `D:\Dev\RAG\MODULAR-RAG-MCP-SERVER` 服务提供 `list_collections`；导入 `docs/knowledge` 后，应能看到包含 wellness 政策文档的 `wellness_service_ops`。这属于部署配置，不是代码依赖：应用只从环境变量读取 collection 名称。

设置 `OPERATIONS_TRACE_STORE_PATH` 后，`/api/operations/chat` 的 trace event 会持久化到 JSONL。之后可以用 replay CLI 检查历史运行：

```bash
python -m observability.replay --trace-id <trace_id> --path data/traces.jsonl
```

## 运行节点

- `initialize_turn`
- `classify_intent`
- `load_customer_context`
- `extract_booking_slots`
- `propose_memory_writes`
- `plan_tool_calls`
- `execute_tools`
- `generate_response`
- `output_policy_check`
- `finalize_turn`

这让意图判断、槽位填充、记忆 proposal、工具规划、确认拦截、响应策略检查和 trace 收尾都保持可观察、可测试。预约槽位来源会写入 state 和 trace metadata，方便审计哪些值来自用户、客户记忆或系统补全。当已存客户偏好被应用到预约槽位时，runtime 会设置 `memory_used`，记录 `applied_customer_memories`，并通过 `/api/operations/chat` 和运营控制台暴露这些字段。
