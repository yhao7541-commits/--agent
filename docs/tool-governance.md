# 工具治理

工具治理网关是运营智能体执行业务动作的控制点。

每个工具都会声明：

- permission：`read`、`write`、`external` 或 `sensitive`
- 是否需要确认
- Pydantic 输入 schema
- Pydantic 输出 schema
- handler 函数

网关会针对未知工具、参数校验失败、需要确认的写操作、handler 超时和 handler 异常返回结构化错误。它也会追加工具级 trace event，让测试和评估可以在不读取日志的情况下检查行为。

每个工具定义都包含超时时间和重试预算。重试只应用于 `read` 工具；`write`、`sensitive` 和 `external` 工具只尝试一次，避免重复产生业务副作用。handler 异常会以通用消息上报，避免通过 API 响应暴露内部异常文本。

## 写操作策略

以下工具需要显式确认：

- `create_booking`
- `reschedule_booking`
- `cancel_booking`
- `write_customer_preference`
- `delete_customer_memory`

`check_schedule`、`find_available_staff`、`lookup_customer_profile`、`search_services`、`search_knowledge_base` 等读取工具可以不经确认直接执行。

`write_customer_preference` 和 `delete_customer_memory` 属于敏感操作，会把已确认的记忆变更路由到客户记忆生命周期。写入状态会返回 `created`、`updated` 或 `conflict`；删除状态会返回 `deleted` 或 `not_found`。

敏感记忆即使经过用户确认，也不会立即进入智能体的客户上下文。`service_contraindication`、`marketing_consent`、`constraint` 和 `policy_note` 会先以 `pending_review` 写入 SQLite memory store，必须通过 `/api/memory/memories/{memory_id}/approve` 或 `/memory` 页面审核通过后，才会被 `lookup_customer_profile` 返回。
