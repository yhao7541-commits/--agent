# 记忆生命周期

客户记忆被视为运营数据，而不是自由形式的聊天历史。智能体可以提出记忆 proposal，但已确认的写入、编辑、审核和删除都必须通过受控工具或记忆管理 API 执行。

```text
用户消息
  -> 确定性高价值规则
  -> 可选 LLM 补充提取
  -> MemoryProposal
  -> 用户确认
  -> 工具网关
  -> SQLite MemoryStore
  -> 审核 / 过期 / 版本 / 事件轨迹
  -> 只有已通过审核的有效记忆会进入客户上下文
```

## 存储

`MemoryStore` 基于 SQLite。运行时持久化路径通过 `CUSTOMER_MEMORY_DB_PATH` 配置，默认值是 `data/customer_memory.sqlite3`。测试会传入 `tmp_path` 数据库或使用内存 store，因此不会写入生产数据。

store 包含两张表：

- `customer_memories`：当前记忆记录、状态、审核状态、来源 trace / conversation、过期时间、版本和软删除时间。
- `customer_memory_events`：追加式生命周期事件，例如 `memory_written`、`memory_pending_review`、`memory_updated`、`memory_approved`、`memory_rejected` 和 `memory_deleted`。

## Proposal 边界

`extract_memory_proposals()` 始终返回 `MemoryProposal` 对象。高价值或高风险场景优先走确定性规则：

- 过敏和服务禁忌
- 不要营销的请求
- 明确的正向或负向服务偏好
- 模糊表达抑制

可选的 LLM extractor 只能在确定性规则未命中时补充识别更灵活的表达。LLM 输出必须经过 schema 校验并转换为 `MemoryProposal`；非法输出会被忽略。LLM 路径永远不能直接写数据库。

## 审核规则

敏感记忆类型包括：

- `service_contraindication`
- `marketing_consent`
- `constraint`
- `policy_note`

敏感写入必须通过 `ToolGateway` 获得用户确认。确认后会以以下状态存储：

- `status=pending_review`
- `review_status=pending`

待审核或已拒绝的记忆不会通过普通客户上下文查询返回。只有运营人员通过 memory API 或 `/memory` 页面审核通过后，它们才会被智能体使用。

## 查询规则

默认客户查询只返回满足以下条件的记忆：

- active
- approved
- 未过期
- 未软删除

过期记忆不会进入智能体查询结果。管理 API 在审核和审计需要时，可以包含 inactive 或 deleted 记录。

## 版本和删除

编辑记忆会递增 `version`，并写入包含修改前后值的 `memory_updated` 事件。

删除记忆是软删除。记录会保留在 SQLite 中，并带有 `status=deleted`、`deleted_at`、递增后的版本号和 `memory_deleted` 事件。已删除记忆不会返回给智能体。

## API

记忆管理 API 注册在 `/api/memory` 下：

- `GET /api/memory/users/{user_id}/memories`
- `PATCH /api/memory/memories/{memory_id}`
- `POST /api/memory/memories/{memory_id}/approve`
- `POST /api/memory/memories/{memory_id}/reject`
- `DELETE /api/memory/memories/{memory_id}`
- `GET /api/memory/memories/{memory_id}/events`

API 返回结构化记忆记录和生命周期事件，不暴露 raw prompt 或内部异常文本。

## UI

`/memory` 页面提供最小可用的运营管理 UI，可用于列出记忆、编辑内容、通过或拒绝敏感记录、软删除记录，以及查看事件历史。它有意与旧聊天入口和 operations chat 控制台保持分离。
