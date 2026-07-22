# Local Service Operations Agent

## 30 秒速览

| 面试官先看 | 内容 |
| --- | --- |
| 项目解决什么问题 | 本地生活服务门店的咨询、预约、排班冲突、客户偏好记忆和人工确认分散在人工前台流程里，容易漏信息、误写预约或无法追踪决策来源。 |
| 目标用户是谁 | 需要处理预约调度、服务咨询、技师排班和客户偏好的本地生活服务门店运营人员；终端顾客通过中文首页操作台发起咨询或预约。 |
| 核心流程是什么 | 用户输入咨询或预约需求 → `OperationsAgent` 识别意图和槽位 → `ToolGateway` 调用知识库、排班、预约、记忆工具 → 写操作先返回确认请求 → trace/replay 记录过程。 |
| 如何启动或查看演示 | 运行 `python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload`，打开首页用户操作台 `http://127.0.0.1:8000`；运营控制台在 `/operations`，记忆管理在 `/memory`。 |
| 如何评测 | 运行 `python -m harness.runners.run_all --smoke`，当前 smoke 数据集覆盖 186 条确定性用例，检查意图、槽位、确认拦截、RAG、记忆、安全策略和人工升级。 |
| 已知限制 | 当前评测仍是 smoke 规模；记忆管理还不是完整 RBAC/审批队列/合规报表；旧 Web/chat 流程仍保留兼容入口，尚未完全切到新的 operations runtime。 |

Local Service Operations Agent 是一个面向本地生活服务门店的状态化 Agent 系统。项目基于 FastAPI、LangGraph、LangChain、FAISS、SQLite 和单一 Operations Agent 编排，实现了服务咨询、多轮预约、员工匹配、排班冲突处理、客户偏好记忆、基于 RAG 的政策问答、人工确认、Tool Gateway 受控工具调用、trace 回放和回归评测骨架。

这个项目的核心目标不是只做一个普通的预约表单，而是把本地服务运营中的高频工作自动化：理解用户想咨询还是预约，判断服务偏好，匹配合适员工，检查可用时间，在写操作前要求明确确认，并在必要时结合知识库、历史行为和偏好数据给出更可靠的服务建议。

## 项目背景

在本地生活服务门店场景中，运营人员需要同时处理大量复杂事务：员工排班、顾客预约、服务项目咨询、价格计算、临时改约、员工偏好记录和收款确认等。随着员工数量和顾客量增加，传统人工前台模式很容易出现沟通成本高、信息遗漏、排班冲突和服务体验不稳定的问题。

因此，本项目尝试用单一服务运营智能体重构这一流程：让系统能够在明确业务约束下理解用户需求，并把咨询、预约、记忆写入和人工兜底统一编排为受控工具调用。它适用于需要人员排班、预约调度和知识咨询的本地生活服务行业。

## 项目价值

本项目重点不是聊天，而是生产级智能体工程问题：状态管理、工具治理、客户记忆、RAG grounding、人工确认、可观测性和评测。基础升级版本新增了 LangGraph 运营运行时、工具治理网关、trace/replay 基础设施和独立的 `/api/operations/chat` 接口，旧有 Web/chat 流程仍保持可用。

### 混合决策与确定性执行

运营运行时支持混合 LLM 决策：模型只返回经过 schema 校验的意图、槽位、置信度、歧义、风险标记、建议动作和决策摘要，这些字段都没有工具执行权；确定性 Tool Gateway 仍负责工具白名单、参数校验、权限、确认 token 和实际副作用。超时重试与 JSON/Schema 修复共用硬性的共享三次调用预算；三次内仍无法得到有效结果、超过总截止时间或置信度不足时进入规则回退。确认、拒绝和硬安全规则不依赖模型，其他结果通过 LangGraph 条件路由进入预约、咨询、记忆、问候、澄清或人工升级分支。

当前可复现证据分为两类：`scripts/demo_decision_resilience.py` 提供不需要 API key 的确定性韧性演示；`harness.runners.run_decision_comparison` 提供需要显式配置凭据的可选的真实模型语义对比。真实模型报告尚未生成，因此这里不声明准确率提升，也不把下方 smoke 指标解释为 LLM 效果。

```text
用户 -> LangGraph 运行时 -> 工具网关 -> 预约/知识库/记忆工具 -> Trace/评测
```

## 不只是聊天机器人

这不是通用聊天机器人。智能体必须在明确业务约束下完成工作：未确认不得创建预约，政策回答必须基于知识来源，客户记忆只能在合适时写入，不确定或敏感场景必须升级人工。

## 运营流程

### 工具治理

所有运营运行时中的业务动作都通过 `ToolGateway` 执行。工具声明 permission、Pydantic 输入/输出 schema 和确认策略；`create_booking`、`reschedule_booking`、`cancel_booking`、`write_customer_preference` 等写入或敏感工具在用户确认前不会执行。

### 记忆生命周期

客户长期记忆先生成 `MemoryProposal`，记录 type、content、evidence、confidence、sensitivity、source、expiry 和确认策略。明确偏好可以生成 proposal，模糊表达会被忽略，过敏、服务禁忌、不要营销等高风险/高价值表达优先走确定性规则；LLM 只能在规则未命中时补充识别，并且输出必须被校验成 `MemoryProposal`。确认后的 `write_customer_preference` 会通过 Tool Gateway 写入 SQLite-backed `MemoryStore`；敏感记忆先进入 `pending_review`，只有 approve 后才会进入 `lookup_customer_profile` 的 customer context。edit 会增加 version 并写 event，delete 是 soft delete。`/api/memory/*` 和 `/memory` 页面提供 list/edit/delete/approve/reject/events 管理入口。

### RAG grounding

咨询和政策类问题会走知识检索路径。默认 `RAG_BACKEND=local` 使用仓库内 `docs/knowledge/` 的确定性知识适配器，保证 CI 和评估不依赖外部服务；设置 `RAG_BACKEND=mcp` 后，`search_knowledge_base` 会通过 stdio MCP 调用外部 RAG 服务。每次 RAG 检索都会在 trace metadata 中保存 query、source、chunk_id、score 和文本预览，用户回复可以自然表达，但审计链路保留引用来源。

MCP 接入的 RAG 通过环境变量配置，不需要修改 graph 或 API：

```env
RAG_BACKEND=mcp
RAG_MCP_COMMAND=python
RAG_MCP_ARGS=-m src.mcp_server.server
RAG_MCP_CWD=D:\Dev\RAG\MODULAR-RAG-MCP-SERVER
RAG_MCP_TOOL=query_knowledge_hub
RAG_MCP_COLLECTION=wellness_service_ops
RAG_MCP_TIMEOUT_SECONDS=45
RAG_LLM_PROVIDER=openai
RAG_LLM_MODEL=qwen-plus
RAG_LLM_API_KEY=your_dashscope_or_openai_compatible_key_here
RAG_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_EMBEDDING_DIMENSIONS=1024
RAG_EMBEDDING_API_KEY=your_dashscope_or_openai_compatible_key_here
RAG_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_VECTOR_STORE_COLLECTION=wellness_service_ops
RAG_RERANK_ENABLED=false
RAG_ANSWER_GENERATION_ENABLED=false
```

诊断真实 MCP 连接和 collection 命中：

```powershell
python scripts/check_mcp_rag.py --collection wellness_service_ops --query "late arrival policy" --min-chunks 1
```

诊断输出会列出 MCP collection、chunk 数量和引用 metadata。`chunk_count > 0` 只能证明 MCP 检索链路可用；仍需检查 `chunks[].source` 是否来自目标本地生活服务知识域。需要强制校验来源时可以加：

```powershell
python scripts/check_mcp_rag.py --collection wellness_service_ops --query "late arrival policy" --min-chunks 1 --require-source booking_policy.md
```

### Trace 回放

每轮运营对话都会生成节点级和工具级 trace event。`observability/trace_store.py` 提供 JSONL store，`observability/replay.py` 可以按顺序输出节点摘要，用于复现工具调用、确认拦截、RAG 检索和人工升级。API 在设置 `OPERATIONS_TRACE_STORE_PATH` 后会把本次运行写入 JSONL：

```powershell
$env:OPERATIONS_TRACE_STORE_PATH="data/traces.jsonl"
python -m observability.replay --trace-id <trace_id> --path data/traces.jsonl
```

## 评估结果

当前评估框架是确定性 smoke 版本，不依赖付费模型调用。运行：

```bash
python -m harness.runners.run_all --smoke
```

最新本地结果：

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| intent_accuracy | 1.00 | 0.85 |
| slot_recall | 1.00 | n/a |
| slot_precision | 1.00 | 0.85 |
| tool_selection_accuracy | 1.00 | 0.85 |
| tool_argument_accuracy | 1.00 | 0.85 |
| confirmation_compliance | 1.00 | 1.00 |
| booking_completion_rate | 1.00 | 0.80 |
| rag_decision_accuracy | 1.00 | 0.85 |
| rag_groundedness | 1.00 | 0.85 |
| memory_write_precision | 1.00 | 0.80 |
| memory_suppression_accuracy | 1.00 | 0.90 |
| memory_recall_accuracy | 1.00 | 0.80 |
| memory_delete_accuracy | 1.00 | 0.80 |
| escalation_accuracy | 1.00 | 0.90 |
| escalation_reason_accuracy | 1.00 | 0.90 |
| security_policy_accuracy | 1.00 | 0.90 |
| p95_latency_ms | 已报告 | 不适用 |

数据集目前覆盖 186 条 smoke 用例。完整路线仍计划继续扩展更复杂的槽位、工具参数、RAG grounding、记忆质量、安全策略和人工升级边界。

## 工程文档

- [架构说明](docs/architecture.md)
- [工具治理](docs/tool-governance.md)
- [记忆生命周期](docs/memory-lifecycle.md)
- [评估](docs/evaluation.md)
- [演示脚本](docs/demo-script.md)
- [安全策略](docs/security-policy.md)

首页用户操作台：`http://127.0.0.1:8000`
运营运行时控制台：`http://127.0.0.1:8000/operations`
记忆管理页面：`http://127.0.0.1:8000/memory`

## 核心能力

- **单一 Operations Agent 编排**：通过 LangGraph 显式节点统一处理意图识别、槽位提取、客户记忆加载、工具规划、确认拦截、响应生成和 trace 收尾。
- **多工具执行**：咨询、预约、排班校验、客户记忆写入和人工升级都收敛为 Tool Gateway 管理的工具调用，避免多个 Agent 各自维护状态。
- **RAG 知识咨询**：支持本地确定性知识适配器和环境变量驱动的 MCP 接入 RAG；咨询路径会保留引用 metadata，便于 trace/replay 审计。
- **智能预约管理**：根据用户需求、员工专长、历史偏好和可用时间进行匹配，辅助完成预约确认。
- **受控工具调用**：新增工具治理网关，对 read、write、external、sensitive 工具进行 schema 校验、确认拦截和 trace 记录。
- **状态化运营运行时**：新增 LangGraph graph skeleton，把意图判断、槽位提取、工具计划、确认请求、响应生成、输出策略检查和 trace 写入建模为显式节点。
- **用户行为分析**：记录用户交互与预约行为，分析偏好模式，并用于后续推荐和个性化反馈。
- **个性化提醒**：在预约完成后，可结合实时天气等外部信息生成更贴近实际场景的提醒。
- **Embedding 缓存优化**：通过数据库缓存和文件缓存减少重复向量计算，提高知识检索性能。
- **数据管理能力**：支持知识库、技师信息和用户行为数据的增删改查，并在数据变化后自动维护索引。
- **日志与兜底机制**：保留关键处理过程日志，在信息不足或异常情况下提供更稳定的降级处理。

## 系统架构

项目采用严格的五层架构，核心原则是：**下层不能反向调用上层**。这样可以避免循环依赖，让业务逻辑、数据访问和接口编排保持清晰边界。

```text
Web 与应用层
    ↓  app.py, web/：页面、路由入口、系统启动
API 层
    ↓  api/：外部接口、请求编排、响应封装
智能体层
    ↓  agents/：AI Agent、任务路由、对话流程控制
服务层
    ↓  services/：业务逻辑、推荐算法、向量处理
DB Layer
    ↓  db/：数据模型、数据库连接、Repository
```

### 允许的调用方向

- Web 层调用 API 层
- API 层调用智能体层或服务层
- 智能体层调用服务层
- Services 层调用 DB 层

### 禁止的调用方式

- 下层反向调用上层
- Web 层绕过 API 直接访问服务层或 DB
- 智能体层绕过服务层直接访问 DB
- 服务层调用智能体层、API 或 Web

## 智能体设计

### 任务分类智能体

任务分类智能体是系统的主调度器，负责分析用户输入、判断任务类型，并把请求分发给合适的专业智能体。

```text
用户输入 → 意图分析 → 智能体路由 → 响应协调
```

主要职责：

- 判断用户意图
- 维护对话状态
- 控制不同智能体之间的切换
- 处理无法分类或超出能力范围的问题

### 咨询智能体

咨询智能体负责知识问答场景，使用 RAG 流程从知识库中检索相关内容，再结合大模型生成回答。

```text
任务分类 → 知识检索 → FAISS 相似度搜索 → 流式回答
```

主要职责：

- 区分咨询问题类型
- 从知识库检索相关内容
- 构建提示词
- 生成自然语言回答

### 预约智能体

预约智能体负责预约相关流程，包括解析用户输入、匹配技师、检查预约信息、生成确认消息等。

```text
任务分类 → 解析预约需求 → 技师匹配 → 预约确认
```

主要职责：

- 提取预约时间、服务项目、技师偏好等信息
- 匹配合适技师
- 处理信息缺失时的追问
- 生成预约结果和提醒

### 用户行为智能体

用户行为智能体更偏向后台智能分析，不完全依赖用户显式请求。它会根据交互记录、预约历史和偏好数据分析用户行为，为后续推荐提供依据。

```text
行为记录 → 模式分析 → 偏好更新 → 个性化推荐
```

主要职责：

- 记录用户行为
- 分析偏好模式
- 生成推荐依据
- 支持主动反馈和个性化服务

## 核心设计思想

### 1. 用任务分类降低系统复杂度

系统并不让一个智能体处理所有事情，而是先判断用户意图，再分发给对应模块。这样可以让咨询、预约、行为分析等逻辑保持独立，也更容易扩展新的智能体。

### 2. 用 RAG 解决专业知识回答

本地生活服务相关的项目介绍、注意事项、适用人群等内容更适合通过知识库维护。RAG 能让回答基于可控知识来源，而不是完全依赖大模型自由生成。

### 3. 用用户行为让推荐更个性化

系统会记录用户的咨询、预约和偏好信息。后续在推荐技师或服务项目时，可以结合历史行为，而不是每次都从零开始询问。

### 4. 用分层架构保证可维护性

智能体负责智能流程，服务层负责业务逻辑，Repository 负责数据访问。每层只关心自己的职责，减少后期修改时的连锁影响。

### 5. 为真实业务场景预留扩展空间

项目目前以本地 SQLite 和单体服务为主，但架构上预留了模型提供商切换、MCP 外部服务接入、后台任务、缓存优化和云端部署的扩展方向。

## 架构图

![系统架构](./architecture%20.jpg)

## 技术栈

- **后端框架**：FastAPI、Uvicorn
- **AI 框架**：LangGraph、LangChain
- **大模型接入**：兼容 OpenAI 格式的模型提供商，例如 Qwen、DeepSeek、Zhipu、OpenAI、Azure OpenAI
- **向量检索**：FAISS
- **数据库**：SQLite、SQLAlchemy
- **RAG 能力**：Embedding、向量索引、知识库检索、提示词构建
- **流式响应**：Python AsyncGenerator
- **前端页面**：Jinja2 模板、静态 CSS
- **外部服务扩展**：MCP，用于天气等外部信息接入
- **配置管理**：python-dotenv
- **后台任务**：schedule

## 项目结构

```text
Local Service Operations Agent/
├── agents/                         # 单一 Operations Agent 层
│   └── operations/                  # LangGraph 状态化运营运行时
│       ├── agent.py                 # OperationsAgent 外观
│       ├── graph.py                 # 状态机编排
│       ├── nodes.py                 # 意图、槽位、工具、响应节点
│       └── state.py                 # 结构化状态
├── api/                             # API 编排层
│   ├── operations.py                # 运营智能体接口
│   ├── appointment.py               # 预约接口
│   ├── consultation.py              # 咨询接口
│   ├── task.py                      # 任务分类接口
│   ├── chat_handler.py              # 流式聊天处理
│   ├── technician.py                # 技师管理接口
│   ├── knowledge.py                 # 知识库管理接口
│   └── user_behavior_analysis.py    # 用户行为分析接口
├── services/                        # 业务逻辑层
│   ├── appointment_service.py       # 预约业务逻辑
│   ├── knowledge_service.py         # 知识库管理
│   ├── recommendation_service.py    # 推荐逻辑
│   ├── technician_service.py        # 技师信息管理
│   ├── text_embedding.py            # Embedding 与向量处理
│   └── user_behavior_service.py     # 用户行为服务
├── db/                              # 数据持久化层
│   ├── models.py                    # SQLAlchemy 模型
│   ├── db_router.py                 # 数据库路由
│   ├── local_db.py                  # 本地数据库操作
│   ├── base/                        # 数据库基础接口
│   └── repositories/                # Repository 数据访问封装
├── config/                          # 配置模块
│   ├── constants.py                 # 常量与枚举
│   ├── database.py                  # 数据库配置
│   ├── model_provider.py            # 模型与 Embedding Provider 工厂
│   ├── settings.py                  # 应用配置
│   └── time_config.py               # 时间与排班配置
├── web/                             # Web 页面层
│   ├── routes.py                    # 页面路由
│   ├── templates/                   # HTML 模板
│   └── static/                      # 静态资源
├── tools/                           # 工具治理网关与受控工具定义
├── memory/                          # 客户记忆生命周期
├── rag/                             # RAG 适配器与引用 metadata
├── observability/                   # Trace schema、JSONL store 与 replay CLI
├── harness/                         # 确定性评估数据集、evaluators 与 runner
├── scripts/                         # 本地工程脚本
├── mcp-server/                      # MCP 外部服务扩展
├── data/                            # 数据库与缓存目录
├── tests/                           # 测试用例
├── app.py                           # 应用入口
├── requirements.txt                 # Python 依赖
├── .env.example                     # 环境变量模板
└── README.md                        # 项目说明
```

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD：

```cmd
.venv\Scripts\activate.bat
```

macOS 或 Linux：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

Windows PowerShell 可以使用：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写模型和数据库配置。项目支持 OpenAI 兼容格式的大模型与 Embedding 服务。

```env
MODEL_PROVIDER=qwen
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=your_openai_compatible_chat_base_url_here
LLM_MODEL=your_chat_model_name_here

EMBEDDING_PROVIDER=qwen
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_BASE_URL=your_openai_compatible_embedding_base_url_here
EMBEDDING_MODEL=your_embedding_model_name_here

DATABASE_URL=sqlite:///./data/smart_appointment.db

DEBUG=True
LOG_LEVEL=INFO

RAG_BACKEND=local
RAG_MCP_COMMAND=python
RAG_MCP_ARGS=-m src.mcp_server.server
RAG_MCP_CWD=D:\Dev\RAG\MODULAR-RAG-MCP-SERVER
RAG_MCP_TOOL=query_knowledge_hub
RAG_MCP_COLLECTION=wellness_service_ops
RAG_MCP_TIMEOUT_SECONDS=45
RAG_LLM_PROVIDER=openai
RAG_LLM_MODEL=qwen-plus
RAG_LLM_API_KEY=your_dashscope_or_openai_compatible_key_here
RAG_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_EMBEDDING_DIMENSIONS=1024
RAG_EMBEDDING_API_KEY=your_dashscope_or_openai_compatible_key_here
RAG_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_VECTOR_STORE_COLLECTION=wellness_service_ops
RAG_RERANK_ENABLED=false
RAG_ANSWER_GENERATION_ENABLED=false
```

常见配置方向：

- Qwen：使用阿里云百炼或 DashScope 的模型、Base URL 和 API Key。
- DeepSeek：可用于聊天模型，Embedding 可搭配其他兼容服务。
- Zhipu：可配置智谱的聊天模型和向量模型。
- Azure OpenAI：将 `MODEL_PROVIDER` 设置为 `azure`，并补充对应的 Azure OpenAI 环境变量。

### 4. 启动服务

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

如果 8000 端口已被占用，可以换成 8001：

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

启动后可以访问：

- 首页用户操作台：http://127.0.0.1:8000
- 运营运行时控制台：http://127.0.0.1:8000/operations
- 记忆管理页面：http://127.0.0.1:8000/memory
- API 文档：http://127.0.0.1:8000/docs
- ReDoc 文档：http://127.0.0.1:8000/redoc

## Docker 设置

本地容器启动：

```bash
docker compose up --build
```

容器默认读取 `.env.example`，监听 `http://127.0.0.1:8000`，并把运行数据写入 Docker volume `wellness-data`。

## 测试

运行全部测试：

```bash
pytest
```

运行单个测试文件：

```bash
pytest tests/test_operations_api.py
```

运行 lint 和评估 smoke：

```bash
python -m ruff check .
python -m harness.runners.run_all --smoke
```

检查外部 MCP RAG 连接：

```powershell
python scripts/check_mcp_rag.py --collection wellness_service_ops --query "late arrival policy" --min-chunks 1
```

如果要防止误用非本地生活服务 collection，可加 `--require-source booking_policy.md`，让诊断在没有命中目标知识域时返回非零退出码。

## 已知限制

- 当前评估数据集是 186 条 smoke 用例规模，还不是最终生产级回归集。
- 模型输出具有非确定性，真实模型路径依赖凭据；当前文档没有可引用的真实模型对比报告，也没有线上业务结果证据。
- 进程内保护没有分布式熔断器；多实例部署仍需共享状态、原子确认消费和写入幂等账本。
- 记忆 store 已接入 SQLite 持久化和最小 `/memory` 管理 UI；还没有实现生产级 RBAC、审批队列分配和完整合规报表。
- MCP 接入的 RAG 适配器已可用；`RAG_MCP_COLLECTION=wellness_service_ops` 时可以使用已灌入的本地生活服务业务资料，诊断脚本会显示命中的来源。
- 新运营接口已可用，但旧 Web/chat 流程尚未切换到新运行时。

## 主要页面

- 首页用户操作台：`web/templates/index.html`，提供常用入口、快速需求填充和主聊天预约入口。
- 运营运行时控制台：`web/templates/operations_console.html`
- 记忆管理页面：`web/templates/memory_management.html`
- 知识库管理：`web/templates/knowledge_management.html`
- 技师管理：`web/templates/technician.html`
- 技师排班：`web/templates/technician_schedule.html`
- 用户行为分析：`web/templates/user_behavior_analysis.html`

## 后续规划

### 更强的智能体自主能力

- 增加智能体自我反思机制，让系统能够评估回答质量和预约成功率。
- 引入更完整的多轮推理链，提升复杂预约和冲突处理能力。
- 根据真实用户反馈优化推荐策略。

### 更完整的单 Agent 工具编排

- 增强 Tool Gateway 的工具权限、幂等键和审计策略。
- 将用户行为分析沉淀为 Service 层定时任务和客户记忆 proposal。
- 把预约、推荐、咨询之间的结构化状态和客户记忆打通得更自然。

### 生产化能力

- 增加用户登录、权限控制和数据隔离。
- 增加更完整的异常处理和边界场景覆盖。
- 优化向量检索性能、缓存策略和响应速度。
- 支持 Docker 部署、云数据库和更标准的日志监控。
