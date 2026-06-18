# Wellness Service Operations Agent

Wellness Service Operations Agent 是一个面向 local wellness service store 的状态化 Agent 系统。项目基于 FastAPI、LangGraph、LangChain、FAISS、SQLite 和多 Agent 协作架构，实现了服务咨询、多轮预约、员工匹配、排班冲突处理、客户偏好记忆、RAG-grounded 政策问答、人工确认、受控工具调用、Trace Replay 和回归评测骨架。

这个项目的核心目标不是只做一个普通的预约表单，而是把 local service operations 中的高频工作自动化：理解用户想咨询还是预约，判断服务偏好，匹配合适员工，检查可用时间，在写操作前要求明确确认，并在必要时结合知识库、历史行为和偏好数据给出更可靠的服务建议。

## 项目背景

在本地生活服务门店场景中，运营人员需要同时处理大量复杂事务：员工排班、顾客预约、服务项目咨询、价格计算、临时改约、员工偏好记录和收款确认等。随着员工数量和顾客量增加，传统人工前台模式很容易出现沟通成本高、信息遗漏、排班冲突和服务体验不稳定的问题。

因此，本项目尝试用 service operations Agent 的方式重构这一流程：让系统能够在明确业务约束下理解用户需求，并把不同任务分发给对应的专业 Agent 或受控工具处理。它适用于需要人员排班、预约调度和知识咨询的本地生活服务行业。

## Why this project matters

本项目重点不是聊天，而是生产级 Agent 工程问题：状态管理、工具治理、客户记忆、RAG grounding、人工确认、可观测性和评测。基础升级版本新增了 LangGraph operations runtime、Tool Governance Gateway、trace/replay 基础设施和独立的 `/api/operations/chat` endpoint，旧有 Web/chat 流程仍保持可用。

```text
User -> LangGraph Runtime -> Tool Gateway -> Booking/Knowledge/Memory Tools -> Trace/Eval
```

## Not a chatbot

这不是通用聊天机器人。Agent 必须在明确业务约束下完成工作：未确认不得创建预约，政策回答必须基于知识来源，客户记忆只能在合适时写入，不确定或敏感场景必须升级人工。

## Operations Workflows

### Tool governance

所有 operations runtime 中的业务动作都通过 `ToolGateway` 执行。工具声明 permission、Pydantic 输入/输出 schema 和确认策略；`create_booking`、`reschedule_booking`、`cancel_booking`、`write_customer_preference` 等写入或敏感工具在用户确认前不会执行。

### Memory lifecycle

客户长期记忆先生成 `MemoryProposal`，记录 type、content、evidence、confidence、sensitivity 和 confirmation policy。明确偏好可以生成 proposal，模糊表达会被忽略，过敏/隐私约束等敏感记忆需要确认；确认后的 `write_customer_preference` 会进入 `MemoryStore.upsert()`，重复偏好会更新，冲突偏好会返回 conflict result。后续预约会通过 `lookup_customer_profile` 读取已确认偏好并用于确认摘要。

### RAG grounding

咨询和政策类问题会走知识检索路径。每次 RAG 检索都会在 trace metadata 中保存 query、source、chunk_id、score 和 text preview，用户回复可以自然表达，但审计链路保留 citations。

### Trace replay

每轮 operations chat 都生成 node-level 和 tool-level trace events。`observability/trace_store.py` 提供 JSONL store，`observability/replay.py` 可以按顺序输出节点摘要，用于复现工具调用、确认拦截、RAG 检索和人工升级。

## Evaluation Results

当前 eval harness 是 deterministic smoke 版本，不依赖付费模型调用。运行：

```bash
python -m harness.runners.run_all --smoke
```

最新本地结果：

| Metric | Result | Threshold |
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
| escalation_accuracy | 1.00 | 0.90 |
| escalation_reason_accuracy | 1.00 | 0.90 |
| security_policy_accuracy | 1.00 | 0.90 |
| p95_latency_ms | reported | n/a |

数据集目前覆盖 50 条 smoke cases。完整路线仍计划扩展到 150+ cases，覆盖更多槽位、工具参数、RAG grounding、memory quality、安全策略和人工升级边界。

## Engineering Docs

- [Architecture](docs/architecture.md)
- [Tool governance](docs/tool-governance.md)
- [Memory lifecycle](docs/memory-lifecycle.md)
- [Evaluation](docs/evaluation.md)
- [Demo script](docs/demo-script.md)
- [Security policy](docs/security-policy.md)

## 核心能力

- **智能任务分类**：自动识别用户是在咨询服务、预约技师、查询无关问题，还是触发用户行为分析，并将请求路由到对应 Agent。
- **多 Agent 协作**：通过任务分类 Agent、咨询 Agent、预约 Agent 和用户行为 Agent 分工处理复杂流程，减少单个模块的职责膨胀。
- **RAG 知识咨询**：使用 FAISS 向量索引检索知识库内容，结合大模型生成自然语言回答，支持流式输出。
- **智能预约管理**：根据用户需求、员工专长、历史偏好和可用时间进行匹配，辅助完成预约确认。
- **受控工具调用**：新增 Tool Governance Gateway，对 read、write、external、sensitive 工具进行 schema 校验、确认拦截和 trace 记录。
- **状态化 Operations Runtime**：新增 LangGraph graph skeleton，把意图判断、槽位提取、工具计划、确认请求、响应生成、输出策略检查和 trace 写入建模为显式节点。
- **用户行为分析**：记录用户交互与预约行为，分析偏好模式，并用于后续推荐和个性化反馈。
- **个性化提醒**：在预约完成后，可结合实时天气等外部信息生成更贴近实际场景的提醒。
- **Embedding 缓存优化**：通过数据库缓存和文件缓存减少重复向量计算，提高知识检索性能。
- **数据管理能力**：支持知识库、技师信息和用户行为数据的增删改查，并在数据变化后自动维护索引。
- **日志与兜底机制**：保留关键处理过程日志，在信息不足或异常情况下提供更稳定的降级处理。

## 系统架构

项目采用严格的五层架构，核心原则是：**下层不能反向调用上层**。这样可以避免循环依赖，让业务逻辑、数据访问和接口编排保持清晰边界。

```text
Web & Application Layer
    ↓  app.py, web/：页面、路由入口、系统启动
API Layer
    ↓  api/：外部接口、请求编排、响应封装
Agents Layer
    ↓  agents/：AI Agent、任务路由、对话流程控制
Services Layer
    ↓  services/：业务逻辑、推荐算法、向量处理
DB Layer
    ↓  db/：数据模型、数据库连接、Repository
```

### 允许的调用方向

- Web 层调用 API 层
- API 层调用 Agents 层或 Services 层
- Agents 层调用 Services 层
- Services 层调用 DB 层

### 禁止的调用方式

- 下层反向调用上层
- Web 层绕过 API 直接访问 Services 或 DB
- Agents 层绕过 Services 直接访问 DB
- Services 层调用 Agents、API 或 Web

## Agent 设计

### Task Classification Agent

任务分类 Agent 是系统的主调度器，负责分析用户输入、判断任务类型，并把请求分发给合适的专业 Agent。

```text
用户输入 → 意图分析 → Agent 路由 → 响应协调
```

主要职责：

- 判断用户意图
- 维护对话状态
- 控制不同 Agent 之间的切换
- 处理无法分类或超出能力范围的问题

### Consultation Agent

咨询 Agent 负责知识问答场景，使用 RAG 流程从知识库中检索相关内容，再结合大模型生成回答。

```text
任务分类 → 知识检索 → FAISS 相似度搜索 → 流式回答
```

主要职责：

- 区分咨询问题类型
- 从知识库检索相关内容
- 构建提示词
- 生成自然语言回答

### Appointment Agent

预约 Agent 负责预约相关流程，包括解析用户输入、匹配技师、检查预约信息、生成确认消息等。

```text
任务分类 → 解析预约需求 → 技师匹配 → 预约确认
```

主要职责：

- 提取预约时间、服务项目、技师偏好等信息
- 匹配合适技师
- 处理信息缺失时的追问
- 生成预约结果和提醒

### User Behavior Agent

用户行为 Agent 更偏向后台智能分析，不完全依赖用户显式请求。它会根据交互记录、预约历史和偏好数据分析用户行为，为后续推荐提供依据。

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

系统并不让一个 Agent 处理所有事情，而是先判断用户意图，再分发给对应模块。这样可以让咨询、预约、行为分析等逻辑保持独立，也更容易扩展新的 Agent。

### 2. 用 RAG 解决专业知识回答

wellness service 相关的项目介绍、注意事项、适用人群等内容更适合通过知识库维护。RAG 能让回答基于可控知识来源，而不是完全依赖大模型自由生成。

### 3. 用用户行为让推荐更个性化

系统会记录用户的咨询、预约和偏好信息。后续在推荐技师或服务项目时，可以结合历史行为，而不是每次都从零开始询问。

### 4. 用分层架构保证可维护性

Agent 负责智能流程，Service 负责业务逻辑，Repository 负责数据访问。每层只关心自己的职责，减少后期修改时的连锁影响。

### 5. 为真实业务场景预留扩展空间

项目目前以本地 SQLite 和单体服务为主，但架构上预留了模型提供商切换、MCP 外部服务接入、后台任务、缓存优化和云端部署的扩展方向。

## 架构图

![System Architecture](./architecture%20.jpg)

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
Smart appointment AI agent/
├── agents/                         # 多 Agent 智能层
│   ├── task_classification_agent.py # 任务分类与主路由
│   ├── consultant_agent.py          # RAG 咨询 Agent
│   ├── appointment_agent.py         # 智能预约 Agent
│   ├── user_behavior_agent.py       # 用户行为分析 Agent
│   ├── operations/                  # LangGraph 状态化运营 Agent runtime
│   ├── task_classification/         # 意图识别、状态管理、路由逻辑
│   ├── consultant/                  # 知识检索、提示词、回答生成
│   ├── appointment/                 # 预约解析、技师匹配、消息构建
│   └── user_behavior/               # 行为记录、偏好管理、模式分析
├── api/                             # API 编排层
│   ├── operations.py                # Operations Agent endpoint
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
├── tools/                           # Tool Governance Gateway 与受控工具定义
├── memory/                          # Customer Memory Lifecycle
├── rag/                             # RAG adapters 与 citation metadata
├── observability/                   # Trace schema、JSONL store 与 replay CLI
├── harness/                         # Deterministic eval datasets、evaluators 与 runner
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

- Web 页面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs
- ReDoc 文档：http://127.0.0.1:8000/redoc

## Docker setup

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
pytest tests/test_task_classification_agent.py
```

运行 lint 和 eval smoke：

```bash
python -m ruff check .
python -m harness.runners.run_all --smoke
```

## Known limitations

- 当前 eval dataset 是 smoke 规模，还不是最终 150+ case 回归集。
- Memory store 是最小可测实现，尚未接入完整用户生命周期审计 UI。
- RAG adapter 是本地 deterministic grounding 层，MCP-backed RAG adapter 仍是扩展点。
- 新 operations endpoint 已可用，但旧 Web/chat 流程尚未切换到新 runtime。

## 主要页面

- 首页聊天与预约入口：`web/templates/index.html`
- 知识库管理：`web/templates/knowledge_management.html`
- 技师管理：`web/templates/technician.html`
- 技师排班：`web/templates/technician_schedule.html`
- 用户行为分析：`web/templates/user_behavior_analysis.html`

## 后续规划

### 更强的 Agent 自主能力

- 增加 Agent 自我反思机制，让系统能够评估回答质量和预约成功率。
- 引入更完整的多轮推理链，提升复杂预约和冲突处理能力。
- 根据真实用户反馈优化推荐策略。

### 更完整的多 Agent 协作

- 增加 Agent-to-Agent 通信机制，减少所有任务都依赖主分类器转发的问题。
- 将用户行为 Agent 的后台分析能力做得更稳定，支持定时任务和主动触达。
- 把预约、推荐、咨询之间的上下文记忆打通得更自然。

### 生产化能力

- 增加用户登录、权限控制和数据隔离。
- 增加更完整的异常处理和边界场景覆盖。
- 优化向量检索性能、缓存策略和响应速度。
- 支持 Docker 部署、云数据库和更标准的日志监控。
