# Agent Runtime Lab

> 事件驱动 AI Agent Runtime · Task Harness · MCP / RAG · Reliability Eval

一个面向可靠执行与工程评测的本地 AI Agent 平台。基于 **LangGraph + FastAPI + MCP**，支持多轮对话、文档 RAG/GraphRAG、联网搜索，以及通过邮件、Excel 等工具完成多步任务。核心 **Task Harness** 使用显式步骤状态、标准化工具事件和完成证据驱动进度，并提供 Baseline vs Harness 可靠性评测；敏感操作由 **HITL** 确认保护。

> 详细架构见 [docs/STRUCTURE.md](docs/STRUCTURE.md) · Task Harness 设计见 [docs/TASK_HARNESS.md](docs/TASK_HARNESS.md)

---

## 功能概览

| 能力 | 说明 |
|------|------|
| **Agent 对话** | LangGraph ReAct，SSE 流式展示 Thought / Action / Observation |
| **Task Harness** | 事件驱动步骤状态机、完成证据、gather/process/deliver 阶段 gate、自动续跑与收敛 |
| **Agent Eval** | 30 条确定性任务集，对比 Baseline/Harness 完成率、工具调用、延迟与失败类型 |
| **HITL** | 发微信、发邮件、导出 Excel 等操作需前端确认后执行 |
| **MCP 工具** | 微信、邮件、本地文件、联网搜索、表格格式化、Excel 导出 |
| **RAG** | Milvus 混合检索（向量 + BM25 + Rerank） |
| **GraphRAG** | Neo4j 知识图谱 + Milvus 向量混合检索 |
| **会话管理** | 多会话、自动标题、长对话摘要 |

---

## 快速开始

### 前置条件

- Python **3.12+**
- 智谱 API Key（[开放平台](https://open.bigmodel.cn/)）
- 可选：Redis、PostgreSQL、Milvus、Neo4j（完整 RAG/GraphRAG/HITL 能力）

### 方式一：本地开发（Windows 推荐）

```powershell
# 1. 克隆并进入项目
cd my_AI_chat

# 2. 安装依赖
pip install -r requirements.txt
pip install -r requirements-postgres.txt   # 可选：Postgres 聊天 + Checkpointer

# 3. 配置环境变量
copy .env.example .env
# 编辑 .env，至少填入 ZHIPUAI_API_KEY=

# 4. 一键启动（自动尝试拉起 MCP）
scripts\start.bat
# 或：python app.py
```

浏览器打开：**http://localhost:5001**

> **微信发消息**依赖 Windows 桌面版微信 + `wxauto4`，仅本地 Windows 环境可用。Docker / Mac / Linux 可使用其余工具（搜索、RAG、邮件等）。

### 方式二：Docker Compose（完整基础设施）

适合体验 RAG / GraphRAG，无需本机单独安装 Milvus、Redis、Neo4j。

```bash
# 1. 配置 .env（至少 ZHIPUAI_API_KEY；可选 TAVILY_API_KEY、EMAIL_*）
cp .env.example .env

# 2. 启动全部服务
docker compose up -d

# 3. 访问
# Web:   http://localhost:5001
# Neo4j: http://localhost:7474  (neo4j / 12345678)
```

`docker compose` 会自动启动：etcd、MinIO、Milvus、Redis、Neo4j、Web（含 MCP）。

### 方式三：仅 MCP 工具服务

```bash
python mcp_server.py
# 默认 http://localhost:8090/mcp
```

---

## 最小可运行配置

只需以下一项即可启动 Web 并进行基础对话（工具能力会随缺失服务降级）：

```env
ZHIPUAI_API_KEY=你的密钥
```

| 配置 | 影响 |
|------|------|
| 无 Redis | 聊天缓存降级，Harness 元数据不可用 |
| 无 Postgres | 无 HITL、无跨重启 Checkpointer |
| 无 Milvus | 无法 RAG / GraphRAG 索引 |
| 无 Tavily | 无法 `web_search` |
| 无 MCP | 降级为纯 LLM（见终端警告） |

完整变量说明见 [.env.example](.env.example)。

---

## Demo 场景

### 1. 多步任务 + Harness + HITL

**输入：**

> 搜索最近一周上海嘉定天气，整理成表格，发到 xxx@example.com

**预期行为：**

1. 检测到复杂任务 → 生成并压缩为 3～5 步执行计划，前端展示 checklist
2. **gather** 阶段：调用 `web_search` / `web_search_batch`
3. **process** 阶段：调用 `format_pretty_table`（HITL 确认）
4. **deliver** 阶段：调用 `send_email`（HITL 确认）
5. 若 Agent 口头说「将要发送」但未调工具 → 系统自动 nudge 续跑

### 2. 文档 RAG 问答

1. 上传 PDF / Word
2. 选择 **普通 RAG** 或 **GraphRAG** 索引模式
3. 提问文档内容，Agent 优先依据检索结果回答

### 3. 跨轮续跑

1. 发起复杂任务，若单轮未完成
2. 发送 **「继续」** → 从 Redis 恢复 Harness 计划与进度，继续执行

---

## 架构简图

```
浏览器 (template/1.html)
    │  HTTP / SSE
    ▼
FastAPI (app.py → routes/*)
    │  asyncio
    ▼
LangGraph ReAct Agent (agent/*)
    │  MCP HTTP
    ▼
mcp_server.py ──→ 微信 / 邮件 / 搜索 / 文件 / Excel

并行存储：
  chat_store  → Postgres + Redis
  rag_service → Milvus (+ Neo4j GraphRAG)
  checkpointer → Postgres (HITL 恢复)
```

---

## 项目结构

```
├── app.py              # Web 入口
├── mcp_server.py       # MCP 工具服务
├── agent/              # LangGraph Agent、Harness、HITL
├── chat/               # 会话存储与摘要
├── rag/                # RAG / GraphRAG
├── app_mcp/            # MCP 生命周期与工具实现
├── routes/             # FastAPI 路由
├── evals/              # Baseline vs Harness 任务集、评分器与报告
├── template/1.html     # 前端单页
├── docs/               # 架构与设计文档
└── scripts/start.bat   # Windows 一键启动
```

详见 [docs/STRUCTURE.md](docs/STRUCTURE.md)。

---

## 常用 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| POST | `/chat/message/stream` | 流式对话（SSE） |
| POST | `/chat/hitl/resume/stream` | HITL 确认后继续 |
| POST | `/chat/cancel` | 取消当前 Agent 运行 |
| GET | `/service/mcp/status` | MCP 服务状态 |

---

## 开发说明

```powershell
# 文档解析依赖（PDF/Office）
scripts\install_doc_deps.bat

# 微信工具测试
python scripts/testwx.py

# 单元测试（Task Harness 逻辑，无需 API Key / 外部服务）
pip install -r requirements-dev.txt
pytest tests/ -v
```

Agent 可靠性评测（使用 Fake Tools，不会真实联网、发邮件或调用微信）：

```powershell
# 快速验证 12 个任务
python -m evals.runner --limit 12 --repeats 1

# 30 个任务各运行 3 次，输出 baseline vs Harness 报告
python -m evals.runner --repeats 3
```

数据集、评分规则与报告说明见 [evals/README.md](evals/README.md)。

GitHub Actions 在 push / PR 时自动运行上述测试（见 [`.github/workflows/python-app.yml`](.github/workflows/python-app.yml)）。

**Task Harness** 默认开启（`AGENT_TASK_HARNESS=1`）。关闭后复杂任务不再拆步规划，仅保留基础 ReAct + HITL。

设计原理、阶段划分、续跑机制见 **[docs/TASK_HARNESS.md](docs/TASK_HARNESS.md)**。

---

## 界面预览

<details>
<summary>点击展开截图</summary>

页面总览：

![页面总览](https://github.com/user-attachments/assets/e21f8905-19f8-4ff1-bcb2-a5443a99e19c)

Agent 推理流程、HITL 确认、Task Harness checklist：

![推理流程](https://github.com/user-attachments/assets/90315bf2-ed5e-46f3-bbbb-68c8f57c412e)

![HITL](https://github.com/user-attachments/assets/7a99b372-786d-47cb-acff-7b2be6ae6b19)

![Harness](https://github.com/user-attachments/assets/d07e35d8-57da-4149-bac8-51cebc08751d)

</details>

---

## License

个人 / 学习用途项目。部署前请妥善保管 `.env` 中的 API Key，勿提交到公开仓库。
