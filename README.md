# almost-agent

FastAPI + PostgreSQL(pgvector) 的多用户 AI Agent 后端：带工具自主循环的对话、
知识库 RAG（含 PDF 表格/图表识别与知识图谱）、四层记忆系统、断点续跑。

```
Python 3.13 · uv · FastAPI · psycopg3 + pgvector · OpenAI 兼容 LLM 接口
源码 ~1.65 万行 · 测试 ~1.09 万行 · 654 个用例
```

---

## 快速开始

### 1. 起依赖服务

```bash
docker compose up -d          # pgvector/pgvector:pg17，端口 5432
```

### 2. 准备配置

```bash
cp .env.example .env          # 填入必填项，.env 已被 gitignore
```

必填 10 项（缺任一项启动即报错）：

| 变量 | 说明 |
|---|---|
| `BASE_URL` / `API_KEY` / `MODEL` | OpenAI 兼容网关，可指向官方、DashScope、Ollama、vLLM |
| `DATABASE_URL` | pgvector 连接串 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 嵌入模型与维度，两者须匹配 |
| `MEMORY_TOP_K` / `MEMORY_SCORE_THRESHOLD` | 长期记忆召回条数与相似度阈值 |
| `SHORT_TERM_MEMORY_MAX_TURNS` / `SHORT_TERM_MEMORY_MAX_TOKENS` | 短期记忆窗口 |

另有 `POSTGRES_PASSWORD`：`docker-compose.yml` 里 postgres 服务用它初始化，
须与 `DATABASE_URL` 中的密码一致。

其余项均有默认值，完整清单见 `.env.example`（按 LLM / 服务 / 数据库 / 记忆 /
知识库 / 图谱 / 检索质量分组注释）。

### 3. 启动

```bash
./scripts/dev.sh              # 开发模式：改 src/ 下任意 .py 自动热重启
uv run almost-agent dev       # 等价于上面
uv run almost-agent serve     # 生产模式（--host / --port 可覆盖）
```

监听 `127.0.0.1:8000`，健康检查 `GET /health`，交互式文档 `/docs`。

---

## 架构

```
                     客户端（SSE / JSON）
                              │
                    ┌─────────▼──────────┐
   api/middleware   │ request_id · auth  │  单 IP 滑动窗口限流
                    │ · 限流 · CORS      │  默认不信任 X-Forwarded-For
                    └─────────┬──────────┘
   api/routers      ┌─────────▼──────────────────────────────────┐
                    │ /v1/auth · /user · /v1/session · /v1/chat  │
                    │ /v1/kb（上传·切片·标签） · /v1/kg（图谱）  │
                    └─────────┬──────────────────────────────────┘
                    ┌─────────▼──────────┐
   agents/          │ loop.py 工具循环   │  max_turns 上限 · 并发信号量
                    │ budget.py 上下文   │  超预算从最旧整轮丢弃
                    └───┬────────────┬───┘
                        │            │
        ┌───────────────▼──┐   ┌─────▼─────────────────────────┐
 llm/   │ client.py 流式   │   │ tools/  工具（模型自主调用）  │
        │ OpenAI 兼容      │   │ kb_search 知识库检索          │
        └──────────────────┘   │ kg_search 图谱检索            │
                               │ memory_search / memory_write  │
                               │ web_search · weather          │
                               └─────┬─────────────────────────┘
                                     │
        ┌────────────────────────────▼─────────────────────────┐
  kb/   │ 解析 parsing → tables / table_cells / figures / office│
        │ 切片 chunking → 嵌入 → retrieval（改写·混合·重排）    │
        │ 图谱 graph / graph_store / align / community          │
        └────────────────────────────┬─────────────────────────┘
  memory/                            │
        短期(会话窗口) · 长期(向量) · 情景(episode) · 画像(profile)
        断点续跑 checkpoint_store · 认证 auth_store
                                     │
                          ┌──────────▼───────────┐
                          │ PostgreSQL + pgvector │
                          └──────────────────────┘
```

## 模块职责

| 目录 | 职责 |
|---|---|
| `api/` | FastAPI 应用、中间件（request_id / 鉴权 / 限流）、路由、请求响应模型 |
| `agents/` | Agent 主循环与周边：`loop.py` 主流程、`rewrite.py` 查询改写与目录注入、`messages.py` 上下文裁剪与断点缝合、`turn.py` 持久化与收尾、`prompts.py`、`budget.py` |
| `llm/` | OpenAI 兼容客户端，流式输出与超时控制 |
| `tools/` | 工具定义与注册：`kb_search`、`kg_search`、`memory_search`、`memory_write`、`web_search`、`weather` |
| `kb/` | 知识库：解析、切片、嵌入、检索、图谱、引用溯源、视觉识别 |
| `memory/` | 存储层：连接池与建表（`db.py`）、短/长期记忆、情景记忆、画像、会话、断点 |
| `migrations/` | Alembic 迁移脚本（见下方「数据库与迁移」） |
| `core/` | 配置（`config.py`）、异常、HTTP 客户端、日志 |
| `tasks/` | 后台任务：会话标题生成、定期清理 |
| `scripts/` | 运维脚本：知识库重建/重解析、重嵌入、检索评测、图谱检查 |

### kb/ 解析链（本项目的核心能力）

| 模块 | 职责 |
|---|---|
| `parsing.py` | 公共入口与 PDF 文本主流程：选项归一化、断行合并、章节识别 |
| `tables.py` | PDF 表格识别与页面组织：找表、跨页/同页续表合并、补表题与元信息、剔签字栏 |
| `table_cells.py` | 单元格清洗与表头对齐：数字断行还原、千分位/小数点归位、分组表头压平 |
| `figures.py` | 图表识别：文字密度/位图面积/矢量路径三道筛 + 视觉模型并行调用 |
| `office.py` | DOCX / XLSX 解析，含内嵌图片与内嵌表格 |
| `chunking.py` | 按章节/页切块，表格整表成片 |
| `retrieval.py` | 查询改写 → 标签预过滤 → 混合检索（关键词 + 向量 RRF）→ LLM 重排 |
| `graph.py` | 图谱构建入口：跨批合并（实体/关系/提及）与分批落库编排 |
| `graph_extract.py` | 抽取调用：进批过滤 → 按章节组批 → 单批容错（重问/拆半）→ 并发编排 |
| `graph_names.py` | 实体名归一化与可用性判据（停用词/谓词/数值/占位词），以及按名字在批内定位 |
| `graph_schema.py` | 抽取输出结构（实体/关系）与模型输出形状的容错解析（紧凑数组 ↔ 完整结构） |
| `graph_store.py` / `graph_write.py` / `kg_match.py` | 图谱存取：查询侧 mixin / 写入侧 mixin / 检索候选打分（均为 MemoryStore 的 mixin） |
| `align.py` / `community.py` | 实体对齐、社区发现 |

> `memory/store.py` 的 `MemoryStore` 由多个 mixin 组合而成（`KbStore` /
> `KgStore` / `KgWriteStore` / `AuthStore` / …），共享同一个连接池。所以
> 知识库、图谱、认证的存取方法都通过全局单例 `store` 调用；新增存储能力时
> 加一个 mixin 并挂进继承列表即可。

---

## 一次问答的链路

`POST /v1/chat` 以 SSE 流式返回：

1. **鉴权**：`AUTH_TOKEN` 非空时校验 `X-API-Key` / Bearer；否则校验用户 token
2. **装配上下文**：短期记忆（会话窗口）+ 长期记忆（向量召回 TopK）+ 情景记忆 + 画像
3. **循环**（上限 `MAX_TURNS`）：LLM 决定直接回答还是调工具 → 执行工具 → 结果拼回上下文
4. **检索**（走 `kb_search` 时）：目录注入 → 标签预过滤 → 混合召回 → 重排 → 带引用片段
5. **输出**：增量 token 流式下发，同时写入消息、更新摘要
6. **记忆落库**：本轮抽取的事实写入长期记忆（相似条目合并），情景与画像增量更新
7. **中断恢复**：连接断开后凭 `run_id` 调 `POST /v1/chat/resume` 续跑（`checkpoint_store`）

## 知识库管线

```
分片上传 /v1/kb/files → complete
   → 解析（PDF / DOCX / XLSX / 纯文本，表格与图表单独成片）
   → 章节识别 → 按章切块 → 批量嵌入 → 落库 chunks（向量 + 关键词）
   → 图谱构建（LLM 抽取实体关系）→ 社区发现 → 实体对齐
```

## 记忆系统

| 层 | 模块 | 内容 |
|---|---|---|
| 短期 | `short_term.py` | 会话窗口内的最近若干轮，超窗口后滚动摘要 |
| 长期 | `long_term.py` | 向量化的事实与偏好，召回按相似度阈值过滤，写入时相似条目合并 |
| 情景 | `episode_store.py` | 带 TTL 的事件片段 |
| 画像 | `profile.py` / `facets.py` | 用户维度的稳定画像与分面 |

## 认证与隔离

- 密码：`pbkdf2_sha256` 哈希 + 随机 salt
- Token：库里只存 SHA-256 摘要，比较用常量时间 `compare_digest`
- 隔离：所有业务查询强制带 `user_id` 条件；`/chat/resume` 校验 run 归属，非本人一律 404
- 可选静态令牌：`AUTH_TOKEN` 非空时 `/v1/**` 需携带
- 对外暴露前建议 `REGISTRATION_ENABLED=false`，改为管理员建号

---

## 数据库与迁移

Schema 由 **Alembic** 版本化管理，应用启动时（lifespan）自动 `upgrade head`。

```bash
uv run alembic upgrade head          # 升到最新
uv run alembic current               # 当前版本
uv run alembic history               # 历史
uv run alembic revision -m "说明"    # 新增迁移
```

**规范**：新增/修改表结构必须写新的迁移脚本，不要往 `memory/db.py` 的
`ensure_schema()` 里加 `ALTER TABLE`——它仅作为回滚锚点保留，被运维脚本与
部分测试使用。迁移脚本用 `op.execute` 透传 SQL 并保留 `IF NOT EXISTS` 幂等性；
`downgrade` 涉及删列删表，执行前先 `pg_dump` 备份。

## 开发命令

```bash
uv run pytest -q                     # 全量测试
uv run pytest tests/test_kb_pdf_tables.py -q   # 单文件
uv run ruff check src tests scripts  # lint
uv run ruff check --fix src          # 自动修复
uv run alembic upgrade head          # 迁移

# 常用运维脚本（参数见各文件 --help）
uv run python scripts/kb.py list          # 知识库维护入口
                          # 子命令：list / reparse / graph / reembed / coverage
uv run python scripts/eval_retrieval.py   # 检索质量回归评测
uv run python scripts/rebuild_graph.py    # 重建知识图谱
```

## 测试与质量

- 654 个用例，覆盖解析、检索、图谱、记忆、鉴权、并发与 SSE 流程
- CI/本地建议门槛：`pytest` 全绿 + `ruff` 零告警
- 解析相关用例取自真实文档（如招股书 PDF），改动抽取规则会立刻在
  `tests/test_kb_pdf_tables.py`、`tests/test_kb.py` 上暴露
