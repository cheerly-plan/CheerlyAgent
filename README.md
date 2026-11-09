# almost-agent

FastAPI + PostgreSQL(pgvector) 的多用户 AI Agent 后端：带工具自主循环的对话、
知识库 RAG（含 PDF 表格/图表识别与知识图谱）、四层记忆系统、断点续跑。

```
Python 3.13 · uv · FastAPI · psycopg3 + pgvector · OpenAI 兼容 LLM 接口
源码 ~2.33 万行 · 测试 ~1.48 万行 · 885 个用例
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

启动时 lifespan 会依次做：Alembic 迁移 → 重排后端自检（未装权重则告警并退回
LLM 重排）→ 把上次进程异常退出卡住的知识库文件标 failed。

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
        │ OpenAI 兼容      │   │ kb_search · kb_list_files     │
        └──────────────────┘   │ kg_search                     │
                               │ search_memory · write_memory  │
                               │ web_search · get_weather      │
                               └─────┬─────────────────────────┘
                                     │
        ┌────────────────────────────▼─────────────────────────┐
  kb/   │ 解析 parsing → tables / table_cells / figures / office│
        │ 章节 sections → 切片 chunking → 嵌入                 │
        │ 检索 router → scope → retrieval（混合·重排 → rerank）│
        │ 图谱 graph / graph_store / align / community          │
        └────────────────────────────┬─────────────────────────┘
  memory/                            │
        短期(会话窗口) · 长期(向量) · 情景(episode) · 画像(profile)
        写入门控 gate · 断点续跑 checkpoint_store · 认证 auth_store
                                     │
                          ┌──────────▼───────────┐
                          │ PostgreSQL + pgvector │
                          └──────────────────────┘
```

## 模块职责

> 下表路径均相对 `src/ai_agent/`。

| 目录 | 职责 |
|---|---|
| `api/` | FastAPI 应用、中间件（request_id / 鉴权 / 限流）、路由、请求响应模型 |
| `agents/` | Agent 主循环与周边：`loop.py` 主流程、`messages.py` 上下文裁剪与断点缝合、`turn.py` 持久化与收尾、`budget.py` |
| `llm/` | OpenAI 兼容客户端，流式输出与超时控制 |
| `tools/` | 工具定义与注册（见下方「工具一览」） |
| `kb/` | 知识库：解析、切片、嵌入、检索、图谱、引用溯源、视觉识别 |
| `report/` | 财报研报流水线：skill 大纲、抽数、同比、撰写、校验、PDF（见下方「研报流水线」） |
| `memory/` | 存储层：连接池与迁移（`db.py`）、短/长期记忆、情景记忆、画像、会话、断点 |
| `prompts/` | 提示词：系统提示与工具描述（`chat.py` / `tools.py`）、知识库提示（`kb.py`）、记忆抽取（`memory.py`）、后台任务（`tasks.py`）、装配器（`builder.py`） |
| `migrations/` | Alembic 迁移脚本（见下方「数据库与迁移」） |
| `core/` | 配置（`config.py`）、异常、HTTP 客户端、日志 |
| `tasks/` | 后台任务：会话标题生成、定期清理 |
| `scripts/` | 运维与评测脚本。日常维护走 `kb.py`，评测用例在 `scripts/cases/` |

### 研报流水线（`report/`）

数字路径（抽数／同比／校验／PDF）是确定性代码；skill 只决定「写什么」。
更细的现状与改文件指引见仓库根目录 [`next.md`](next.md)。

| 交付类型 | skill 示例 | 对外称呼 |
|---|---|---|
| 单公司／单银行 | `company_financial_report`、`bank_financial_report` | 财报深度分析 |
| 赛道行业横比 | `robotics_industry_report`、`semiconductor_industry_report`、`bank_industry_report` | **财报对比底稿（赛道分组版）** |

赛道底稿已含：组内对比表、结构梯队、利润—现金流—合同负债交叉、同比／不可比说明、有出处原文引述、下期外部数据源清单。不含估值／买卖建议；景气宏观未接数据源前不推断。

聊天入口的 `generate_report` 是**异步**的：只提交后台任务并返回 `job_id`（深度层约 5–8 分钟），结果用 `report_job_status(job_id)` 查，也可用同一 `job_id` 轮询 `GET /v1/report/jobs/{id}`；聊天断线不会中断生成。同步入口 `POST /v1/report` 契约不变。

| 改什么 | 去哪 |
|---|---|
| 代表公司、赛道、大纲 | `report/skills/<id>/SKILL.md` |
| 指标集、组内列、下期数据源文案 | `report/config/industries.yaml` |
| 新增行业脚手架 | `scripts/scaffold_industry.py` |

### 工具一览

模型自主调用，注册在 `tools/registry.py` 的 `TOOLS` 列表里。

| 工具名 | 模块 | 作用 |
|---|---|---|
| `kb_search` | `kb_search.py` | 知识库检索，返回带文件名/页码/章节的引用片段 |
| `kg_search` | `kg_search.py` | 知识图谱检索，返回实体邻域（三元组与关系描述） |
| `kb_list_files` | `kb_list.py` | 列出库内文件清单（替代「目录注入」，模型需要时才主动查） |
| `search_memory` / `write_memory` | `memory.py` | 长期/情景记忆的检索与写入 |
| `web_search` | `web_search.py` | 联网搜索（`ddgs`） |
| `get_weather` | `weather.py` | 天气查询 |

### kb/ 解析链（本项目的核心能力）

| 模块 | 职责 |
|---|---|
| `parsing.py` | 公共入口与 PDF 文本主流程：选项归一化、断行合并、章节识别 |
| `sections.py` | 标题与页眉甄别：页眉与真标题文本可能一字不差，靠字号 + 纵向位置 + 编号正则三者组合判定 |
| `tables.py` | PDF 表格识别与页面组织：找表、跨页/同页续表合并、补表题与元信息、剔签字栏 |
| `table_cells.py` | 单元格清洗与表头对齐：数字断行还原、千分位/小数点归位、分组表头压平 |
| `otsl.py` | OTSL 表格结构表示（单元格合并/表头语义） |
| `figures.py` | 图表识别：文字密度/位图面积/矢量路径三道筛 + 视觉模型并行调用 |
| `vision.py` / `ocr.py` | 视觉模型与 OCR 兜底 |
| `office.py` | DOCX / XLSX 解析，含内嵌图片与内嵌表格 |
| `chunking.py` | 按章节/页切块，表格整表成片；跨页续片重复携带表题与表头以保持自解释 |
| `retrieval.py` | 混合检索（关键词 + 向量 RRF，按主体保底配额）→ 重排 |
| `rerank.py` | 重排后端：默认 `local_ce`（`BAAI/bge-reranker-v2-m3`），未装则退回 LLM 重排 |
| `router.py` | 文档路由：问句点名即收口，一句都没点名时才用摘要向量补 |
| `scope.py` | 检索作用域：文件集合 / 报告期 / 主体的收窄与可达性兜底 |
| `meta.py` | 检索元数据：从文件名规则抽取主体 / 报告期，并给多主体文档的切片做主体归属 |
| `summary.py` | 文档级摘要：把一份文件压成一段「它讲的是什么」，单独 embedding 供路由使用 |
| `doc_profiles.py` + `kb/skills/*/SKILL.md` | 文档类型 skill：实体范围与关系谓词注入抽取 prompt；JSON 可运行时覆盖 |
| `citations.py` | 引用溯源 |
| `lexicon.py` | 词表（科目词、短词表等） |
| `pipeline.py` | 入库编排：解析 → 切片 → 嵌入 → 图谱，含卡住文件的恢复 |
| `store.py` | `KbStore` mixin：切片与关键词的存取、邻接、词频统计 |

### kb/ 图谱链

| 模块 | 职责 |
|---|---|
| `graph.py` | 图谱构建入口：跨批合并（实体/关系/提及）与分批落库编排 |
| `graph_extract.py` | 抽取调用：进批过滤 → 按章节组批 → 单批容错（重问/拆半）→ 并发编排 |
| `graph_names.py` | 实体名归一化与可用性判据（停用词/谓词/数值/占位词），以及按名字在批内定位 |
| `graph_schema.py` | 抽取输出结构（实体/关系）与模型输出形状的容错解析（紧凑数组 ↔ 完整结构） |
| `graph_store.py` / `graph_write.py` / `kg_match.py` | 图谱存取：查询侧 mixin / 写入侧 mixin / 检索候选打分（均为 MemoryStore 的 mixin） |
| `align.py` / `community.py` | 实体对齐、社区发现 |
| `graph_metrics.py` | 图谱构建质量指标（供 `scripts/eval_graph.py` 用） |

> `memory/store.py` 的 `MemoryStore` 由多个 mixin 组合而成（`KbStore` /
> `KgStore` / `KgWriteStore` / `AuthStore` / …），共享同一个连接池。所以
> 知识库、图谱、认证的存取方法都通过全局单例 `store` 调用；新增存储能力时
> 加一个 mixin 并挂进继承列表即可。

---

## 一次问答的链路

`POST /v1/chat` 以 SSE 流式返回：

1. **鉴权**：`AUTH_TOKEN` 非空时校验 `X-API-Key` / Bearer；否则校验用户 token
2. **装配上下文**：短期记忆（会话窗口）+ 长期记忆（向量召回 TopK）+ 情景记忆 + 画像 + 动态「知识库能力」块（按库内实际文档类型给出实体范围与关系谓词，并附库内已有报告期清单）
3. **图谱预检索**（命中关系词时）：`loop.py` 在模型决策**之前**先跑一遍图谱，把邻域塞进 system message。所以很多关系问句不必再调 `kg_search`；但报表科目金额仍必须走 `kb_search`
4. **循环**（上限 `MAX_TURNS`）：LLM 决定直接回答还是调工具 → 执行工具 → 结果拼回上下文。问句点名了文件主体或图谱实体时，用 `tool_choice="required"` 强制本轮必须查库，但**选 kb 还是 kg 仍归模型**
5. **检索**（走 `kb_search` 时）：
   - **文档路由**（`kb/router.py`）点名与摘要**二选一，不混用**：问句点名了主体/别名/文件名 → 命中即收口，不再用摘要补（补进来的常是别的公司）；一句都没点名 → 才按摘要向量选候选，距离超过 `kb_doc_route_max_distance` 的不进
   - **作用域收窄**（`kb/scope.py`）：按报告期与主体收敛。**精确实体只在旧文件里时改查那份文件**，避免「没提年份就只看最新一期」把历史期实体切掉
   - **混合召回**：关键词 + 向量 RRF，按主体保底配额（够近的主体各先占一个名额，「A 和 B 的营收」两家都出得来）→ 重排 → 距离阈值 → 带引用片段
6. **输出**：增量 token 流式下发，同时写入消息、更新摘要
7. **记忆落库**（后台线程池，不阻塞 SSE）：门控判断本轮值不值得抽取 → LLM 抽取（每轮 ≤ 2 条）→ 噪声过滤 → 三层去重 → 写入长期记忆 / 情景记忆 / 画像
8. **中断恢复**：连接断开后凭 `run_id` 调 `POST /v1/chat/resume` 续跑（`checkpoint_store`）

## 知识库管线

```
分片上传 /v1/kb/files → complete
   → 解析（PDF / DOCX / XLSX / 纯文本，表格与图表单独成片）
   → 章节识别 → 按章切块 → 批量嵌入 → 落库 chunks（向量 + 关键词）
   → 文档摘要单独 embedding（路由依赖它）
   → 图谱构建（LLM 抽取实体关系）→ 实体对齐 → 社区发现
```

## 记忆系统

| 层 | 模块 | 内容 |
|---|---|---|
| 短期 | `short_term.py` | 会话窗口内的最近若干轮，超窗口后滚动摘要；摘要伪装成一轮 user/assistant 注入 |
| 长期 | `long_term.py` | 向量化的事实与偏好，召回按相似度阈值过滤；写入走门控 → 抽取 → 噪声过滤 → 三层去重 |
| 情景 | `episode_store.py` / `episode_merge.py` | 带 TTL 的事件片段，支持相近情节合并 |
| 画像 | `profile.py` / `facets.py` / `profile_store.py` | 用户维度的稳定画像与分面；分面分互斥（覆盖）与累积（追加）两类，删除语义按分面区分 |

写入链路每轮最多产出 2 条，跑在 `_turn_finalizer` 线程池里，不阻塞 SSE。
门控（`gate.py`）用外部概率模型判断「本轮值不值得抽取」，低于
`memory_gate_threshold` 就跳过，省掉一次抽取调用；该服务不可用时 fail-open
（宁可多抽一次，不可静默丢记忆）。

## 认证与隔离

- 密码：`pbkdf2_sha256` 哈希 + 随机 salt
- Token：库里只存 SHA-256 摘要，比较用常量时间 `compare_digest`
- 隔离：所有业务查询强制带 `user_id` 条件；`/chat/resume` 校验 run 归属，非本人一律 404
- 可选静态令牌：`AUTH_TOKEN` 非空时 `/v1/**` 需携带
- 对外暴露前建议 `REGISTRATION_ENABLED=false`，改为管理员建号

### 主要端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat` | SSE 流式对话 |
| POST | `/v1/chat/resume` | 凭 `run_id` 断点续跑 |
| GET | `/v1/chat/pending_run` | 查询可续跑的 run |
| GET | `/v1/chat/memory` | 会话历史分页（不是长期记忆管理） |
|  | `/v1/session`、`/v1/auth`、`/user` | 会话、登录注册、用户信息 |
|  | `/v1/kb/**` | 知识库上传、切片、标签、维护 |
|  | `/v1/kg/**` | 图谱查询与维护 |

> 长期记忆目前**没有对外的查看/删除接口**，只有 `search_memory` / `write_memory`
> 两个工具。做隐私合规前需要先补一个管理端点。

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

---

## 评测

三套评测各自测不同的一层，**改动对应能力前先跑一遍，改完再跑一遍**。

| 脚本 | 用例集 | 条数 | 测什么 |
|---|---|---|---|
| `scripts/eval_retrieval.py` | `scripts/cases/retrieval_eval_cases_corpus.json` | 69 | 知识库检索：hit@1/@3、MRR、recall@10。一次只问一家 |
| `scripts/eval_graph_retrieval.py` | `scripts/cases/graph_eval_cases.json` | 29 | 图谱检索：问句能否查到正确的三元组，同时测**工具路径**与**预检索路径**两条 |
| `scripts/eval_tool_route.py` | `scripts/cases/tool_route_eval_cases.json` | 11 | 模型选工具：科目走 `kb_search`，关系走 `kg_search`，两者都要时两个都调。不执行检索 |
| `scripts/eval_graph.py` | `scripts/cases/gold_graph.json` | 1 份文件金标 | 图谱**构建**质量：实体/别名/关系召回率、孤立点比例、最大连通分量 |
| `scripts/eval_memory.py` | `scripts/cases/memory_eval_cases.json` | 15 | 长期记忆抽取：该写的写对、不该写的不写 |
| `scripts/eval_memory.py --recall` | `scripts/cases/episode_eval_cases.json` | 2 | 情景记忆召回（写入后检索，用完删除） |

另有 `scripts/cases/retrieval_eval_cases.json`（22 条，检索脚本的默认集）、
`scripts/cases/retrieval_eval_cases_3f.json`（20 条）是早期小样本集，保留作对照。

```bash
# 知识库检索（默认关闭重排，跑得稳；--rerank 打开会调 LLM）
uv run python scripts/eval_retrieval.py --cases scripts/cases/retrieval_eval_cases_corpus.json --top-k 10
uv run python scripts/eval_retrieval.py --validate        # 先校验用例与语料是否对齐

# 图谱检索（6 秒跑完，--out 落 json）
uv run python scripts/eval_graph_retrieval.py --validate
uv run python scripts/eval_graph_retrieval.py --out data/eval/graph_retrieval.json

# 工具选择（会调主对话模型；工具结果是桩，不执行检索）
uv run python scripts/eval_tool_route.py --list
uv run python scripts/eval_tool_route.py

# 图谱构建质量（只读，但要求库里有金标对应的那份文件；--before 传上次报告可对比）
uv run python scripts/eval_graph.py --gold scripts/cases/gold_graph.json --file-id <id>
```

### 当前基线（2026-11-03）

知识库检索（2026-11-03，当时 72 条、重排关、top_k=10）：hit@1 50/72、hit@3 64/72（89%）、
hit@10 69/72、MRR@10 0.800。通过线 hit@3 ≥ 85%。两家公司放在同一问里的 3 条已从用例集拿掉：评测只打一次 `kb_search`，工具不会把两个主体拆成两次查询。
用例集与语料强绑定，换语料（重新解析后）要先跑 `--validate`。

图谱检索（2026-11-03，当时 31 条，分 historical / historical_explicit / distractor / alias /
direct / negative 六组）。「中科飞测的注册资本是多少」「中化装备的董事长年薪是多少」已从用例集拿掉：这两问是科目金额，对话里由模型识别后走 `kb_search`，不会调用 `kg_search`。现在剩 29 条，负例 4 条都是库里没有的实体或编造的关系。

| 分组 | 条数 | 工具路径 | 预检索（仅触发的） |
|---|---|---|---|
| `historical` 历史期实体（不提年份） | 8 | 8/8 | 5/5 |
| `historical_explicit` 明说年份 | 3 | 3/3 | 3/3 |
| `distractor` 跨公司同类干扰 | 5 | 5/5 | 4/4 |
| `alias` 别名/简称 | 3 | 3/3 | 2/2 |
| `direct` 直查关系 | 6 | 6/6 | 2/2 |
| **`negative` 库里没有的实体** | 6 | **0/6** | 0/3 |
| 合计 | 31 | 25/31 | 16/19 |

**已知缺口（已被评测量化）**：问库里不存在的实体（「贵州茅台的前十大股东」）时，
图谱仍会返回别家公司的邻域并带精确数字。当时负例 6 条全部复现；拿掉 2 条科目金额后，
剩下的 4 条（库外实体或编造关系）在那次结果里也是 0/4。
`search_entities` 已返回 `matched` 字段（`name` / `alias` / `keyword` / `vector`）
可用于判定「库里到底有没有问的那个实体」，修复后这组应从 0/4 变成 4/4，
且正例 25 条不能掉。**别用调 `kb_graph_min_score` 的方式解决**——正负例的
score 区间重叠，动阈值必然误伤。

## 开发命令

```bash
uv run pytest -q                     # 全量测试
uv run pytest tests/test_kb_pdf_tables.py -q   # 单文件
uv run ruff check src tests scripts  # lint
uv run ruff check --fix src          # 自动修复
uv run alembic upgrade head          # 迁移

# 知识库维护入口（参数见 --help）
uv run python scripts/kb.py list                 # 子命令：list / reparse / graph
                                                 #       / reembed / meta / summary
                                                 #       / coverage / lexicon
uv run python scripts/kb.py meta --all           # 存量文件回填主体/报告期（切片跟着刷）
uv run python scripts/kb.py summary --all        # 存量文件补文档摘要索引（文档路由依赖它）

# 评测（详见上方「评测」）
uv run python scripts/eval_retrieval.py
uv run python scripts/eval_graph_retrieval.py
uv run python scripts/eval_memory.py             # 会调记忆模型
uv run python scripts/eval_memory.py --recall    # 情景记忆召回评测（写入后检索，用完删除）

# 图谱
uv run python scripts/kb.py graph --name 中行    # 只重建图谱；整份重解析加 --with-graph
uv run python scripts/check_kg_alignment.py      # 图谱对齐质量检查（清理历史错并的别名）
uv run python scripts/eval_graph.py              # 图谱构建质量评测
```

## 测试与质量

- 885 个用例，覆盖解析、检索、图谱、记忆、鉴权、并发与 SSE 流程
- CI/本地门槛：`pytest` 全绿 + `ruff` 零告警
- 解析相关用例取自真实文档（如招股书 PDF），改动抽取规则会立刻在
  `tests/test_kb_pdf_tables.py`、`tests/test_kb.py` 上暴露

**当前 lint 未清零**（4 项，均为存量问题，修掉即可回到零告警）：

| 位置 | 规则 | 说明 |
|---|---|---|
| `scripts/eval_retrieval.py:245` ×3 | E702 | 一行多条语句（分号），拆行即可 |
| `src/ai_agent/kb/graph.py:47` | F401 | `_subject_from_filename` 导入未使用，`--fix` 可自动删 |
