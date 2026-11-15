# cheely-agent

FastAPI + PostgreSQL(pgvector) 的多用户 AI Agent 后端：带工具自主循环的对话、
知识库 RAG（含 PDF 表格/图表识别与知识图谱）、四层记忆系统、断点续跑。

```
Python 3.13 · uv · FastAPI · psycopg3 + pgvector · OpenAI 兼容 LLM 接口
源码 ~4.55 万行 · 测试 ~2.86 万行 · 1490 个用例
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

必填 9 项（缺任一项启动即报错）：

| 变量 | 说明 |
|---|---|
| `BASE_URL` / `API_KEY` / `MODEL` | OpenAI 兼容网关，可指向官方、DashScope、Ollama、vLLM |
| `DATABASE_URL` | pgvector 连接串 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 嵌入模型与维度，两者须匹配 |
| `MEMORY_TOP_K` / `MEMORY_SCORE_THRESHOLD` | 长期记忆召回条数与相似度阈值 |
| `SHORT_TERM_MEMORY_MAX_TOKENS` | 摘要输入（被挤出窗口的历史）的 token 预算 |

另有 `POSTGRES_PASSWORD`：`docker-compose.yml` 里 postgres 服务用它初始化，
须与 `DATABASE_URL` 中的密码一致。

其余项均有默认值。配置项的**唯一来源**是 `src/ai_agent/core/config.py` 的
`Settings`（全部字段，默认值与说明都在字段上方注释里）；`.env.example`
只列必填项、常用可选覆盖（注释示例）和按域分组的键名索引，不再重复抄写默认值。
`.env` 里出现 `Settings` 不认识的键时，启动会打 WARNING 但不会拒绝启动。

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
   api/routers      ┌─────────▼──────────────────────────────────────────┐
                    │ /v1/auth · /user · /v1/session · /v1/chat          │
                    │ /v1/kb · /v1/kg · /v1/report · /v1/inspect         │
                    └─────────┬──────────────────────────────────────────┘
                    ┌─────────▼──────────┐
   agents/          │ loop.py 工具循环   │  max_turns 上限 · 并发信号量
                    │ budget.py 上下文   │  超预算从最旧整轮丢弃
                    └───┬────────────┬───┘
                        │            │
        ┌───────────────▼──┐   ┌─────▼─────────────────────────────┐
 llm/   │ client.py 流式   │   │ tools/  工具（模型自主调用）      │
        │ OpenAI 兼容      │   │ kb_search · kb_list_files · kg_search│
        └──────────────────┘   │ search_memory · write_memory      │
                               │ web_search                        │
                               │ generate_report · report_job_status│
                               │ report_extract / compare / validate│
                               └─────┬─────────────────────────────┘
                                     │
        ┌────────────────────────────▼─────────────────────────────┐
  kb/   │ ingest/ 解析·切片·OCR   retrieval/ 路由·作用域·混合检索  │
        │ graph/ 抽取·对齐·社区   pipeline · ingest_worker · cninfo │
        └────────────────────────────┬─────────────────────────────┘
  report/                            │ 事实层底稿 + 可选深度层 deep.pdf
  memory/                            │ 短/长/情景/画像 · 门控 · 断点
  storage/                           │ 连接池 · 认证 · embeddings · store
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
| `tools/` | 工具定义与注册，按 `kb/` · `memory/` · `misc/` · `report/` 分包（见下方「工具一览」） |
| `kb/` | 知识库：`ingest/` 解析切片、`retrieval/` 检索、`graph/` 图谱、`pipeline` 入库编排、巨潮补数 |
| `report/` | 财报研报流水线：事实层底稿 + 可选深度层、异步任务、审核门禁（见下方「研报流水线」） |
| `memory/` | 短/长期记忆、情景记忆、画像、会话、断点续跑、写入门控 |
| `storage/` | 连接池与迁移入口（`db.py`）、认证、embeddings、组合式 `MemoryStore` |
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

默认可交付**双 PDF**：事实层底稿（纯数字／原文，可自动定稿）+ 深度层 `<stem>.deep.pdf`（推断结论，独立审核记录，下载走门禁）。深度层现状见 [`next.md`](next.md)，事实契约见 [`docs/deep-facts-schema.md`](docs/deep-facts-schema.md)。

聊天入口的 `generate_report` 是**异步**的：只提交后台任务并返回 `job_id`（深度层约 5–8 分钟），结果用 `report_job_status(job_id)` 查，也可用同一 `job_id` 轮询 `GET /v1/report/jobs/{id}`；聊天断线不会中断生成。同步入口 `POST /v1/report` 契约不变。审核队列读接口限管理员；批准／驳回／定稿对**生成者本人 + 管理员**开放。

| 改什么 | 去哪 |
|---|---|
| 代表公司、赛道、大纲 | `report/skills/<id>/SKILL.md` |
| 指标集、组内列、下期数据源文案 | `report/config/industries.yaml` |
| 深度层章节／评审／PDF 附录 | `report/deep/` |
| 新增行业脚手架 | `scripts/scaffold_industry.py` |

### 工具一览

模型自主调用，注册在 `tools/registry.py` 的 `TOOLS` 列表里。

| 工具名 | 模块 | 作用 |
|---|---|---|
| `kb_search` | `tools/kb/kb_search.py` | 知识库检索，返回带文件名/页码/章节的引用片段 |
| `kg_search` | `tools/kb/kg_search.py` | 知识图谱检索，返回实体邻域（三元组与关系描述） |
| `kb_list_files` | `tools/kb/kb_list.py` | 列出库内文件清单（替代「目录注入」，模型需要时才主动查） |
| `kb_read` | `tools/kb/kb_search.py` | 按「文件名 · 第N页」读回知识库原文与上下文 |
| `read_tool_result` | `tools/misc/tool_result.py` | 按 `tool_call_id` 读回任意历史工具结果全文，支持 offset 分页 |
| `search_history` | `tools/misc/history.py` | 检索当前会话旧对话：关键词精确匹配 + 语义召回补换词场景 |
| `read_history` | `tools/misc/history.py` | 按 seq 读取命中处前后上下文 |
| `search_memory` / `write_memory` | `tools/memory/memory.py` | 长期/情景记忆的检索与写入 |
| `web_search` | `tools/misc/web_search.py` | 联网搜索（`ddgs`） |
| `generate_report` | `tools/report/report_generate.py` | 提交研报后台任务，返回 `job_id`（不阻塞 SSE） |
| `report_job_status` | `tools/report/report_status.py` | 按 `job_id` 查进度与交付链接 |
| `report_extract_indicators` | `tools/report/report_extract.py` | 积木：从已入库财报抽指标 |
| `report_compare` | `tools/report/report_compare.py` | 积木：同比／环比等确定性对比 |
| `validate_figures` | `tools/report/report_validate.py` | 积木：正文数字白名单与出处校验。名字不带 `report_` 前缀——对话里答一段业绩分析同样要过它 |

> `report_pdf` 仍在仓库里，但不注册进 `TOOLS`——渲染含落盘与审核门禁，只能由服务端流水线触发。

### kb/ 解析链（`kb/ingest/`，本项目的核心能力）

| 模块 | 职责 |
|---|---|
| `parsing.py` | 公共入口与 PDF 文本主流程：选项归一化、断行合并、章节识别；密页走文字层、稀字页才送 OCR |
| `sections.py` | 标题与页眉甄别：页眉与真标题文本可能一字不差，靠字号 + 纵向位置 + 编号正则三者组合判定 |
| `tables.py` | PDF 表格识别与页面组织：找表、跨页/同页续表合并、补表题与元信息、剔签字栏 |
| `table_cells.py` | 单元格清洗与表头对齐：数字断行还原、千分位/小数点归位、分组表头压平 |
| `otsl.py` | OTSL 表格结构表示（单元格合并/表头语义） |
| `figures.py` | 图表识别：文字密度/位图面积/矢量路径三道筛 + 视觉模型并行调用 |
| `vision.py` / `ocr.py` | 视觉模型与 OCR 兜底 |
| `office.py` | DOCX / XLSX 解析，含内嵌图片与内嵌表格 |
| `chunking.py` | 按章节/页切块，表格整表成片；跨页续片重复携带表题与表头以保持自解释 |
| `meta.py` | 检索元数据：从文件名规则抽取主体 / 报告期，并给多主体文档的切片做主体归属 |
| `summary.py` | 文档级摘要：把一份文件压成一段「它讲的是什么」，单独 embedding 供路由使用 |
| `doc_profiles.py` + `kb/skills/*/SKILL.md` | 文档类型 skill：实体范围与关系谓词注入抽取 prompt；JSON 可运行时覆盖 |
| `lexicon.py` | 词表（科目词、短词表等） |

### kb/ 检索链（`kb/retrieval/`）

| 模块 | 职责 |
|---|---|
| `retrieval.py` | 混合检索（关键词 + 向量 RRF，按主体保底配额）→ 重排 |
| `rerank.py` | 重排后端：默认 `local_ce`（`BAAI/bge-reranker-v2-m3`），未装则退回 LLM 重排 |
| `router.py` | 文档路由：问句点名即收口，一句都没点名时才用摘要向量补 |
| `scope.py` | 检索作用域：文件集合 / 报告期 / 主体的收窄与可达性兜底 |
| `citations.py` | 引用溯源 |

### kb/ 图谱链（`kb/graph/`）

| 模块 | 职责 |
|---|---|
| `graph.py` | 图谱构建入口：跨批合并（实体/关系/提及）与分批落库编排 |
| `graph_extract.py` | 抽取调用：进批过滤 → 按章节组批 → 单批容错（重问/拆半）→ 并发编排 |
| `graph_names.py` | 实体名归一化与可用性判据（停用词/谓词/数值/占位词），以及按名字在批内定位 |
| `graph_schema.py` | 抽取输出结构（实体/关系）与模型输出形状的容错解析（紧凑数组 ↔ 完整结构） |
| `graph_store.py` / `graph_write.py` / `kg_match.py` | 图谱存取：查询侧 mixin / 写入侧 mixin / 检索候选打分（均为 MemoryStore 的 mixin） |
| `align.py` / `community.py` | 实体对齐、社区发现 |
| `graph_metrics.py` | 图谱构建质量指标（供 `scripts/eval_graph.py` 用） |

> 入库编排在 `kb/pipeline.py`；重解析可走独立 `kb/ingest_worker.py`，避免 OCR 拖死 API。
> `storage/store.py` 的 `MemoryStore` 由多个 mixin 组合而成（`KbStore` /
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
7. **记忆落库**（后台线程池，不阻塞 SSE）：LLM 抽取（每轮 ≤ 2 条）→ 噪声过滤 → 三层去重 → 写入长期记忆 / 情景记忆 / 画像。Jev 门控默认关闭（见「记忆系统」）
8. **中断恢复**：连接断开后凭 `run_id` 调 `POST /v1/chat/resume` 续跑（`checkpoint_store`）

### 业绩分析 skill（脚本化预取）

问「某公司业绩怎么样」时，模型原来要自己决定调哪些工具、调几次——实测同一
个问题调 7–13 次，每次不同：抽取分几次、重复读原文、同指标反复算。把顺序
写进提示词只能降低概率，保证不了。

`agents/_loop/performance.py` 改成脚本化：命中「点名库内主体 + 含业绩词」
时，**在模型决策之前**按固定顺序预取（`report_extract_indicators` 抽单季与
累计两套口径），把结果作为一条 system 消息注入，然后**把
`report_extract_indicators` 从本轮工具列表里摘掉**。

```text
命中条件   主体在库 + 含业绩词 + 不是要研报/导出
报告期     问句里取（2026Q3 / 2026H1）；取不到用库内最新一期
降级       主体识别不到 / 抽不到数据 → 什么都不做，模型自主决策
幂等       消息上打 performance_prefetch 标记，续跑不重复预取
```

**为什么摘工具而不是靠提示词**：prefetch 之后提示词里写了「不要重复抽」，
实测三轮模型每轮还是调了一次 `report_extract_indicators`。摘掉工具是确定性
的，不受模型随机性影响。

与 `report/skills/` 的区别：那边是**产物模板**（研报长什么样、该抽哪些指标、
公司名单），这里编的是**行为顺序**（先调谁再调谁）。

实测同一问题的工具调用次数：13 次 → **5/6/6 次**（三次）。

### 工具调用的事件与防打转

循环里每次工具调用会下发 `tool_start` / `tool_progress` / `tool_end`。`tool_start`
带的可读信息：

```json
{"type": "tool_start", "name": "web_search",
 "args": {"query": "本周热点新闻"},
 "label": "本周热点新闻",
 "call_index": 2}
```

`label` 是从该工具的关键参数里抽的一句话（`tools_exec._tool_call_label`），
`call_index` 是本轮里这个工具第几次被调用。

**`tool_start` 之后紧跟一条 `tool_progress`**：前端（`ai_chat` 的 `useChat.ts`）
渲染的其实是 `tool_progress.message`，`tool_start` 只会套一句静态文案
（`toolStatusHint`）。原来只有 `generate_report` 会发 `tool_progress`，
其余工具全程停在「正在处理…」。现在每次调用前统一补一条：

```json
{"type": "tool_progress", "name": "web_search",
 "message": "正在搜索「本周热点新闻」（第 3 次）…"}
```

文案由 `_tool_progress_hint(name, label, index)` 拼：工具→中文动词、参数摘要、
第几次（首次不显示「第 1 次」避免噪音）。未收录的工具回退成「正在处理「X」…」。

**长工具的中间进度**：`kb_search` 实测冷态 9s、热态 2.7s，耗时几乎全在重排
（向量召回 < 0.1s），只给开头一句的话前端要干等近十秒。它现在走
`_PROGRESS_TOOLS` 的线程通道，检索层在重排前推一条：

```json
{"type": "tool_progress", "message": "正在重排 12 条候选结果…"}
```

用户看到的状态因此是「正在检索知识库「X」…」→「正在重排 N 条候选结果…」→ 答案。
其余工具实测都在 0.5s 内（`kg_search` 0.07s、`search_memory` 0.09s、
`web_search` 0.43s），不需要中间进度。

**防打转**：同一工具返回**完全相同的结果**累计 `RETRY_LIMIT`（默认 3）次后，
本轮不再执行它，改为回一条终止提示（`RUNTIME_TOOL_EXHAUSTED`），让模型基于
已有结果作答并如实说明哪部分没查到。上面那次事故里，同一份行情数据换了
6 种关键词搜了 6 遍（那天是周六、本就没有行情），只追加提示不足以制止。
禁用按工具粒度生效，不影响同轮其它工具。

**单工具次数预算**：`MAX_TOOL_CALLS_PER_RUN`（默认 6，0 = 不限）按工具分别计数。
`MAX_TURNS` 管的是模型决策步数，而**一个步骤里可以并行发起多个工具调用**，
光靠步数上限兜不住——上面那次 13 次调用就发生在 10 步预算之内。

三个上限的关系：

| 机制 | 管什么 | 触发后的行为 |
|---|---|---|
| `MAX_TURNS` | 模型决策步数 | 整轮强制收尾 |
| `MAX_TOOL_CALLS_PER_RUN` | 单工具累计次数 | 该工具不再执行，回预算提示 |
| `RETRY_LIMIT` | 单工具**相同结果**次数 | 该工具不再执行，回终止提示 |

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
| 抽取纪律 | `prompts/memory.py` + `memory/long_term.py` | 领域中立：只记「用户本人的长期事实」，任务对象（公司/病患/订单/合同…）一律不记。规则层**不含任何行业词**，只靠结构特征判定（见下方「抽取规则」） |
| 短期 | `short_term.py` | 会话窗口的最近历史；超 token 预算才滚动摘要（不按固定轮数提前摘要）；摘要伪装成一轮 user/assistant 注入 |
| 长期 | `long_term.py` | 向量化的事实与偏好，召回按相似度阈值过滤；写入走门控 → 抽取 → 噪声过滤 → 三层去重 |
| 情景 | `episode_store.py` / `episode_merge.py` | 带 TTL 的事件片段；相近情节合并，相似但不同的事互相链接（召回一跳扩展） |
| 画像 | `profile.py` / `facets.py` / `profile_store.py` | 用户维度的稳定画像与分面；分面分互斥（覆盖）与累积（追加）两类，删除语义按分面区分 |
| 候选 | `promotion.py` / `promotion_store.py` | 观察态证据、跨会话复现、候选审核与自动晋升；候选 facet 由抽取结果或确定性线索词推断 |
| 归档 | `archive.py` / `archive_store.py` | L1/L2 的只读副本，按需展开；重复蒸馏会更新文本/证据，不会产生副本 |
| 开关 | `setting_store.py` | 用户级记忆开关：关闭后不写、不注入、不挂记忆工具；已有数据保留，可查看/导出/删除 |

写入链路每轮最多产出 2 条，跑在 `_turn_finalizer` 线程池里，不阻塞 SSE。

**Jev 门控（`gate.py`）默认关闭**（`MEMORY_GATE_ENABLED=false`）。它用外部概率
模型判断「本轮值不值得抽取」，设计意图是省掉一次小模型调用；端到端评测
（104 条黄金集）实测它是**净负收益**：

| 配置 | 放行正例 | 拦截负例 |
|---|---|---|
| 开（阈值 0.65，原措辞） | 1/40 | 44/44 |
| 开（最佳措辞 + 阈值 0.10） | 36/40 | 29/44 |
| **关** | **40/40** | **43/44** |

「放行正例」= 该记的没被门控拦掉。用户说「我叫魏佳楠」时门控概率只有 0.05
（阈值 0.65）——根因不是阈值，而是把「值得记」与「不值得记」写在同一句里问，
决策模型无法回答。抽取模型本身已能拒掉 43/44 的噪声，门控只是多丢一次。
代码与开关保留，换决策模型后可用同一套评测重测再决定。

L1/L2 支持分页查看与单条/全部删除，`GET /v1/memory/export` 导出 L1–L4 全量数据；
`DELETE /v1/memory` 清空长期记忆。删除 L1/L2 单条时会同步清理同文候选与归档副本，
避免删掉后又被后台晋升/蒸馏带回来。归档重复蒸馏走 `ON CONFLICT DO UPDATE`，
文本、出处与证据会随源数据刷新；查询支持 `limit` / `offset`。

### 抽取规则（领域中立）

判断标准只有一条：**这是关于用户本人的长期事实，还是本次任务的对象？**
这条 prompt 原则适用于任何领域，规则层则用结构特征把它落地：

| 规则 | 命中即丢 | 覆盖的形态 |
|---|---|---|
| `_ONGOING` | `(用户)?(正在\|已\|刚刚)(动词)` | 本轮动作：查询、翻译、排查、运行… |
| `_OBJECT_SUBJECT` | 句首是 `患者\|该案\|订单\|客户\|本合同\|该学生…` | 任务对象主语句（任何领域） |
| `_DOC_FACT` | 主语是文档本身 + 记载/披露/显示 | 「病历记载」「判决书载明」「工单显示」 |
| `_IMPERATIVE_TASK` | 「帮我写」「把X翻译」 | 一次性任务请求 |
| `_TEMPORAL` / `_NOW_ACTING` | 下周出差、我正在跑 | 短期安排与当前状态 |
| `_ASSISTANT_RECAP` | 助手建议/回答/决定 | 对话纪要 |
| `_MEMORY_QUESTION` | 「你还记得我是谁吗」 | 询问记忆本身 |

历史：早期这套规则按金融 + 知识库写成（股东、招股书、年报、股票代码、
`MEMORY_DOMAIN_NOISE_ENABLED` 开关），换领域要手动关。三组金融词在 169 条
用例上命中 0 次后删除，改由 `_DOC_FACT` 一条通用规则承接——它天然覆盖
医疗的病历、法律的判决书、客服的工单。

**规则边界**：同一结构不同含义（「我在做一份研报」vs「我在做仓库盘点」）、
领域特有的隐私边界、模糊定义，规则都判不了，交给抽取模型；这也是为什么
评测用例要按 `owner=rule|model` 分组，而不是把失败一律算给规则。

### 上下文管理

- 所有上下文预算按 token 计数：`LLM_CONTEXT_TOKEN_BUDGET` 是硬上限（默认 40000 token），先扣
  `CONTEXT_RESERVE_TOKENS`（输出/工具参数预留）和工具 schema 的 token
  （启动时按真实 schema 测量，见 `CONTEXT_TOOL_SCHEMA_RESERVE_TOKENS`）
  得到实际输入预算；`CONTEXT_SOFT_LIMIT_RATIO` 是软水位，达到后按 token
  成本优先收缩工具结果。计数走 Qwen tokenizer（`QWEN_TOKENIZER_PATH`）。
- 工具结果优先收缩：按「可省 token × 新旧 × 是否被引用」排序，优先收
  大而旧、未被引用的结果，换成元数据占位（工具名、命中文件/页码、参数、
  `tool_call_id`）；当前轮永不老化。
- 完整原文进入进程内 `tool_result_pool`（LRU + TTL + 单条上限），
  `read_tool_result` 优先从池里按 `tool_call_id` 读回，池子没有才回落到
  当前消息；池里的内容是裁剪前原文，不受单工具输出上限影响。
- 软水位仍不够时整轮丢弃最旧对话；`assistant(tool_calls)` 与对应 `tool`
  消息始终成对，不产生孤儿 tool 消息。
- 摘要和 `agent_session.summary_until_seq` 一起更新；冷启动只加载游标之后
  的消息，避免已摘要的原文与摘要重复进入上下文。
- 工具结果按 pool → `agent_tool_result`（裁剪前全文）→ 当前 messages →
  `agent_chat_history` 查找；进程重启或池子淘汰后仍能回读完整原文。
- 多路召回按 RRF 融合（`MEMORY_FUSION=rrf`）：向量 / 词面 / 语义历史按名次
  累加，不依赖分数量纲；一次性内容融合后再降权。
- 摘要只作索引；旧对话细节用 `search_history` 回原文（关键词 + 语义双路），
  再用 `read_history` 按 seq 看前后文，数字/条款不直接引用摘要。
- 历史对话向量存在 `agent_chat_embedding`：新消息落库后异步建索引，
  老消息由维护线程增量回填；`HISTORY_SEMANTIC_TOP_K=0` 可退回纯关键词。
- 已读过的占位、低价值旧工具对按整对（assistant(tool_calls)+tool）删除，
  不产生孤儿 tool 消息；移出活跃窗口的快照同步从结果池淘汰。
- 启动时预热 Qwen tokenizer；不可用时告警并退回启发式计数。

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
| GET/PUT | `/v1/memory/settings` | 用户级记忆开关 |
| GET/DELETE | `/v1/memory/profile` | L1 画像分页查看 / 清空 |
| DELETE | `/v1/memory/profile/{id}` | 删除单条画像 |
| GET/DELETE | `/v1/memory/episodes` | L2 情景分页查看 / 清空 |
| DELETE | `/v1/memory/episodes/{id}` | 删除单条情景 |
| GET | `/v1/memory/export` | 导出 L1–L4 全量记忆 |
| DELETE | `/v1/memory` | 清空长期记忆（画像/情景/候选/归档） |
|  | `/v1/memory/candidates`、`/v1/memory/archive` | 候选审核、归档查询与分页 |
|  | `/v1/session`、`/v1/auth`、`/user` | 会话、登录注册、用户信息 |
|  | `/v1/kb/**` | 知识库上传、切片、标签、维护 |
|  | `/v1/kg/**` | 图谱查询与维护 |
| POST | `/v1/report` | 同步生成研报 |
| POST | `/v1/report/jobs` | 异步提交研报任务（202） |
| GET | `/v1/report/jobs/{id}` | 轮询任务进度 |
| GET | `/v1/report/files/{name}` | 研报附件下载／`?preview=1` 在线预览 |
|  | `/v1/report/reviews/**` | 审核队列、批准／驳回、定稿 |
|  | `/v1/inspect/**` | 管理员回看用户会话与每轮工具调用 |

> 普通用户只能操作自己的记忆，管理员可通过 `user_id` 指定用户；关闭记忆开关
> 只停止读写与工具挂载，不会隐式删除已有数据。

---

## 数据库与迁移

Schema 由 **Alembic** 版本化管理，应用启动时（lifespan）自动 `upgrade head`。

```bash
uv run alembic upgrade head          # 升到最新
uv run alembic current               # 当前版本
uv run alembic history               # 历史
uv run alembic revision -m "说明"    # 新增迁移
```

**规范**：新增/修改表结构必须写新的迁移脚本，不要往 `storage/db.py` 的
`ensure_schema()` 里加 `ALTER TABLE`——它仅作为回滚锚点保留，被运维脚本与
部分测试使用。迁移脚本用 `op.execute` 透传 SQL 并保留 `IF NOT EXISTS` 幂等性；
`downgrade` 涉及删列删表，执行前先 `pg_dump` 备份。

---

## 评测

三套评测各自测不同的一层，**改动对应能力前先跑一遍，改完再跑一遍**。

| 脚本 | 用例集 | 条数 | 测什么 |
|---|---|---|---|
| `scripts/eval_retrieval.py` | `scripts/cases/retrieval_eval_cases_corpus.json`（需 `--cases` 指定） | 69 | 知识库检索：hit@1/@3、MRR、recall@10。一次只问一家 |
| `scripts/eval_graph_retrieval.py` | `scripts/cases/graph_eval_cases.json` | 29 | 图谱检索：问句能否查到正确的三元组，同时测**工具路径**与**预检索路径**两条 |
| `scripts/eval_tool_route.py` | `scripts/cases/tool_route_eval_cases.json` | 11 | 模型选工具：科目走 `kb_search`，关系走 `kg_search`，两者都要时两个都调。不执行检索 |
| `scripts/eval_graph.py` | `scripts/cases/gold_graph.json` | 1 份文件金标 | 图谱**构建**质量：实体/别名/关系召回率、孤立点比例、最大连通分量 |
| `scripts/eval_memory.py` | `scripts/cases/memory_eval_cases.json` | 15 | 长期记忆抽取：该写的写对、不该写的不写（需模型） |
| `scripts/eval_memory.py --recall` | `scripts/cases/episode_eval_cases.json` | 2 | 情景记忆召回（写入后检索，用完删除） |
| `scripts/eval_memory_rules.py` | `scripts/cases/memory_rule_cases.json` | 69 | **规则层离线**：该记未记 / 不该记记了，含逐规则归因 |
| `scripts/eval_memory_rules.py --cases cases/memory_crossdomain_cases.json` | 同左 | 60 | 跨领域（医疗/法律/教育/客服），防止规则过拟合 |
| `scripts/eval_memory_rules.py --cases cases/memory_realworld_cases.json` | 同左 | 10 | **真实对话**（线上 170 条用户消息里的自述片段） |
| `scripts/eval_memory_golden.py` | `scripts/cases/memory_golden_cases.json` | 104 | 端到端（模型 + 规则），按 `owner=rule/model/either` 分组 |
| `scripts/eval_multiturn_compress.py` | `scripts/cases/multiturn_compress_cases.json` | 28 | 多轮压缩：历史被压掉后，追问指向的事实还在不在 |
| `tests/test_memory_e2e.py` | — | 1 条生命周期 | 写入 → 注入 → 晋升 → 归档 → 删除；真库 + 确定性桩，不调模型 |

### 记忆评测的四个层次

记忆质量不能只测一层。现在的分工：

| 层 | 套件 | 需要 | 能测什么 |
|---|---|---|---|
| 规则层 | `eval_memory_rules` | 无 | 过滤逻辑有没有误杀/漏放；脚本输出**逐规则归因**（哪条正则拦的 / 根本没覆盖） |
| 跨领域 | `eval_memory_rules_crossdomain` | 无 | 医疗/法律/教育/客服四域，防止规则过拟合到金融+技术 |
| 真实分布 | `eval_memory_rules_realworld` | 无 | 线上真实对话：混合意图、撤回指令、意向 vs 事实 |
| 端到端 | `eval_memory_golden` | 模型 | 模型 + 规则的整体行为，按 `owner` 分组 |

**为什么要分层**：`memory_health` 的「噪声拦截 100%」是规则验证规则，同义反复，
证明不了规则本身对不对。规则层用人工标注的 `extracted` 做独立真值；端到端层
再验证模型实际输出。两者互不掩盖。

用例字段（规则层）：

```json
{"id": 1, "expect": "remember", "owner": "rule", "user": "原始问句",
 "extracted": [{"kind": "profile", "facet": "name", "content": "模型抽出的值"}]}
```

- `expect`: `remember` 全部保留 / `forget` 全部拦掉 / `split` 混合意图（配
  `expect_keep` + `expect_drop`）/ `either` 不计分
- `owner`: `rule` 归确定性规则；`model` 是开放式提问，规则不该管，放进规则层
  评测只会制造假失败

```bash
# 规则层（零依赖，CI 里跑的就是这三条）
uv run python scripts/eval_memory_rules.py
uv run python scripts/eval_memory_rules.py --cases cases/memory_crossdomain_cases.json --min-forget 0.75
uv run python scripts/eval_memory_rules.py --cases cases/memory_realworld_cases.json --verbose

# 端到端（需要模型；按 owner 分组看哪一层在出错）
uv run python scripts/eval_memory_golden.py --owner rule
uv run python scripts/eval_memory_golden.py --verbose

# 多轮压缩（确定性模式进 CI；--summarize llm 看真实摘要的事实保留率）
uv run python scripts/eval_multiturn_compress.py --keep-last 1
uv run python scripts/eval_multiturn_compress.py --sweep 1,2 --summarize llm
```

**写评测的三条经验**（都是踩过的坑）：

1. **断言不能用字面匹配**。摘要把「关索引」写成「关闭索引」是正常改写，
   字面匹配会假报丢失；用同义组 `must_any: [["关索引", "关闭索引"]]`。
2. **用例要有防呆**。第一版多轮压缩用例 23/28 条的关键事实恰好落在保留窗口里，
   根本测不出问题；现在 `test_facts_live_in_early_turns` 会断言这一点。
3. **标注错了和代码错了同样要修**。真实数据那次 2 条「误杀」其实是标注写了
   不该写的期待（短期行程当常住地）。

`eval_memory` 在 `eval/config.json` 里是 **blocking**：nightly 或显式
`--allow-external` 跑，通过率跌破阈值会阻断发布。三个评测脚本的通过线集中在
`eval/config.json` 的 `thresholds`，由 runner 统一以 `--threshold` 注入；脚本里的
同名常量只作本地直接运行时的默认值。记忆生命周期门禁由 CI 独立的
`memory-e2e` job 承担（自带 pgvector 服务、不依赖模型密钥），不再在
`config.json` 里重复注册 suite，避免 nightly 在两个 job 各跑一遍。

另有 `scripts/cases/retrieval_eval_cases.json`（22 条，检索脚本的默认集）、
`scripts/cases/retrieval_eval_cases_3f.json`（20 条）是早期小样本集，保留作对照；
上表第一行的 69 条 corpus 集需要显式 `--cases` 指定。

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

# 记忆端到端（需要 Postgres；CI 的 memory-e2e job 会自动起 pgvector）
uv run pytest tests/test_memory_e2e.py -q
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

- 1490 个用例，覆盖解析、检索、图谱、记忆、研报流水线／深度层、鉴权、并发与 SSE 流程
- CI/本地门槛：`pytest` 全绿 + `ruff` 零告警
- 解析相关用例取自真实文档（如招股书 PDF），改动抽取规则会立刻在
  `tests/test_kb_pdf_tables.py`、`tests/test_kb.py` 上暴露

**当前 lint 未清零**（11 项，多为存量问题；其中 6 项可用 `--fix` 自动修）：

| 位置 | 规则 | 说明 |
|---|---|---|
| `agents/loop.py` ×2 | E402 | 模块级 import 不在文件顶部 |
| `kb/ingest/table_cells.py` ×2 | F541 | f-string 无占位符 |
| `kb/ingest_worker.py` | F401 | `time` 未使用 |
| `report/analyze_evidence.py` | F401 | 未使用导入 |
| `report/compose_segment.py` ×5 | F401 / F811 | 未使用导入与同名符号重定义 |
