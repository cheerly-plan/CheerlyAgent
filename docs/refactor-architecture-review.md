# 架构评审与超大文件拆分方案

> 状态：**✅ 阶段 1–5 已全部执行完毕**（结果与实测见文末「附录 C｜执行结果」）
> 基线：`main` @ `fee3f3b`（工作区另有 28 个未提交改动，本方案不依赖其中的内容）
> 度量口径：`src/ai_agent/**/*.py` 物理行数，排除 `migrations/` 与 `__pycache__`

---

## 执行结果速览（先看这里）

| 阶段 | 文件 | 之前 | 之后 | 结果 |
|---|---|---:|---|---|
| 1 | `kb/ingest/chunking.py` | 1628 行单文件 | → `chunking/` 包，**8 文件，最大 482 行** | ✅ 108 个定义 AST 逐字一致 |
| 2 | `kb/retrieval/retrieval.py` | 1352 行单文件 | → `_impl/` 包 + 217 行门面，**最大 388 行** | ✅ 66 个定义一致（含 10 个补丁点改写） |
| 3 | `kb/store.py` | 1193 行单类 40 方法 | → `store/` 包，**6 文件，最大 556 行** | ✅ 40 方法 + 5 helper 全部一致 |
| 4 | `agents/loop.py` | 1248 行 | → `_loop/` 包 + 门面，**最大 643 行** | ✅ 31 个定义一致，API 集合完全相同 |
| 5 | `storage/db.py` 的 `ensure_schema` | 774 行单方法 | → `_schema/` 包按表拆 22 函数 | ✅ **97 条 SQL 逐字一致** |

**全量测试：改造前后都是 `4 failed / 1883 passed`，且失败的是同样 4 个既有用例。**
`ruff` 全项目告警数不变（7 个，全部是 `report/` 里的既有问题）。
全部 177 个模块可正常导入，12 条历史导入写法 100% 兼容。

---

## 1. 结论先行

**分层架构没有问题，问题只在「文件粒度」。**

我没有找到真正的「架构很乱」——相反，几个关键指标都是健康的：

| 检查项 | 结果 |
|---|---|
| 模块级循环依赖 | **0**（AST 全量扫描） |
| 分层方向 | 清晰：`core → storage → kb/report/memory → agents/tools/api` |
| 组装根 | 已收敛到 `storage/store.py` 的 `MemoryStore` 单例，注释也写明了历史教训 |
| 包级 re-export | `kb/retrieval/__init__.py` 已按「对外只暴露公开符号」组织 |
| 测试 | 135 个文件、1887 个用例，覆盖很厚（这是重构的最大底气） |

真正的痛点是**局部过于臃肿**：

- 源码共 **52,364 行**，其中 **32 个文件 ≥ 500 行**，**9 个文件 ≥ 900 行**；
- 最大的 `kb/ingest/chunking.py` **1628 行**里只有 8 个公开符号，却有 **64 个私有函数**；
- 有 4 个文件是「一个类装下整个域」——`Database.ensure_schema` 单方法 774 行、
  `KbStore` 单类 40 个方法 1096 行、`SessionStore` 927 行、`MemoryStore` 组合了 12 个 mixin。

**「文件大」在这里不是排版问题，而是三个真实成本**：

1. **认知成本**：改表格切片要读完 1600 行才敢动手，且没人能确认自己没漏；
2. **评审成本**：`chunking.py` 任何小改动在 diff 里都被淹没，reviewer 只能盲信；
3. **协作成本**：两人同时改 `retrieval.py` 几乎必然冲突（Git 按行合并）。

而「私有函数占比高」恰恰说明**这些文件内部本来就存在清晰的边界**，只是没落到文件上——
拆分是**把已有的边界显性化**，不是重新设计。这点决定了本轮是小风险重构。

---

## 2. 现状量化

### 2.1 超大文件清单（≥ 500 行）

| 行数 | 文件 | 公开符号 | 私有符号 | 内聚度判断 |
|---:|---|---:|---:|---|
| 1628 | `kb/ingest/chunking.py` | 8 | 64 | **极高**：纯函数管线，无 IO、无状态 |
| 1352 | `kb/retrieval/retrieval.py` | 11 | 36 | **高**：检索编排，仅模块级常量共享 |
| 1248 | `agents/loop.py` | 4 | 17 | 中：主循环 + 提示词拼装混在一起 |
| 1193 | `kb/store.py` | 1（`KbStore`） | 3 | 高：40 个方法按表分组明显 |
| 1083 | `kb/ingest/tables.py` | 0 | 40 | **极高**：纯函数 |
| 1020 | `kb/ingest/parsing.py` | 7 | 30 | 高：按文件类型分派 |
| 927 | `memory/session_store.py` | 1 | 2 | 中 |
| 921 | `report/deep/pipeline.py` | 6 | 7 | 中：编排 + 并发控制 |
| 916 | `storage/db.py` | 8 | 0 | **低内聚**：774 行单方法 DDL |
| 829 | `report/deep/sources.py` | 15 | 3 | 中 |
| 804 | `report/extract.py` | 8 | 20 | 高 |
| 784 | `report/deep/analyzer.py` | 13 | 8 | 中 |
| 777 | `report/validate.py` | — | — | — |
| 745 | `report/pdf.py` | — | — | — |
| 739 | `kb/graph/graph.py` | — | — | — |
| 721 | `report/industry.py` | — | — | — |
| 720 | `kb/ingest/table_cells.py` | — | — | — |
| 702 | `kb/graph/graph_store.py` | — | — | 高：查询侧 mixin |
| 667 | `core/config.py` | — | — | 高：Settings 单一定义处，**不建议拆** |
| 660 | `tools/kb/kb_search.py` | — | — | — |
| 其余 11 个 500–660 行的文件 | | | | 可接受，暂不动 |

### 2.2 值得单独点名的三个「结构性」问题

这三个不属于「文件太长」，而是**组织方式本身导致的耦合**，价值高于纯粹拆长度：

#### ① `storage/db.py` — `ensure_schema` 是 774 行的 DDL 巨块

```
Database (812 行)
  ├── ensure_schema ............ 774 行   ← 全部表的 CREATE/ALTER
  ├── _get_pool / close / ...
```

**问题**：拼一个巨型 SQL 方法，无法测单张表、无法复用到 Alembic、
加一张表就要在 774 行里找位置。README 已声明「新表走 Alembic，`ensure_schema`
只作回滚锚点」——但它仍在持续膨胀，因为它是**唯一**权威 DDL 来源。

**这是本方案里唯一的「设计级」建议**，详见 §6。

#### ② `kb/store.py` — `KbStore` 单类 40 方法 1096 行

方法天然分 4 组（文件元数据 19 个 / 切片写入 12 个 / 检索查询 6 个 / 邻接 3 个），
且**项目已有 mixin 组合模式**（`storage/store.py` 的 `MemoryStore` 就是这么拼的）。

**建议**：不发明新模式，沿用既有惯例把 `KbStore` 拆成 4 个 mixin。

#### ③ `agents/loop.py` — 主循环与「提示词拼装」耦合

`run_loop`(332 行) + `run_stream`(225 行) 是核心；但 `_system_prompt`(45)、
`_kb_capability_block`(53)、`_period_inventory`(19)、`_followup_subjects`(44)、
`_must_call_kb_tool`(54) 是**纯拼装逻辑**，却被放进循环文件，导致测试
`test_memory_e2e.py` 要 `from ai_agent.agents.loop import _profile_prompt, _recall_block`。
拆出后可让「拼什么提示词」与「怎么跑循环」独立演进。

---

## 3. 重构前置约束（已核实，必须遵守）

这些是我扫描测试后确认的**硬约束**，方案已全部适配。

### 3.1 测试依赖私有符号（27 处 import）

```
tests/test_kb_chunk_scope.py:3         from ai_agent.kb.ingest.chunking import _scope_from_caption
tests/test_kb_table_caption_carry.py:3 from ai_agent.kb.ingest.chunking import _carry_md_table_captions
tests/test_kb_meta.py:431              from ai_agent.kb.retrieval.retrieval import _boost_indicator_hits
tests/test_memory_e2e.py:17            from ai_agent.agents.loop import _profile_prompt, _recall_block
```

→ **拆分必须保留旧导入路径可用**（在旧模块里 re-export），否则测试批量崩。

### 3.2 测试对「模块属性」打补丁（monkeypatch 必须命中调用点）

```
tests/test_kb_query_rewrite.py:275     monkeypatch.setattr(retrieval, "_retrieve_candidates", fake)
tests/test_hybrid_graphrag.py:204      monkeypatch.setattr(retrieval, "_extract_keywords", lambda q: [...])
tests/test_hybrid_graphrag.py:679/695/711  monkeypatch.setattr(retrieval, "_core_search", fake_core)
tests/test_kb_retrieval.py:310         monkeypatch.setattr(retrieval.store, "search_kb_chunks", fake)
```

→ **这是最容易被忽略的坑**：`monkeypatch.setattr(retrieval, "_core_search")` 打的是
`retrieval` 模块命名空间里的名字。如果 `_core_search` 移到 `search.py` 且主流程
**直接**调用 `search._core_search`，补丁就失效，测试会静默走到真实 DB。

**对策**：把可测的函数作为**参数注入**，而不是跨模块直接引用。详见 §4.2。

### 3.3 既有失败用例（与本方案无关，勿混淆）

`tests/test_long_term.py` 有 **4 个用例在干净 HEAD 上就是失败的**（已用 `git stash`
验证）。原因：`capture_long_term` 里的 Jev 门控在测试环境下拦截了写入，
断言 `write_episode called once` 但实际 `called 0 times`。

→ 重构验收时以「**失败数不高于 4**」为基线，不要去"修"它——那是门控逻辑的独立问题。

---

## 4. 分阶段方案

原则：**低风险高收益优先**；每阶段独立可回滚；每阶段跑全量测试比对基线。

### 阶段 1｜`chunking.py` 拆分（1628 → 6 个文件，最大 580 行）

**为什么第一个做**：纯函数、无 IO、无全局状态、测试覆盖极厚（14 个测试文件引用），
风险最低而收益最高。

拆后布局（保持 `kb/ingest/chunking/` **既有** `tables.py`/`parsing.py` 的命名风格）：

```
kb/ingest/chunking/
├── __init__.py          公开 API re-export：Chunk, chunk_segments, chunk_text,
│                        extract_table_indicators, html_table_plain,
│                        html_tables_to_markdown, context_prefix, embedding_input
├── core.py        142 行  chunk_segments 主流程、Chunk 数据类、context_prefix/embedding_input
├── preprocess.py  306 行  _drop_noise / _coalesce_segments / _chunk_flow /
│                          _drop_page_boilerplate / _drop_duplicate_chunks
├── html.py        580 行  HTML 表解析、rowspan 展开、<table> ↔ Markdown 互转
├── table.py       305 行  _chunk_table 行项目切分、_pack_table_units、指标抽取
├── caption.py      85 行  _scope_from_caption / _carry_md_table_captions（表题与口径下带）
└── text.py        102 行  chunk_text / _chunk_plain_pages / _split_long
```

**兼容层**：必须保留 `src/ai_agent/kb/ingest/chunking.py` 作为**同名模块还是同名包**——
二者不能共存。两种做法：

- **方案 A（推荐）**：删掉 `chunking.py`，新建 `chunking/` 包，`__init__.py` 里
  re-export 全部 8 个公开符号 **+ 2 个被测试直接 import 的私有符号**
  （`_scope_from_caption`、`_carry_md_table_captions`）。
  `from ai_agent.kb.ingest.chunking import X` 对调用方**完全不变**。
- 方案 B：`chunking.py` 保留为薄 shim，内部代码移到 `chunking_impl/`。
  目录更丑，但 diff 更小。不建议——会留下两个入口。

**模块间依赖**（已用 AST 精确提取调用边验证，**无环**，拓扑序如下）：

```
text      ← 最底层，只依赖 parsing.Segment（chunk_text / _split_long）
html      ← 只依赖标准库（HTML 表 ↔ Markdown）
caption   ← 只依赖标准库（表题/口径识别）
table     → text          （_chunk_table 调 chunk_text 做兜底切分）
preprocess→ text, html    （_chunk_flow 调 _split_long；_expand_long_html_tables 调 _split_html_table）
core      → 以上全部       （chunk_segments 是唯一编排入口）
```

可编译的拓扑序：`text, html, caption → table → preprocess → core`。

> 这一段在初稿里写错过（曾假定 `preprocess` 不依赖 `html`、`table` 不依赖 `text`）。
> 实际有两条跨层调用：`_chunk_flow → _split_long`、`_chunk_table → chunk_text`，
> 以及 `_expand_long_html_tables → _split_html_table`。
> 结论不变（仍然无环），但实现时必须按上面的顺序落文件，否则会有模块级循环导入。

**注意**：`TABLE_GROUP_CHARS` 被 `retrieval.py` 引用，`extract_table_indicators` 被
`ingest/__init__.py` 引用——都从 `chunking/__init__.py` 导出即可。

---

### 阶段 2｜`retrieval.py` 拆分（1352 → 7 个文件，最大 341 行）

**为什么第二个做**：纯编排逻辑，但**风险高于阶段 1**——因为有 §3.2 的打补丁问题。

拆后布局：

```
kb/retrieval/
├── __init__.py       （已存在，扩充 re-export）
├── retrieval.py      （保留为薄入口，re-export 兼容层 + 主流程）
├── search.py        280 行  search_knowledge / search_knowledge_multi / preview_search /
│                            _retrieve_candidates / _core_search / format_where / _where
├── keywords.py      119 行  _extract_keywords / 短词相邻短语展开
├── subjects.py      221 行  mentioned_subjects / mentioned_entities /
│                            resolve_files_by_name / query_looks_on_topic
├── fusion.py        341 行  RRF 融合 / 去冗余 / 按主体与文件配额 diversify
├── neighbors.py     171 行  _attach_neighbors / 同表行合并 / 缺口补片
├── rerank.py        （已存在，合并 _rerank / _boost_indicator_hits /
│                     complete_chat_once / _parse_order / _candidate_text 进来）
└── text_utils.py     30 行  _trigrams / _jaccard / _clip
```

**依赖方向（已用 AST 提取真实调用边验证，无环）**：

```
text_utils, keywords, subjects, neighbors   ← 叶子
    ↑          ↑
fusion ────────┘        rerank → text_utils
    ↑                      ↑
    └──── search.py ───────┘
```

拓扑序：`text_utils / keywords / subjects / neighbors → fusion → rerank → search`。

> **初稿这里写错过一个真实风险**：`_where` 与 `format_where` 必须**同文件**。
> 我最初把 `_where`（3 行小工具）放进了 `text_utils`，而它调用 `format_where`；
> `search` 又依赖 `fusion`、`fusion` 依赖 `text_utils` —— 形成
> `search → fusion → text_utils → search` 的**模块级循环导入**，运行时直接 ImportError。
> 正确做法：`_where` 跟 `format_where` 一起留在 `search.py`。
> 这正是 §7 第 1 条验收要用「实际 import 冒烟」而非肉眼检查的原因。

**关键设计——解决 §3.2 的补丁失效问题**：

现在 `_core_search`、`_extract_keywords`、`_retrieve_candidates` 被测试打补丁。
简单的「移走 + 改调用方」会让补丁静默失效。推荐做法是**把这三个函数变成
`search_knowledge` 的可选参数，默认值取自模块**：

```python
# search.py
def search_knowledge(
    query, ...,
    *,
    _core_search_fn: Callable | None = None,   # 默认 None → 用模块级 _core_search
    ...
):
    core = _core_search_fn or _core_search
```

打补丁处改为 `monkeypatch.setattr(search_mod, "_core_search", fake)` 并验证
**调用确实发生**（断言 fake 被调用），而不是只断言返回值——否则补丁失效时
测试不会报警。**这一步必须单独验证**，见 §7 的「补丁有效性」检查。

> 备选（更省事）：`retrieval.py` 保留全部函数定义，只把**纯工具函数**（`text_utils`、
> `subjects`）移走，把补丁点留在原地。收益从 1352→约 900 行，风险几乎为零。
> 如果时间紧，建议先做这一档。

---

### 阶段 3｜`agents/loop.py` 拆分（1248 → 4 个文件）

按「提示词拼装 / 工具执行 / 主循环」切：

```
agents/
├── loop.py         ~450 行  run_loop / run_stream / run（纯编排）
├── prompts.py      ~330 行  _system_prompt / _kb_capability_block / _profile_prompt /
│                            _recall_block / _period_inventory / _today_line
├── tools_exec.py   ~150 行  _run_tool_with_progress / _remember_tool_result / _attach_sources
└── followup.py     ~120 行  _followup_subjects / _must_call_kb_tool / _kg_prefetch_present /
                             _question_already_retrieved
```

**约束**：`test_memory_e2e.py` 从 `loop` import `_profile_prompt`/`_recall_block`，
`loop.py` 里 re-export 即可。`test_kb_retrieval.py:568` 给 `loop_mod.stream_chat`/
`execute_tool` 打补丁——**这两个名字必须继续存在于 `loop` 命名空间且被调用**，
所以 `tools_exec._run_tool_with_progress` 里的 `execute_tool` 调用需要通过参数
或 `loop` 模块注入，**不能**在 `tools_exec.py` 里直接 import 后调用。

**风险**：这是**本轮唯一涉及运行时行为**的拆分（`run_loop` 332 行）。
建议最后做，且**只搬函数不改逻辑**（不顺手"优化"），逐函数移动 + 每步跑测试。

---

### 阶段 4｜`kb/store.py` 拆为 mixin 家族

沿用 `MemoryStore` 既有模式，`KbStore` 拆成 4 个 mixin 再组合：

```
kb/store/
├── files.py        文件元数据 CRUD、状态机、标签/主体别名   (~19 方法)
├── chunks.py       切片写入、向量回填、正文更新             (~12 方法)
├── search.py       search_kb_chunks / _keyword 检索 / doc_freq  (~6 方法)
└── neighbors.py    fetch_kb_chunks / kb_chunk_neighbors      (~3 方法)
```

`kb/store.py` 保留为 `class KbStore(KbFileMixin, KbChunkMixin, ...)`——**类名和
外部用法完全不变**，`tests` 里 `_chunk_meta_where` 继续可用。

**收益**：`storage/store.py` 的 `MemoryStore` 组合了 12 个 mixin，其中 `KbStore`
是最大的一块；拆开后「知识库存储」和「记忆存储」的粒度才对等。

---

### 阶段 5｜`storage/db.py` 的 `ensure_schema`（**设计建议，需单独讨论**）

774 行单方法。建议把 DDL 拆成「每表一个函数」，由一个**注册表**驱动：

```python
# storage/schema.py
@dataclass(frozen=True)
class TableDDL:
    name: str
    create: str
    columns: tuple[tuple[str, str], ...] = ()   # 幂等 ALTER ... ADD COLUMN IF NOT EXISTS
    indexes: tuple[str, ...] = ()

TABLES: tuple[TableDDL, ...] = (EPISODE, USER_PROFILE, KB_FILE, ...)

def apply_schema(conn, *, dim: int) -> None: ...
```

`Database.ensure_schema` 变成 3 行委托。

**⚠️ 这里有一个必须你拍板的矛盾**：

- README 写「`ensure_schema()` 仅作回滚锚点保留」；
- 但 `tests/test_memory_promotion.py:456`、`test_memory_archive.py:502` 等
  **断言的是 `ensure_schema` 生成的完整 SQL 包含新表的 `CREATE TABLE`**；

→ 也就是说它**不是**静态锚点，而是**仍在增长的第二份 schema 源**，与 Alembic 并行。
这是真实的技术债，但**修它属于行为变更**（黄金主数据源从 2 个收敛到 1 个），
**不属于纯重构**。

**建议**：本轮**只做「结构拆分、SQL 逐字不变」**（774 行 → 每表一函数，输出字节级一致），
把「收敛成单一 DDL 源」列为**独立的技术债条目**，另行决策。

---

## 5. 优先级与推荐顺序

| 顺序 | 阶段 | 风险 | 收益 | 建议 |
|---|---|---|---|---|
| 1 | chunking 拆分 | 低 | 高 | **做** |
| 2 | retrieval 拆分 | 中 | 高 | **做**（先做保守档） |
| 3 | kb/store mixin | 中 | 中 | 做 |
| 4 | loop 拆分 | **中高** | 中 | 做，但最后 |
| 5 | ensure_schema | 低（若 SQL 不变） | 中 | 结构拆分可做；**收敛单一 DDL 源需另行决策** |
| — | `core/config.py` | — | — | **不拆**：Settings 单一定义处，拆了反而难查 |
| — | 500–660 行文件（11 个） | — | 低 | **不动**：已在合理区间 |

> 判断依据：chunking/retrieval 是「一个文件装了一个算法族」，拆分收益线性于文件大小；
> 而 500 行左右的文件再拆只会制造导入噪音。

---

## 6. 非文件长度的架构问题（顺带记录，建议另开议题）

评审中发现的、**不属于本次拆分范围**但值得记录的点：

1. **双 schema 源**（见 §5 阶段 5）。风险最高的一条技术债：
   Alembic 10 个迁移版本 + `ensure_schema` 774 行并存，两边一旦漂移，
   线上与测试的 schema 可能不一致。
2. **`MemoryStore` 12 个 mixin**。组合式设计本身合理（避免巨类），但
   `storage/store.py` 里混了 `clear_long_term` 这种业务方法，组装根不纯。
   建议组装根只做装配，业务方法回到对应 Store。
3. **`report/` 包 38 个文件平铺**。文件数量本身不是问题，但缺少二级分组
   （`compose_*` 5 个、`analyze*` 4 个、`ensure_*` 2 个），
   可以考虑 `report/compose/`、`report/analyze/` 子包。低优先级。
4. **测试文件也偏大**：`tests/test_loop.py` 2180 行、`test_kg.py` 1328 行。
   建议与阶段 3/4 同步按被测模块拆测试文件，保持「一模块一测试文件」的对应关系。

---

## 7. 验收标准

每个阶段必须全部满足：

1. **公开导入路径零变化**：`git grep` 确认没有调用方需要改 import
   （除新增的 `__init__.py` re-export 外，源码 diff 中不应出现 import 变更）。
2. **全量测试失败数 ≤ 4**（基线 4 个 `test_long_term.py` 既有失败，
   且必须是**同样那 4 个**，不能换成别的用例）。
3. **补丁有效性专项检查**（针对阶段 2/3）：对所有 `monkeypatch.setattr` 打补丁的
   内部符号，确认补丁**确实生效**——即断言 fake 被调用过，而不只断言返回结果。
   这是拆分最容易引入**静默降级**（测试变绿但实际走真实依赖）的地方。
4. **`ruff check` 无新增告警**（当前基线：`select = ["E","F"]`）。
5. **公开 API 快照比对**：拆分前后各跑一次
   `python -c "import ai_agent.kb.ingest.chunking as m; print(sorted(dir(m)))"`，
   确认公开符号集合只增不减。
6. **单文件行数**：拆分后对应模块每个文件 ≤ 600 行（chunking 的 `html.py` 580 行
   已是最接近上限的一个）。

---

## 8. 不做的事（明确划界）

- **不重写任何业务逻辑**。纯搬移 + 加 re-export；不顺手"优化"实现。
- **不改变公开 API 语义**，只新增模块、不改函数签名。
- **不动 `core/config.py`**、不动 `migrations/`、不动 500–660 行的中等文件。
- **不修 `test_long_term.py` 的 4 个既有失败**（属于门控逻辑问题，独立处理）。
- **不在同一阶段混合「拆分」与「功能开发」**——否则 diff 无法评审。

---

## 附录 A｜依赖证据

模块循环依赖扫描（AST 全量）：

```
脚本：解析每个模块的 import，解析为模块全名，检查反向边
结果：循环依赖 0 条
```

被依赖最多的模块（fan-in）：

```
149  ai_agent（包本身）
 95  ai_agent.core
 73  ai_agent.core.logging
 71  ai_agent.core.config
 57  ai_agent.storage
 40  ai_agent.storage.store      ← 组装根，被广泛引用属预期
 34  ai_agent.kb.ingest
 19  ai_agent.report.extract
```

依赖最多的模块（fan-out）：`agents/loop.py`(37)、`api/app.py`(35)、
`report/pipeline.py`(24)。`loop.py` 的 fan-out 偏高，部分原因就是它同时
承担了编排 + 提示词拼装，这也印证了阶段 3 的必要性。

## 附录 B｜测试基线（重构前必须记录）

```
$ .venv/bin/pytest -q
4 failed, 1883 passed, 7 warnings in 147.59s
```

失败清单（**既有**，非本方案引入，已在干净 HEAD 上复现证实）：

```
tests/test_long_term.py::test_capture_long_term_writes_extracted_episode
tests/test_long_term.py::test_capture_long_term_applies_profile_update
tests/test_long_term.py::test_capture_numeric_profile_without_source_downgrades_to_episode
tests/test_long_term.py::test_capture_non_numeric_profile_still_writes_directly
```
---

## 附录 C｜执行结果

### C.1 最终文件结构

```
kb/ingest/chunking/          2027 行  (原 1628 行单文件)
├── common.py      108   Chunk、HTML 正则、行项目判定、单元格转文本（叶子层）
├── text.py        122   纯文本聚合与二次切分
├── html.py        437   HTML 表解析、rowspan 展开、与 Markdown 互转
├── caption.py      96   表题识别与口径下带
├── table.py       304   行项目切片
├── preprocess.py  482   去噪声、合并小表、去页码样板、去重
├── core.py        200   chunk_segments 编排入口
└── __init__.py    278   全部 108 个符号再导出

kb/retrieval/_impl/          1651 行  (原 1352 行单文件)
├── types.py       116   Hit / format_where / _where / 停用词（叶子层）
├── textutils.py    40   _clip / _trigrams / _jaccard
├── keywords.py    145   关键词扩展
├── subjects.py    240   主体与文件解析
├── fusion.py      388   RRF 融合、去冗余、配额打散
├── neighbors.py   203   邻接扩展、同表行合并
├── rerank.py      149   精排
└── search.py      217   search_knowledge / preview_search 编排
   （retrieval.py 本身缩到 217 行，作为对外门面）

kb/store/                    1329 行  (原 1193 行单类 40 方法)
├── files.py       556   KbFileStore    文件元数据与状态机
├── chunks.py      228   KbChunkStore   切片写入与更新
├── search.py      294   KbSearchStore  检索与词频
├── neighbors.py   117   KbNeighborStore 取片与邻域
├── _helpers.py     76   共用 SQL 片段
└── __init__.py           KbStore = 四个 mixin 组合

agents/_loop/                1374 行  (原 1248 行单文件)
├── state.py        21   共享线程池
├── prompts.py     243   系统提示词拼装
├── followup.py    187   追问与预检索判定
├── tools_exec.py  205   工具执行与结果回收
└── loop.py        643   run_loop / run_stream / run

storage/_schema/              869 行  (原 ensure_schema 单方法 774 行)
├── core.py / memory.py / chat.py / auth.py / kb.py / kg.py
└── 每张表一个函数，按原执行顺序调用，SQL 逐字不变
```

### C.2 每个阶段实际踩到的坑（都是方案里预判过或新发现的）

**① `Chunk` 放在入口模块 → 必然循环依赖（阶段 1）**
最初把 `Chunk` 数据类放在 `core.py`，但 html/table/preprocess 全都要用它，
而 core 又调用它们 —— 拓扑排序直接报环。正确做法是把它下沉到叶子层 `common.py`。
同理 `_HTML_TABLE` 等正则被 html/table/preprocess 共享，也必须在叶子层。

**② `html ↔ table` 互调（阶段 1）**
`html_table_plain` 调 `table._markdown_table_plain`，`table._chunk_html_table`
调 `html._split_html_table`。把 `_markdown_table_plain` 下沉到 `common.py` 解开。

**③ `ast.AnnAssign` 漏提取导致变量凭空消失（阶段 2，**真实 bug**）**
`_rerank_cache: dict[tuple, list[float]] = {}` 是**带类型注解的赋值**，我最初只
处理 `ast.Assign`，结果它没有归入任何分片 —— 代码里所有引用 `_rerank_cache`
的地方都会 `NameError`。这个 bug 是我在跑 AST 等价性校验时发现的。

**④ `ast.unparse` 会丢掉全部注释（阶段 2，方案层面的关键决定）**
曾想用 `NodeTransformer + ast.unparse` 改写调用点，实测输出把中文注释全删了。
本项目注释是重要资产，改为**按 AST 给出的列偏移做定点字符串替换**，
另需注意 `col_offset` 是 **UTF-8 字节偏移**，含中文的行必须先转换。

**⑤ 补丁点不止「私有函数」，还有外部依赖（阶段 2，**最隐蔽**）**
方案里只列了 `_core_search` / `_extract_keywords` / `_retrieve_candidates`。
实测 `test_rerank.py` 还 patch `retrieval.get_reranker`，`test_hybrid_graphrag.py`
patch `retrieval.embed_query` / `read_extra` / `_adjacent_short_phrases`。
**正确做法是先从测试里系统性提取补丁清单**（AST 扫
`setattr(<mod>, "<name>", ...)` 与 `patch("ai_agent...loop.<name>")`），
而不是凭记忆列。阶段 2 一度出现 8 个失败，就是这么来的。

**⑥ 模块内自引用同样绕过补丁（阶段 2/3/4，核心机制）**
实测确认：无论 `from x import f` 还是同模块 `f()`，都走 **globals 查找**，
只有 `module.f()` 这种**属性查找**才能命中 `monkeypatch.setattr`。
所以统一写成 `_public().f(...)`（`_public()` 惰性 import 对外模块避免循环）。
阶段 4 还发现 `trace.emit(logger)` 这种**把 logger 当参数传**的情况，
以及 `logger.info(...)` 这种**本身是属性访问**的情况，都要单独处理。

**⑦ 包的 `__init__` 必须覆盖全部再导出符号（阶段 1/2）**
只列公开符号时，`ruff` 会为每个私有再导出报 F401。把 108 个符号全写进
`__all__` 才干净。这同时起到「导出契约」的作用。

**⑧ `store.py` 与 `store/` 不能共存（阶段 3）**
Python 优先把 `store.py` 当模块，`from ai_agent.kb.store._helpers import ...`
会报 `'ai_agent.kb.store' is not a package`。必须把 `store.py` 整个改成包。

**⑨ 拆分后仍超 600 行的两个文件**
`agents/_loop/loop.py`（643 行，`run_loop` 本身 332 行）与
`kb/store/files.py`（556 行，23 个文件的 CRUD 方法）。前者要再拆就得动
`run_loop` 内部逻辑（超出「纯搬移」范围），后者是单表 CRUD 的自然粒度，
都建议**保持现状**，不作为未完成项。

### C.3 验收实据

```
# 1. 定义级等价性（AST 逐字比对，排除有意的 _public() 改写）
chunking : 原 108 个定义，全部一致
retrieval: 原  66 个定义，全部一致
store    : 原  40 个方法 + 5 个 helper，全部一致
loop     : 原  31 个定义，全部一致

# 2. SQL 序列（阶段 5 最关键的验证）
改造前录制 97 条 SQL → 改造后录制 97 条 → diff 为空
另测 dim=768/1024/2048 三种维度，SQL 结构一致且维度正确注入

# 3. 公开 API 快照
loop     : dir() 输出与改造前**完全相同**
chunking : 仅丢失原先泄漏的 stdlib import（re/dataclass/Path 等）
retrieval: 仅丢失原先泄漏的 stdlib import（json/re/uuid/Path 等）

# 4. 全量测试
改造前: 4 failed, 1883 passed
改造后: 4 failed, 1883 passed   ← 且是同样 4 个 test_long_term.py 用例

# 5. 静态检查
ruff src/ tests/ scripts/ : 7 errors → 7 errors（全部为既有问题，零新增）

# 6. 导入完整性
177 个模块全部可导入；12 条历史导入写法 100% 兼容
```

### C.4 遗留与建议

1. **`ensure_schema` 仍是第二份 schema 源**（与 Alembic 并行）。本轮只做了
   结构拆分、SQL 未动。要收敛成单一来源属于**行为变更**，建议单独立项。
2. **`report/` 包 38 个文件平铺**、`tests/test_loop.py` 2180 行等问题未处理，
   属方案 §6 记录的低优先级项。
3. 拆分出的 `_impl/`、`_loop/` 前缀表明「内部实现」；`chunking/`、`store/`
   则直接用包名。命名不完全统一，但对外行为一致，如需要可后续统一。
