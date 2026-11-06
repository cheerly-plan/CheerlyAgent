# 外网财报补齐 tool：代码审查与接入方案

> 日期：2026-11-06
> 输入：用户提供的巨潮资讯网（cninfo）公告抓取脚本，目标是做成「本地 KB 缺财报时自动补齐」的 tool。
> 性质：**评估 + 方案，未改动任何 `src/` 代码**。接口参数来自公开资料，标注了需实测项。

---

## 一、结论

原脚本作为**一次性下载工具**思路正确（按代码查 orgId → 查公告列表 → 拼直链下载），主链路没问题。
但直接包装成 tool 接进当前后端，会踩三类坑，其中 **P0 会静默写错报告期**，金融场景不可接受：

| 级别 | 问题 | 后果 |
|---|---|---|
| **P0** | 文件名 `{公告日期}_{标题}.pdf` 前缀日期污染 KB 元数据 | `resolve_file_identity` 把 `2024-04-02_贵州茅台2023年年度报告.pdf` 解析成 `subject=20240402贵州茅台`、`period=2024`。**报告期错一年**，`parse_request` 按 period 路由会拿不到或拿错文件 |
| **P0** | 只下载、不走 KB ingest | PDF 落盘 ≠ 可检索。`extract_indicators` 查的是 pgvector 切片，磁盘上的文件它看不见 |
| **P0** | 默认 `category_ndbg_szsh` 只抓年报 | 项目当前真实缺口是 **2025Q3 季报基期**和半导体标的，抓年报补不上 |
| P1 | `pageSize=50` 被服务端硬限为 30 且未翻页 | 同一报告期多份公告（更正/摘要）会漏，或只拿到前 30 条 |
| P1 | 不过滤摘要 / 英文版 / 更正公告 / 审计报告 | KB 里同时存在「原文」和「更正后」，抽取会命中错误口径——正是上一轮刚修过的那类问题 |
| P1 | 缺 `X-Requested-With: XMLHttpRequest`、用 `http://` | 巨潮查询接口按 Ajax 调用校验，缺头可能被拒；全站已 https |
| P1 | `szse_stock.json` 顶层结构不稳定 | 有时是 list 有时是嵌套 dict，`for s in d.get("stockList", [])` 可能遍历到字符串或直接空结果 |
| P2 | 下载不验 `status_code` / `Content-Type` / `%PDF` magic | 403 HTML 被写成 `.pdf`，后续解析报一堆莫名错误 |
| P2 | 无限速、无重试、无幂等、无域名白名单 | 批量下容易被限流；`adjunctUrl` 来自外部响应，拼接前不校验 host 有 SSRF 面 |
| P2 | 板块只按 `startswith("6")` 判断 | 8/4 开头的北交所会误判为深市 |
| — | 项目无 `requests` 依赖 | `pyproject.toml` 里没有 requests / httpx，直接 import 在干净环境会失败 |

另外：粘贴过来的代码压成了一行（`def get_org_id(code):for f in ...:try:`），Python 里复合语句这样写是 `SyntaxError`。判断是聊天窗口丢格式，不展开。

### P0 文件名坑的实证

直接在项目里跑 `kb/meta.py` 的规则：

```
'2024-04-02_贵州茅台2023年年度报告.pdf'
   subject = '20240402贵州茅台'   period = '2024'        ← 应为 2023
'贵州茅台2023年年度报告.pdf'
   subject = '贵州茅台'           period = '2023'        ← 正确
'中国银行股份有限公司2025年第三季度报告.pdf'
   subject = '中国银行股份有限公司'  period = '2025Q3'   ← 正确
```

`resolve_file_identity` 是**文件名规则优先，封面只做条件补缺**；period 一旦从文件名抽出（哪怕抽错）就不会再用封面纠正。所以只要按原脚本命名，`period=2024` 会一路带进 DB 和检索路由。

**修正**：落盘名改成 `{公司全称}{YYYY年}{报告类型}.pdf`，例如 `贵州茅台酒股份有限公司2023年年度报告.pdf`、`中国银行股份有限公司2025年第三季度报告.pdf`。日期/公告 ID 放 `tags` 或元数据，不要进文件名前缀。

### 精确优先级（补充核实，三个字段的回退条件不对称）

`kb/meta.py::resolve_file_identity(filename, cover)` 的实现：

```python
meta = extract_file_meta(filename)                       # 先全部按文件名算
if not looks_like_company(meta["subject"]):             # 主体：判「像不像公司」
    found = dominant_company(cover)
    if found: meta["subject"] = found; ...
if not meta["period"]:                                   # 报告期：只判空
    meta["period"] = extract_period(cover)
meta["doc_type"] = infer_doc_type(filename) or infer_doc_type(cover)
```

| 字段 | 先取 | 何时回退封面 | 陷阱 |
|---|---|---|---|
| `subject` | 文件名 `extract_subject` | **仅当 `looks_like_company()` 为 False** | 判定是 `COMPANY_SUFFIX` **末尾锚定**，含 `银行/证券/保险/基金/集团/控股/有限公司…`。「中国银行」自带「银行」后缀 ⇒ `20251030中国银行` 被判为合法公司，**封面救不回来** |
| `period` | 文件名 `extract_period` | **仅当结果为空串** | 只判空不判错；抽成 `2024` 就一直是 `2024`，封面写 `2023年年度报告` 也不会纠正 |
| `doc_type` | 文件名 `infer_doc_type` | 文件名推断不出才用封面 | 三条路径里唯一有「人工值优先」保护的：pipeline 里 `if not record["doc_type"].strip()` 才写 |

封面文本来自 `kb/pipeline.py::_cover_text(segments, pages=2)`：**前两页**正文拼接、截 4000 字符；无页码信息的文档退化为全部文本前 4000 字符。主体补缺用的是 `dominant_company`（正则找出现次数最多的公司全称），**不是**更精细的 `cover_company`（带 Markdown 标题行优先、括号处理）——后者只在 `kb/graph_names.py` 的图谱实体命名里用。

实测（`resolve_file_identity` 传真实封面文本）：

```
中国银行股份有限公司2025年第三季度报告.pdf  → subject=中国银行股份有限公司  period=2025Q3   ← 规范名，封面不参与
2025-10-30_中国银行2025年第三季度报告.pdf   → subject=20251030中国银行      period=2025Q3   ← 银行后缀让脏名字过关
2024-04-02_贵州茅台2023年年度报告.pdf       → subject=贵州茅台酒股份有限公司 period=2024    ← 主体被封面救回，报告期仍错一年
贵州茅台酒股份有限公司年度报告.pdf         → subject=贵州茅台酒股份有限公司 period=2023    ← 文件名无年份，封面补对
最终版.pdf                                  → subject=中国银行股份有限公司   period=2025Q3   ← 全靠封面
```

另外两条容易忽略的路径：
- **改名 / 存量回填完全不看封面**：`store.rename_kb_file` 和 backfill 只跑 `extract_file_meta(filename)` 纯文件名规则，允许人工显式传 `subject` / `period` 覆盖。
- **`_process` 无条件写 subject/period，但这不是隐患**（二次核查后修正）：
  `kb_pipeline.submit()` 全仓只有 `complete_upload` 一个调用点，且该端点要求
  `status == "uploaded"`；文件 `ready` 之后**没有任何重新解析入口**。人工指定
  `subject`/`period` 的入口只有 CLI `scripts/kb.py` 的 backfill（作用于已 ready 的存量文件）。
  改名走 `store.rename_kb_file`（显式重算主体，设计如此），重嵌走 `_reembed`（只换向量，不碰元数据）。
  所以「人工值被 `_process` 冲掉」在正常流程里触发不了。
  `doc_type` 有 `if not record["doc_type"].strip()` 保护，是因为它属于**图谱抽取配置**
  （`api/routers/kg.py` 会在文件 ready 后显式设置，跨阶段被写入），与只在解析时算一次的
  **索引元数据**角色不同——保护不对称是合理设计，不需要对齐。**结论：不改。**

---

## 二、需要实测确认的接口事实（不编造 API）

以下来自公开技术资料（两个独立来源对 category 一致），**巨潮会不定期调整，落地前必须用一个已知代码实测**：

| 项 | 值 | 可信度 |
|---|---|---|
| 查询接口 | `POST https://www.cninfo.com.cn/new/hisAnnouncement/query`，`application/x-www-form-urlencoded` | 高，多来源一致 |
| 年报 | `category_ndbg_szsh` | 高 |
| 一季报 | `category_yjdbg_szsh` | 高 |
| 半年报 | `category_bndbg_szsh` | 高 |
| 三季报 | `category_sjdbg_szsh` | 高 |
| `stock` | `{code},{orgId}`，缺 orgId 返回空列表 | 高 |
| `seDate` | `YYYY-MM-DD~YYYY-MM-DD` | 高 |
| `column` | `sse` / `szse` / `bj`；`plate` 对应 `sh` / `sz` / `bj` | 中，需实测 |
| `pageSize` 硬上限 30 | 传 50 实际只回 30 | **中，单一来源，必须实测** |
| orgId 推荐来源 | `topSearch/query`（按代码反查，返回含 code/orgId/zwjc 的对象列表） | 中，路径与 GET/POST 需实测 |
| 必需请求头 | `User-Agent` + `Referer` + `X-Requested-With: XMLHttpRequest` + `Accept: application/json` | 中，需实测 |

**验证顺序**：先用手上已知代码（如 `600519`）跑一次「查 orgId → 查一页 → 下一份 PDF → pymupdf 能打开」，全通再写批量逻辑。

---

## 三、建议的 tool 设计

### 3.1 边界（和上一轮 `generate_report` 同一套原则）

- **工具名**：`fetch_missing_reports`，**不暴露 `output_path` / `outdir` / URL**。
- **入参**：`code`（6 位数字，正则白名单）、`period`（`YYYYQn` / `YYYYH1` / `YYYY`）、`category`（枚举白名单）、`max_files`（默认 3，上限 10）。不给自由文本 URL。
- **落盘**：由服务端固定到 `settings.kb_upload_dir / {file_id}/`，复用 `store.create_kb_file` 建档。
- **返回**：紧凑文本（入库文件数、`file_id` 列表、subject/period 解析结果、跳过原因），**不回传 PDF 二进制**，`execute_tool` 会截到 3000 字符。
- **默认关闭**：加配置开关 `enable_cninfo_fetch=False`。理由见 3.3。
- **不自动联动报告流水线**：`generate_report` 里不加「发现 gaps 就自动去外网补」的隐式行为，由用户显式触发。

### 3.2 入库必须走完整链路

复用 `api/routers/kb.py` 既有约定，别另起炉灶：

```
create_kb_file(user_id, 规范文件名, size, ".pdf", tags=[来源标记])
  → 落盘 kb_upload_dir/{file_id}/{规范文件名}.pdf
  → _file_sha256 / update_kb_file_hash（内容指纹去重，同内容拒绝重复入库）
  → kb_pipeline.submit(file_id, user_id, path)   # 解析→切片→批量嵌入→灌库
  → 前端轮询 agent_kb_file.status（parsing → embedding → ready / failed）
```

`kb_pipeline` 内部会调 `resolve_file_identity(filename, cover)` 写 `subject` / `period` / `doc_type`——这就是 3.1 里文件名规范的必要性。

**来源溯源**：`create_kb_file` 已有 `tags` 字段，建议写入 `cninfo:{announcementId}`、`公告日期`、`来源域名`；更规范的做法是在 `agent_kb_file` 加 `source_url` / `source_id` 列（需 alembic 迁移）。否则 `Evidence` 的溯源链只到「本地文件」，追不到原始公告。

### 3.3 两个架构判断

**① 别做成同步工具。** 下载 1–5 MB 还好，但 `kb_pipeline` 的解析 + 切片 + 批量嵌入是秒级到分钟级；一份 300 页年报更久。聊天主循环同步等待会卡住 SSE 流。建议：工具只负责「查到公告 → 下载校验 → 建档 → submit」，立即返回 `status=parsing` 和 `file_id`，让前端按现有文件列表轮询。这也是 `kb/pipeline.py` 注释里写明的设计（"进度即 status，前端轮询，不单独建 job 表"），顺着它走最省事。

**② 自动补齐默认关闭。** 外部公告的口径、合并范围、更正版本可能与本地库不一致；一旦让模型"缺数就自己上网补"，报告会变成混合来源却看不出来，比现在老老实实写 gaps 更危险。建议做成：用户看到 gaps → 显式说「补一下 2025Q3」→ 工具入库 → 用户再手动重跑报告。

### 3.4 适配层要点（推荐参数）

| 项 | 推荐值 | 调参方向 |
|---|---|---|
| 请求节流 | 1 次/秒 + 0.3–0.8s 随机抖动，单任务串行 | 被 403 就退避到 3–5s，不要降并发硬冲 |
| 超时 | 查询 10s，下载 30s（连接/读取共用） | 大年报可放到 60s |
| 重试 | 2 次，指数退避，仅对超时/5xx | 4xx 不重试 |
| 单文件上限 | 50 MB（低于现有 `kb_max_file_mb`，需确认该值） | 超限跳过并记录 |
| `max_files` | 默认 3，硬上限 10 | 防止一次抓全历史 |
| orgId 缓存 | 内存 + 落盘 JSON，TTL 24h | 股票列表结构不稳，缓存里存 `code → (orgId, name, column)` |
| 分页 | 以 `totalAnnouncement` / 实际返回条数为准翻页，`pageNum` 递增，最多 10 页 | 不要假设 pageSize=50 生效 |
| 公告过滤 | 排除标题含「摘要 / 英文 / 更正 / 修订 / 审计 / 补充 / 取消」的条目；同一 `(period, company)` 只留最新一份正文 | 规则要可配置，允许人工覆盖 |
| 去重 | `announcementId` 集合 + 内容 SHA256 | 两层都要 |
| 下载校验 | 域名白名单 `static.cninfo.com.cn` + `urlparse().path` 后缀 + `status==200` + `Content-Type` 含 pdf + 前 4 字节 `%PDF` + pymupdf 能打开 | 任一不过就跳过并把原因写进返回 |
| 标题清洗 | 剥 HTML 标签 → 去 `\/:*?"<>|` 和换行 → 截到 80 字 | 但**最终落盘名不用标题**，用 3.1 的规范名 |

### 3.5 HTTP 客户端选型

| 方案 | 优点 | 代价 | 建议 |
|---|---|---|---|
| 扩展 `core/http.py`（`request_form` + `request_bytes`） | 零新依赖，沿用项目「零依赖共享客户端」约定，超时语义统一 | 要改公共模块，需补单测 | **推荐** |
| `uv add requests>=2.32` | 改动最小，脚本几乎照搬 | 新增运行时依赖；项目现有模块都不用它，风格割裂 | 备选 |
| `httpx` | 原生支持流式与连接池 | 新依赖，且项目已有 urllib 封装 | 不推荐 |

`core/http.py` 目前只有 `request_json`（GET / POST JSON），不支持 form-urlencoded 和二进制下载，需要加两个函数。

### 3.6 测试策略（吸取全量测试卡住的教训）

上一轮全量 pytest 在 `test_hybrid_graphrag.py`（有外部调用）约 7% 处卡了 6 分钟。新模块的单测**必须全部 mock 网络**：

- monkeypatch `core.http.request_form` / `request_bytes`，fixture 提供固定的公告 JSON 和 `%PDF` 字节；
- 覆盖：分页、公告过滤、orgId 缓存命中、`%PDF` magic 失败、域名白名单拦截、文件名解析断言（`period` 必须等于公告所属报告期）、SHA256 重复跳过；
- **一个用例都不允许真连 cninfo**。CI 里也别放集成测试，改成手动 smoke 脚本。

---

## 四、成本与延迟粗估（单份季报）

| 环节 | 延迟 | 成本 |
|---|---|---|
| orgId 查询 + 公告列表 | 0.2–1.5s（缓存命中 ~0） | 0 |
| PDF 下载（1–5 MB） | 1–5s | 0 |
| KB 解析 + 切片 + 嵌入 | 5–40s（季报）；年报 1–5 min | 0 LLM token，但嵌入模型按 token 计费，年报可达数万 token |
| 图谱抽取（可选） | 几十分钟，**建议新入库文件先不建图** | 高，别自动触发 |
| 报告流水线重跑 | 1–2s | 0（全确定性，零 LLM） |

结论：**瓶颈在 ingest 不在下载**。所以工具要异步返回，且默认只补最小集合（单股 + 单期 + 正文一份），别批量抓全历史。

---

## 五、合规提醒（别跳过）

- 巨潮资讯网是深交所法定信息披露平台，由深圳证券信息有限公司运营。其法律声明明确：**个人、非商业用途可免费查看和下载；商业用途（复制、入库、再分发、对外提供服务）需事先取得书面同意**。
- 平台用户协议对自动化抓取有限制。合规底线通常被概括为三条：不突破技术防护措施（验证码 / 加密接口）、不造成服务器过载（高频请求）、不用于非法目的。
- 落地要求：限速、串行、日请求配额上限、只访问官方域名、保留来源与抓取时间日志、不二次分发原始 PDF。
- **如果这套后端要对外部用户提供研报服务，先确认授权范围，或改用已获授权的商业数据 API。** 这部分是业务决策，不是技术问题。
- 版本提醒：Python 3.13 + `uv` 环境；若选 requests，装前确认版本与 3.13 兼容；巨潮接口参数随时可能变，加一层健康检查比悄悄失败好。

---

## 六、阶段建议

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0 · 先别写 tool** | 用离线脚本按规范文件名手工补 `2025Q3` + 一个半导体标的入库，走完整 `kb_pipeline` | `list_kb_files` 里 `subject` / `period` 正确；重跑 `parse_request`，gaps 对应项消失；`extract_indicators` 能抽到基期数字 |
| **P1** | `cninfo_client.py`（纯函数、缓存、分页、过滤、校验）+ 全 mock 单测 | 单测全绿，CI 不连外网 |
| **P2** | 入库函数 + `fetch_missing_reports` 门面工具 + 配置开关（默认关闭）+ 端点 | 工具调用返回 `status=parsing` 和 file_id；重复调用被 SHA256 拦住 |
| **P3** | 与报告流水线联动（仍显式触发）+ 来源溯源字段 + 评测集验证「补数据后同比走跨期计算」 | 研报评测集数值一致率不退化 |

**最省事的验证路径其实是 P0**：先只改文件名规则，手工把缺的季报传进 KB，看报告能不能自动算出跨期同比。如果这条闭环成立，再决定要不要投工具化的工。

---

## 七、待确认

1. 是否接受「自动补齐默认关闭、需显式触发」？还是希望 gaps 出现时自动补（我不建议）。
2. HTTP 客户端走扩展 `core/http.py` 还是加 `requests` 依赖？
3. 来源溯源用现有 `tags` 够用，还是要加 `source_url` / `source_id` 列（涉及 alembic 迁移）？
4. 商业授权边界：这套系统是内部研究用，还是会对外提供服务？
5. ~~是否顺手修 `_process` 覆盖问题~~ —— 已核查，**不需要改**（见第一节末条）。真正待办的仍是 P0：规范下载文件名。
