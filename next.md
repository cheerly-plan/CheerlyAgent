# 研报流水线指引（现状）

用户需求示例：
- 单公司：`生成中国能源建设 2025Q3 财报深度分析 PDF 研报`
- 赛道行业：`生成机器人行业 2026Q3 财报对比分析` → 标题落为 **财报对比底稿（赛道分组版）**

定位约定：

| 类型 | skill | 对外称呼 | 说明 |
|---|---|---|---|
| 单公司／单银行 | `company_financial_report` / `bank_financial_report` | 财报深度分析 | 可含经营解读 LLM（受约束） |
| 赛道行业（机器人／半导体／银行分组） | `*_industry_report` | **财报对比底稿** | 不做行业深度研究、不做估值买卖建议 |

## 交付形态

默认双 PDF：

| 层 | 产物 | 门禁 |
|---|---|---|
| 事实层底稿 | `<stem>.pdf` + Markdown + 溯源 | 正文数字白名单；可自动定稿 |
| 深度推断层 | `<stem>.deep.pdf` | 独立审核记录；未定稿可 `?preview=1` 预览，下载仍走门禁 |

深度层方案见 `docs/financial-deep-analysis-upgrade.md`，结构化事实契约见 `docs/deep-facts-schema.md`。

## 流水线

    ↓
1️⃣ 需求解析（`report/parse.py` + `report/skills/*/SKILL.md`）
    ├ 路由 skill、展开花名册／赛道、大纲、指标集
    └ 赛道 skill 标题固定为「财报对比底稿（赛道分组版）」
    ↓
2️⃣ 检索 & 抽数（确定性代码，复用知识库）
    ├ 召回本期／同比基期季报；深度层可拉近 3 年年报进 `deep_facts`
    ├ 抽取营收、扣非、毛利率、合同负债、经营现金流等
    └ 每条数字带文件、页码、原文片段（溯源）
    ↓
3️⃣ 计算 & 对比（`report/compare.py`，不交给 LLM 口算）
    ├ 同口径同比／环比；口径不齐 → 显式「不可比／缺基期」，不编造增减
    └ 赛道报告大纲含「同比与趋势」节
    ↓
4️⃣ 撰写（`compose*` + `analyze*`）
    ├ 组内宽表、结构比率梯队
    ├ 表内「扣非 × 经营现金流 × 合同负债」质量交叉（有数才写）
    ├ 赛道报告经营解读：**不跑 LLM**；有季报原文则「原文披露」引述，无原文则归因从略
    └ 下期预告：列建议外部数据源清单；接入前不推断景气周期
    ↓
5️⃣ 深度层（可选，`report/deep/`）
    ├ 章节并发 + 多轮评审；空响应自动升级主模型
    ├ 推断必须带证据／证伪信号；红框结论区进 deep.pdf
    └ 进度事件与事实层共用 `report/progress.py`
    ↓
6️⃣ PDF 渲染（独立工具，LLM 不直接出二进制）
    ├ 组内横比柱状图为主；不足四期不画伪趋势折线
    └ 溯源附件；深度层独立成 deep.pdf
    ↓
7️⃣ 校验与审核（`report/validate.py` + `report/review.py`）
    ├ 正文数字必须在已授权 facts／计算／原文引述白名单内
    ├ 无出处推断一律不放行（事实层）；深度层走人工审核
    └ 审核：生成者本人 + 管理员可批准／驳回／定稿；队列读接口限管理员
    ↓
✅ 输出 Markdown／PDF／deep.pdf + provenance；交付文本写回聊天消息

## 聊天入口（异步）

`generate_report` 只提交后台任务并返回 `job_id`（深度层约 5–8 分钟）：

1. SSE 下发 `report_job` 事件 → 前端渲染任务卡片并轮询
2. 进度用 `report_job_status(job_id)` 或 `GET /v1/report/jobs/{id}`
3. 聊天断线不中断生成；重启后靠消息里的 `report_job_id`／交付文本续上

同步入口 `POST /v1/report` 契约不变。

## 赛道报告「写深一点」已落地的四层

1. **时间序列**：有基期写同比；不可比明确写出，不整段放弃
2. **财务质量交叉**：扣非／经营现金流／合同负债表内关系
3. **有出处归因**：原文证据引述；禁止无出处推断
4. **命名与预期**：称「对比底稿」；下期预告给数据源清单（见 `config/industries.yaml` notes.next_period）

## 改什么去哪改

| 想改什么 | 文件 |
|---|---|
| 代表公司／赛道／大纲章节 | `src/ai_agent/report/skills/<id>/SKILL.md` |
| 行业指标集、组内列、下期数据源文案 | `src/ai_agent/report/config/industries.yaml` |
| 经营解读（梯队／质量交叉／原文引述） | `analyze_structure.py` / `analyze.py` |
| 同比不可比表述 | `compose_lead.py` |
| 标题／阅读说明「底稿」措辞 | `parse.py` / `compose_common.py` |
| 深度层章节／评审／附录 PDF | `report/deep/` |
| 异步任务与进度 | `report/jobs.py` / `report/progress.py` |
| 审核门禁 | `report/review.py` / `api/routers/report.py` |

新增行业：优先 `scripts/scaffold_industry.py` + 一份 skill，尽量不改管线 Python。
