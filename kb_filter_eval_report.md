# 知识库 chunk 级元数据过滤：黄金集对照评测

> 任务：跑 `scripts/eval_retrieval.py` 用黄金集对比「过滤开 / 关」的 hit@k / MRR，
> 确认精度提升且不丢召回。
> 结论先行：**在当前语料上无法确认精度提升**——生产实际下推的过滤是 no-op，
> 唯一能跑的 `subjects` 维度反而掉精度（hit@3 跌破门禁），但**召回不丢**。

## 0. 前置发现：旧黄金集与现语料已对不上
- 旧黄金集 `scripts/retrieval_eval_cases.json`（131 条）针对 顺丰控股 / 安克创新 /
  中国能源建设 / 中国银行，但当前 KB 只有 3 份报告：中化装备、中国黄金、长安汽车
  （且原 4 家的源 PDF 已不在 `data/uploads`）。
- `--validate` 直接报 131 条全部「文件不在库里」，旧集在本语料上无法运行。
- 已为当前语料新建 `scripts/retrieval_eval_cases_3f.json`（20 条，已 validate 对齐），
  覆盖 中国黄金 + 长安汽车两家的指标/封面/股东问法（中化装备切片 `file_id` 孤儿、
  join 取不到，暂未纳入）。

## 1. 改动：eval_retrieval.py 增加过滤开关
- 新增 `--filters`（单跑「开」模式）、`--compare-filters`（关/开各跑一轮出对照）、
  `--cases`（指定用例集，默认仍指旧黄金集）。
- 新增 `infer_filters(query, case)`：镜像生产 `kb_search._kb_search` 的过滤推断——
  case 自带 `filters` 字段（模拟模型正确传参）+ query 命中 `_PARENT_INTENT` 时
  `scopes=[PARENT]`。返回 `None` 即等同「过滤关」。

## 2. 关键事实：当前语料的过滤列是空的
`agent_kb_chunk` 列 `kind/section/scope/subject/period/meta` 都存在，但实数据：

| 列 | 现状 |
|---|---|
| `kind` | 全为 `'text'`——**表格切片没被标成 `table`** |
| `section` | 全 NULL |
| `scope` | 全 NULL |
| `subject` | 有值（公司名），99/99 |
| `period` | 有值（报告期），99/99 |

生产 `kb_search` 实际下推的过滤是 **kinds / sections / scopes**——这三列全空，
所以生产里「过滤开」在当前语料上是**纯 no-op**。验证（对「中国黄金的货币资金」）：

```
OFF (no filter)       : [(长安汽车,25),(中国黄金,12),(中国黄金,14),(中国黄金,0),(中国黄金,10)]
kinds=[table] (prod)  : 与 OFF 逐条一致   ← 空列→召回空→安全回退不过滤
scopes=[parent](prod) : 与 OFF 逐条一致   ← 同上
kinds==off ? True | scopes==off ? True
```

即：召回零损失，但也没有任何精度收益——因为根本没东西可滤。

## 3. 对照结果（top_k=5，rerank 关，纯检索质量）
用唯一有值的维度 `subjects`（公司名）当过滤器跑对照（20 条，18 条应用了过滤）：

| 指标 | 关 | 开 | Δ |
|---|---|---|---|
| hit@1 | 60% | 55% | -5pp |
| hit@3 | 85% | 75% | **-10pp（跌破 85% 门禁）** |
| hit@5 | 95% | 95% | 0 |
| recall@10 | 95% | 95% | 0 |
| MRR@10 | 0.722 | 0.680 | -0.044 |

名次变化的 7/20 条中，3 条上升、4 条下降（非全向下），但聚合的 hit@1/3/MRR
整体下滑——典型「顶部精度被稀释、召回不掉」。

## 4. 解读：为什么「过滤开」反而掉精度
- **关模式搜全库 99 片**：跨 3 家公司时，目标公司的指标片在嵌入空间里「鹤立鸡群」，
  排得很靠前。
- **开模式用 `subjects` 收窄到本公司 33 片**：候选集内全是同一家公司、同套术语、
  同结构，嵌入彼此高度相似，向量判别力反而下降 → 金 chunk 在公司内被别的表片挤到
  后面（hit@1/3 掉），但仍在前 5/10（recall 不掉）。
- 即：**实体收窄在「嵌入已经能区分实体」的小语料上是冗余甚至有害的**；它的价值在
  大规模、实体易混淆、或纯语义无范围检索时。本语料恰好嵌入分离良好。
- 顺带发现：关模式本身就有跨公司串味——「中国黄金的货币资金」第 1 名竟是
  **长安汽车**的货币资金片。这正说明生产靠 `file_ids` 收窄（而非本元数据过滤）才是
  修串味的正确手段；本功能若再叠一层 `subjects` WHERE，既无收益又添成本。

## 5. 结论与建议
1. **不能确认「精度提升」**：现语料上，生产实际下推的过滤（kinds/sections/scopes）
   是 no-op；唯一能跑的 `subjects` 维度反而掉精度（hit@3 跌破门禁）。但**召回不丢**
   （recall@10 95% 持平，安全回退机制生效）。
2. 功能实现本身正确：SQL WHERE 生效、空过滤→回退保护召回的机制已验证可用。
3. **要真正拿到价值，先做 ingestion 补全元数据**：
   - 表格切片标 `kind='table'`（OTSL / 表格检测已有，未接到 kind）；
   - 从 `## 节标题` 抽 `section`；
   - 区分合并 / 母公司报表填 `scope`（consolidated / parent）。
   三者补齐后 kinds / sections / scopes 才有东西可滤。
4. 即便补齐，收窄也不是免费精度：建议保持「仅模型显式传参 / 母公司口径时」才下推
   （当前代码已是该保守策略），避免对答案在正文里的问法误加 `kinds=['table']`。
5. 当前 1 条 recall 漏检（长安汽车应收账款）与过滤无关，是切片/嵌入本身问题，单独排查。

## 6. 未提交改动
- `scripts/eval_retrieval.py`（新增过滤对比开关：`--filters` / `--compare-filters` / `--cases`）
- `scripts/retrieval_eval_cases_3f.json`（当前语料黄金集，20 条，已 validate）
- 工作区另有 5 个非本次改动文件（chunking.py / ocr.py / otsl.py / test_kb.py /
  test_kb_otsl.py）未提交，本次未混入。
