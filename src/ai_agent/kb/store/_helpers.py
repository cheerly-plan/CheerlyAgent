"""kb/store 家族共用的 SQL 片段与模块级小工具。"""

from __future__ import annotations

from ai_agent.kb.ingest.chunking import Chunk, extract_table_indicators

# 检索分桶键：按主体（公司）分桶，每个主体的头名在相关性带宽内保底一个
# 名额（见 search_kb_chunks）。「A 和 B 的营业收入」这种一次调用跨两个
# 主体时，两边的头名都进得来，不再是一家吃满、一家落空。
# 主体为空（老数据 / 文件名里抽不出公司名）的切片全部落在同一个桶里，
# 行为与分桶前逐字一致——回填 subject 之后才生效，不会突然改掉存量结果。
_SUBJECT_BUCKET = "COALESCE(NULLIF(subject, ''), '')"

# 行项目存在 meta.indicators。关键词命中正文或这一列都算，不拿它做 WHERE。
_INDICATOR_LIST = (
    "CASE WHEN jsonb_typeof(c.meta->'indicators') = 'array' "
    "THEN c.meta->'indicators' ELSE '[]'::jsonb END"
)

def _keyword_hit(i: int) -> str:
    return (
        f"(CASE WHEN c.text ILIKE %(kw{i})s OR EXISTS ("
        f"SELECT 1 FROM jsonb_array_elements_text({_INDICATOR_LIST}) AS item "
        f"WHERE item ILIKE %(kw{i})s) THEN 1 ELSE 0 END)"
    )

def _chunk_indicators(chunk: Chunk) -> dict:
    """只有表格切片记行项目。正文没有这一列，meta 留空对象。"""
    if chunk.kind != "table":
        return {}
    names = extract_table_indicators(chunk.text)
    return {"indicators": names} if names else {}

def _chunk_meta_where(
    *,
    kinds: list[str] | None,
    subjects: list[str] | None,
    periods: list[str] | None,
    scopes: list[str] | None,
    sections: list[str] | None,
    params: dict,
) -> str:
    """生成 chunk 元数据过滤的 SQL 片段（前导 AND 连接），并把参数写入 params。

    列名全部硬编码（kind/subject/period/scope/section），值走参数化，无注入面。
    空列表 / None 退化为不过滤（与 file_ids 现有「空列表退化为 NULL」一致）。语义：
    - kinds / periods / scopes：精确 = ANY（枚举 / 标准格式，安全）
    - subjects：ILIKE 模糊，兼容「中国银行」缩写匹配「中国银行股份有限公司」
    - sections：ILIKE 模糊，章节名写法多（「第十节 管理层讨论」vs「十、管理层讨论」）
    """

    clauses: list[str] = []
    if kinds:
        clauses.append("AND c.kind = ANY(%(kinds)s)")
        params["kinds"] = list(kinds)
    if subjects:
        sub = " OR ".join(
            f"c.subject ILIKE %(subject{i})s" for i in range(len(subjects))
        )
        clauses.append(f"AND ({sub})")
        for i, s in enumerate(subjects):
            params[f"subject{i}"] = f"%{s}%"
    if periods:
        clauses.append("AND c.period = ANY(%(periods)s)")
        params["periods"] = list(periods)
    if scopes:
        clauses.append("AND c.scope = ANY(%(scopes)s)")
        params["scopes"] = list(scopes)
    if sections:
        sub = " OR ".join(
            f"c.section ILIKE %(section{i})s" for i in range(len(sections))
        )
        clauses.append(f"AND ({sub})")
        for i, s in enumerate(sections):
            params[f"section{i}"] = f"%{s}%"
    return " ".join(clauses)
