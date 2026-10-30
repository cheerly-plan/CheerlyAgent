"""报表口径（合并 / 母公司）的识别与检索偏好。

实录背景：能建三季报 p18「母公司现金流量表」期末现金 2,721,444，
p13「合并现金流量表」期末现金 77,436,294，差 28 倍。两片都含
「期末现金及现金等价物余额」这串字，母公司那片只有 4 行、语义更纯，
向量相似度反而更高——不区分口径就会把单体数当集团数答出去。
"""

from ai_agent.kb.chunking import Chunk, chunk_segments
from ai_agent.kb.parsing import Segment
from ai_agent.kb.tables import SCOPE_CONSOLIDATED, SCOPE_PARENT, _table_scope
from ai_agent.kb.retrieval import _apply_scope_preference, _prefer_scope


# ——— 表题 → 口径 ———


def test_parent_caption_is_recognized():
    assert _table_scope("母公司利润表") == SCOPE_PARENT
    assert _table_scope("母公司现金流量表") == SCOPE_PARENT
    assert _table_scope("母公司资产负债表") == SCOPE_PARENT


def test_consolidated_caption_is_recognized():
    assert _table_scope("合并资产负债表") == SCOPE_CONSOLIDATED
    assert _table_scope("合并利润表") == SCOPE_CONSOLIDATED
    # 目录页的写法带序号，同样要认出来
    assert _table_scope("3、合并年初到报告期末现金流量表") == SCOPE_CONSOLIDATED


def test_caption_without_scope_stays_none():
    """判不出就不表态：这类表（主要指标、非经常性损益）两张表都可能引用。"""
    assert _table_scope("（一）主要会计数据和财务指标") is None
    assert _table_scope("（二）非经常性损益项目和金额") is None
    assert _table_scope(None) is None


def test_scope_is_not_guessed_from_table_body():
    """只看表题。表体里的「其中：」小计项在两张表都有，用它判会全判成同一个。"""
    body = "| 其中：利息费用 | 332,791 |"
    assert _table_scope(body) is None


# ——— 解析 → 切片 的透传 ———


def test_scope_passes_from_segment_to_chunk():
    segs = [
        Segment(
            "母公司利润表\n| 项目 | 本期 |\n|---|---|\n| 研发费用 | 37 |",
            page=16,
            kind="table",
            scope=SCOPE_PARENT,
        ),
        Segment("普通正文，没有口径", page=17, kind="text"),
    ]
    chunks = chunk_segments(segs, max_chars=2000)
    tables = [c for c in chunks if c.kind == "table"]
    assert tables and tables[0].scope == SCOPE_PARENT
    # 正文段不带口径，不能被顺手标成某个值
    assert all(c.scope is None for c in chunks if c.kind == "text")


def test_long_table_chunks_all_inherit_scope():
    """整表切成多片时，每片都是同一张表，口径必须一起带过去。"""
    rows = "\n".join(f"| 科目{i} | {i} |" for i in range(60))
    text = "合并现金流量表\n| 项目 | 本期 |\n|---|---|\n" + rows
    chunks = chunk_segments(
        [Segment(text, page=13, kind="table", scope=SCOPE_CONSOLIDATED)],
        max_chars=300,
    )
    assert len(chunks) > 1
    assert all(c.scope == SCOPE_CONSOLIDATED for c in chunks)


# ——— 检索偏好 ———


def _rows():
    """模拟实测排名：母公司 4 行片第 1，合并 11 行片第 3。"""
    return [
        {"seq": 68, "scope": SCOPE_PARENT, "_fused_score": 1 / 61, "text": "母公司"},
        {"seq": 44, "scope": SCOPE_CONSOLIDATED, "_fused_score": 1 / 62, "text": "合并"},
        {"seq": 49, "scope": SCOPE_CONSOLIDATED, "_fused_score": 1 / 63, "text": "合并"},
    ]


def test_default_query_prefers_consolidated():
    assert _prefer_scope("中国能建期末现金及现金等价物余额") == SCOPE_CONSOLIDATED
    out = _apply_scope_preference("中国能建期末现金及现金等价物余额", _rows())
    assert out[0]["seq"] in (44, 49)  # 合并口径上来了
    assert out[-1]["seq"] == 68  # 母公司沉到最后


def test_query_asking_parent_keeps_parent_on_top():
    assert _prefer_scope("中国能建母公司期末现金余额") == SCOPE_PARENT
    out = _apply_scope_preference("中国能建母公司期末现金余额", _rows())
    assert out[0]["seq"] == 68


def test_rows_without_scope_are_untouched():
    """正文段、无表题的表格与口径无关，不能被这条规则牵连。"""
    rows = [
        {"seq": 1, "scope": None, "_fused_score": 1 / 61},
        {"seq": 2, "scope": None, "_fused_score": 1 / 62},
    ]
    assert [r["seq"] for r in _apply_scope_preference("随便问点什么", rows)] == [1, 2]


def test_bias_is_small_enough_not_to_overturn_big_gaps():
    """只压得动 RRF 头部的 3% 差距，压不动真实的相关性差距。"""
    rows = [
        {"seq": 1, "scope": SCOPE_PARENT, "_fused_score": 1 / 61},  # 明显相关
        {"seq": 2, "scope": SCOPE_CONSOLIDATED, "_fused_score": 1 / 90},  # 差一截
    ]
    out = _apply_scope_preference("期末现金", rows)
    assert out[0]["seq"] == 1


def test_missing_score_falls_back_to_rank():
    """纯向量路径没有 RRF 分，按名次折算，不能因为缺分就原地不动。"""
    rows = [
        {"seq": 68, "scope": SCOPE_PARENT},
        {"seq": 44, "scope": SCOPE_CONSOLIDATED},
    ]
    out = _apply_scope_preference("期末现金", rows)
    assert out[0]["seq"] == 44
