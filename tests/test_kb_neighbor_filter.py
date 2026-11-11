"""邻接片段去重：只去掉重复表头，真正的跨片续写要留住。"""

from ai_agent.tools.kb.kb_search import _useful_neighbor


BODY = (
    "合并利润表\n"
    "| 项目 | 2026年前三季度 | 2025年前三季度 |\n"
    "|---|---|---|\n"
    "| 五、净利润 | 1,589,422,902.20 | 951,408,366.04 |"
)


def test_repeated_table_header_is_dropped():
    piece = (
        "合并利润表\n"
        "| 项目 | 2026年前三季度 | 2025年前三季度 |\n"
        "|---|---|---|\n"
        "| （一）归属母公司所有者的其他综合收益 | -512,385,879.05 | -141,623,765.26 |\n"
        "| 1.不能重分类进损益的其他综合收益 | -5,499,524.43 | 1,463,269.66 |"
    )
    kept = _useful_neighbor(piece, BODY)
    assert "|---|---|---|" not in kept
    assert "（一）归属母公司所有者的其他综合收益" in kept


def test_pure_duplicate_is_dropped_entirely():
    piece = "合并利润表\n| 项目 | 2026年前三季度 | 2025年前三季度 |\n|---|---|---|"
    assert _useful_neighbor(piece, BODY) == ""


def test_novel_continuation_is_kept():
    """「原因有三：」这种跨片续写不在命中片里，必须保留。"""
    piece = "原因有三：一是需求回暖，二是产能利用率提升，三是产品结构改善。"
    assert _useful_neighbor(piece, BODY) == piece


def test_tiny_remainder_is_dropped():
    piece = "| 项目 | 2026年前三季度 | 2025年前三季度 |\n|---|---|---|\n余"
    assert _useful_neighbor(piece, BODY) == ""
