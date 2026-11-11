"""同表多片合并：不相邻命中（1 和 5）也要合成一条。"""

from ai_agent.kb.retrieval.retrieval import _merge_same_table_rows

HEADER = "2026年1—9月\n合并利润表\n| 项目 | 2026前三季度 |"


def _row(seq: int, text: str, page: int = 10, kind: str = "table", distance: float = 0.2):
    return {
        "file_id": "11111111-1111-1111-1111-111111111111",
        "seq": seq,
        "page": page,
        "kind": kind,
        "text": text,
        "distance": distance,
        "filename": "某公司2026年第三季度报告",
    }


def test_adjacent_slices_merge_into_one():
    rows = [
        _row(37, HEADER + "\n| 营业收入 | 673,041,663.13 |"),
        _row(38, HEADER + "\n| 净利润 | 32,307,014.95 |"),
        _row(1, "别的文件", page=3, kind="text"),
    ]
    merged = _merge_same_table_rows(rows)
    tables = [r for r in merged if r["kind"] == "table"]
    assert len(tables) == 1
    assert "673,041,663.13" in tables[0]["text"]
    assert "32,307,014.95" in tables[0]["text"]


def test_header_repeated_only_once():
    rows = [
        _row(37, HEADER + "\n| 营业收入 | 1 |"),
        _row(38, HEADER + "\n| 净利润 | 2 |"),
    ]
    text = _merge_same_table_rows(rows)[0]["text"]
    assert text.count("合并利润表") == 1


def test_non_adjacent_hits_still_merge():
    """命中落在 1 和 5：中间空档读取失败也不影响合并本身。"""
    rows = [
        _row(37, HEADER + "\n| 营业收入 | 1 |"),
        _row(41, HEADER + "\n| 每股收益 | 0.15 |"),
    ]
    merged = _merge_same_table_rows(rows)
    assert len(merged) == 1
    assert "营业收入" in merged[0]["text"]
    assert "每股收益" in merged[0]["text"]


def test_different_tables_same_page_not_merged():
    other = "2026年1—9月\n合并现金流量表\n| 项目 | 本期 |"
    rows = [_row(37, HEADER + "\n| 营业收入 | 1 |"), _row(50, other + "\n| 现金 | 2 |")]
    assert len(_merge_same_table_rows(rows)) == 2


def test_text_chunks_not_merged():
    rows = [
        _row(1, "第一段正文", kind="text"),
        _row(2, "第一段正文", kind="text"),
    ]
    assert len(_merge_same_table_rows(rows)) == 2


def test_merged_row_keeps_best_distance_and_position():
    rows = [
        _row(37, HEADER + "\n| a | 1 |", distance=0.30),
        _row(38, HEADER + "\n| b | 2 |", distance=0.10),
        _row(99, "其他", page=2, kind="table", distance=0.05),
    ]
    merged = _merge_same_table_rows(rows)
    assert merged[0]["distance"] == 0.10   # 合并结果占最佳成员的位置
    assert len(merged) == 2


def test_zero_distance_is_treated_as_best():
    """distance=0.0 是完全匹配，不能被 `or 1.0` 当成最差。"""
    rows = [
        _row(37, HEADER + "\n| a | 1 |", distance=0.4),
        _row(38, HEADER + "\n| b | 2 |", distance=0.0),
    ]
    merged = _merge_same_table_rows(rows)
    assert merged[0]["distance"] == 0.0
    assert merged[0]["seq"] == 37          # 合并后从头开始
