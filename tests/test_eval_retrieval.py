"""评测脚本的用例校验与命中判定。

这两块逻辑决定了"评测分数可不可信"，比检索代码本身更该有测试：
校验漏判 → 失效用例伪装成检索 miss；判定过松 → 分数虚高。
"""

from types import SimpleNamespace

from scripts.eval_retrieval import hit_rank, normalize_expect, validate_cases


def _hit(filename: str, text: str):
    return SimpleNamespace(filename=filename, text=text)


def _case(query: str, expect: dict) -> dict:
    return {"query": query, "expect": expect}


# ——— expect 的三种写法统一成 sources ———


def test_normalize_single_text():
    sources = normalize_expect({"filename": "甲.md", "text": "原文"})
    assert sources == [{"filename": "甲.md", "texts": ["原文"]}]


def test_normalize_text_all():
    sources = normalize_expect({"filename": "甲.md", "text_all": ["A", "B"]})
    assert sources == [{"filename": "甲.md", "texts": ["A", "B"]}]


def test_normalize_sources_keeps_file_per_source():
    sources = normalize_expect(
        {
            "sources": [
                {"filename": "甲.md", "text": "A"},
                {"filename": "乙.md", "text_all": ["B", "C"]},
            ]
        }
    )
    assert sources == [
        {"filename": "甲.md", "texts": ["A"]},
        {"filename": "乙.md", "texts": ["B", "C"]},
    ]


# ——— 命中判定 ———


def test_hit_rank_returns_position_of_first_match():
    hits = [_hit("乙.md", "无关"), _hit("甲.md", "含原文")]
    assert hit_rank(hits, {"filename": "甲.md", "text": "原文"}) == 2


def test_hit_rank_miss_when_text_absent():
    hits = [_hit("甲.md", "别的内容")]
    assert hit_rank(hits, {"filename": "甲.md", "text": "原文"}) is None


def test_hit_rank_miss_when_text_in_wrong_file():
    hits = [_hit("乙.md", "含原文")]
    assert hit_rank(hits, {"filename": "甲.md", "text": "原文"}) is None


def test_text_all_requires_both_pieces_in_same_hit():
    """多段证据是"同一片里都有"，不是"分散在两条切片里各有一半"。"""
    hits = [_hit("甲.md", "只有A"), _hit("甲.md", "只有B")]
    assert hit_rank(hits, {"filename": "甲.md", "text_all": ["A", "B"]}) is None

    hits = [_hit("甲.md", "A 和 B 都在")]
    assert hit_rank(hits, {"filename": "甲.md", "text_all": ["A", "B"]}) == 1


def test_cross_file_sources_rank_is_when_last_piece_arrives():
    hits = [_hit("甲.md", "A"), _hit("乙.md", "无关"), _hit("乙.md", "B")]
    rank = hit_rank(
        hits,
        {
            "sources": [
                {"filename": "甲.md", "text": "A"},
                {"filename": "乙.md", "text": "B"},
            ]
        },
    )
    assert rank == 3


def test_cross_file_sources_miss_if_one_side_missing():
    hits = [_hit("甲.md", "A")]
    rank = hit_rank(
        hits,
        {
            "sources": [
                {"filename": "甲.md", "text": "A"},
                {"filename": "乙.md", "text": "B"},
            ]
        },
    )
    assert rank is None


# ——— 用例与语料的对齐校验 ———


_ROWS = [
    {"filename": "甲.md", "text": "第一条：A；B 都在这里"},
    {"filename": "乙.md", "text": "第二条：C"},
]


def test_validate_passes_when_evidence_exists():
    cases = [
        _case("问一", {"filename": "甲.md", "text": "A；B"}),
        _case("问二", {"filename": "乙.md", "text": "C"}),
    ]
    assert validate_cases(cases, _ROWS) == []


def test_validate_reports_missing_file():
    problems = validate_cases(
        [_case("问", {"filename": "丙.md", "text": "A"})], _ROWS
    )
    assert len(problems) == 1
    assert "文件不在库里" in problems[0][1]


def test_validate_reports_stitched_text_with_hint():
    """把两句用「；」拼起来是最常见的失效原因，错误信息要能提示这一点。"""
    problems = validate_cases(
        [_case("问", {"filename": "甲.md", "text": "A；C"})], _ROWS
    )
    assert len(problems) == 1
    assert "拼接串" in problems[0][1]


def test_validate_reports_each_missing_piece_in_text_all():
    problems = validate_cases(
        [_case("问", {"filename": "甲.md", "text_all": ["A", "缺的一段"]})], _ROWS
    )
    assert len(problems) == 1
    assert "缺的一段" in problems[0][1]


def test_validate_checks_every_source_separately():
    problems = validate_cases(
        [
            _case(
                "问",
                {
                    "sources": [
                        {"filename": "甲.md", "text": "A"},
                        {"filename": "乙.md", "text": "不存在"},
                    ]
                },
            )
        ],
        _ROWS,
    )
    assert len(problems) == 1
    assert "乙.md" in problems[0][1]
