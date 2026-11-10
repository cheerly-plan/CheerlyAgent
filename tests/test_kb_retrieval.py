"""检索质量四件套：语境前缀、去冗余、重排、邻接扩展。"""

import uuid

import pytest

from ai_agent.core.config import settings
from ai_agent.kb.retrieval import retrieval
from ai_agent.kb.ingest.chunking import context_prefix, embedding_input, html_tables_to_markdown
from ai_agent.tools.kb.kb_search import _format_hit, _format_hit_with_context

FILE_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


def _row(seq: int, text: str, *, distance: float = 0.3) -> dict:
    return {
        "id": uuid.uuid4(),
        "file_id": FILE_ID,
        "filename": "半导体报告.pdf",
        "seq": seq,
        "page": seq + 1,
        "kind": "text",
        "text": text,
        "distance": distance,
    }


def test_context_prefix_carries_document_and_position():
    assert context_prefix("半导体报告.pdf", page=3, kind="text") == "《半导体报告》第3页"
    assert context_prefix("年报.pdf", page=16, kind="table") == "《年报》第16页"
    assert context_prefix("财务表.xlsx", page=2, kind="table") == "《财务表》表2"
    assert context_prefix("年报.pdf", page=5, kind="figure") == "《年报》第5页图表"
    # 无页概念的文档（Word）只带文档名，不硬编段号
    assert context_prefix("考勤制度.docx", page=None, kind="text") == "《考勤制度》"


def test_embedding_input_prefixes_context_but_keeps_body():
    text = "HBM 供需缺口扩大"
    assert embedding_input(text, filename="a.pdf", page=2, kind="text") == (
        f"《a》第2页\n{text}"
    )


def test_embedding_and_rerank_read_table_cells_not_tags():
    html = (
        "<table><tr><td>项目</td><td>期末</td></tr>"
        "<tr><td>货币资金</td><td>634,706,310.18</td></tr></table>"
    )
    embedded = embedding_input(html, filename="季报.pdf", page=8, kind="table")
    assert "<td>" not in embedded
    assert "货币资金 | 634,706,310.18" in embedded
    shown = retrieval._candidate_text(
        {"filename": "季报.pdf", "kind": "table", "page": 8, "seq": 1, "text": html}
    )
    assert "<td>" not in shown
    assert "货币资金 | 634,706,310.18" in shown
    assert retrieval._RERANK_CLIP == 800


def test_embedding_flattens_markdown_table_separator():
    """修订切片若存成 Markdown 表，分隔行不能进向量。"""
    md = "| 项目 | 期末 |\n| --- | --- |\n| 货币资金 | 1 |"
    embedded = embedding_input(md, filename="季报.pdf", page=8, kind="table")
    assert "---" not in embedded
    assert "货币资金 | 1" in embedded


def test_oversized_rowspan_does_not_invent_rows():
    """合并格被切剩一行时，不能按 rowspan 补出只有原因、没有项目名的空行。"""
    html = (
        "<table>"
        "<tr><td>项目名称</td><td>变动比例(%)</td><td>主要原因</td></tr>"
        "<tr><td>利润总额_本报告期</td><td>-76.60</td>"
        '<td rowspan="9">利润波动</td></tr>'
        "</table>"
    )
    md = html_tables_to_markdown(html)
    assert md.count("利润波动") == 1
    assert "|  |  | 利润波动 |" not in md


def test_html_table_becomes_markdown_and_keeps_merged_headers():
    """模型看到的是 Markdown 表。合并列表头要复制到每一列，分组标题不横着复制。"""
    html = (
        "<table>"
        '<tr><td rowspan="2">项目</td><td colspan="2">期末余额</td></tr>'
        "<tr><td>合并</td><td>母公司</td></tr>"
        '<tr><td colspan="3">流动资产：</td></tr>'
        "<tr><td>货币资金</td><td>100</td><td>80</td></tr>"
        "</table>"
    )
    md = html_tables_to_markdown(f"合并资产负债表\n{html}\n附注")
    assert "<td>" not in md and "<table>" not in md
    assert md.startswith("合并资产负债表\n")
    assert md.endswith("附注")
    assert "| 项目 | 期末余额 | 期末余额 |" in md
    assert "| 项目 | 合并 | 母公司 |" in md
    assert "| 流动资产： |  |  |" in md
    assert "| 货币资金 | 100 | 80 |" in md
    # 竖线是列分隔，单元格里的竖线不能再被当成一列
    assert html_tables_to_markdown("<table><tr><td>a|b</td><td>1</td></tr></table>") == (
        "| a\\|b | 1 |\n| --- | --- |"
    )


def test_format_hit_sends_markdown_table_to_the_model():
    html = (
        "<table><tr><td>项目</td><td>本期</td></tr>"
        "<tr><td>营业收入</td><td>169,462</td></tr></table>"
    )
    text = _format_hit(
        {
            "filename": "季报.pdf",
            "kind": "table",
            "page": 8,
            "seq": 3,
            "text": html,
        }
    )
    assert "<td>" not in text
    assert "| 营业收入 | 169,462 |" in text
    assert text.startswith("[季报.pdf · 第8页]")


def test_citations_keep_html_so_the_preview_can_merge_cells():
    """前端预览靠 colspan 画合并格，引用正文不能跟模型上下文一起改成 Markdown。"""
    from ai_agent.tools.kb.kb_search import _citations_of

    html = "<table><tr><td colspan=\"2\">流动资产：</td></tr></table>"
    hit = retrieval.Hit(
        file_id=str(FILE_ID),
        filename="季报.pdf",
        seq=3,
        page=8,
        kind="table",
        text=html,
        distance=0.2,
    )
    [citation] = _citations_of([hit])
    assert citation.text == html


def test_dedupe_keeps_similar_table_from_another_file():
    """两年的股东表长得很像，去重不能把另一年整份丢掉。"""
    body = "前10名股东持股情况表苏州翌流明光电科技有限公司37800963" * 4
    rows = [
        {**_row(0, body), "file_id": uuid.uuid4()},
        {**_row(1, body + "尾"), "file_id": uuid.uuid4()},
    ]
    assert len(retrieval._dedupe(rows)) == 2


def test_multi_period_query_keeps_both_report_years(monkeypatch):
    """重排把 2026 排满时，点名的 2025 三季报仍要占一个名额，并且排在前面。"""
    monkeypatch.setattr(settings, "kb_subject_quota_band", 0.12)
    newer = uuid.uuid4()
    older = uuid.uuid4()
    rows = []
    for seq in range(3):
        rows.append(
            {
                **_row(seq, f"2026股东{seq}", distance=0.2 + seq * 0.01),
                "file_id": newer,
                "filename": "深圳中科飞测科技股份有限公司2026年第三季度报告.pdf",
                "subject": "苏州翌流明光电科技有限公司",
            }
        )
    rows.append(
        {
            **_row(3, "2025股东", distance=0.45),
            "file_id": older,
            "filename": "深圳中科飞测科技股份有限公司2025年第三季度报告.pdf",
            "subject": "深圳市创新投资集团有限公司",
        }
    )
    picked = retrieval._diversify(
        "中科飞测2025年和2026年三季度股东持股对比", rows, top_k=3
    )
    assert {row["file_id"] for row in picked} >= {newer, older}
    assert picked[0]["file_id"] == newer
    assert picked[1]["file_id"] == older


def test_two_named_companies_each_keep_a_slot_despite_rewritten_subjects():
    """两家一起查时，股东表把主体改成股东名，也不能让一家占满 top_k。"""
    bank = uuid.uuid4()
    auto = uuid.uuid4()
    rows = []
    for seq in range(4):
        rows.append(
            {
                **_row(seq, f"长安股东{seq}", distance=0.20 + seq * 0.01),
                "file_id": auto,
                "filename": "长安汽车有限公司2026年第三季度报告.pdf",
                "subject": f"股东{seq}",
            }
        )
    rows.append(
        {
            **_row(4, "41,108,449", distance=0.25),
            "file_id": bank,
            "filename": "中国银行股份有限公司2026年第三季度报告.pdf",
            "subject": "中国银行股份有限公司",
        }
    )
    picked = retrieval._diversify(
        "中国银行和长安汽车的总资产分别是多少",
        rows,
        top_k=3,
        per_file=True,
    )
    assert [row["file_id"] for row in picked[:2]] == [auto, bank]


def test_dedupe_drops_near_duplicate_chunks():
    """切片 overlap 会让相邻片高度重合，只保留信息量最大的那条。"""
    body = "2026 年 HBM 市场规模约 546 亿美元，同比增速 58%，产能供不应求。" * 3
    rows = [
        _row(0, body),
        _row(1, body + "略有改动"),  # 与上一条三字组重合度极高
        _row(2, "公司考勤制度规定，迟到超过 30 分钟按旷工处理。"),
    ]
    kept = retrieval._dedupe(rows)
    assert [row["seq"] for row in kept] == [0, 2]


class _FakeReranker:
    """可插拔重排后端的测试桩：_rerank 现在走 get_reranker().rerank()。"""

    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return self._scores


def test_rerank_follows_model_order_and_keeps_unlisted(monkeypatch):
    rows = [_row(0, "甲"), _row(1, "乙"), _row(2, "丙")]
    # 分数制：rank 越高分越高；这里让丙(2)最相关、甲(0)次之、乙(1)最低
    fake = _FakeReranker([1.0, 0.0, 2.0])
    monkeypatch.setattr(retrieval, "get_reranker", lambda: fake)
    ordered = retrieval._rerank("问题", rows, top_k=2)
    assert [row["seq"] for row in ordered] == [2, 0, 1]


def test_diversify_by_subject_keeps_second_company_after_rerank(monkeypatch):
    """重排把一家排满 top_k 时，带宽内的另一家头名仍要留下。"""
    monkeypatch.setattr(settings, "kb_subject_quota_band", 0.12)
    rows = [
        {**_row(0, "甲营收", distance=0.20), "subject": "甲公司"},
        {**_row(1, "甲利润", distance=0.21), "subject": "甲公司"},
        {**_row(2, "甲现金", distance=0.22), "subject": "甲公司"},
        {**_row(3, "乙营收", distance=0.30), "subject": "乙公司"},
    ]
    picked = retrieval._diversify_by_subject(rows, top_k=3)
    assert [row["seq"] for row in picked] == [0, 1, 3]


def test_diversify_by_subject_skips_far_company(monkeypatch):
    monkeypatch.setattr(settings, "kb_subject_quota_band", 0.12)
    rows = [
        {**_row(0, "甲营收", distance=0.20), "subject": "甲公司"},
        {**_row(1, "甲利润", distance=0.21), "subject": "甲公司"},
        {**_row(2, "甲现金", distance=0.22), "subject": "甲公司"},
        {**_row(3, "乙无关", distance=0.50), "subject": "乙公司"},
    ]
    picked = retrieval._diversify_by_subject(rows, top_k=3)
    assert [row["seq"] for row in picked] == [0, 1, 2]


def test_rerank_falls_back_on_garbage_or_error(monkeypatch):
    rows = [_row(0, "甲"), _row(1, "乙")]

    # 分数数量不符（garbage）→ 退回向量顺序
    fake = _FakeReranker([1.0])  # 只有 1 个分数，候选有 2 条
    monkeypatch.setattr(retrieval, "get_reranker", lambda: fake)
    assert retrieval._rerank("问题", rows, top_k=2) == rows

    # 后端抛错 → 退回向量顺序，绝不丢候选
    def boom(query, docs):
        raise RuntimeError("模型超时")

    fake_err = _FakeReranker([])
    fake_err.rerank = boom
    monkeypatch.setattr(retrieval, "get_reranker", lambda: fake_err)
    assert retrieval._rerank("问题", rows, top_k=2) == rows


def test_parse_order_ignores_out_of_range_and_duplicates():
    assert retrieval._parse_order("[2,2,9,0,-1,1]", 3) == [2, 1]
    assert retrieval._parse_order("no json here", 3) == []


def test_search_knowledge_applies_threshold_and_neighbors(monkeypatch):
    calls: dict = {}

    def fake_search(embedding, *, file_ids=None, top_k=3, max_distance=None):
        calls["top_k"] = top_k
        calls["max_distance"] = max_distance
        return [_row(4, "命中片段")]

    def fake_neighbors(pairs, *, radius=1):
        calls["radius"] = radius
        return {
            (str(FILE_ID), 3): [{"seq": 3, "page": 3, "kind": "text", "text": "上文"}],
            (str(FILE_ID), 5): [{"seq": 5, "page": 5, "kind": "text", "text": "下文"}],
        }

    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(retrieval.store, "search_kb_chunks", fake_search)
    monkeypatch.setattr(retrieval.store, "kb_chunk_neighbors", fake_neighbors)
    monkeypatch.setattr(settings, "kb_rerank_enabled", False)
    monkeypatch.setattr(settings, "kb_hybrid_enabled", False)
    monkeypatch.setattr(settings, "kb_search_candidates", 16)
    monkeypatch.setattr(settings, "kb_search_max_distance", 0.72)
    monkeypatch.setattr(settings, "kb_neighbor_expand", 1)

    hits = retrieval.search_knowledge("HBM 缺口", top_k=3)

    assert len(hits) == 1
    assert hits[0].before == ["上文"] and hits[0].after == ["下文"]
    assert calls["top_k"] == 16  # 召回放大，再重排/去重后取 top_k
    assert calls["max_distance"] == pytest.approx(0.72)
    assert calls["radius"] == 1


def test_search_knowledge_survives_neighbor_failure(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(
        retrieval.store, "search_kb_chunks", lambda *a, **k: [_row(0, "片段")]
    )

    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(retrieval.store, "kb_chunk_neighbors", boom)
    monkeypatch.setattr(settings, "kb_rerank_enabled", False)
    monkeypatch.setattr(settings, "kb_hybrid_enabled", False)
    monkeypatch.setattr(settings, "kb_neighbor_expand", 1)

    hits = retrieval.search_knowledge("问题")
    assert hits[0].text == "片段"
    assert hits[0].before == [] and hits[0].after == []


def test_to_hit_keeps_missing_distance_as_none():
    row = _row(0, "关键词独有")
    row.pop("distance")
    assert retrieval._to_hit(row).distance is None
    row["distance"] = 0.0
    assert retrieval._to_hit(row).distance == 0.0


def test_format_hit_puts_period_before_the_page():
    text = _format_hit(
        {
            "filename": "甲公司2025年第三季度报告.pdf",
            "kind": "table",
            "page": 4,
            "seq": 13,
            "text": "苏州翌流明 37,800,963",
            "period": "2025Q3",
        }
    )
    assert text.startswith("[甲公司2025年第三季度报告.pdf · 2025Q3 · 第4页]")


def test_format_hit_with_context_marks_neighbors_and_position():
    hit = retrieval.Hit(
        file_id=str(FILE_ID),
        filename="半导体报告.pdf",
        seq=4,
        page=5,
        kind="text",
        text="命中正文",
        distance=0.2,
        before=["上一段结尾"],
        after=["下一段开头"],
    )
    out = _format_hit_with_context(hit)
    assert out.startswith("[半导体报告.pdf · 第5页]")
    assert "〔上文〕上一段结尾" in out
    assert "命中正文" in out
    assert "〔下文〕下一段开头" in out
    # 顺序必须是 上文 → 命中 → 下文，模型才能顺着读
    assert out.index("〔上文〕") < out.index("命中正文") < out.index("〔下文〕")


def test_format_hit_with_context_labels_figure_evidence():
    hit = retrieval.Hit(
        file_id=str(FILE_ID),
        filename="年报.pdf",
        seq=0,
        page=2,
        kind="figure",
        text="图中显示 2026 年营收",
        distance=0.4,
    )
    out = _format_hit_with_context(hit)
    assert "第2页图表（图像识别，可能有误差）" in out


def test_kb_search_follows_session_scope_when_model_omits_file_ids(monkeypatch):
    """输入框里选了「只查这几份文件」，工具默认就只查这些，模型不传也一样。"""
    from ai_agent.kb.retrieval.scope import reset_kb_scope, set_kb_scope
    from ai_agent.tools.kb import kb_search as kb_tool

    seen: dict = {}

    def fake_search(queries, *, file_ids=None, top_k=3, filters=None):
        seen["file_ids"] = file_ids
        return []

    monkeypatch.setattr(kb_tool.store, "has_ready_kb_file", lambda: True)
    monkeypatch.setattr(kb_tool, "search_knowledge_multi", fake_search)
    monkeypatch.setattr("ai_agent.kb.retrieval.scope._route", lambda q: [])
    latest = uuid.uuid4()
    monkeypatch.setattr(
        "ai_agent.storage.store.store.list_kb_files",
        lambda **k: [
            {
                "file_id": uuid.uuid4(),
                "status": "ready",
                "subject": "甲公司",
                "period": "2025",
                "filename": "甲公司2025年年度报告.pdf",
                "subject_aliases": [],
            },
            {
                "file_id": latest,
                "status": "ready",
                "subject": "甲公司",
                "period": "2026Q3",
                "filename": "甲公司2026年第三季度报告.pdf",
                "subject_aliases": [],
            },
        ],
    )

    token = set_kb_scope([str(FILE_ID)])
    try:
        kb_tool._kb_search("问题")
    finally:
        reset_kb_scope(token)

    assert seen["file_ids"] == [FILE_ID]

    # 模型显式给了文件名时以解析结果为准（它可能只查其中一份）
    other = uuid.UUID("99999999-9999-9999-9999-999999999999")
    monkeypatch.setattr(
        "ai_agent.kb.retrieval.resolve_files_by_name", lambda names: [other]
    )
    token = set_kb_scope([str(FILE_ID)])
    try:
        kb_tool._kb_search("问题", files=["别的文件.pdf"])
    finally:
        reset_kb_scope(token)
    assert seen["file_ids"] == [other]

    # 没圈范围、问句也没写时期：同一主体只留最新一期，不再把年报和季报一起搜
    kb_tool._kb_search("问题")
    assert seen["file_ids"] == [latest]


def test_kg_search_follows_session_scope_when_model_omits_file_ids(monkeypatch):
    """图谱检索也要跟输入框的文件范围，否则圈定文件后仍会搜到库里别的实体。"""
    from ai_agent.kb.retrieval.scope import reset_kb_scope, set_kb_scope
    from ai_agent.tools.kb import kg_search as kg_tool

    seen: dict = {}

    monkeypatch.setattr(kg_tool.store, "graph_stats", lambda **k: {"entities": 3})
    monkeypatch.setattr(kg_tool, "embed_query", lambda q: [0.1])
    monkeypatch.setattr("ai_agent.kb.retrieval.scope._route", lambda q: [])
    latest = uuid.uuid4()
    monkeypatch.setattr(
        "ai_agent.storage.store.store.list_kb_files",
        lambda **k: [
            {
                "file_id": uuid.uuid4(),
                "status": "ready",
                "subject": "甲公司",
                "period": "2025",
                "filename": "甲公司2025年年度报告.pdf",
                "subject_aliases": [],
            },
            {
                "file_id": latest,
                "status": "ready",
                "subject": "甲公司",
                "period": "2026Q3",
                "filename": "甲公司2026年第三季度报告.pdf",
                "subject_aliases": [],
            },
        ],
    )

    def fake_entities(embedding, *, query="", top_k=5, file_ids=None, min_score=0.0):
        seen["file_ids"] = file_ids
        return []

    monkeypatch.setattr(kg_tool.store, "search_entities", fake_entities)

    token = set_kb_scope([str(FILE_ID)])
    try:
        kg_tool._kg_search("问题和谁有关")
    finally:
        reset_kb_scope(token)

    assert seen["file_ids"] == [FILE_ID]

    kg_tool._kg_search("问题和谁有关")
    assert seen["file_ids"] == [latest]




def test_kb_search_schema_exposes_files_not_ids():
    from ai_agent.tools.registry import tool_schemas

    schema = next(s for s in tool_schemas() if s["function"]["name"] == "kb_search")
    props = schema["function"]["parameters"]["properties"]
    assert "files" in props
    assert "file_ids" not in props
    kg = next(s for s in tool_schemas() if s["function"]["name"] == "kg_search")
    kg_props = kg["function"]["parameters"]["properties"]
    assert "files" in kg_props
    assert "file_ids" not in kg_props


def test_kb_scope_is_isolated_and_resettable():
    from ai_agent.kb.retrieval.scope import kb_scope, reset_kb_scope, set_kb_scope

    assert kb_scope() == ()
    token = set_kb_scope([" a ", "", "b"])
    try:
        assert kb_scope() == ("a", "b")
    finally:
        reset_kb_scope(token)
    assert kb_scope() == ()


def test_run_loop_applies_kb_scope_around_tool_execution(monkeypatch):
    """范围必须落到「工具执行那一刻」：run_stream 的生成器会被不同线程
    轮流 next()，外层设置的 contextvar 传不进去，只能在执行点同栈设置。"""
    from ai_agent.agents import loop as loop_mod
    from ai_agent.kb.retrieval.scope import kb_scope

    seen: list[tuple[str, ...]] = []

    def fake_stream(messages, tools, **kwargs):
        if not any(m.get("role") == "tool" for m in messages):
            yield "text", ""
            yield "result", {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "kb_search", "arguments": {"query": "q"}}
                ],
            }
        else:
            yield "text", "好的"
            yield "result", {"content": "好的", "tool_calls": []}

    def fake_execute(name, arguments):
        seen.append(kb_scope())
        return "检索结果"

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)

    events = list(
        loop_mod.run_loop(
            [{"role": "user", "content": "问个问题"}],
            user_id="u1",
            kb_file_ids=(str(FILE_ID),),
        )
    )

    assert seen == [(str(FILE_ID),)]
    assert any(e.get("type") == "text" for e in events)
    # 执行完必须复位，不能泄漏到同一线程里的下一次调用
    assert kb_scope() == ()


def test_query_looks_on_topic_needs_company_name(monkeypatch):
    """「深圳带伞」不能因为文件名里有「深圳市顺丰」就算在查库。"""
    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            {
                "file_id": FILE_ID,
                "filename": "顺丰控股股份有限公司2026年第三季度报告.pdf",
                "status": "ready",
                "tags": ["季报"],
            }
        ],
    )
    assert retrieval.query_looks_on_topic("顺丰控股投了哪些基金") is True
    assert retrieval.query_looks_on_topic("深圳宝安明天出门需要带伞吗") is False
