"""kb_read：按位置（文件名 + 页码）读回切片原文。

位置就是命中抬头里那串「文件名 · 第N页」——模型回答时本来就要交代出处。
早期版本把内部引用（``file_id:seq``）塞进工具正文，模型照着抄进了回答，
所以那一套（ContextVar 旁路 + 消息元数据 + 输出侧正则过滤）整体删掉了。
"""

import uuid
from unittest.mock import patch

from ai_agent.kb.retrieval.retrieval import Hit
from ai_agent.tools.kb import kb_search as ks

FILE_ID = "3823ff6e-1111-2222-3333-444455556666"


def _hit(seq: int, page: int, text: str) -> Hit:
    return Hit(
        file_id=FILE_ID,
        filename="某公司2026年第三季度报告",
        seq=seq,
        page=page,
        kind="table",
        text=text,
        distance=0.2,
        section="四、季度财务报表",
        period="2026Q3",
    )


def _row(seq: int, page: int, text: str) -> dict:
    return {
        "file_id": FILE_ID,
        "seq": seq,
        "page": page,
        "page_end": page,
        "kind": "table",
        "section": "四、季度财务报表",
        "scope": None,
        "period": "2026Q3",
        "text": text,
        "filename": "某公司2026年第三季度报告",
    }


def test_preview_masks_numbers():
    """清单预览不能漏出金额，否则模型会拿预览直接作答。"""
    preview = ks._preview("| 货币资金 | 5,060,456,799.16 | 5,574,955,555.50 |")
    assert "5,060,456,799" not in preview
    assert "5,574,955,555" not in preview
    assert "…" in preview


def test_list_format_has_index_and_location():
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")]
    lines = ks._format_hit_list(hits)
    assert lines.startswith("[1] ")
    assert "第10页" in lines
    assert "1,589,422,902.20" not in lines       # 金额仍然打码


def test_tool_output_has_no_internal_reference():
    """工具正文里不能出现内部标识 —— 模型会照着复述进回答。"""
    hit = _hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")
    out = ks._format_hit_with_context(hit)
    assert "ref=" not in out
    assert "3823ff6e" not in out


def test_read_by_header_location():
    header = "[某公司2026年第三季度报告 · 2026Q3 · 第10页 · 四、季度财务报表]"
    with patch.object(
        ks.store, "fetch_kb_chunks_by_location",
        return_value=[_row(39, 10, "| 五、净利润 | 1,589,422,902.20 |")],
    ) as spy:
        out = ks._read_hits(location=header)
    assert "1,589,422,902.20" in out
    name, page = spy.call_args.args
    assert name == "某公司2026年第三季度报告"
    assert page == 10


def test_read_by_short_location():
    """模型只写「文件名 · 第3页」也能读回。"""
    with patch.object(
        ks.store, "fetch_kb_chunks_by_location",
        return_value=[_row(3, 3, "股东总数 428,827")],
    ):
        out = ks._read_hits(location="某公司2026年第三季度报告 · 第3页")
    assert "428,827" in out


def test_read_without_location_asks_for_it():
    assert "请给出位置" in ks._read_hits(location="")


def test_read_unknown_file_asks_research():
    with patch.object(ks.store, "fetch_kb_chunks_by_location", return_value=[]):
        out = ks._read_hits(location="不存在的公司 · 第3页")
    assert "重新 kb_search" in out


def test_read_is_stateless():
    """同样的位置读两次都回库里查，结果一致。"""
    payload = [_row(39, 10, "| 净利润 | 100 |")]
    with patch.object(
        ks.store, "fetch_kb_chunks_by_location", return_value=payload
    ) as spy:
        first = ks._read_hits(location="某公司 · 第10页")
        second = ks._read_hits(location="某公司 · 第10页")
    assert first == second
    assert spy.call_count == 2


def test_aged_stub_points_to_location_not_reference():
    from ai_agent.agents.messages import age_tool_results

    content = "[某公司 · 第10页]\n" + "数" * 900
    messages = [
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "kb_search", "arguments": {"query": "q"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "kb_search", "content": content},
        {"role": "user", "content": "问2"},
        {"role": "user", "content": "问3"},
    ]
    stub = age_tool_results(messages, keep_turns=2)[2]["content"]
    assert "原文已省略" in stub
    assert "kb_read(location=" in stub
    assert "ref=" not in stub


def test_list_only_flag_declared_off_by_default():
    from ai_agent.core.config import Settings

    assert Settings.model_fields["kb_search_list_only"].default is False


def test_search_output_has_no_reference_tokens(monkeypatch):
    monkeypatch.setattr(ks.store, "has_ready_kb_file", lambda: True)
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")]
    monkeypatch.setattr(
        ks, "retrieve_for_question", lambda *a, **kw: ([uuid.uuid4()], hits)
    )
    monkeypatch.setattr(ks.settings, "kb_search_list_only", False)
    out = ks._kb_search("长电科技 2026Q3 净利润")
    assert "ref=" not in out
    assert "3823ff6e" not in out
