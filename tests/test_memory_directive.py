"""长期指令（directive）：回答方式的要求要单独成段、不被索引截断。"""

from ai_agent.memory.facets import FACETS, FACET_LABELS, normalize_facet
from ai_agent.memory.progressive import _DIRECTIVE_HEADER, build_memory_index
from ai_agent.storage.db import MemoryItem


def _item(id_: str, text: str, facet: str) -> MemoryItem:
    return MemoryItem(
        id=id_,
        user_id="u1",
        text=text,
        created_at="2026-11-14T00:00:00+00:00",
        facet=facet,
    )


def test_directive_is_a_facet():
    assert "directive" in FACETS
    assert FACET_LABELS["directive"] == "长期指令"
    assert normalize_facet("长期指令") == "directive"
    assert normalize_facet("回答要求") == "directive"


def test_directive_gets_its_own_block():
    items = [
        _item("d1", "以后都先给结论再展开", "directive"),
        _item("p1", "喜欢用表格对比", "preference"),
    ]
    block, _trace = build_memory_index(items, user_input="随便聊聊", top_k=8)
    assert block.startswith(_DIRECTIVE_HEADER)
    assert "以后都先给结论再展开" in block
    assert "喜欢用表格对比" in block


def test_directive_survives_top_k_truncation():
    items = [_item("d1", "每次回答都要附出处", "directive")] + [
        _item(f"p{i}", f"偏好条目 {i}", "preference") for i in range(10)
    ]
    block, _trace = build_memory_index(items, user_input="问一个偏好条目", top_k=2)
    # 索引只留 2 条偏好，但指令必须还在
    assert "每次回答都要附出处" in block
    assert block.count("偏好条目") <= 2


def test_no_directive_keeps_original_layout():
    items = [_item("p1", "喜欢葡萄", "preference")]
    block, _trace = build_memory_index(items, user_input="水果")
    assert _DIRECTIVE_HEADER not in block
