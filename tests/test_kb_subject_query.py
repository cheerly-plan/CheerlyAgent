"""subject 作为查询入口。

files 参数现接受主体名（含简称/别名），底层按 subject 元数据自动路由到对应文件，
而不必先喂一份文件清单再让模型挑文件名。同时验证：编造的主体名应被识别为
「限定但空」（返回 []），供工具层给出拒绝提示，而不是静默退回全库。
"""

import uuid

from ai_agent.agents import loop as loop_mod
from ai_agent.kb.retrieval import retrieval
from ai_agent.kb.retrieval.scope import resolve_tool_file_ids


def _ready(file_id, filename, subject="", aliases=None, period=""):
    return {
        "file_id": file_id,
        "filename": filename,
        "status": "ready",
        "description": "",
        "subject": subject,
        "period": period,
        "subject_aliases": aliases or [],
    }


def test_resolve_files_by_subject_name(monkeypatch):
    """模型直接说主体名（而非文件名）也能路由到该主体的全部文件。"""
    a = uuid.uuid4()
    b = uuid.uuid4()
    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            _ready(
                a,
                "中国能源建设股份有限公司2026年第三季度报告.pdf",
                "中国能源建设股份有限公司",
                ["中国能源建设", "中国能建"],
                "2026Q3",
            ),
            _ready(
                b,
                "中国能源建设股份有限公司2025年年度报告.pdf",
                "中国能源建设股份有限公司",
                ["中国能源建设", "中国能建"],
                "2025",
            ),
            _ready(
                uuid.uuid4(),
                "顺丰控股股份有限公司2026年第三季度报告.pdf",
                "顺丰控股股份有限公司",
                ["顺丰", "顺丰控股"],
                "2026Q3",
            ),
        ],
    )
    # 用主体名 → 路由到中国能源建设的两份文件
    assert set(resolve_tool_file_ids(["中国能源建设"])) == {a, b}
    # 用别名「中国能建」同样命中
    assert set(resolve_tool_file_ids(["中国能建"])) == {a, b}


def test_resolve_files_by_subject_and_filename_mixed(monkeypatch):
    """files 可混用主体名与文件名简称。"""
    a = uuid.uuid4()
    b = uuid.uuid4()
    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            _ready(
                a,
                "中国能源建设股份有限公司2026年第三季度报告.pdf",
                "中国能源建设股份有限公司",
                ["中国能源建设", "中国能建"],
                "2026Q3",
            ),
            _ready(
                b,
                "顺丰控股股份有限公司2026年第三季度报告.pdf",
                "顺丰控股股份有限公司",
                ["顺丰", "顺丰控股"],
                "2026Q3",
            ),
            # 一份不会被输入匹配到的就绪文件，避免"命中全部=未圈范围"误触发
            _ready(uuid.uuid4(), "某新能源行业研究报告.pdf", "新能源行业", ["新能源"], "2026"),
        ],
    )
    ids = resolve_tool_file_ids(["中国能源建设", "顺丰控股"])
    assert set(ids) == {a, b}


def test_fabricated_subject_returns_empty_list(monkeypatch):
    """编造的主体名：files 指定了但解析不到 → 返回空列表（限定但空），
    供工具层给出拒绝提示，而不是静默退回全库。
    """
    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            _ready(
                uuid.uuid4(),
                "中国能源建设股份有限公司2026年第三季度报告.pdf",
                "中国能源建设股份有限公司",
                ["中国能源建设", "中国能建"],
                "2026Q3",
            ),
        ],
    )
    assert resolve_tool_file_ids(["不存在的某公司"]) == []
    # 未指定 files（空/None）则退回会话范围/全库（None）
    assert resolve_tool_file_ids([]) is None
    assert resolve_tool_file_ids(None) is None


def test_no_subject_falls_back_to_filename(monkeypatch):
    """无 subject 字段的文件（如旧数据）仍按文件名匹配，行为不变。"""
    a = uuid.uuid4()
    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            _ready(a, "中国能源建设股份有限公司2026年第三季度报告.pdf", subject=""),
            # 一份不会被输入匹配到的就绪文件，避免"命中全部=未圈范围"误触发
            _ready(uuid.uuid4(), "某无关文件.pdf", subject=""),
        ],
    )
    # 文件名里的核心名仍能命中；简称不在别名表里就对不上
    assert resolve_tool_file_ids(["中国能源建设"]) == [a]
    assert resolve_tool_file_ids(["中国能建"]) == []


def test_period_narrows_to_latest_named_year_or_keeps_both(monkeypatch):
    """没写时期只留最新一期；写了年份或同比按问句处理；不带 query 不收窄。"""
    a = uuid.uuid4()
    b = uuid.uuid4()
    rows = [
        _ready(
            a,
            "中国能源建设股份有限公司2026年第三季度报告.pdf",
            "中国能源建设股份有限公司",
            ["中国能源建设", "中国能建"],
            "2026Q3",
        ),
        _ready(
            b,
            "中国能源建设股份有限公司2025年年度报告.pdf",
            "中国能源建设股份有限公司",
            ["中国能源建设", "中国能建"],
            "2025",
        ),
        _ready(uuid.uuid4(), "某无关文件.pdf", "无关公司", [], "2026"),
    ]
    monkeypatch.setattr(retrieval.store, "list_kb_files", lambda **k: rows)
    assert set(resolve_tool_file_ids(["中国能源建设"])) == {a, b}
    assert resolve_tool_file_ids(["中国能源建设"], query="营业收入") == [a]
    assert resolve_tool_file_ids(["中国能源建设"], query="2025年营收") == [b]
    assert set(resolve_tool_file_ids(["中国能源建设"], query="营收同比")) == {a, b}
    # 点名的文件被时期滤空时保留原范围，避免一份明确点到的文件被丢掉
    assert set(resolve_tool_file_ids(["中国能源建设"], query="2020年营收")) == {a, b}


def test_exact_entity_in_older_file_overrides_latest(monkeypatch):
    """没写时期时，精确点到的实体只在旧文件里，就改查那份，不留最新一期的别家。"""
    latest = uuid.uuid4()
    older = uuid.uuid4()
    other = uuid.uuid4()
    rows = [
        _ready(latest, "中科飞测2026年第三季度报告.pdf", "深圳中科飞测科技股份有限公司", [], "2026Q3"),
        _ready(older, "中科飞测2025年第三季度报告.pdf", "深圳中科飞测科技股份有限公司", [], "2025Q3"),
        _ready(other, "中国银行2026年第三季度报告.pdf", "中国银行股份有限公司", [], "2026Q3"),
    ]
    monkeypatch.setattr(retrieval.store, "list_kb_files", lambda **k: rows)
    monkeypatch.setattr("ai_agent.kb.retrieval.router.route_files", lambda *a, **k: [])
    monkeypatch.setattr(
        "ai_agent.storage.store.store.exact_entity_files", lambda query: [older]
    )
    assert resolve_tool_file_ids(None, query="诺安成长混合型证券投资基金") == [older]


def test_exact_entity_already_in_latest_keeps_period_scope(monkeypatch):
    """实体在最新一期里还有一份时，期收窄的结果不动。"""
    latest = uuid.uuid4()
    older = uuid.uuid4()
    rows = [
        _ready(latest, "中科飞测2026年第三季度报告.pdf", "深圳中科飞测科技股份有限公司", [], "2026Q3"),
        _ready(older, "中科飞测2025年第三季度报告.pdf", "深圳中科飞测科技股份有限公司", [], "2025Q3"),
    ]
    monkeypatch.setattr(retrieval.store, "list_kb_files", lambda **k: rows)
    monkeypatch.setattr("ai_agent.kb.retrieval.router.route_files", lambda *a, **k: [])
    monkeypatch.setattr(
        "ai_agent.storage.store.store.exact_entity_files", lambda query: [latest, older]
    )
    assert resolve_tool_file_ids(None, query="营业收入") == [latest]


def test_exact_entity_does_not_override_named_period_or_explicit_other_file(monkeypatch):
    """写了年份，或用户点的文件里本来没有这个实体，都不把范围改走。"""
    latest = uuid.uuid4()
    older = uuid.uuid4()
    rows = [
        _ready(
            latest,
            "中国能源建设股份有限公司2026年第三季度报告.pdf",
            "中国能源建设股份有限公司",
            ["中国能源建设"],
            "2026Q3",
        ),
        _ready(
            older,
            "中国能源建设股份有限公司2025年年度报告.pdf",
            "中国能源建设股份有限公司",
            ["中国能源建设"],
            "2025",
        ),
    ]
    monkeypatch.setattr(retrieval.store, "list_kb_files", lambda **k: rows)
    monkeypatch.setattr(
        "ai_agent.storage.store.store.exact_entity_files", lambda query: [older]
    )
    assert resolve_tool_file_ids(["中国能源建设"], query="2026年营收") == [latest]
    # 只点了 2026 这份，旧文件不在候选里，精确实体不能把范围换成它
    assert resolve_tool_file_ids(
        ["中国能源建设股份有限公司2026年第三季度报告.pdf"],
        query="诺安成长混合型证券投资基金",
    ) == [latest]


def test_span_quarter_keeps_both_years_and_pulls_the_missing_one(monkeypatch):
    """对比两年三季度时，只点到 2026 也要把同一家的 2025 三季报补上。"""
    y2026 = uuid.uuid4()
    y2025 = uuid.uuid4()
    annual = uuid.uuid4()
    other = uuid.uuid4()
    rows = [
        _ready(
            y2026,
            "深圳中科飞测科技股份有限公司2026年第三季度报告.pdf",
            "深圳中科飞测科技股份有限公司",
            ["深圳中科飞测科技"],
            "2026Q3",
        ),
        _ready(
            y2025,
            "深圳中科飞测科技股份有限公司2025年第三季度报告.pdf",
            "深圳中科飞测科技股份有限公司",
            ["深圳中科飞测科技"],
            "2025Q3",
        ),
        _ready(
            annual,
            "深圳中科飞测科技股份有限公司2025年年度报告.pdf",
            "深圳中科飞测科技股份有限公司",
            ["深圳中科飞测科技"],
            "2025",
        ),
        _ready(other, "中国银行2026年第三季度报告.pdf", "中国银行股份有限公司", [], "2026Q3"),
    ]
    monkeypatch.setattr(retrieval.store, "list_kb_files", lambda **k: rows)
    both = resolve_tool_file_ids(
        ["深圳中科飞测科技股份有限公司2026年第三季度报告"],
        query="2025年和2026年三季度股东持股对比",
    )
    assert set(both) == {y2025, y2026}
    # 简称不在别名表里，也要圈到两年的三季报，不把年报和别的公司带上
    named = resolve_tool_file_ids(None, query="中科飞测2025年和2026年三季度持股对比")
    assert set(named) == {y2025, y2026}


def test_capability_block_lists_doc_types_not_file_inventory(monkeypatch):
    """提示词只声明文档类型的实体和关系，不注入公司名单和文件清单。"""
    monkeypatch.setattr(
        loop_mod.store,
        "list_kb_files",
        lambda **k: [
            {
                "file_id": "a",
                "filename": "中国能源建设2026三季报.pdf",
                "status": "ready",
                "description": "三季报",
                "doc_type": "quarterly_report",
                "subject": "中国能源建设股份有限公司",
                "period": "2026Q3",
                "subject_aliases": ["中国能建"],
            },
            {
                "file_id": "b",
                "filename": "顺丰2026三季报.pdf",
                "status": "ready",
                "description": "",
                "doc_type": "quarterly_report",
                "subject": "顺丰控股股份有限公司",
                "period": "2026Q3",
                "subject_aliases": ["顺丰"],
            },
        ],
    )
    block = loop_mod._kb_capability_block()
    assert "【知识库能力】" in block
    assert "上市公司季报" in block
    assert "实体范围=" in block
    assert "关系=" in block
    assert "中国能源建设股份有限公司" not in block
    assert "中国能源建设2026三季报.pdf" not in block
    assert "库内已有报告期：2026Q3（2 份）" in block
    assert "【知识库主体】" not in block
    assert "【知识库可用文件】" not in block
