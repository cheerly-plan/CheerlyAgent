"""subject 作为查询入口。

files 参数现接受主体名（含简称/别名），底层按 subject 元数据自动路由到对应文件，
而不必先喂一份文件清单再让模型挑文件名。同时验证：编造的主体名应被识别为
「限定但空」（返回 []），供工具层给出拒绝提示，而不是静默退回全库。
"""

import uuid

from ai_agent.agents import loop as loop_mod
from ai_agent.kb import retrieval
from ai_agent.kb.scope import resolve_tool_file_ids


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
    # 无 subject 时，文件名路径仍能命中（从文件名抽主体）
    assert resolve_tool_file_ids(["中国能建"]) == [a]


def test_kb_files_block_injects_subject_list(monkeypatch):
    """主体清单作为新的查询入口被注入；无 subject 的文件不进主体清单。"""
    monkeypatch.setattr(
        loop_mod.store,
        "list_kb_files",
        lambda **k: [
            {
                "file_id": "a",
                "filename": "中国能源建设2026三季报.pdf",
                "status": "ready",
                "description": "",
                "subject": "中国能源建设股份有限公司",
                "period": "2026Q3",
                "subject_aliases": ["中国能建"],
            },
            {
                "file_id": "b",
                "filename": "顺丰2026三季报.pdf",
                "status": "ready",
                "description": "",
                "subject": "顺丰控股股份有限公司",
                "period": "2026Q3",
                "subject_aliases": ["顺丰"],
            },
        ],
    )
    block = loop_mod._kb_files_block()
    assert "【知识库主体】" in block
    assert "中国能源建设股份有限公司" in block
    assert "顺丰控股股份有限公司" in block
    # 小库：文件清单仍全列，且含原【知识库可用文件】标题（菜单保留）
    assert "【知识库可用文件】" in block
    assert "中国能源建设2026三季报.pdf" in block


def test_kb_files_block_large_library_folds_file_list(monkeypatch):
    """文件很多时，文件清单折叠为摘要，但主体清单仍注入（避免撑爆上下文）。"""
    files = [
        {
            "file_id": str(i),
            "filename": f"公司{i}报告.pdf",
            "status": "ready",
            "description": "",
            "subject": f"公司{i}股份有限公司",
            "period": "2026Q3",
            "subject_aliases": [],
        }
        for i in range(40)
    ]
    monkeypatch.setattr(loop_mod.store, "list_kb_files", lambda **k: files)
    block = loop_mod._kb_files_block()
    assert "【知识库主体】" in block
    # 大库：文件清单折叠，不再逐条列出全部文件名
    assert "公司39报告.pdf" not in block
    assert "共 40 份" in block
    assert "kb_list_files" in block
