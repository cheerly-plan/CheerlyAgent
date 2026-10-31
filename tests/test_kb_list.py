"""kb_list_files 工具：模型按需查询知识库文件清单（替代原目录注入）。"""

from ai_agent.tools import kb_list
from ai_agent.tools.kb_list import _kb_list_files


def test_kb_list_filters_to_ready_and_shows_description(monkeypatch):
    monkeypatch.setattr(
        kb_list.store,
        "list_kb_files",
        lambda q="", tag="": [
            {"filename": "a.pdf", "status": "ready", "description": "年报财务数据"},
            {"filename": "b.pdf", "status": "parsing", "description": ""},
            {"filename": "c.pdf", "status": "ready", "description": ""},
        ],
    )
    out = _kb_list_files()
    assert "知识库当前已就绪的文件" in out
    assert "a.pdf：年报财务数据" in out  # 有描述则带描述
    assert "c.pdf" in out  # 无描述也列出
    assert "b.pdf" not in out  # 非 ready 不出现


def test_kb_list_empty(monkeypatch):
    monkeypatch.setattr(kb_list.store, "list_kb_files", lambda q="", tag="": [])
    assert "没有处理完成" in _kb_list_files()


def test_kb_list_q_filters_filename(monkeypatch):
    def fake(q="", tag=""):
        files = [
            {"filename": "中国银行2024年报.pdf", "status": "ready", "description": "财务"},
            {"filename": "考勤制度.docx", "status": "ready", "description": ""},
        ]
        if q:
            files = [f for f in files if q in f["filename"]]
        return files

    monkeypatch.setattr(kb_list.store, "list_kb_files", fake)
    out = _kb_list_files("银行")
    assert "中国银行2024年报.pdf" in out
    assert "考勤制度.docx" not in out
