"""kb_list_files 工具：模型按需查询知识库文件清单（替代原目录注入）。"""

from ai_agent.tools.kb import kb_list
from ai_agent.tools.kb.kb_list import _kb_list_files


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
    assert "a.pdf：年报财务数据" in out  # 没写报告期时，描述仍紧跟文件名
    assert "c.pdf" in out  # 无描述也列出
    assert "b.pdf" not in out  # 非 ready 不出现


def test_kb_list_shows_period_before_description(monkeypatch):
    monkeypatch.setattr(
        kb_list.store,
        "list_kb_files",
        lambda q="", tag="": [
            {
                "filename": "中科飞测2025年第三季度报告.pdf",
                "status": "ready",
                "period": "2025Q3",
                "description": "股东与财务",
            }
        ],
    )
    out = _kb_list_files()
    assert "中科飞测2025年第三季度报告.pdf（报告期 2025Q3）：股东与财务" in out


def test_kb_list_empty(monkeypatch):
    monkeypatch.setattr(kb_list.store, "list_kb_files", lambda q="", tag="": [])
    assert "没有处理完成" in _kb_list_files()


def test_kb_list_q_filters_filename(monkeypatch):
    monkeypatch.setattr(
        kb_list.store,
        "list_kb_files",
        lambda q="", tag="": [
            {"filename": "中国银行2024年报.pdf", "status": "ready", "description": "财务"},
            {"filename": "考勤制度.docx", "status": "ready", "description": ""},
        ],
    )
    out = _kb_list_files("银行")
    assert "中国银行2024年报.pdf" in out
    assert "考勤制度.docx" not in out


def test_kb_list_q_matches_subject_and_description(monkeypatch):
    monkeypatch.setattr(
        kb_list.store,
        "list_kb_files",
        lambda q="", tag="": [
            {
                "filename": "2026三季报.pdf",
                "status": "ready",
                "subject": "中国能源建设股份有限公司",
                "subject_aliases": ["中国能建"],
                "description": "",
            },
            {
                "filename": "notes.txt",
                "status": "ready",
                "subject": "",
                "subject_aliases": [],
                "description": "考勤与休假制度",
            },
            {
                "filename": "其他.pdf",
                "status": "ready",
                "subject": "顺丰控股",
                "description": "",
            },
        ],
    )
    by_alias = _kb_list_files("中国能建")
    assert "2026三季报.pdf" in by_alias
    assert "其他.pdf" not in by_alias
    by_desc = _kb_list_files("考勤")
    assert "notes.txt" in by_desc
    assert "2026三季报.pdf" not in by_desc


def test_kb_list_empty_query_over_limit_returns_count_only(monkeypatch):
    files = [
        {"filename": f"公司{i}报告.pdf", "status": "ready", "description": ""}
        for i in range(31)
    ]
    monkeypatch.setattr(kb_list.store, "list_kb_files", lambda q="", tag="": files)
    out = _kb_list_files()
    assert "31 份" in out
    assert "公司0报告.pdf" not in out
    assert "请用 q" in out


def test_kb_list_query_caps_at_limit(monkeypatch):
    files = [
        {
            "filename": f"银行{i}.pdf",
            "status": "ready",
            "subject": "银行股份",
            "description": "",
        }
        for i in range(40)
    ]
    monkeypatch.setattr(kb_list.store, "list_kb_files", lambda q="", tag="": files)
    out = _kb_list_files("银行")
    assert out.count("\n- ") == 30
    assert "还有 10 份未列出" in out


def test_kb_list_query_miss_is_not_empty_library(monkeypatch):
    monkeypatch.setattr(
        kb_list.store,
        "list_kb_files",
        lambda q="", tag="": [
            {"filename": "考勤制度.docx", "status": "ready", "description": ""},
        ],
    )
    out = _kb_list_files("银行")
    assert "没有处理完成" not in out
    assert "没有文件名、主体或描述匹配" in out
