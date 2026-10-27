"""知识库 P1：权限守卫、检索工具注册与切片页码溯源。"""

import inspect
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

from ai_agent.api.dependencies import get_current_user, require_admin
from ai_agent.api.routers.auth import _to_auth_response, me
from ai_agent.api.routers.kb import (
    _clean_tags,
    delete_file,
    download_file,
    edit_chunk,
    init_upload,
    list_chunks,
    list_files,
    set_file_tags,
)
from ai_agent.api.routers.kg import build_graphs, entity_chunks, file_graph, global_graph, rebuild_graph, search_graph
from ai_agent.api.routers.user import user_info
from ai_agent.core.config import settings
from ai_agent.kb import vision
from ai_agent.kb.chunking import chunk_segments, chunk_text
from ai_agent.kb.parsing import Segment, parse_segments, supported
from ai_agent.kb.pipeline import KbPipeline
from ai_agent.memory.db import UserRecord
from ai_agent.tools.kb_search import _format_hit
from ai_agent.tools.registry import TOOLS, tool_label, tool_schemas


def _user(username: str) -> UserRecord:
    return UserRecord(user_id="uid-1", username=username, display_name=username)


def test_require_admin_allows_whitelisted(monkeypatch):
    monkeypatch.setattr(settings, "admin_usernames", "alice, bob")
    assert require_admin(_user("bob")).username == "bob"


def test_require_admin_rejects_others(monkeypatch):
    monkeypatch.setattr(settings, "admin_usernames", "alice")
    with pytest.raises(HTTPException) as exc:
        require_admin(_user("mike"))
    assert exc.value.status_code == 403


def test_require_admin_empty_whitelist_denies_everyone(monkeypatch):
    # 空配置 = 没有管理员，写操作全 403（只读浏览不受影响）
    monkeypatch.setattr(settings, "admin_usernames", "")
    with pytest.raises(HTTPException):
        require_admin(_user("alice"))


def test_user_info_reports_admin_flag(monkeypatch):
    monkeypatch.setattr(settings, "admin_usernames", "alice")
    assert user_info(_user("alice")).is_admin is True
    assert user_info(_user("mike")).is_admin is False


def test_login_response_reports_admin_flag(monkeypatch):
    # 登录响应直接带 is_admin，前端才能立即决定是否渲染知识库入口
    monkeypatch.setattr(settings, "admin_usernames", "alice")
    monkeypatch.setattr(
        "ai_agent.api.routers.auth.store.issue_token",
        lambda _uid: ("tok", datetime.now(UTC)),
    )
    assert _to_auth_response(_user("alice")).user.is_admin is True
    assert _to_auth_response(_user("mike")).user.is_admin is False


def test_me_reports_admin_flag(monkeypatch):
    # 登录响应、/v1/auth/me、/user/info 三个入口必须同构带上 is_admin，
    # 前端换任何一个来源取用户信息都不能丢知识库入口
    monkeypatch.setattr(settings, "admin_usernames", "alice")
    assert me(_user("alice")).is_admin is True
    assert me(_user("mike")).is_admin is False


def test_kb_reads_open_to_all_users():
    """读接口（列表/切片/下载/看图/图谱检索）对所有登录用户开放，普通用户也能查。"""
    for func, param in [
        (list_files, "_user"),
        (list_chunks, "_user"),
        (download_file, "_user"),
        (global_graph, "_user"),
        (file_graph, "_user"),
        (search_graph, "_user"),
        (entity_chunks, "_user"),
    ]:
        dep = inspect.signature(func).parameters[param].default
        assert dep.dependency is get_current_user, func.__name__


def test_kb_writes_require_admin():
    """写操作（上传建档/删除/改标签/切片修订/建图/重建）仍然卡在管理员身后。"""
    for func, param in [
        (init_upload, "user"),
        (delete_file, "_user"),
        (set_file_tags, "_user"),
        (edit_chunk, "user"),
        (build_graphs, "user"),
        (rebuild_graph, "user"),
    ]:
        dep = inspect.signature(func).parameters[param].default
        assert dep.dependency is require_admin, func.__name__


def test_kb_chunk_edit_requires_admin():
    # 切片修订会改写检索库内容，必须卡在管理员身后
    dep = inspect.signature(edit_chunk).parameters["user"].default
    assert dep.dependency is require_admin


def test_kg_build_requires_admin():
    # 批量建图触发 LLM 抽取，同样只对管理员开放
    dep = inspect.signature(build_graphs).parameters["user"].default
    assert dep.dependency is require_admin


def test_pipeline_does_not_autobuild_graph():
    # 上传后不再自动建图：图谱由管理员核对切片后选文件显式触发，
    # 防止「建完又改切片 → 整份重抽」的浪费
    source = inspect.getsource(KbPipeline._process)
    assert "build_file_graph" not in source
    assert "_build_graph" not in source


def test_kb_search_registered():
    names = [item["function"]["name"] for item in tool_schemas()]
    assert "kb_search" in names
    assert "kb_search" in [tool.name for tool in TOOLS]
    assert tool_label("kb_search") == "知识库检索"


def test_format_hit_uses_page_for_pdf():
    assert _format_hit(
        {"filename": "手册.pdf", "page": 3, "seq": 7, "text": "正文"}
    ) == "[手册.pdf · 第3页]\n正文"


def test_format_hit_falls_back_to_seq_without_page():
    assert _format_hit(
        {"filename": "笔记.md", "page": None, "seq": 0, "text": "正文"}
    ) == "[笔记.md · 第1段]\n正文"


def test_chunk_segments_keeps_page_of_each_chunk():
    segments = [
        Segment("第一页内容。" * 30, 1),
        Segment("第二页内容。", 2),
        Segment("无页文档"),
    ]
    chunks = chunk_segments(segments, max_chars=100, overlap=10)
    # 第 1 页长文拆成多片，页码全部保留为 1；页面不跨页合并
    assert [c.page for c in chunks][:2] == [1, 1]
    assert chunks[-2].page == 2
    assert chunks[-1].page is None
    assert all(c.text for c in chunks)


def test_chunk_text_short_input_untouched():
    assert chunk_text("你好世界", max_chars=500, overlap=50) == ["你好世界"]


def test_chunk_text_blank_returns_empty():
    assert chunk_text("   \n\n ") == []


# ——— 表格与图表：内容类型化 ———


def test_docx_table_is_extracted_as_markdown():
    # 之前的实现只遍历 paragraphs，Word 里的数据表整表丢失
    from docx import Document

    doc = Document()
    doc.add_paragraph("2024 年营收同比增长 18%。")
    table = doc.add_table(rows=3, cols=3)
    rows = [
        ["年份", "营收(亿)", "增速"],
        ["2023", "120", "12%"],
        ["2024", "142", "18%"],
    ]
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    path = Path(tempfile.mkdtemp()) / "年报.docx"
    doc.save(path)

    segments = parse_segments(path)
    assert any(seg.kind == "table" for seg in segments)
    table_text = next(seg.text for seg in segments if seg.kind == "table")
    assert "| 年份 | 营收(亿) | 增速 |" in table_text
    assert "| 2024 | 142 | 18% |" in table_text


def test_xlsx_sheet_becomes_table_segment():
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "营收"
    sheet.append(["年份", "营收(亿)"])
    sheet.append([2023, 120])
    sheet.append([2024, 142.0])
    path = Path(tempfile.mkdtemp()) / "数据.xlsx"
    workbook.save(path)

    segments = parse_segments(path)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.kind == "table"
    assert seg.page == 1  # sheet 序号，展示为「表1」
    assert "工作表：营收" in seg.text
    assert "| 2024 | 142 |" in seg.text  # 142.0 归一成 142


def test_xlsx_supported_suffix():
    assert supported(".xlsx")
    assert supported(".XLSM")
    assert not supported(".xls")


def test_chunk_table_kept_whole_when_short():
    text = "| 年份 | 营收 |\n|---|---|\n| 2024 | 142 |"
    chunks = chunk_segments([Segment(text, 1, "table")], max_chars=500)
    assert len(chunks) == 1
    assert chunks[0].kind == "table"
    assert chunks[0].page == 1


def test_chunk_long_table_repeats_header():
    rows = "\n".join(f"| {year} | {year * 10} |" for year in range(2000, 2060))
    text = "| 年份 | 营收 |\n|---|---|\n" + rows
    chunks = chunk_segments([Segment(text, 1, "table")], max_chars=200)
    assert len(chunks) > 1
    # 每片都带表头，否则行列对应关系丢失
    assert all(c.text.startswith("| 年份 | 营收 |") for c in chunks)
    assert all(c.kind == "table" for c in chunks)


def test_format_hit_labels_content_kind():
    base = {"filename": "报告.pdf", "seq": 0, "text": "内容"}
    assert _format_hit({**base, "kind": "text", "page": 3}) == "[报告.pdf · 第3页]\n内容"
    assert _format_hit({**base, "kind": "table", "page": 2}) == "[报告.pdf · 表2]\n内容"
    figure = _format_hit({**base, "kind": "figure", "page": 5})
    assert "第5页图表" in figure
    assert "图像识别" in figure


def test_vision_disabled_by_default(monkeypatch):
    # 未配置模型即关闭：默认不产生任何视觉调用费用
    monkeypatch.setattr(settings, "kb_vision_model", "")
    assert vision.enabled() is False
    assert vision.describe_image(b"fake-png") is None


# ——— 管理能力：标签 / 内容去重 / 下载 ———


def test_clean_tags_strips_dedupes_and_caps():
    assert _clean_tags(["  考勤 ", "", "考勤", "制" * 30]) == ["考勤", "制" * 24]
    assert _clean_tags([f"t{i}" for i in range(20)]) == [f"t{i}" for i in range(8)]
    assert _clean_tags([]) == []


def _prepare_complete(monkeypatch):
    """complete_upload 的公共桩：建档记录 + 分片落盘 + store/pipeline 打桩。

    临时目录放项目内（.tmp-kb-test）：系统 tmp 目录在沙箱里写不进去。
    """
    import shutil
    import uuid as _uuid

    from ai_agent.api.routers import kb as kb_router

    base = Path(__file__).resolve().parent.parent / ".tmp-kb-test"
    base.mkdir(exist_ok=True)
    upload_root = base / _uuid.uuid4().hex
    record = {
        "id": _uuid.uuid4(),
        "filename": "a.pdf",
        "size_bytes": 4,
        "status": "uploaded",
    }
    monkeypatch.setattr(kb_router, "_get_file", lambda fid: record)
    monkeypatch.setattr(kb_router.settings, "kb_upload_dir", upload_root)
    part_dir = upload_root / str(record["id"]) / "parts"
    part_dir.mkdir(parents=True)
    (part_dir / "000000.part").write_bytes(b"ABCD")
    # 测试收尾清理整个临时根目录（record 目录已由被测代码自己删）
    monkeypatch.finalize(lambda: shutil.rmtree(base, ignore_errors=True))
    return kb_router, record


def test_check_duplicate_rejects_same_content(monkeypatch):
    """同内容文件拒绝入库：409 + 删档案，重复检查在解析管线之前。"""
    import uuid as _uuid

    from ai_agent.api.routers import kb as kb_router

    fid = _uuid.uuid4()
    monkeypatch.setattr(
        kb_router.store,
        "find_kb_file_by_hash",
        lambda h, *, exclude_file_id=None: {
            "file_id": _uuid.uuid4(),
            "filename": "已有.pdf",
        },
    )
    removed: list = []
    monkeypatch.setattr(kb_router.store, "delete_kb_file", removed.append)
    monkeypatch.setattr(kb_router.shutil, "rmtree", lambda *a, **k: None)

    with pytest.raises(HTTPException) as exc:
        kb_router._check_duplicate("a" * 64, fid, "a.pdf", "alice")

    assert exc.value.status_code == 409
    assert "已有.pdf" in exc.value.detail
    assert removed == [fid]


def test_check_duplicate_passes_new_content(monkeypatch):
    import uuid as _uuid

    from ai_agent.api.routers import kb as kb_router

    monkeypatch.setattr(
        kb_router.store, "find_kb_file_by_hash", lambda h, *, exclude_file_id=None: None
    )
    deleted: list = []
    monkeypatch.setattr(kb_router.store, "delete_kb_file", deleted.append)

    # 不抛异常即通过
    kb_router._check_duplicate("a" * 64, _uuid.uuid4(), "a.pdf", "alice")
    assert deleted == []

