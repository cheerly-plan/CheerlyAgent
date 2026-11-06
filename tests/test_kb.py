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
    rename_file,
    set_file_tags,
)
from ai_agent.api.routers.kg import build_graphs, entity_chunks, file_graph, global_graph, rebuild_graph, search_graph
from ai_agent.api.routers.user import user_info
from ai_agent.core.config import settings
from ai_agent.kb.ingest import vision
from ai_agent.kb.ingest.chunking import chunk_segments, chunk_text
from ai_agent.kb.ingest.parsing import Segment, parse_segments, supported
from ai_agent.kb.pipeline import KbPipeline
from ai_agent.storage.db import UserRecord
from ai_agent.tools.kb.kb_search import _format_hit
from ai_agent.tools.registry import TOOLS, tool_schemas


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


def test_html_table_stays_one_chunk():
    rows = "".join(
        f"<tr><td>项目{i}</td><td>{i * 1000}.00</td></tr>" for i in range(40)
    )
    table = f"<table><tr><td>项目</td><td>本期</td></tr>{rows}</table>"
    text = f"单位：元\n\n{table}\n\n以上为合并口径。"
    chunks = chunk_text(text, max_chars=180, overlap=50)
    assert chunks == ["单位：元", table, "以上为合并口径。"]


def test_html_table_chunk_kind_is_table():
    table = "<table><tr><td>营业收入</td><td>1</td></tr></table>"
    text = f"前文说明。\n\n{table}\n\n后文。"
    chunks = chunk_segments([Segment(text, 4)], max_chars=500, overlap=50)
    assert [(c.kind, c.text) for c in chunks] == [
        ("text", "前文说明。"),
        ("table", table),
        ("text", "后文。"),
    ]
    assert chunks[1].page == 4


def test_cross_page_html_tables_merge_and_drop_repeated_header():
    header = "<tr><td>项目</td><td>本期金额</td></tr>"
    page4 = (
        "单位：元\n\n"
        f"<table>{header}<tr><td>营业收入</td><td>1</td></tr></table>"
    )
    page5 = (
        f"<table>{header}<tr><td>净利润</td><td>2</td></tr></table>\n\n"
        "以上为合并口径。"
    )
    chunks = chunk_segments(
        [Segment(page4, 4), Segment(page5, 5)],
        max_chars=500,
        overlap=0,
    )
    tables = [c for c in chunks if c.kind == "table"]
    assert [c.kind for c in chunks] == ["text", "table", "table", "text"]
    assert all(c.page == 4 and c.page_end == 5 for c in tables)
    assert all(c.text.count(header) == 1 for c in tables)
    assert "营业收入" in tables[0].text
    assert "净利润" in tables[1].text
    assert "营业收入" not in tables[1].text


def test_cross_page_tables_stay_apart_when_columns_differ():
    page4 = "<table><tr><td>项目</td><td>本期</td></tr><tr><td>收入</td><td>1</td></tr></table>"
    page5 = "<table><tr><td>甲</td><td>乙</td><td>丙</td></tr></table>"
    chunks = chunk_segments(
        [Segment(page4, 4), Segment(page5, 5)],
        max_chars=500,
        overlap=0,
    )
    assert [(c.page, c.page_end) for c in chunks] == [(4, 4), (5, 5)]


def test_cross_page_html_table_copies_header_onto_continuation():
    header = "<tr><td>项目</td><td>2026年前三季度</td><td>2025年前三季度</td></tr>"
    page1 = (
        f"合并利润表\n单位：元\n<table>{header}"
        "<tr><td>营业收入</td><td>30,439.00</td><td>45,763.00</td></tr></table>"
    )
    page2 = (
        "<table><tr><td>资产减值损失</td><td>-507.00</td><td>3.00</td></tr></table>"
    )
    chunks = chunk_segments(
        [
            Segment(page1, 11, "table", "四、财务报表", scope="consolidated"),
            Segment(page2, 12, "table", "四、财务报表", scope="consolidated"),
        ],
        max_chars=500,
        overlap=0,
    )
    assert chunks[1].text.startswith("合并利润表\n单位：元\n<table>" + header)
    assert "资产减值损失" in chunks[1].text
    assert chunks[1].page == 12
    assert chunks[1].scope == "consolidated"


def test_cross_page_html_table_copies_header_onto_blank_item():
    """页首是金额为空的科目（其他债权投资），也要补上一页的列名。"""
    header = "<tr><td>项目</td><td>期末余额</td><td>期初余额</td></tr>"
    page5 = (
        f"<table>{header}"
        "<tr><td>发放贷款和垫款</td><td></td><td></td></tr>"
        "<tr><td>债权投资</td><td></td><td></td></tr></table>"
    )
    page6 = (
        "<table><tr><td>其他债权投资</td><td></td><td></td></tr>"
        "<tr><td>长期应收款</td><td></td><td>1,274,106,334.49</td></tr></table>"
    )
    chunks = chunk_segments(
        [Segment(page5, 5, "table"), Segment(page6, 6, "table")],
        max_chars=500,
        overlap=0,
    )
    page6 = [c for c in chunks if c.page == 6]
    assert page6
    assert all(c.text.startswith("<table>" + header) for c in page6)
    assert any("其他债权投资" in c.text for c in page6)


def test_header_keeps_moving_across_later_pages():
    header = "<tr><td>项目</td><td>期末余额</td><td>期初余额</td></tr>"
    first = (
        f"<table>{header}"
        "<tr><td>货币资金</td><td>100</td><td>90</td></tr></table>"
    )
    later = [
        "<table><tr><td>长期股权投资</td><td>20</td><td>18</td></tr></table>",
        "<table><tr><td>应交税费</td><td>6</td><td>8</td></tr></table>",
        "<table><tr><td>盈余公积</td><td>4</td><td>4</td></tr></table>",
    ]
    chunks = chunk_segments(
        [Segment(first, 5, "table"), *[Segment(text, 6 + i, "table") for i, text in enumerate(later)]],
        max_chars=500,
        overlap=0,
    )
    assert [c.page for c in chunks] == [5, 6, 7, 8]
    assert all(header in c.text for c in chunks)
    assert chunks[3].text.index("项目") < chunks[3].text.index("盈余公积")


def test_new_statement_header_replaces_the_carried_one():
    old = "<tr><td>项目</td><td>期末余额</td><td>期初余额</td></tr>"
    new = "<tr><td>项目</td><td>本期发生额</td><td>上期发生额</td></tr>"
    chunks = chunk_segments(
        [
            Segment(
                f"<table>{old}<tr><td>货币资金</td><td>100</td><td>90</td></tr></table>",
                8,
                "table",
            ),
            Segment(
                f"合并利润表\n<table>{new}"
                "<tr><td>营业收入</td><td>101</td><td>114</td></tr></table>",
                9,
                "table",
            ),
            Segment(
                "<table><tr><td>营业利润</td><td>-510</td><td>2,398</td></tr></table>",
                10,
                "table",
            ),
        ],
        max_chars=500,
        overlap=0,
    )
    assert old not in chunks[2].text
    assert chunks[2].text.startswith("合并利润表\n<table>" + new)
    assert "营业利润" in chunks[2].text


def test_cross_page_html_table_copies_the_latest_header():
    # 页中换了列名，续表沿用最后一组，不把资产负债表的列名安到财务费用上
    page1 = (
        "<table>"
        "<tr><td>合并资产负债表项目</td><td>2026年9月30日</td><td>变动比例</td></tr>"
        "<tr><td>货币资金</td><td>35,914.00</td><td>-33.52%</td></tr>"
        "<tr><td>合并利润表项目</td><td>2026年1-9月</td><td>变动比例</td></tr>"
        "<tr><td>税金及附加</td><td>1,420.00</td><td>-45.20%</td></tr>"
        "</table>"
    )
    page2 = "<table><tr><td>财务费用</td><td>647.00</td><td>131.33%</td></tr></table>"
    chunks = chunk_segments(
        [Segment(page1, 2, "table"), Segment(page2, 3, "table")],
        max_chars=500,
        overlap=0,
    )
    continued = next(c for c in chunks if c.page == 3)
    assert "合并利润表项目" in continued.text
    assert "合并资产负债表项目" not in continued.text
    assert continued.text.index("合并利润表项目") < continued.text.index("财务费用")
    tax = next(c for c in chunks if "税金及附加" in c.text)
    assert "合并资产负债表项目" not in tax.text


def test_clipped_header_is_replaced_by_previous_page_header():
    """页顶第一列空着的残头不是新表头，换成上一页的列名。"""
    header = (
        "<tr><td>股东名称</td><td>股东性质</td><td>持股数量</td>"
        "<td>持股比例(%)</td></tr>"
    )
    page3 = (
        f"<table>{header}"
        "<tr><td colspan=\"4\">前10名股东持股情况</td></tr></table>"
    )
    remnant = (
        "<tr><td></td><td></td><td>(%)</td><td>份数量</td></tr>"
    )
    page4 = (
        f"<table>{remnant}"
        "<tr><td>苏州翌流明光电科技有限公司</td><td>境内非国有法人</td>"
        "<td>37,800,963</td><td>11.75</td></tr>"
        "<tr><td>深圳市创新投资集团有限公司</td><td>国有法人</td>"
        "<td>8,929,927</td><td>2.78</td></tr></table>"
    )
    chunks = chunk_segments(
        [Segment(page3, 3, "table"), Segment(page4, 4, "table")],
        max_chars=500,
        overlap=0,
    )
    holders = [c for c in chunks if c.page == 4]
    assert holders
    assert all("股东名称" in c.text and "持股比例(%)" in c.text for c in holders)
    assert all("份数量" not in c.text for c in holders)
    first = next(c for c in holders if "翌流明" in c.text)
    assert "11.75" in first.text
    assert first.text.index("股东名称") < first.text.index("翌流明")


def test_clipped_header_is_dropped_when_previous_page_has_no_header():
    """上一页没有可继承的列名时，残头也不再当成表头重复。"""
    remnant = "<tr><td></td><td></td><td>(%)</td><td>份数量</td></tr>"
    table = (
        f"<table>{remnant}"
        "<tr><td>苏州翌流明光电科技有限公司</td><td>境内非国有法人</td>"
        "<td>37,800,963</td><td>11.75</td></tr>"
        "<tr><td>深圳市创新投资集团有限公司</td><td>国有法人</td>"
        "<td>8,929,927</td><td>2.78</td></tr></table>"
    )
    chunks = chunk_segments(
        [Segment(table, 4, "table")],
        max_chars=80,
        overlap=0,
    )
    assert chunks
    assert all("份数量" not in c.text for c in chunks)
    assert any("翌流明" in c.text and "11.75" in c.text for c in chunks)


def test_cross_page_html_table_does_not_copy_header_after_prose():
    page1 = (
        "<table><tr><td>项目</td><td>本期</td></tr>"
        "<tr><td>营业收入</td><td>100</td></tr></table>"
    )
    page2 = "<table><tr><td>净利润</td><td>18</td></tr></table>"
    chunks = chunk_segments(
        [
            Segment(page1, 2, "table"),
            Segment("标的公司产权清晰，不存在抵押。", 2, "text"),
            Segment(page2, 3, "table"),
        ],
        max_chars=500,
        overlap=0,
    )
    assert "项目" not in chunks[-1].text
    assert chunks[-1].text.startswith("<table><tr><td>净利润</td>")


def test_captioned_html_table_does_not_count_caption_as_table():
    header = "<tr><td>项目</td><td>本期</td><td>上期</td></tr>"
    rows = "".join(
        "<tr><td>科目"
        f"{i:02d}名称比较长用来撑开这一行</td><td>{i},234,567.89</td>"
        f"<td>{i},876,543.21</td></tr>"
        for i in range(25)
    )
    table = f"<table>{header}{rows}</table>"
    chunks = chunk_segments(
        [
            Segment("报表前的说明文字。", 8, "text", "第十节 财务报告"),
            Segment(
                f"母公司利润表\n{table}",
                8,
                "table",
                "第十节 财务报告",
                scope="parent",
            ),
        ],
        max_chars=500,
        overlap=0,
    )
    texts = [c for c in chunks if c.kind == "text"]
    tables = [c for c in chunks if c.kind == "table"]
    assert [c.text for c in texts] == ["报表前的说明文字。"]
    assert len(tables) > 1
    assert all(c.text.startswith("母公司利润表\n<table>") for c in tables)
    assert all(c.scope == "parent" for c in tables)


def test_long_html_table_splits_by_row_and_repeats_header():
    from ai_agent.kb.ingest.chunking import TABLE_GROUP_CHARS, html_table_plain

    header = "<tr><td>项目</td><td>本期</td><td>上期</td></tr>"
    rows = "".join(
        "<tr><td>科目"
        f"{i:02d}名称比较长用来撑开这一行</td><td>{i},234,567.89</td>"
        f"<td>{i},876,543.21</td></tr>"
        for i in range(25)
    )
    table = f"<table>{header}{rows}</table>"
    chunks = chunk_segments([Segment(table, 8)], max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(c.kind == "table" and c.page == 8 for c in chunks)
    assert all(header in c.text for c in chunks)
    joined = "".join(c.text for c in chunks)
    for i in range(25):
        assert f"科目{i:02d}" in joined
    for chunk in chunks:
        plain = html_table_plain(chunk.text)
        lines = [line for line in plain.splitlines() if line.strip()]
        if len(lines) > 2:
            assert len(plain) <= TABLE_GROUP_CHARS


def test_html_table_section_row_starts_a_new_group():
    header = "<tr><td>项目</td><td>期末</td></tr>"
    body = "".join(
        "<tr><td>流动项"
        f"{i:02d}的名字比较长用来把这一组撑过上限</td><td>{i},000,000.00</td></tr>"
        for i in range(24)
    )
    section = "<tr><td colspan=\"2\">非流动资产：</td></tr>"
    tail = (
        "<tr><td>投资性房地产</td><td>2,578,854.37</td></tr>"
        "<tr><td>固定资产</td><td>9,000.00</td></tr>"
    )
    table = f"<table>{header}{body}{section}{tail}</table>"
    chunks = chunk_segments([Segment(table, 8)], max_chars=500, overlap=0)
    assert len(chunks) > 1
    hosts = [c for c in chunks if "非流动资产" in c.text]
    assert len(hosts) == 2
    assert "投资性房地产" in hosts[0].text
    assert "固定资产" in hosts[1].text
    assert all("流动项00" not in c.text for c in hosts)


def test_html_income_statement_account_is_its_own_chunk():
    """营业利润单独成片，表题和列名跟着走；加减行留在这一片里。"""
    header = "<tr><td>项目</td><td>本期发生额</td><td>上期发生额</td></tr>"
    rows = "".join(
        [
            "<tr><td>一、营业总收入</td><td>101</td><td>114</td></tr>",
            "<tr><td>三、营业利润(亏损以“-”号填列)</td><td>-510</td><td>2,398</td></tr>",
            "<tr><td>加:营业外收入</td><td>95</td><td>131</td></tr>",
            "<tr><td>四、利润总额(亏损总额以“-”号填列)</td><td>-451</td><td>2,499</td></tr>",
            "<tr><td>五、净利润(净亏损以“-”号填列)</td><td>-726</td><td>2,144</td></tr>",
        ]
    )
    text = f"## 2、合并年初到报告期末利润表\n单位：元\n<table>{header}{rows}</table>"
    chunks = chunk_segments([Segment(text, 9, "table")], max_chars=500, overlap=0)
    assert len(chunks) == 4
    assert all(
        c.text.startswith("## 2、合并年初到报告期末利润表\n单位：元\n<table>" + header)
        for c in chunks
    )
    profit = next(c for c in chunks if "营业利润" in c.text)
    assert "营业外收入" in profit.text
    assert "净利润" not in profit.text
    assert "营业总收入" not in profit.text


def test_html_table_one_long_row_stays_with_header():
    header = "<tr><td>项目</td><td>说明</td></tr>"
    long = "甲" * 900
    table = f"<table>{header}<tr><td>备注</td><td>{long}</td></tr></table>"
    chunks = chunk_segments([Segment(table, 1)], max_chars=500, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].kind == "table"
    assert long in chunks[0].text


def test_short_html_table_not_merged_into_prose():
    table = "<table><tr><td>营业收入</td><td>1</td></tr></table>"
    text = f"前文说明。\n\n{table}\n\n后文。"
    assert chunk_text(text, max_chars=500, overlap=50) == ["前文说明。", table, "后文。"]


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


def test_chunk_long_table_repeats_caption_too():
    """表题（「母公司利润表」）要跟着每一片走。

    否则长表切完之后，后半张表的切片只剩「项目 | 本期 | 上期 + 数字」，
    「母公司」三个字不在正文里，检索和实体抽取都看不出这张表属于谁。
    """
    rows = "\n".join(f"| 项目{year} | {year} |" for year in range(2000, 2060))
    text = "母公司利润表\n| 项目 | 本期 |\n|---|---|\n" + rows
    chunks = chunk_segments([Segment(text, 16, "table")], max_chars=200)
    assert len(chunks) > 1
    assert all(c.text.startswith("母公司利润表\n| 项目 | 本期 |") for c in chunks)


def test_chunk_short_table_with_caption_stays_one_piece():
    text = "母公司利润表\n| 项目 | 本期 |\n|---|---|\n| 营业收入 | 1 |"
    chunks = chunk_segments([Segment(text, 16, "table")], max_chars=500)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("母公司利润表")


def test_chunk_long_table_repeats_unit_meta():
    """单位和审计状态跟表题一样，切开后每一片都要带着。"""
    rows = "\n".join(f"| 项目{year} | {year} |" for year in range(2000, 2060))
    text = (
        "母公司利润表\n单位：千元 · 未经审计\n"
        "| 项目 | 本期 |\n|---|---|\n" + rows
    )
    chunks = chunk_segments([Segment(text, 16, "table")], max_chars=200)
    assert len(chunks) > 1
    assert all(
        c.text.startswith("母公司利润表\n单位：千元 · 未经审计\n| 项目 | 本期 |")
        for c in chunks
    )


def test_chunk_table_keeps_nested_financial_rows_together():
    """金融投资和三个计量子项是一组，不能按字数切到两片里。"""
    pad = "\n".join(f"| 填充科目{i} | {i}000 | {i}111 |" for i in range(8))
    group = "\n".join(
        [
            "| 金融投资 | 100 | 90 |",
            "| 其中：以公允价值计量且其变动计入当期损益的金融资产 | 10 | 9 |",
            "| 以公允价值计量且其变动计入其他综合收益的金融资产 | 20 | 19 |",
            "| 以摊余成本计量的金融资产 | 70 | 62 |",
        ]
    )
    text = (
        "中国银行资产表\n单位：百万元人民币\n"
        "| 项目 | 本期 | 上期 |\n|---|---|---|\n"
        f"{pad}\n{group}"
    )
    chunks = chunk_segments([Segment(text, 8, "table")], max_chars=220)
    assert len(chunks) > 1
    joined = "\n".join(c.text for c in chunks)
    assert "金融投资" in joined
    holders = [c for c in chunks if "金融投资" in c.text]
    assert len(holders) == 1
    body = holders[0].text
    assert "以公允价值计量且其变动计入当期损益的金融资产" in body
    assert "以公允价值计量且其变动计入其他综合收益的金融资产" in body
    assert "以摊余成本计量的金融资产" in body
    assert "单位：百万元人民币" in holders[0].text


def test_chunk_table_does_not_isolate_total_row():
    """资产总计不能单独成片，必须跟本表最后一组明细走。"""
    pad = "\n".join(f"| 明细科目{i} | {i}00 | {i}11 |" for i in range(12))
    text = (
        "合并资产负债表\n单位：千元\n"
        "| 项目 | 期末 | 期初 |\n|---|---|---|\n"
        f"{pad}\n| 资产总计 | 999 | 888 |"
    )
    chunks = chunk_segments([Segment(text, 7, "table")], max_chars=180)
    assert len(chunks) > 1
    total_chunks = [c for c in chunks if "资产总计" in c.text]
    assert len(total_chunks) == 1
    data_rows = [
        ln
        for ln in total_chunks[0].text.splitlines()
        if ln.startswith("| ") and "---" not in ln and "项目" not in ln
    ]
    assert any("明细科目" in ln for ln in data_rows)


def test_chunk_merged_table_keeps_page_span():
    text = "| 项目 | 本期 |\n|---|---|\n| 营业收入 | 1 |\n| 营业成本 | 2 |"
    chunks = chunk_segments(
        [Segment(text, 7, "table", page_end=8)], max_chars=500
    )
    assert len(chunks) == 1
    assert chunks[0].page == 7
    assert chunks[0].page_end == 8


def test_table_group_mode_section_aggregates_by_section():
    """section 模式按分组聚合：分组边界干净，且切片数少于 row 模式。"""
    header = "<tr><td>项目</td><td>期末</td><td>期初</td></tr>"
    flow = "".join(
        f"<tr><td>流动项{i:02d}比较长用来把这一组撑过上限</td><td>{i},000</td><td>{i},111</td></tr>"
        for i in range(24)
    )
    section = "<tr><td colspan=\"3\">非流动资产：</td></tr>"
    nonflow = "".join(
        f"<tr><td>非流动项{i:02d}比较长用来把这一组撑过上限</td><td>{i},000</td><td>{i},111</td></tr>"
        for i in range(24)
    )
    table = f"<table>{header}{flow}{section}{nonflow}</table>"
    row_chunks = chunk_segments([Segment(table, 8)], max_chars=500, overlap=0)
    sec_chunks = chunk_segments(
        [Segment(table, 8)], max_chars=500, overlap=0, table_group_mode="section"
    )
    assert len(sec_chunks) < len(row_chunks)
    hosts = [c for c in sec_chunks if "非流动资产" in c.text]
    assert hosts
    # 分组边界干净：非流动组的片里不该出现流动项（用 ">流动项" 锚，
    # 避开「非流动项」里的「流动项」子串误匹配）
    assert all(">流动项" not in c.text for c in hosts)
    assert all(header in c.text for c in sec_chunks)


def test_table_group_mode_section_slides_plain_table():
    """无分组标题的扁平长表：按窗口在科目边界断、每片重复表头，片数少于 row。"""
    header = "<tr><td>项目</td><td>本期</td><td>上期</td></tr>"
    rows = "".join(
        f"<tr><td>科目{i:02d}名称较长</td><td>{i},234,567.89</td><td>{i},876,543.21</td></tr>"
        for i in range(25)
    )
    table = f"<table>{header}{rows}</table>"
    row_chunks = chunk_segments([Segment(table, 8)], max_chars=500, overlap=0)
    sec_chunks = chunk_segments(
        [Segment(table, 8)], max_chars=500, overlap=0, table_group_mode="section"
    )
    assert len(sec_chunks) > 1
    assert len(sec_chunks) < len(row_chunks)
    assert all(header in c.text for c in sec_chunks)


def test_rowspan_reason_stays_with_following_rows():
    """变动原因表的合并格不能被「不适用」行切开，也不能丢掉最后一行。"""
    table = (
        "<table>"
        "<tr><td>项目名称</td><td>变动比例(%)</td><td>主要原因</td></tr>"
        "<tr><td>营业收入_本报告期</td><td>43.30</td>"
        '<td rowspan="2">营收增长</td></tr>'
        "<tr><td>营业收入_年初至报告期末</td><td>47.92</td></tr>"
        "<tr><td>利润总额_本报告期</td><td>-76.60</td>"
        '<td rowspan="3">利润波动</td></tr>'
        "<tr><td>利润总额_年初至报告期末</td><td>不适用</td></tr>"
        "<tr><td>归属于上市公司股东的净利润_本报告期</td><td>-77.33</td></tr>"
        "<tr><td>研发投入_年初至报告期末</td><td>31.32</td>"
        '<td rowspan="2">研发增长</td></tr>'
        "<tr><td>研发投入占营业收入的比例_本报告期</td><td>减少5.75个百分点</td></tr>"
        "</table>"
    )
    chunks = chunk_segments(
        [Segment(table, 3, "table")],
        max_chars=80,
        overlap=0,
        table_group_mode="section",
    )
    profit = next(c for c in chunks if "利润波动" in c.text)
    assert "利润总额_年初至报告期末" in profit.text
    assert "净利润_本报告期" in profit.text
    assert 'rowspan="3"' in profit.text
    revenue = next(c for c in chunks if "营收增长" in c.text)
    assert "营业收入_年初至报告期末" in revenue.text
    joined = "\n".join(c.text for c in chunks)
    assert "研发投入占营业收入的比例_本报告期" in joined
    assert "研发增长" in next(c for c in chunks if "研发投入_年初至报告期末" in c.text).text


def test_table_group_mode_section_keeps_subtotal_with_parent():
    """section 模式下合计/加减行仍跟所属科目走，不单独成片。"""
    header = "<tr><td>项目</td><td>本期</td><td>上期</td></tr>"
    rows = (
        "<tr><td>一、营业总收入</td><td>101</td><td>114</td></tr>"
        "<tr><td>三、营业利润</td><td>-510</td><td>2,398</td></tr>"
        "<tr><td>加:营业外收入</td><td>95</td><td>131</td></tr>"
        "<tr><td>四、利润总额</td><td>-451</td><td>2,499</td></tr>"
    )
    table = f"<table>{header}{rows}</table>"
    chunks = chunk_segments(
        [Segment(table, 9, "table")],
        max_chars=200,
        overlap=0,
        table_group_mode="section",
    )
    profit = [c for c in chunks if "营业利润" in c.text]
    assert len(profit) == 1
    assert "营业外收入" in profit[0].text


def test_format_hit_labels_content_kind():
    base = {"filename": "报告.pdf", "seq": 0, "text": "内容"}
    assert _format_hit({**base, "kind": "text", "page": 3}) == "[报告.pdf · 第3页]\n内容"
    assert _format_hit({**base, "kind": "table", "page": 2}) == "[报告.pdf · 第2页]\n内容"
    assert (
        _format_hit({**base, "filename": "财务.xlsx", "kind": "table", "page": 2})
        == "[财务.xlsx · 表2]\n内容"
    )
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


def _kb_workdir(request):
    """项目内临时目录。系统 tmp 在沙箱里写不进去。

    每个用例只占一个子目录，收尾只删这个子目录：整棵删掉会和并发用例抢同
    一路径。根目录已存在时不再调 mkdir(exist_ok=True)：当前沙箱的 shim
    直接替换了 Path.mkdir，exist_ok 会被无视、照样抛 EEXIST；先 exists()
    再建同样安全（用例串行跑，无并发竞争）。
    """
    import shutil
    import uuid as _uuid

    base = Path(__file__).resolve().parent.parent / ".tmp-kb-test"
    if not base.exists():
        base.mkdir()
    path = base / _uuid.uuid4().hex
    path.mkdir()
    request.addfinalizer(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


def _prepare_complete(request, monkeypatch):
    """complete_upload 的公共桩：建档记录 + 分片落盘 + store/pipeline 打桩。

    临时目录放项目内（.tmp-kb-test）：系统 tmp 目录在沙箱里写不进去。
    """
    import uuid as _uuid

    from ai_agent.api.routers import kb as kb_router

    upload_root = _kb_workdir(request)
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



# ---------- 改名 ----------


def _rename_record(request, monkeypatch, status="ready", filename="旧报告", mime=".pdf"):
    """rename_file 的公共桩：ready 记录 + 磁盘上放好旧名文件 + pipeline 打桩。

    文件名是纯名（不含后缀），类型由 mime 单独存——这是当前数据模型。
    """
    import uuid as _uuid

    from ai_agent.api.routers import kb as kb_router

    upload_root = _kb_workdir(request)
    fid = _uuid.uuid4()
    record = {
        "id": fid,
        "filename": filename,
        "mime": mime,
        "size_bytes": 4,
        "status": status,
    }
    monkeypatch.setattr(kb_router, "_parse_file_id", lambda _fid: fid)
    monkeypatch.setattr(kb_router, "_get_file", lambda _fid: record)
    monkeypatch.setattr(kb_router.settings, "kb_upload_dir", upload_root)
    renamed: list = []
    monkeypatch.setattr(
        kb_router.store,
        "rename_kb_file",
        lambda _fid, name: renamed.append(name) or True,
    )
    submitted: list = []
    monkeypatch.setattr(
        kb_router.kb_pipeline, "submit_reembed", lambda _fid: submitted.append(_fid)
    )
    return kb_router, record, renamed, submitted


def test_rename_strips_suffix(request, monkeypatch):
    """新名带后缀时剥掉：文件名存纯名，类型由 mime 单独管，后缀不再进 filename。"""
    from ai_agent.api.schemas import KbRenameRequest

    kb_router, record, renamed, submitted = _rename_record(request, monkeypatch)
    result = rename_file("x", KbRenameRequest(filename="新报告.docx"), _admin())
    assert result == {"ok": True, "filename": "新报告", "reembedding": True}
    assert renamed == ["新报告"]
    assert submitted == [record["id"]]


def test_rename_requires_ready(request, monkeypatch):
    from ai_agent.api.schemas import KbRenameRequest

    kb_router, record, renamed, submitted = _rename_record(
        request, monkeypatch, status="parsing"
    )
    with pytest.raises(HTTPException) as exc:
        rename_file("x", KbRenameRequest(filename="新报告.pdf"), _admin())
    assert exc.value.status_code == 409
    assert renamed == [] and submitted == []


def test_rename_updates_db_disk_and_reembeds(request, monkeypatch):
    """名变了：DB 存纯名 + 磁盘文件重命名（拼回后缀）+ 后台重嵌，三件事都发生。"""
    import uuid as _uuid

    from ai_agent.api.schemas import KbRenameRequest

    kb_router, record, renamed, submitted = _rename_record(request, monkeypatch)
    upload_dir = kb_router.Path(kb_router.settings.kb_upload_dir) / str(record["id"])
    upload_dir.mkdir(parents=True)
    (upload_dir / "旧报告.pdf").write_bytes(b"ABCD")

    result = rename_file(
        str(record["id"]), KbRenameRequest(filename="2026 新报告.pdf"), _admin()
    )

    assert result == {"ok": True, "filename": "2026 新报告", "reembedding": True}
    assert renamed == ["2026 新报告"]
    # 磁盘文件拼回 mime 后缀重命名
    assert (upload_dir / "2026 新报告.pdf").is_file()
    assert not (upload_dir / "旧报告.pdf").exists()
    assert submitted == [record["id"]]


def test_rename_same_name_skips_reembed(request, monkeypatch):
    """新名剥后缀后与现名相同：名字没变，不值得整份重嵌。"""
    from ai_agent.api.schemas import KbRenameRequest

    kb_router, record, renamed, submitted = _rename_record(
        request, monkeypatch, filename="报告", mime=".PDF"
    )
    result = rename_file("x", KbRenameRequest(filename="报告.pdf"), _admin())
    assert result["reembedding"] is False
    assert renamed == [] and submitted == []


def _admin():
    return UserRecord(
        user_id="u1", username="alice", display_name="Alice",
        created_at=datetime.now(UTC).isoformat(),
    )


# ---------- 并行向量化与并行视觉 ----------


def test_embed_texts_parallel_keeps_order(monkeypatch):
    """多批并发执行，但结果顺序与输入严格一致（写错位会毁掉全部检索）。"""
    from ai_agent.storage import embeddings

    def fake_embed(texts):
        return [[len(t), 0] for t in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed)
    texts = [f"t{i:02d}" for i in range(25)]  # 10/批 → 3 批
    out = embeddings.embed_texts_parallel(texts, concurrency=3)
    assert out == fake_embed(texts)


def test_embed_texts_parallel_single_batch_passthrough(monkeypatch):
    """不足两批直接走串行路径，不开线程池。"""
    from ai_agent.storage import embeddings

    called = []
    monkeypatch.setattr(
        embeddings, "embed_texts", lambda ts: called.append(list(ts)) or [[1]]
    )
    assert embeddings.embed_texts_parallel(["a"], concurrency=4) == [[1]]
    assert embeddings.embed_texts_parallel(["a", "b"], concurrency=0) == [[1]]
    assert embeddings.embed_texts_parallel([], concurrency=4) == []


def test_embed_texts_parallel_failure_propagates(monkeypatch):
    """任一批失败抛 LLMError：与串行版失败语义一致，调用方决定怎么兜。"""
    import pytest as _pytest

    from ai_agent.core.exceptions import LLMError
    from ai_agent.storage import embeddings

    def flaky(texts):
        if texts and texts[0] == "t10":
            raise LLMError("boom")
        return [[0] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", flaky)
    with _pytest.raises(LLMError):
        embeddings.embed_texts_parallel([f"t{i}" for i in range(20)], concurrency=3)


def test_describe_parallel_keeps_order(monkeypatch):
    """视觉并行：结果顺序与输入一致（页码归属不能乱）。"""
    from ai_agent.kb.ingest import figures

    def fake_describe(blob, mime):
        return f"desc-{blob.decode()}"

    monkeypatch.setattr(figures.vision, "describe_image", fake_describe)
    jobs = [(b"1", "image/png"), (b"2", "image/png"), (b"3", "image/png")]
    assert figures._describe_parallel(jobs) == ["desc-1", "desc-2", "desc-3"]

    monkeypatch.setattr(figures.settings, "kb_vision_concurrency", 1)
    assert figures._describe_parallel(jobs) == ["desc-1", "desc-2", "desc-3"]


# ——— 图表页筛选：别把表格页当图表送视觉模型 ———


class _FakePage:
    """只实现 _page_has_chart 用到的那几个方法。"""

    def __init__(self, images=(), drawings=0, size=(600, 800)):
        self._images = list(images)
        self._drawings = drawings
        self.rect = type("Rect", (), {"width": size[0], "height": size[1]})()

    def get_image_info(self):
        return [{"bbox": (0, 0, w, h)} for w, h in self._images]

    def get_images(self, full=False):
        return [(1, 0, w, h, 8, "", "", "", "") for w, h in self._images]

    def get_drawings(self):
        if self._drawings == "boom":
            raise ValueError("坏页")
        return [{}] * self._drawings


def test_chart_page_skips_dense_text_page():
    """整页表格：几百条矢量边框 + 上千字正文 —— 不是图表页。

    实测 331 页招股书按「路径数 ≥ 150」误判 124 页，20 页预算全烧在
    目录和财务报表上，每页还要等一次视觉调用。
    """
    from ai_agent.kb.ingest import figures

    page = _FakePage(drawings=215, images=[(80, 60)])
    assert figures._page_has_chart(page, text_len=1085) is False


def test_chart_page_accepts_large_image_with_little_text():
    from ai_agent.kb.ingest import figures

    page = _FakePage(images=[(500, 600)])  # 占页面大半
    assert figures._page_has_chart(page, text_len=180) is True


def test_chart_page_ignores_logo_sized_image():
    """页眉 logo / 装饰线不算图表：面积占比够不上阈值。"""
    from ai_agent.kb.ingest import figures

    page = _FakePage(images=[(60, 40)], drawings=3)
    assert figures._page_has_chart(page, text_len=200) is False


def test_chart_page_accepts_vector_only_chart():
    """ECharts 走 SVG 导出时没有位图，靠路径数识别（但要求文字少）。"""
    from ai_agent.kb.ingest import figures

    page = _FakePage(drawings=200)
    assert figures._page_has_chart(page, text_len=120) is True


def test_chart_page_survives_broken_page():
    """单页取图信息报错不能把整份解析带崩。"""
    from ai_agent.kb.ingest import figures

    page = _FakePage(drawings="boom")
    assert figures._page_has_chart(page, text_len=100) is False


def test_rebuild_graph_reads_chunks_from_db(monkeypatch):
    """图谱重建不再重新解析源文件：PDF 重新解析要再渲染 + 视觉识别一遍，
    几分钟白花，而且重切的 seq 与库里的切片对不上。"""
    import uuid as _uuid

    from ai_agent.kb import pipeline as pl

    seen: dict = {}
    monkeypatch.setattr(
        pl.store,
        "list_kb_chunk_inputs",
        lambda fid: [
            {"id": 1, "text": "第一段正文，公司主营装饰工程。", "page": 1, "kind": "text"},
            {"id": 2, "text": "第二段正文，报告期内营收增长。", "page": 2, "kind": "text"},
        ],
    )
    monkeypatch.setattr(pl.store, "update_graph_status", lambda *a, **k: seen.setdefault("status", a[1]))
    # 抽取配置跟着文件走：重建时读库里的 doc_type/level 传给抽取器，
    # 不读库就得每次重建都让人重选一遍
    monkeypatch.setattr(
        pl.store,
        "get_kb_file",
        lambda fid: {"doc_type": "prospectus", "extract_level": "light"},
    )
    monkeypatch.setattr(
        pl,
        "build_file_graph",
        lambda fid, chunks, **k: seen.update(chunks=chunks, **k) or {"entities": 3},
    )

    def boom(*a, **k):
        raise AssertionError("重建图谱不该重新解析源文件")

    monkeypatch.setattr(pl, "parse_segments", boom)

    pl.kb_pipeline._rebuild_graph(_uuid.uuid4())

    assert [c.text for c in seen["chunks"]] == [
        "第一段正文，公司主营装饰工程。",
        "第二段正文，报告期内营收增长。",
    ]
    assert [c.page for c in seen["chunks"]] == [1, 2]
    # 库里的抽取配置要原样传下去，否则重建会悄悄换成默认 prompt
    assert seen["doc_type"] == "prospectus"
    assert seen["level"] == "light"


def test_format_hit_renders_page_range_and_section():
    """跨页切片按「第134–135页 · 第十节 …」展示：只写起始页会误导溯源。"""
    from ai_agent.tools.kb.kb_search import _format_hit

    assert _format_hit(
        {
            "filename": "招股书.pdf",
            "page": 134,
            "page_end": 135,
            "section": "第十节 管理层讨论与分析",
            "seq": 0,
            "text": "正文",
        }
    ) == "[招股书.pdf · 第134–135页 · 第十节 管理层讨论与分析]\n正文"
    # 单页片不带区间
    assert _format_hit(
        {"filename": "招股书.pdf", "page": 12, "page_end": 12, "seq": 0, "text": "正文"}
    ) == "[招股书.pdf · 第12页]\n正文"
