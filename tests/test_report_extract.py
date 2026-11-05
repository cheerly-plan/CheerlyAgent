"""研报抽取：数字必须来自表格原文，上年同期列不能当成本期。"""

import uuid

from ai_agent.report.extract import extract_indicators, read_indicator

_TABLE = """
单位：元
<table>
<tr><td>项目</td><td>本报告期</td><td>上年同期</td></tr>
<tr><td>营业收入</td><td>100,924,954</td><td>111,000,000</td></tr>
<tr><td>营业成本</td><td>80,000</td><td>70,000</td></tr>
<tr><td>其中：营业收入</td><td>1</td><td>2</td></tr>
<tr><td>归属于上市公司股东的扣除非经常性损益的净利润</td><td>1,234.56</td><td>1,000</td></tr>
<tr><td>毛利率</td><td>12.30%</td><td>11.00%</td></tr>
</table>
"""


def test_read_indicator_takes_current_period_cell():
    got = read_indicator(_TABLE, ["营业收入"], period="2025Q3")
    assert got is not None
    assert got["value"] == "100,924,954"
    assert got["column"] == "本报告期"
    assert got["unit"] == "元"
    assert "100,924,954" in got["quote"]
    assert "111,000,000" not in got["value"]


def test_read_indicator_prefers_the_line_itself_over_a_breakdown():
    detail_first = _TABLE.replace(
        "<tr><td>营业收入</td><td>100,924,954</td><td>111,000,000</td></tr>\n"
        "<tr><td>营业成本</td><td>80,000</td><td>70,000</td></tr>\n"
        "<tr><td>其中：营业收入</td><td>1</td><td>2</td></tr>\n",
        "<tr><td>其中：营业收入</td><td>1</td><td>2</td></tr>\n"
        "<tr><td>营业收入</td><td>100,924,954</td><td>111,000,000</td></tr>\n",
    )
    both = read_indicator(detail_first, ["营业收入"], period="2025Q3")
    assert both is not None and both["value"] == "100,924,954"


def test_read_indicator_matches_long_deducted_profit_name():
    got = read_indicator(_TABLE, ["扣除非经常性损益的净利润"], period="2025Q3")
    assert got is not None
    assert got["value"] == "1,234.56"


def test_read_indicator_rejects_number_missing_from_source():
    text = _TABLE.replace("100,924,954", "100924954")
    got = read_indicator(text, ["营业收入"], period="2025Q3")
    assert got is not None
    assert got["value"] == "100924954"
    swapped = text.replace("100924954", "999")
    assert read_indicator(swapped, ["营业收入"], period="2025Q3")["value"] == "999"


def test_extract_indicators_keeps_file_and_page(monkeypatch):
    file_id = uuid.uuid4()
    files = [
        {
            "file_id": file_id,
            "filename": "中国能源建设2025年第三季度报告.pdf",
            "status": "ready",
            "subject": "中国能源建设股份有限公司",
            "subject_aliases": ["中国能建"],
            "period": "2025Q3",
        },
        {
            "file_id": uuid.uuid4(),
            "filename": "中国能源建设2024年第三季度报告.pdf",
            "status": "ready",
            "subject": "中国能源建设股份有限公司",
            "subject_aliases": [],
            "period": "2024Q3",
        },
    ]

    def search(keywords, **kwargs):
        assert kwargs["file_ids"] == [file_id]
        assert kwargs["kinds"] == ["table"]
        return [
            {
                "file_id": file_id,
                "filename": files[0]["filename"],
                "subject": files[0]["subject"],
                "page": 12,
                "page_end": 12,
                "section": "主要会计数据",
                "scope": "consolidated",
                "text": _TABLE,
            }
        ]

    monkeypatch.setattr("ai_agent.report.extract.store.list_kb_files", lambda **k: files)
    monkeypatch.setattr(
        "ai_agent.report.extract.store.search_kb_chunks_keyword", search
    )
    result = extract_indicators(["中国能建"], "2025Q3", ["营业收入", "毛利率"])
    assert [fact.value for fact in result.facts] == ["100,924,954", "12.30%"]
    assert result.missing == []
    revenue = result.facts[0]
    assert revenue.subject == "中国能源建设股份有限公司"
    assert revenue.period == "2025Q3"
    assert revenue.evidence.file_id == str(file_id)
    assert revenue.evidence.page == 12
    assert revenue.evidence.section == "主要会计数据"
    assert revenue.scope == "consolidated"


def test_extract_indicators_reports_a_missing_line(monkeypatch):
    file_id = uuid.uuid4()
    files = [
        {
            "file_id": file_id,
            "filename": "中国黄金2025年第三季度报告.pdf",
            "status": "ready",
            "subject": "中国黄金集团黄金珠宝股份有限公司",
            "subject_aliases": ["中国黄金"],
            "period": "2025Q3",
        }
    ]
    monkeypatch.setattr("ai_agent.report.extract.store.list_kb_files", lambda **k: files)
    monkeypatch.setattr(
        "ai_agent.report.extract.store.search_kb_chunks_keyword",
        lambda *a, **k: [],
    )
    result = extract_indicators(["中国黄金"], "2025Q3", ["毛利率"])
    assert result.facts == []
    assert result.missing == ["毛利率：表格里没有对上的一行"]
