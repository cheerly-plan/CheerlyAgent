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
    # 需求简称「中国能建」能对上库内别名时，正文主体用简称，便于缺口对账
    assert revenue.subject == "中国能建"
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
    assert result.missing == ["毛利率：表格和正文里都没有对上的一行"]


def test_extract_gross_margin_from_narrative_when_table_missing(monkeypatch):
    """中芯等季报常把毛利率写在经营分析正文，主表没有独立行。"""
    file_id = uuid.uuid4()
    files = [
        {
            "file_id": file_id,
            "filename": "中芯国际2025年第三季度报告.pdf",
            "status": "ready",
            "subject": "中芯国际",
            "subject_aliases": ["中芯国际"],
            "period": "2025Q3",
        }
    ]
    prose = (
        "三季度，公司整体实现营业收入人民币171.62亿元，环比增长6.9%；"
        "公司毛利率为25.5%，环比上升4.8个百分点；产能利用率上升至95.8%。"
    )

    def search(keywords, **kwargs):
        kinds = kwargs.get("kinds") or []
        if kinds == ["table"]:
            return []
        return [
            {
                "file_id": file_id,
                "filename": files[0]["filename"],
                "subject": files[0]["subject"],
                "page": 8,
                "page_end": 8,
                "section": "经营情况讨论与分析",
                "scope": "",
                "text": prose,
            }
        ]

    monkeypatch.setattr("ai_agent.report.extract.store.list_kb_files", lambda **k: files)
    monkeypatch.setattr(
        "ai_agent.report.extract.store.search_kb_chunks_keyword", search
    )
    result = extract_indicators(["中芯国际"], "2025Q3", ["毛利率"])
    assert len(result.facts) == 1
    fact = result.facts[0]
    assert fact.name == "毛利率"
    assert fact.value == "25.5"
    assert fact.unit == "%"
    assert fact.evidence.page == 8
    assert "25.5" in fact.evidence.quote


# 季报「主要财务数据」的真实形态：单季与累计并排，且各自带一列官方同比。
_DUAL_COLUMN_TABLE = """
单位：百万元人民币（另有说明者除外）
<table>
<tr><td>项目</td><td>本报告期</td><td>本报告期比上年同期增减(%)</td><td>年初至报告期末</td><td>年初至报告期末比上年同期增减(%)</td></tr>
<tr><td>营业收入</td><td>169,462</td><td>4.48</td><td>526,365</td><td>7.16</td></tr>
<tr><td>净利润</td><td>67,201</td><td>5.91</td><td>199,232</td><td>5.09</td></tr>
</table>
"""


def test_dual_column_table_picks_cumulative_value_with_matching_yoy():
    """研报说「前三季度营收」指累计列，不能报成单季的 169,462。"""
    got = read_indicator(_DUAL_COLUMN_TABLE, ["营业收入"], period="2026Q3")
    assert got is not None
    assert got["value"] == "526,365"
    assert got["column"] == "年初至报告期末"
    assert got["reported_yoy"] == "7.16%"
    assert "169,462" not in got["value"]
    assert "526,365" in got["quote"] and "7.16" in got["quote"]


def test_single_quarter_column_keeps_its_own_yoy():
    """只有单季列时，同比必须配单季那一列，不能串到累计同比去。"""
    table = """
单位：百万元
<table>
<tr><td>项目</td><td>本报告期</td><td>本报告期比上年同期增减(%)</td></tr>
<tr><td>营业收入</td><td>169,462</td><td>4.48</td></tr>
</table>
"""
    got = read_indicator(table, ["营业收入"], period="2026Q3")
    assert got is not None
    assert got["value"] == "169,462"
    assert got["column"] == "本报告期"
    assert got["reported_yoy"] == "4.48%"
    assert got["cumulative"] is False


def test_q1_benbaogaoqi_counts_as_cumulative():
    """一季报「本报告期」覆盖即 1–3 月，应视同累计口径，供 Q3 环比推导。"""
    table = """
单位：元
<table>
<tr><td>项目</td><td>本报告期</td><td>上年同期</td><td>本报告期比上年同期增减变动幅度(%)</td></tr>
<tr><td>营业收入</td><td>4,188,075,574.04</td><td>1,909,062,290.61</td><td>119.38</td></tr>
</table>
"""
    got = read_indicator(table, ["营业收入"], period="2026Q1")
    assert got is not None
    assert got["value"] == "4,188,075,574.04"
    assert got["column"] == "本报告期"
    assert got["cumulative"] is True


def test_extract_indicators_carries_column_and_reported_yoy(monkeypatch):
    file_id = uuid.uuid4()
    files = [
        {
            "file_id": file_id,
            "filename": "中国银行股份有限公司2026年第三季度报告.pdf",
            "status": "ready",
            "subject": "中国银行股份有限公司",
            "subject_aliases": ["中国银行"],
            "period": "2026Q3",
        }
    ]

    def search(keywords, **kwargs):
        return [
            {
                "file_id": file_id,
                "filename": files[0]["filename"],
                "subject": files[0]["subject"],
                "page": 2,
                "page_end": 2,
                "section": "一、主要财务数据",
                "scope": "",
                "text": _DUAL_COLUMN_TABLE,
            }
        ]

    monkeypatch.setattr("ai_agent.report.extract.store.list_kb_files", lambda **k: files)
    monkeypatch.setattr("ai_agent.report.extract.store.search_kb_chunks_keyword", search)
    result = extract_indicators(["中国银行"], "2026Q3", ["营业收入"])
    assert len(result.facts) == 1
    fact = result.facts[0]
    assert fact.value == "526,365"
    assert fact.column == "年初至报告期末"
    assert fact.reported_yoy == "7.16%"
