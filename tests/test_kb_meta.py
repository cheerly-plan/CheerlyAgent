"""检索元数据（主体 / 报告期）与「按主体分桶配额」。

背景：切片此前只有 file_id 一个结构维度，问「A 和 B 的营业收入」时两家
共用同一个 query 向量抢 top_k 个名额，离混合向量更近的那家会吃满、另一家
一条都回不来。元数据给检索补上 subject 维度，配额再按主体分桶轮转。
"""

import uuid
from contextlib import contextmanager

from ai_agent.kb.ingest import meta
from ai_agent.kb.graph.kg_match import take_by_subject
from ai_agent.storage.store import store


# ---------- 元数据抽取 ----------


def test_extract_subject_strips_year_and_report_type():
    assert (
        meta.extract_subject("中国能源建设股份有限公司2026年第三季度报告.pdf")
        == "中国能源建设股份有限公司"
    )
    assert meta.extract_subject("顺丰控股股份有限公司2026年半年度报告.pdf") == (
        "顺丰控股股份有限公司"
    )
    # 没有公司名的文档也能抽出可读主体（制度/研报），只是不构成公司维度
    assert meta.extract_subject("员工考勤制度.docx") == "员工考勤制度"


def test_extract_period_quarter_half_and_year():
    assert meta.extract_period("XX公司2026年第三季度报告.pdf") == "2026Q3"
    assert meta.extract_period("XX公司2026年半年度报告.pdf") == "2026H1"
    assert meta.extract_period("XX公司2026年年度报告.pdf") == "2026"
    assert meta.extract_period("员工考勤制度.docx") == ""


def test_extract_aliases_covers_full_and_core_name():
    aliases = meta.extract_aliases("中国能源建设股份有限公司")
    assert "中国能源建设股份有限公司" in aliases
    assert "中国能源建设" in aliases
    assert meta.extract_aliases("") == []


def test_extract_file_meta_keys_match_columns():
    out = meta.extract_file_meta("安克创新科技股份有限公司2026年第三季度报告.pdf")
    assert set(out) == {"subject", "subject_aliases", "period", "meta"}
    assert out["subject"] == "安克创新科技股份有限公司"
    assert out["period"] == "2026Q3"
    assert out["meta"] == {}


def test_count_mentions_sums_surface_forms():
    text = "中国能源建设股份有限公司营收增长，中国能源建设子公司同步增长"
    assert meta.count_mentions(text, ["中国能源建设"]) == 2


# ---------- 多主体文档的切片归属 ----------


def test_pick_chunk_subject_follows_dominant_mention():
    candidates = [
        ("宁德时代股份有限公司", ["宁德时代股份有限公司", "宁德时代"]),
        ("比亚迪股份有限公司", ["比亚迪股份有限公司", "比亚迪"]),
    ]
    text = "宁德时代营收 1000 亿，宁德时代毛利率 25%，宁德时代出货量增长"
    assert meta.pick_chunk_subject(text, candidates, default="新能源行业") == (
        "宁德时代股份有限公司"
    )
    # 反过来归给比亚迪
    assert (
        meta.pick_chunk_subject(
            "比亚迪营收 2000 亿，比亚迪海外扩张", candidates, default="新能源行业"
        )
        == "比亚迪股份有限公司"
    )


def test_pick_chunk_subject_falls_back_to_default_without_mention():
    candidates = [("宁德时代股份有限公司", ["宁德时代"])]
    # 没提到任何候选 → 退回文件主体，不猜
    assert meta.pick_chunk_subject("行业整体增速放缓", candidates, default="新能源行业") == (
        "新能源行业"
    )
    # 都提到时按「先被提到的」定，不再退回文件主体——那个桶两家都查不到
    assert (
        meta.pick_chunk_subject(
            "宁德时代与比亚迪对比",
            [("宁德时代股份有限公司", ["宁德时代"]), ("比亚迪股份有限公司", ["比亚迪"])],
            default="新能源行业",
        )
        == "宁德时代股份有限公司"
    )
    assert (
        meta.pick_chunk_subject(
            "比亚迪与宁德时代对比",
            [("宁德时代股份有限公司", ["宁德时代"]), ("比亚迪股份有限公司", ["比亚迪"])],
            default="新能源行业",
        )
        == "比亚迪股份有限公司"
    )


def test_pick_chunk_subject_keeps_default_when_it_wins():
    """自家年报里自家名字提得最多时，不该被别家的零星提及抢走。"""
    candidates = [
        ("顺丰控股股份有限公司", ["顺丰控股股份有限公司", "顺丰控股"]),
        ("安克创新科技股份有限公司", ["安克创新"]),
    ]
    text = "顺丰控股营收增长，顺丰控股时效件领先；公司亦向安克创新采购设备"
    assert meta.pick_chunk_subject(text, candidates, default="顺丰控股股份有限公司") == (
        "顺丰控股股份有限公司"
    )


# ---------- 种子/切片名额按主体轮转 ----------


def _seed(subject: str, score: float, name: str = "") -> dict:
    return {
        "subject": subject,
        "score": score,
        "name": name or subject,
        "file_id": uuid.uuid4(),
    }


def test_take_by_subject_reserves_a_slot_for_close_subject():
    """对比问句：分数略低的另一家也拿得到名额。"""
    rows = [_seed("A", 0.90), _seed("A", 0.88), _seed("A", 0.86), _seed("B", 0.85)]
    picked = take_by_subject(rows, top_k=3, band=0.12)
    # B 的头名在带宽内 → 先占位，剩下的名额才回到 A 的第 2 条
    assert [r["subject"] for r in picked] == ["A", "B", "A"]


def test_take_by_subject_does_not_pad_far_subject():
    """只问一家时，差得远的别家不该硬塞进来（否则单主体问句被稀释）。"""
    rows = [_seed("A", 0.90), _seed("A", 0.88), _seed("A", 0.86), _seed("B", 0.52)]
    picked = take_by_subject(rows, top_k=3, band=0.12)
    assert [r["subject"] for r in picked] == ["A", "A", "A"]


def test_take_by_subject_without_subject_keeps_score_order():
    """主体为空（老数据）时退化成按分数取前 N 个，与分桶前逐字一致。"""
    rows = [_seed("", 0.9), _seed("", 0.8), _seed("", 0.7)]
    assert [r["score"] for r in take_by_subject(rows, top_k=2)] == [0.9, 0.8]


def test_take_by_subject_stops_when_exhausted():
    rows = [_seed("A", 0.9), _seed("B", 0.8)]
    assert len(take_by_subject(rows, top_k=10)) == 2


# ---------- 检索 SQL：分桶而不是全局抢名额 ----------


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows
        self.rowcount = len(rows)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, sink):
        self._sink = sink

    def execute(self, sql, params=None):
        self._sink.append(sql)
        return _FakeResult([])

    def executemany(self, sql, rows):
        self._sink.append(sql)
        return _FakeResult([])

    def commit(self):
        return None


class _FakePool:
    """假连接池：只记录 SQL，不连库（单测没有 Postgres）。"""

    def __init__(self, sink: list[str]):
        self._sink = sink

    @contextmanager
    def connection(self):
        yield _FakeConn(self._sink)


def _fake_pool(sink: list[str]) -> _FakePool:
    return _FakePool(sink)


def _sql_of(monkeypatch, call, sink: list[str]) -> str:
    monkeypatch.setattr(store, "_get_pool", lambda: _fake_pool(sink))
    call()
    return "\n".join(sink)


def test_search_kb_chunks_reserves_slot_per_subject(monkeypatch):
    """向量路：主体的头名只要够近就先占位，其余按距离竞争。"""
    sink: list[str] = []
    sql = _sql_of(
        monkeypatch,
        lambda: store.search_kb_chunks([0.1, 0.2], file_ids=[uuid.uuid4()], top_k=3),
        sink,
    )
    assert "PARTITION BY COALESCE(NULLIF(subject, ''), '')" in sql
    assert "ROW_NUMBER() OVER" in sql
    # 保底的关键：头名 + 在带宽内 → 排在「其余按距离竞争」之前
    assert "ORDER BY (rn = 1 AND subject_best <= global_best + %(band)s) DESC" in sql
    assert "distance" in sql and "LIMIT %(top_k)s" in sql


def test_search_kb_chunks_keyword_reserves_slot_per_subject(monkeypatch):
    sink: list[str] = []
    sql = _sql_of(
        monkeypatch,
        lambda: store.search_kb_chunks_keyword(["营业收入"]),
        sink,
    )
    assert "PARTITION BY COALESCE(NULLIF(subject, ''), '')" in sql
    assert "ORDER BY (rn = 1 AND subject_best >= global_best - %(band)s) DESC" in sql


def test_backfill_file_meta_writes_file_and_chunks(monkeypatch):
    """回填要同时刷文件行和切片行：检索分桶用的是切片上那份。"""
    sink: list[str] = []
    file_id = uuid.uuid4()
    monkeypatch.setattr(store, "_get_pool", lambda: _fake_pool(sink))
    out, chunks = store.backfill_file_meta(file_id, "顺丰控股股份有限公司2026年第三季度报告.pdf")
    assert out["subject"] == "顺丰控股股份有限公司"
    assert out["period"] == "2026Q3"
    assert chunks == 0  # 假连接没有真实行数，只断言不抛错
    assert any("UPDATE agent_kb_file" in s for s in sink)
    assert any("UPDATE agent_kb_chunk" in s for s in sink)


def test_backfill_file_meta_honours_manual_override(monkeypatch):
    sink: list[str] = []
    monkeypatch.setattr(store, "_get_pool", lambda: _fake_pool(sink))
    out, _ = store.backfill_file_meta(
        uuid.uuid4(), "最终版.pdf", subject="中国银行股份有限公司", period="2026Q3"
    )
    # 文件名不规范时规则抽不出主体，人工指定优先
    assert out["subject"] == "中国银行股份有限公司"
    assert "中国银行" in out["subject_aliases"]
    assert out["period"] == "2026Q3"


def test_infer_doc_type_checks_half_year_before_annual():
    assert meta.infer_doc_type("甲公司2026年半年度报告.pdf") == "quarterly_report"
    assert meta.infer_doc_type("甲公司2026年第三季度报告.pdf") == "quarterly_report"
    assert meta.infer_doc_type("甲公司2026年年度报告.pdf") == "annual_report"
    assert meta.infer_doc_type("甲公司招股说明书.pdf") == "prospectus"
    assert meta.infer_doc_type("员工考勤制度.docx") == ""


def test_cover_company_prefers_heading_and_keeps_parentheses():
    assert meta.cover_company(
        "证券代码：000625\n# 重庆长安汽车股份有限公司\n2026年第三季度报告"
    ) == "重庆长安汽车股份有限公司"
    assert meta.cover_company(
        "证券代码：600579 证券简称：中化装备\n# 中化装备科技（青岛）股份有限公司"
    ) == "中化装备科技（青岛）股份有限公司"
    # 括号把字面截断后，不能把剩下的「股份有限公司」当成公司名
    assert meta.cover_company("中化装备科技（青岛）股份有限公司") == (
        "中化装备科技（青岛）股份有限公司"
    )
    assert meta.cover_company("股份有限公司") == ""
    assert meta.cover_company("") == ""
    assert meta.cover_company("重要内容提示：本公司董事会保证真实") == ""


def test_resolve_file_identity_fills_company_and_period_from_cover():
    cover = "中国能源建设股份有限公司\n中国能源建设股份有限公司2026年第三季度报告"
    out = meta.resolve_file_identity("扫描件.pdf", cover)
    assert out["subject"] == "中国能源建设股份有限公司"
    assert out["period"] == "2026Q3"
    assert out["doc_type"] == "quarterly_report"
    assert out["meta"] == {}


def test_resolve_file_identity_keeps_short_filename_over_cover_boilerplate():
    """短名文件名不能被封面「公开发行证券」套话覆盖。"""
    cover = "本公司及全体董事保证信息披露真实准确完整。公开发行证券的公司信息披露编报规则"
    out = meta.resolve_file_identity("北方华创2025年第三季度报告.pdf", cover)
    assert out["subject"] == "北方华创"
    assert out["period"] == "2025Q3"


def test_resolve_file_identity_upgrades_short_name_when_cover_has_full_legal():
    cover = "北方华创科技集团股份有限公司\n2025年第三季度报告"
    out = meta.resolve_file_identity("北方华创2025年第三季度报告.pdf", cover)
    assert out["subject"] == "北方华创科技集团股份有限公司"


def test_is_plausible_subject_rejects_boilerplate():
    assert not meta.is_plausible_subject("公开发行证券")
    assert not meta.is_plausible_subject("股份有限公司")
    assert meta.is_plausible_subject("北方华创")
    assert meta.is_plausible_subject("中芯国际集成电路制造有限公司")


def test_name_in_text_is_whole_string_not_shared_prefix():
    full = "中国能源建设股份有限公司"
    assert meta.name_in_text("中国能源建设的营业收入", full)
    assert not meta.name_in_text("中国能建前三季度营收", full)
    assert not meta.name_in_text("公司营收", full)
    assert meta.name_in_text("中国能建前三季度营收", "中国能建")


def test_period_rank_annual_is_later_than_same_year_quarter():
    assert meta.period_rank("2026") > meta.period_rank("2026Q3")
    assert meta.period_rank("2026Q3") > meta.period_rank("2025")
    assert meta.period_rank("2026H1") == (2026, 2)
    assert meta.period_rank("") == (0, 0)


def test_query_period_policy():
    from datetime import date

    today = date(2026, 11, 2)
    assert meta.query_period_policy("营业收入", today=today) == ("latest", "")
    assert meta.query_period_policy("", today=today) == ("all", "")
    assert meta.query_period_policy("营收同比", today=today) == ("all", "")
    assert meta.query_period_policy("2025年和2026年营收", today=today) == ("all", "")
    assert meta.query_period_policy("2025年营收", today=today) == ("year", "2025")
    assert meta.query_period_policy("去年营收", today=today) == ("year", "2025")
    assert meta.query_period_policy("2026年三季度营收", today=today) == ("exact", "2026Q3")
    assert meta.query_period_policy("三季度营收", today=today) == ("exact", "2026Q3")
    assert meta.query_period_policy("半年报营收", today=today) == ("exact", "2026H1")
    # 两年都点了名，又指定了季度：两年的这一季都要，不能收成其中一年
    assert meta.query_period_policy("2025年和2026年三季度持股", today=today) == (
        "span_q",
        "Q3",
    )
    assert meta.query_period_policy("三季度持股和去年对比", today=today) == (
        "span_q",
        "Q3",
    )
    assert meta.query_period_policy("去年和今年的营收", today=today) == ("all", "")
    assert meta.query_period_policy("宁德时代与比亚迪对比", today=today) == (
        "latest",
        "",
    )
    assert meta.period_matches("2025Q3", "span_q", "Q3")
    assert meta.period_matches("2026Q3", "span_q", "Q3")
    assert not meta.period_matches("2025", "span_q", "Q3")
    assert not meta.period_matches("2026H1", "span_q", "Q3")


def test_fragment_subject_matches_unique_short_name():
    """「中科飞测」不在别名表里，但是全称里独有的一段。"""
    catalog = [
        ("深圳中科飞测科技股份有限公司", ["深圳中科飞测科技"]),
        ("中国银行股份有限公司", ["中国银行"]),
        ("长安汽车有限公司", ["长安汽车"]),
    ]
    assert meta.fragment_subjects("中科飞测2025年和2026年三季度持股对比", catalog) == {
        "深圳中科飞测科技股份有限公司"
    }
    # 整段就是核心名时交给 name_in_text，片段匹配不重复认
    assert "中国银行股份有限公司" not in meta.fragment_subjects("中国银行营收", catalog)
    assert meta.fragment_subjects("今年季度报告里毛利率怎么算", catalog) == set()


def test_assign_chunk_subjects_splits_only_when_two_companies_appear():
    a = "中国能源建设股份有限公司营业收入 12 亿元。"
    b = "顺丰控股股份有限公司营业收入 8 亿元。"
    library = [
        ("中国能源建设股份有限公司", ["中国能源建设股份有限公司", "中国能源建设"]),
        ("顺丰控股股份有限公司", ["顺丰控股股份有限公司", "顺丰控股"]),
    ]
    assert meta.assign_chunk_subjects([a, b], "中国能源建设股份有限公司", library) == [
        "中国能源建设股份有限公司",
        "顺丰控股股份有限公司",
    ]
    assert meta.assign_chunk_subjects(
        [a, a], "中国能源建设股份有限公司", library
    ) == ["中国能源建设股份有限公司", "中国能源建设股份有限公司"]


def test_extract_table_indicators_keeps_line_items_only():
    from ai_agent.kb.ingest.chunking import extract_table_indicators

    text = """母公司利润表
单位：万元
| 项目 | 本期 | 上期 |
| --- | --- | --- |
| 营业收入 | 12 | 10 |
| 营业成本 | 8 | 7 |
| 流动资产： |  |  |
| 合计 | 20 | 17 |
| 123 | 1 | 2 |
"""
    assert extract_table_indicators(text) == ["营业收入", "营业成本"]


def test_extract_table_indicators_reads_html_first_column():
    from ai_agent.kb.ingest.chunking import extract_table_indicators

    text = (
        "<table><tr><td>项目</td><td>本期</td></tr>"
        "<tr><td>净利润</td><td>3</td></tr>"
        "<tr><td>流动资产：</td><td></td></tr>"
        "</table>"
    )
    assert extract_table_indicators(text) == ["净利润"]


def test_keyword_search_also_matches_indicator_names(monkeypatch):
    sink: list[str] = []
    sql = _sql_of(
        monkeypatch,
        lambda: store.search_kb_chunks_keyword(["营业收入"]),
        sink,
    )
    assert "jsonb_array_elements_text" in sql
    assert "c.meta->'indicators'" in sql
    assert "WHERE item ILIKE" in sql


def test_rerank_boosts_chunk_whose_indicator_is_named():
    from ai_agent.kb.retrieval.retrieval import _boost_indicator_hits

    rows = [
        {"indicators": ["营业成本"]},
        {"indicators": ["营业收入", "净利润"]},
        {"indicators": []},
    ]
    boosted = _boost_indicator_hits("营业收入是多少", rows, [0.80, 0.74, 0.90])
    assert boosted[1] > boosted[0]
    assert boosted[0] == 0.80
    assert boosted[2] == 0.90
    # 「营收」不是「营业收入」的整段，不加分
    plain = _boost_indicator_hits("营收是多少", rows, [0.80, 0.74, 0.90])
    assert plain == [0.80, 0.74, 0.90]
