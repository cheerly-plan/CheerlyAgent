"""检索元数据（主体 / 报告期）与「按主体分桶配额」。

背景：切片此前只有 file_id 一个结构维度，问「A 和 B 的营业收入」时两家
共用同一个 query 向量抢 top_k 个名额，离混合向量更近的那家会吃满、另一家
一条都回不来。元数据给检索补上 subject 维度，配额再按主体分桶轮转。
"""

import uuid
from contextlib import contextmanager

from ai_agent.kb import meta
from ai_agent.kb.kg_match import take_by_subject
from ai_agent.memory.store import store


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
    assert out["meta"]["report_kind"] == "季度报告"


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
