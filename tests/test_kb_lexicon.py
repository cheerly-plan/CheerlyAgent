"""四字短词表：从行项目和公司名抽出，手改文件只留例外。"""

import jieba

from ai_agent.kb import lexicon


def test_core_four_strips_index_and_notes():
    assert lexicon.core_four("三、营业利润(亏损以“-”号填列)") == "营业利润"
    assert lexicon.core_four("三、营业利润（亏损以“-”号填列）") == "营业利润"
    assert lexicon.core_four("其中：营业收入") == "营业收入"
    assert lexicon.core_four("减:营业成本") == "营业成本"
    assert lexicon.core_four("营业收入(元)") == "营业收入"
    assert lexicon.core_four("营业收入_本报告期") == "营业收入"
    assert lexicon.core_four("加:营业外收入") == ""
    assert lexicon.core_four("一、营业总收入") == ""
    assert lexicon.core_four("合并利润表项目") == ""
    assert lexicon.core_four("净利润") == ""


def test_org_four_takes_the_leading_four_characters():
    assert lexicon.org_four("中化装备科技(青岛)股份有限公司") == "中化装备"
    assert lexicon.org_four("华宝新能-关于收购境外公司100%股权的公告") == "华宝新能"
    assert lexicon.org_four("中国银行") == "中国银行"
    assert lexicon.org_four("深圳市华宝新能源股份有限公司") == "深圳市华"
    assert lexicon.org_four("") == ""


def test_choose_words_keeps_bounded_splits_and_applies_exceptions():
    def cutter(word: str):
        table = {
            "营业利润": ["营业", "利润"],
            "中国银行": ["中国银行"],
            "合并利润": ["合并", "利润"],
        }
        return table[word]

    chosen = lexicon.choose_words(
        ["营业利润", "中国银行", "合并利润"],
        {"营业利润": 6, "中国银行": 4, "合并利润": 0},
        100,
        keeps=["证券代码"],
        drops=["营业利润"],
        cutter=cutter,
    )
    assert chosen == ["证券代码"]


def test_parse_extra_reads_keep_and_drop():
    keeps, drops = lexicon.parse_extra(
        "# 注释\n+证券代码\n-合并利润\n营业外收入\n"
    )
    assert keeps == ["证券代码", "营业外收入"]
    assert drops == {"合并利润"}


def test_loaded_word_is_not_split(tmp_path, monkeypatch):
    generated = tmp_path / "short_words.txt"
    generated.write_text("造词试例\n", encoding="utf-8")
    monkeypatch.setattr(lexicon, "generated_path", lambda: generated)
    monkeypatch.setattr(lexicon, "extra_path", lambda: tmp_path / "missing.txt")
    monkeypatch.setattr(lexicon, "_loaded", False)
    lexicon.load_short_words()
    try:
        tokens = [token for token in jieba.cut("看造词试例多少") if token.strip()]
        assert "造词试例" in tokens
    finally:
        jieba.del_word("造词试例")
        monkeypatch.setattr(lexicon, "_loaded", False)
