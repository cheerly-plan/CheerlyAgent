"""自动生成，请勿手工编辑：sections.yaml 的静态降级快照。

由 scripts/sync_sections_snapshot.py 导出；sections.yaml 损坏时 sections
回退到这份纯 Python 数据，不依赖 YAML 解析。改章节别名只需改 sections.yaml，
再跑一次同步脚本；lint 会拦住「改了配置忘了同步」。
"""

ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reading_note", ("阅读说明",)),
    ("lead", ("本期结论", "覆盖范围", "行业概览")),
    ("indicator_table", ("主要指标与出处",)),
    ("change", ("同比与环比",)),
    ("quarter", ("单季节奏",)),
    ("year_end", ("较上年末变动",)),
    ("observation", ("数据观察", "核心主线")),
    ("analysis", ("经营解读",)),
    ("segment_roster", ("赛道分组",)),
    ("segment_peer", ("组内对比",)),
    ("chart_guide", ("趋势与对比图",)),
    ("next_period", ("下期预告", "景气周期", "细分赛道")),
    ("key_companies", ("重点公司",)),
    ("trend", ("趋势",)),
    ("peer", ("公司间对比",)),
    ("gaps", ("缺失说明", "缺失与口径", "风险与缺失")),
)

NOTE = "这一节没有对应的数据，未写。"
