"""生成 ``classification_defaults.py``：分类规则的只读降级快照。

与 ``sections_snapshot.py`` 同款设计：只依赖 PyYAML 与标准库，**刻意不 import
registry / industry**，避免「生成物要读被生成模块」的鸡生蛋问题。

单一数据源是两份配置：

* ``config/indicators.yaml`` 的 ``classification``（name_hints / segment_to_type）；
* ``config/industries.yaml`` 的 ``profiles``（公司类型白名单）与 ``aliases``（行业名别名）。

生成物是纯 Python 字面量，主配置与 YAML 镜像都读不到时仍能完成分类；
由 ``scripts/sync_indicators_snapshot.py`` 导出，lint 校验它不是过期生成物。
这样新增行业只需改 YAML 再跑一次同步脚本，不必再手工维护三份 Python 常量。
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULTS_PY = Path(__file__).resolve().parent / "classification_defaults.py"

_HEADER = (
    "# 自动生成，请勿手工编辑：分类规则的只读降级快照。\n"
    "# 由 scripts/sync_indicators_snapshot.py 导出；主配置不可用时回退到这里。\n"
    "\n"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _render_pairs(name: str, pairs) -> str:
    lines = [f"{name} = ("]
    for left, right in pairs:
        lines.append(f"    ({str(left)!r}, {str(right)!r}),")
    lines.append(")")
    return "\n".join(lines)


def render_defaults(indicators: Path, industries: Path) -> str:
    """按固定格式渲染生成物全文；文本内容稳定，可直接与磁盘文件比较。"""
    ind_raw = _load(indicators)
    industry_raw = _load(industries)

    classification = ind_raw.get("classification") or {}
    name_hints = [
        (str(pair[0]), str(pair[1]))
        for pair in (classification.get("name_hints") or [])
    ]
    # name_hints 的顺序有语义（长词优先、先命中先返回），不能排序。
    segment_to_type = [
        (str(key), str(value))
        for key, value in (classification.get("segment_to_type") or {}).items()
    ]

    profiles = industry_raw.get("profiles") or {}
    aliases = industry_raw.get("aliases") or {}
    company_types = sorted({str(key) for key in profiles})

    type_aliases = "\n".join(
        ["TYPE_ALIASES = {"]
        + [f"    {str(key)!r}: {str(value)!r}," for key, value in aliases.items()]
        + ["}"]
    )
    types = "\n".join(
        ["COMPANY_TYPES = frozenset(", "    {"]
        + [f"        {item!r}," for item in company_types]
        + ["    }", ")"]
    )

    return _HEADER + "\n\n".join(
        [
            types,
            type_aliases,
            _render_pairs("NAME_HINTS", name_hints),
            _render_pairs("SEGMENT_TO_TYPE", segment_to_type),
        ]
    ) + "\n"
