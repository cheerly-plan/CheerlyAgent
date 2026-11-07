"""把 sections.yaml 导出成纯 Python 的静态降级快照。

为什么是 Python 而不是第二份 YAML：主配置坏掉最常见的就是 YAML 语法错，
降级数据不该再依赖 YAML 解析器。这份生成物只有字面量，能被 ``sections.py``
直接 import，且**不手工维护**——改章节别名只改 ``sections.yaml``，再跑一次
``scripts/sync_sections_snapshot.py``。

本模块刻意不 import ``sections``，避免「生成物还不存在 → sections 导入失败 →
没法生成」的鸡生蛋问题。完整 schema 校验由 ``lint`` / 测试负责。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

_REPORT_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _REPORT_DIR / "config"
SECTIONS_YAML = _CONFIG_DIR / "sections.yaml"
DEFAULTS_PY = _REPORT_DIR / "sections_defaults.py"

_SECTION_ID_RE = re.compile(r"[a-z][a-z0-9_]*\Z")

_HEADER = '''"""自动生成，请勿手工编辑：sections.yaml 的静态降级快照。

由 scripts/sync_sections_snapshot.py 导出；sections.yaml 损坏时 sections
回退到这份纯 Python 数据，不依赖 YAML 解析。改章节别名只需改 sections.yaml，
再跑一次同步脚本；lint 会拦住「改了配置忘了同步」。
"""
'''


def _py_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _titles_repr(titles: tuple[str, ...]) -> str:
    body = ", ".join(_py_str(title) for title in titles)
    return f"({body},)" if len(titles) == 1 else f"({body})"


def render_snapshot(config_path: Path | None = None) -> str:
    """读取章节配置，返回生成物应有的完整文本；配置结构不合法就抛 ValueError。"""
    path = config_path or SECTIONS_YAML
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError(f"{path}: version 必须为 1")

    aliases = raw.get("aliases")
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError(f"{path}: aliases 必须是非空映射")

    lines = [_HEADER.rstrip(), "", "ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = ("]
    for section_id, value in aliases.items():
        if not isinstance(section_id, str) or not _SECTION_ID_RE.fullmatch(section_id):
            raise ValueError(f"{path}: 非法 section_id {section_id!r}")
        titles = (value,) if isinstance(value, str) else tuple(value)
        if not titles or not all(isinstance(title, str) and title.strip() for title in titles):
            raise ValueError(f"{path}: {section_id} 的标题列表不合法")
        lines.append(f"    ({_py_str(section_id)}, {_titles_repr(titles)}),")
    lines.append(")")

    fallback = raw.get("fallback") or {}
    note = fallback.get("note") if isinstance(fallback, dict) else None
    if not isinstance(note, str) or not note or "\n" in note:
        raise ValueError(f"{path}: fallback.note 必须是非空单行文本")
    if any(ch.isdigit() for ch in note) or "{" in note or "}" in note:
        raise ValueError(f"{path}: fallback.note 不得含数字或模板占位")

    lines.extend(["", f"NOTE = {_py_str(note)}", ""])
    return "\n".join(lines)
