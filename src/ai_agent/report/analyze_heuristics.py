"""经营解读启发式名单：主源 config/analysis.yaml，读坏时用内置兜底。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)

_ANALYSIS_YAML = Path(__file__).resolve().parent / "config" / "analysis.yaml"

# 与迁出前 analyze.py 字面量一致，供配置损坏时降级。
_BUILTIN_EVIDENCE_KEYWORDS = (
    "经营情况简要分析",
    "经营情况讨论与分析",
    "主要会计数据、财务指标发生变动的情况、原因",
    "净息差",
    "不良贷款",
    "拨备覆盖率",
    "营业收入",
    "净利润",
    "毛利率",
    "研发费用",
    "研发投入",
    "产能利用率",
    "量价齐升",
)
_BUILTIN_STRUCTURE_CUES = (
    ("量价齐升", "量价"),
    ("毛利润增加", "毛利润增加"),
)
_BUILTIN_OPERATING_KEYS = (
    "量价",
    "存储",
    "出货",
    "毛利润增加",
    "经营业绩",
    "产能",
    "研发",
)
_BUILTIN_CROSS_INDUSTRY = (
    "退保金",
    "赔付支出",
    "保单红利",
    "提取保险责任准备金",
    "分保费用",
    "净息差",
    "不良贷款",
    "拨备覆盖率",
    "利息净收入",
)


def _as_str_tuple(raw: Any) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw if str(item).strip())


def _as_cue_pairs(raw: Any) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    pairs: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, token = str(item[0]).strip(), str(item[1]).strip()
            if label and token:
                pairs.append((label, token))
    return tuple(pairs)


@lru_cache(maxsize=1)
def _load() -> dict[str, tuple]:
    try:
        raw = yaml.safe_load(_ANALYSIS_YAML.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("analysis.yaml 顶层必须是映射")
        return {
            "evidence_keywords": _as_str_tuple(raw.get("evidence_keywords"))
            or _BUILTIN_EVIDENCE_KEYWORDS,
            "structure_narrative_cues": _as_cue_pairs(
                raw.get("structure_narrative_cues")
            )
            or _BUILTIN_STRUCTURE_CUES,
            "operating_snippet_keys": _as_str_tuple(raw.get("operating_snippet_keys"))
            or _BUILTIN_OPERATING_KEYS,
            "cross_industry_accounts": _as_str_tuple(raw.get("cross_industry_accounts"))
            or _BUILTIN_CROSS_INDUSTRY,
        }
    except Exception:
        logger.exception("analysis.yaml 读取失败，经营解读启发式回退内置名单")
        return {
            "evidence_keywords": _BUILTIN_EVIDENCE_KEYWORDS,
            "structure_narrative_cues": _BUILTIN_STRUCTURE_CUES,
            "operating_snippet_keys": _BUILTIN_OPERATING_KEYS,
            "cross_industry_accounts": _BUILTIN_CROSS_INDUSTRY,
        }


def evidence_keywords() -> tuple[str, ...]:
    return _load()["evidence_keywords"]  # type: ignore[return-value]


def structure_narrative_cues() -> tuple[tuple[str, str], ...]:
    return _load()["structure_narrative_cues"]  # type: ignore[return-value]


def operating_snippet_keys() -> tuple[str, ...]:
    return _load()["operating_snippet_keys"]  # type: ignore[return-value]


def cross_industry_accounts() -> tuple[str, ...]:
    return _load()["cross_industry_accounts"]  # type: ignore[return-value]


def reload_heuristics() -> None:
    """测试用：清缓存后下次按磁盘配置重读。"""
    _load.cache_clear()
