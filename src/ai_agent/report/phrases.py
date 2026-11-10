"""研报共用文案：避免 compose / analyze 各抄一份导致漂移。

本模块不依赖 compose / analyze，可被两侧安全 import，不会形成环。
"""

from __future__ import annotations

# 经营解读无可用段落时的从略句；compose 渲染空节时也用同一句。
ANALYSIS_FALLBACK = "经营解读：本期季报未提供可核对的归因表述，本节从略。"
