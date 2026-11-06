"""研报流水线。解析需求，抽取指标，计算增减，写成正文，排成 PDF，再校验。

包根做 PEP 562 惰性导出（``__getattr__``）：按需再加载子模块，不在 import
``ai_agent.report`` 时就拖起整条流水线。原因——``prompts/tools.py`` 为了渲染
report_type 可选清单要读 ``skills`` 注册表（轻叶子，只依赖 industry），但包根
原来急切 import compose/extract/pipeline，链条是 ``prompts → report 包根 →
compose → extract → memory.store → episode_merge → prompts``，转一圈撞上
还没初始化完的 ``prompts``，直接 ImportError。惰性化后这条边安全了，顺带让
``from ai_agent.report import skills`` 这类轻量使用不再背上 PDF/校验全家桶。

``__getattr__`` 对 ``from ai_agent.report import run_report`` 这种写法同样
生效：模块属性缺失时 Python 会走这里再 import 子模块。
"""

from __future__ import annotations

__all__ = [
    "ReportResult",
    "compare_measures",
    "compose_report",
    "extract_indicators",
    "parse_request",
    "render_report_pdf",
    "run_report",
    "validate_report",
]

# 惰性导出名 → 子模块。新增公开 API 记得同步加进 __all__。
_LAZY_EXPORTS: dict[str, str] = {
    "compare_measures": "compare",
    "compose_report": "compose",
    "extract_indicators": "extract",
    "parse_request": "parse",
    "render_report_pdf": "pdf",
    "run_report": "pipeline",
    "validate_report": "validate",
    "ReportResult": "pipeline",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(f"ai_agent.report.{module_name}")
    value = getattr(module, name)
    # 缓存回模块属性，后续访问不再走 __getattr__
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
