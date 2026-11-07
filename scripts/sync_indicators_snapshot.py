"""改完研报主配置后，同步生成只读降级镜像。

用法（项目根目录）：

    uv run python scripts/sync_indicators_snapshot.py

``registry`` 在主配置损坏时会回退到 ``indicators_defaults.yaml``，所以这份镜像
必须与主配置一致。脚本先做严格 schema 校验（等价于一次配置 lint），通过后才
重写镜像；``tests/test_report_registry.py::test_snapshot_matches_shipped_config``
会在 CI 里拦住「改了主配置忘了同步」。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_agent.report import registry

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src/ai_agent/report/config"
_MAIN = _CONFIG_DIR / "indicators.yaml"
_SNAPSHOT = _CONFIG_DIR / "indicators_defaults.yaml"

_HEADER = (
    "# 自动生成，请勿手工编辑：indicators.yaml 的只读降级镜像。\n"
    "# 由 scripts/sync_indicators_snapshot.py 导出；主配置损坏时 registry 回退到这里。\n"
)


def main() -> None:
    loaded = registry.validate_registry(_MAIN)  # 配置不合法就在这里 fail-fast
    raw = yaml.safe_load(_MAIN.read_text(encoding="utf-8"))
    body = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, width=200)
    _SNAPSHOT.write_text(_HEADER + body, encoding="utf-8")
    print(
        f"已同步 {_SNAPSHOT.relative_to(_CONFIG_DIR.parents[2])}："
        f"{len(loaded.indicators)} 个指标、{len(loaded.aliases)} 条别名"
    )


if __name__ == "__main__":
    main()
