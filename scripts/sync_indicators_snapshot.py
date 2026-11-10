"""改完研报配置后，同步生成两份只读降级快照。

用法（项目根目录）：

    uv run python scripts/sync_indicators_snapshot.py

产出：

* ``config/indicators_defaults.yaml`` —— indicators.yaml 的 YAML 镜像；
* ``report/classification_defaults.py`` —— 分类规则（公司类型、行业别名、
  name_hints / segment_to_type）的纯 Python 字面量。

``registry`` / ``industry`` 在主配置损坏时会回退到这两份快照，所以必须与配置一致。
脚本先做严格 schema 校验（等价于一次配置 lint），通过后才重写；lint 的
``snapshot_sync`` / ``classification_snapshot_sync`` 会在启动时拦住「改了配置忘了同步」。
这样新增一个行业只改 YAML 再跑一次本脚本，无需手工维护 Python 常量。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_agent.report import classification_snapshot, registry

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src/ai_agent/report/config"
_MAIN = _CONFIG_DIR / "indicators.yaml"
_SNAPSHOT = _CONFIG_DIR / "indicators_defaults.yaml"
_INDUSTRIES = _CONFIG_DIR / "industries.yaml"

_HEADER = (
    "# 自动生成，请勿手工编辑：indicators.yaml 的只读降级镜像。\n"
    "# 由 scripts/sync_indicators_snapshot.py 导出；主配置损坏时 registry 回退到这里。\n"
)


def main() -> None:
    loaded = registry.validate_registry(_MAIN)  # 配置不合法就在这里 fail-fast
    raw = yaml.safe_load(_MAIN.read_text(encoding="utf-8"))
    body = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, width=200)
    _SNAPSHOT.write_text(_HEADER + body, encoding="utf-8")

    expected = classification_snapshot.render_defaults(_MAIN, _INDUSTRIES)
    classification_snapshot.DEFAULTS_PY.write_text(expected, encoding="utf-8")

    print(
        f"已同步 {_SNAPSHOT.relative_to(_CONFIG_DIR.parents[2])}："
        f"{len(loaded.indicators)} 个指标、{len(loaded.aliases)} 条别名"
    )
    print(
        f"已同步 {classification_snapshot.DEFAULTS_PY.relative_to(_CONFIG_DIR.parents[2])}"
    )


if __name__ == "__main__":
    main()
