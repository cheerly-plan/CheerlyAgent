"""研报配置硬门禁（CI / 预发布）。

用法（项目根目录）：

    uv run python scripts/lint_report_config.py

与启动 lint 用同一个检查集，但强制 strict：配置有任何问题都以非零码退出。
服务启动时默认不阻断（``REPORT_CONFIG_STRICT=false``），这里负责在上线前拦住。
"""

from __future__ import annotations

import sys

from ai_agent.report.lint import ReportConfigError, lint_report_config


def main() -> int:
    try:
        report = lint_report_config(strict=True)
    except ReportConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"研报配置 lint 通过：{len(report.checked)} 项检查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
