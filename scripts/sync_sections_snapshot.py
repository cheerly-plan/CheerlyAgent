"""改完 sections.yaml 后，同步生成静态降级快照 sections_defaults.py。

用法（项目根目录）：

    uv run python scripts/sync_sections_snapshot.py            # 重新生成
    uv run python scripts/sync_sections_snapshot.py --check    # CI：只校验，不改文件

``sections.py`` 在 YAML 损坏时回退到这份纯 Python 数据，所以它必须与主配置
一致。启动 lint 的 ``sections_snapshot_sync`` 检查会拦住「改了配置忘了同步」。
"""

from __future__ import annotations

import argparse
import sys

from ai_agent.report.sections_snapshot import DEFAULTS_PY, render_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写文件；不一致返回 1")
    args = parser.parse_args(argv)

    try:
        source = render_snapshot()
    except Exception as exc:
        print(f"sections.yaml 不合法，拒绝生成降级快照：{exc}", file=sys.stderr)
        return 1

    if args.check:
        current = DEFAULTS_PY.read_text(encoding="utf-8") if DEFAULTS_PY.exists() else ""
        if current != source:
            print(
                "sections_defaults.py 与 sections.yaml 不一致，请运行 "
                "uv run python scripts/sync_sections_snapshot.py",
                file=sys.stderr,
            )
            return 1
        print("sections_defaults.py 与 sections.yaml 一致")
        return 0

    DEFAULTS_PY.write_text(source, encoding="utf-8")
    print(f"已生成 {DEFAULTS_PY.name}：{source.count(chr(10) + '    (')} 个章节别名组")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
