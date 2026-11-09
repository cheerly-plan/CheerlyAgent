"""结构化事实清单静态门禁（人工录入后 / CI 预发布）。

用法（项目根目录）::

    uv run python scripts/validate_deep_facts.py
    uv run python scripts/validate_deep_facts.py --dir data/deep_facts --check-kb
    uv run python scripts/validate_deep_facts.py --no-strict  # 只查三项必填

严格档会把「缺页码/原文片段/回溯链」「level 非法」「清单重复」当成错误，
以非零码退出。``--check-kb`` 额外核对 ``source_file`` 能否对应到知识库已
入库原件（按文件名互相包含或 file_id 匹配），挡住「引用了不存在的出处」。

本脚本只读清单与知识库元数据，**不解析 PDF、不生成数字、不调用 LLM**。
"""

from __future__ import annotations

import argparse

from ai_agent.report.deep.sources import default_facts_dir, lint_facts_dir


def _known_filenames() -> list[str]:
    from ai_agent.storage.store import store

    rows = list(store.list_kb_files())
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if name := str(row.get("filename") or ""):
            names.append(name)
        if file_id := str(row.get("file_id") or ""):
            names.append(file_id)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default=str(default_facts_dir()),
        help="清单目录（默认 data/deep_facts）",
    )
    parser.add_argument(
        "--no-strict", action="store_true",
        help="只检查 item/value/source_file 三项必填，其余降级为警告",
    )
    parser.add_argument(
        "--check-kb", action="store_true",
        help="校验 source_file 是否能在知识库已入库文件名/file_id 里找到",
    )
    args = parser.parse_args()

    known = _known_filenames() if args.check_kb else None
    report = lint_facts_dir(
        args.dir, strict=not args.no_strict, known_files=known
    )
    print(f"清单目录：{args.dir}（{'严格档' if not args.no_strict else '宽松档'}）")
    print(f"文件 {len(report.files)} 个 / 条目 {report.rows} 条 / 可入账 {report.accepted} 条")
    if report.files:
        print("文件：" + "、".join(report.files))
    if report.errors:
        print(f"\n错误 {len(report.errors)} 项：")
        for error in report.errors[:40]:
            print(f"  ✗ {error}")
        if len(report.errors) > 40:
            print(f"  … 另有 {len(report.errors) - 40} 项")
    if report.duplicates:
        print(f"\n重复条目 {len(report.duplicates)} 项：")
        for duplicate in report.duplicates[:20]:
            print(f"  ! {duplicate}")
    if report.warnings:
        print(f"\n警告 {len(report.warnings)} 项：")
        for warning in report.warnings[:20]:
            print(f"  · {warning}")
        if len(report.warnings) > 20:
            print(f"  … 另有 {len(report.warnings) - 20} 项")
    if report.ok and not report.duplicates:
        print("\n事实清单校验通过：每条都带原文出处与页码，可进入台账门禁。")
        return 0
    print("\n事实清单校验未通过：请按 schema 修正后重跑，不要绕过门禁直接生成报告。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
