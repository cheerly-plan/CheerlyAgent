"""从研报「溯源附件」确定性导出结构化事实清单（零 LLM、零 PDF 解析）。

用法（项目根目录）::

    uv run python scripts/export_deep_facts.py
    uv run python scripts/export_deep_facts.py --exclude-period 2026Q3
    uv run python scripts/export_deep_facts.py --dry-run

**为什么需要这一步**：深度层的事实台账只认「已核对原文的结构化事实」。
知识库里躺着 PDF 不等于附注数字已抽出，也绝不能让 LLM 现场从 PDF 抽数
（那会把幻觉重新引入防幻觉门禁）。本脚本只解析**项目自己已经生成并校验过**
的 ``*.provenance.md`` 表格：每行自带公司、报告期、指标、原文数值、文件
（含知识库 file_id）、页码、章节、原文片段，正好是清单所需的全部溯源字段。

严格边界：

* 只取「文件列含知识库 file_id」且「数值列是原文金额（``-?数字元``）」的行；
  akshare 摘要、推算比率、百分比、空页码一律跳过；
* 不换算单位、不算同比、不补缺值、不改写 quote，纯字符串搬运；
* 输出落在 ``data/deep_facts/``（该目录被 gitignore 忽略，属于本地数据，
  不随代码提交），并写清生成来源与条数。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.report.deep.sources import DOC_CHAPTERS, classify_doc, default_facts_dir

_ROW_SPLIT = re.compile(r"(?<!\\)\|")
# 溯源附件里 file_id 用全角括号「（…）」包裹，报告里也可能出现半角括号。
_FILE_ID = re.compile(r"[(（]([0-9a-fA-F-]{36})[)）]")
_ORIGINAL_AMOUNT = re.compile(r"^-?[\d,]+(?:\.\d+)?元$")


def _cells(line: str) -> list[str]:
    """拆 Markdown 表格行；只去掉行首/行尾竖线产生的空项，保留中间空单元格。"""
    parts = _ROW_SPLIT.split(line)
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [part.strip().replace(r"\|", "|") for part in parts]


def parse_provenance(path: Path) -> tuple[list[dict], dict[str, int]]:
    """把一份溯源附件转成清单行；返回 (行, 跳过原因计数)。"""
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _cells(line)
        if len(cells) != 9:
            continue
        subject, period, item, value, _basis, source_file, page, section, quote = cells
        if subject in {"公司", "---"} or set(subject) <= {"-", " "}:
            continue  # 表头 / 分隔行
        if not _FILE_ID.search(source_file):
            skipped["无知识库 file_id"] = skipped.get("无知识库 file_id", 0) + 1
            continue
        if "akshare" in source_file.lower():
            skipped["第三方摘要"] = skipped.get("第三方摘要", 0) + 1
            continue
        if not _ORIGINAL_AMOUNT.match(value):
            skipped["非原文金额（推算值/比率/百分比）"] = (
                skipped.get("非原文金额（推算值/比率/百分比）", 0) + 1
            )
            continue
        if not page.isdigit() or not quote:
            skipped["缺页码或原文片段"] = skipped.get("缺页码或原文片段", 0) + 1
            continue
        doc_type = classify_doc(source_file)
        rows.append({
            "item": item[:120],
            "value": value[:120],
            "source_file": source_file[:200],
            "doc_type": doc_type,
            "level": "L1",
            "location": f"第{page}页 {section}"[:120],
            "quote": quote[:400],
            "subject": subject[:80],
            "period": period[:40],
            "acquisition": "知识库定期报告原文／溯源附件确定性导出（零 LLM）",
            "chapters": list(DOC_CHAPTERS.get(doc_type, (3, 4))),
        })
    return rows, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", default=str(settings.report_output_dir),
        help="溯源附件所在目录（默认 settings.report_output_dir）",
    )
    parser.add_argument(
        "--out-dir", default=str(default_facts_dir()),
        help="清单输出目录（默认 data/deep_facts）",
    )
    parser.add_argument(
        "--exclude-period", action="append", default=[], metavar="PERIOD",
        help="排除指定报告期（可重复），用于避免与本期事实层台账重复",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    source = Path(args.source_dir)
    paths = sorted(source.glob("*.provenance.md"))
    if not paths:
        print(f"未找到溯源附件：{source}/*.provenance.md", file=sys.stderr)
        return 1

    excluded = set(args.exclude_period)
    by_key: dict[tuple[str, ...], dict] = {}
    skipped_totals: dict[str, int] = {}
    for path in paths:
        rows, skipped = parse_provenance(path)
        for reason, count in skipped.items():
            skipped_totals[reason] = skipped_totals.get(reason, 0) + count
        for row in rows:
            if row["period"] in excluded:
                continue
            key = (row["subject"], row["period"], row["item"],
                   row["value"], row["source_file"])
            by_key.setdefault(key, row)

    facts = [by_key[key] for key in sorted(by_key)]
    print(f"溯源附件：{len(paths)} 份 → 结构化事实 {len(facts)} 条（已去重）")
    if skipped_totals:
        print("跳过：" + "；".join(f"{k} {v} 行" for k, v in sorted(skipped_totals.items())))
    from collections import Counter
    print("按报告期：" + "；".join(
        f"{p} {n} 条" for p, n in sorted(Counter(f["period"] for f in facts).items())
    ))
    if excluded:
        print(f"排除报告期：{'、'.join(sorted(excluded))}")
    if args.dry_run:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "schema": "deep-facts/v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_by": "scripts/export_deep_facts.py（从 *.provenance.md "
                           "确定性解析，零 LLM、零 PDF 解析、零数值计算）",
            "source_files": [path.name for path in paths],
            "fact_count": len(facts),
            "note": "value 为原文披露金额，未做任何换算或推导；"
                    "修改前请对照原文与页码核对，禁止手工填入心算值",
        },
        "facts": facts,
    }
    target = out_dir / "periodic_facts.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已写入：{target}")
    print("下一步：uv run python scripts/validate_deep_facts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
