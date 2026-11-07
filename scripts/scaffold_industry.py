"""给研报新增一个行业：生成 SKILL.md + 追加配置片段，零 Python 改动。

用法（项目根目录）：

    # 先看将要写什么（默认 dry-run，不改任何文件）
    uv run python scripts/scaffold_industry.py \\
        --name 光伏 --set photovoltaics --title 光伏行业财报对比分析 \\
        --companies "隆基绿能|601012|组件, 通威股份|600438|硅料" \\
        --segment-types "组件=manufacturing,硅料=cyclical"

    # 确认后写入，并自动同步降级快照 + 跑配置 lint
    uv run python scripts/scaffold_industry.py ... --apply

刻意用「文本插入」而不是 ``yaml.safe_dump`` 重写配置：industries.yaml 里有
``&bank_only`` 锚点与大量注释，dump 会把它们展开/丢掉。插入点用固定锚点，
找不到就报错退出，绝不猜测位置。

新增行业本身不需要改 .py：公司类型白名单、行业别名、指标集、定性文案、
赛道语义都来自 industries.yaml，渲染层通过 industry.profile_key_for_plan()
按 indicator_set 取。赛道映射写进新 profile 自己的 segments，不追加到
indicators.yaml 的全局 segment_to_type —— 那会污染其它行业的同名赛道。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "src" / "ai_agent" / "report" / "config"
_SKILLS_ROOT = _PROJECT_ROOT / "src" / "ai_agent" / "report" / "skills"

_SKILL_TEMPLATE = """---
name: {skill_id}
kind: industry
title: {title}
industries: {name}
indicator_set: {set_key}
description: {name}代表公司按赛道分组做财报对比；行业景气待接外部数据源，本期不推断
companies: {companies}
---

# {title}

管「写什么」与「盯哪几家、归哪条赛道」。数字只来自已入库/自动补齐的公司财报；
不做估值分位、不做买卖建议。行业景气与出货量等外部数据源未接时，只在「下期预告」
留白，不编造宏观增速。

代表公司及赛道分组以文件头 `companies` 字段为唯一来源，由系统按赛道自动展开；
正文不重复列举，新增公司只改该字段一处。

## 大纲

### 阅读说明

section: reading_note

交代期间口径、合并／母公司范围；金额默认统一折算为元（原单位见指标表脚注）；写明三级归因边界——①已抽取数字事实可写；②原文经营说明须标出处；③无出处的第三方推断一律不写。

### 赛道分组

section: segment_roster

代表公司（按赛道）：{{companies}}。只陈述分组与是否抽到本期财报，不写赛道景气度。

### 组内对比

section: segment_peer

只在同一赛道内对比{{indicators}}；跨赛道不混排同一张表。没抽到的公司／指标进缺失说明。

### 经营解读

section: analysis

只根据已抽取指标与季报原文解释经营现象；每个因果句必须能指回已授权数字或原文证据，否则不写；不写同业均值、估值与投资建议。

### 趋势与对比图

section: chart_guide

两点同比对比用柱状图；不足四期不画折线趋势。图注标明对比期与口径。

### 缺失与口径

section: gaps

列出未调到的季报、未抽到的指标与口径不齐项；不写投资建议。

### 下期预告

section: next_period

景气周期待接入外部数据源；本期不推断周期位置。下一期需明确：数据源清单、入库表结构、更新频率。
"""

_PROFILE_BLOCK = """  {set_key}:
    include: *shared_include
    exclude: *bank_only
{segments}
"""


def _parse_pairs(raw: str, sep: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in (raw or "").split(","):
        text = item.strip()
        if not text:
            continue
        if sep not in text:
            raise ValueError(f"片段 {text!r} 缺少分隔符 {sep!r}")
        left, _, right = text.partition(sep)
        out.append((left.strip(), right.strip()))
    return out


def _insert_before(text: str, anchor: str, block: str) -> str:
    if anchor not in text:
        raise ValueError(f"找不到插入锚点 {anchor!r}，配置格式可能已改动")
    return text.replace(anchor, block + anchor, 1)


def build_changes(args) -> dict[Path, str]:
    """返回「目标文件 → 写入后的完整文本」，不落盘。"""
    root: Path = args.root
    config = root / "src/ai_agent/report/config" if root != _PROJECT_ROOT else _CONFIG_DIR
    skills_root = root / "src/ai_agent/report/skills" if root != _PROJECT_ROOT else _SKILLS_ROOT
    skill_id = f"{args.set_key}_industry_report"

    skill_path = skills_root / skill_id / "SKILL.md"
    if skill_path.exists():
        raise ValueError(f"skill 已存在：{skill_path}")

    changes: dict[Path, str] = {}
    base = args.name if args.name.endswith("行业") else f"{args.name}行业"
    changes[skill_path] = _SKILL_TEMPLATE.format(
        skill_id=skill_id,
        title=args.title or f"{base}财报对比分析",
        name=args.name,
        set_key=args.set_key,
        companies=args.companies,
    )

    # 赛道语义属于这个行业自己：全部写进 profile.segments。
    # 带赛道花名册却漏配映射，会让渲染层回退 generic 或串到别的行业模板，
    # 所以这里直接拒绝生成，而不是等 lint 事后报错。
    segment_pairs = _parse_pairs(args.segment_types, "=")
    for segment, company_type in segment_pairs:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", company_type):
            raise ValueError(f"赛道 {segment!r} 的公司类型 {company_type!r} 非法")
    roster_segments = {
        row.split("|")[2].strip()
        for row in (args.companies or "").split(",")
        if row.strip() and len(row.split("|")) >= 3 and row.split("|")[2].strip()
    }
    mapped = {segment for segment, _ in segment_pairs}
    missing = sorted(roster_segments - mapped)
    if missing:
        raise ValueError(
            f"花名册赛道 {missing} 没有 --segment-types 映射；"
            "新增行业必须声明自己的赛道语义"
        )
    unknown = sorted(mapped - roster_segments)
    if unknown:
        raise ValueError(f"--segment-types 含花名册里不存在的赛道 {unknown}")
    segments_block = ""
    if segment_pairs:
        segments_block = "    segments:\n" + "".join(
            f"      {segment}: {company_type}\n"
            for segment, company_type in segment_pairs
        )

    industries = config / "industries.yaml"
    text = industries.read_text(encoding="utf-8")
    text = _insert_before(
        text,
        "\nsets:\n",
        f"\n  {args.name}: {args.set_key}\n  {args.set_key}: {args.set_key}\n",
    )
    text = _insert_before(
        text,
        "\n  generic:\n",
        _PROFILE_BLOCK.format(set_key=args.set_key, segments=segments_block),
    )
    changes[industries] = text
    # 不再改 indicators.yaml：全局 segment_to_type 只作无行业上下文时的兜底。
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="新增研报行业（生成配置，零 Python 改动）")
    parser.add_argument(
        "--name", required=True, help="中文行业名，不要带「行业」后缀，如 光伏"
    )
    parser.add_argument("--set", dest="set_key", required=True, help="profile key，如 photovoltaics")
    parser.add_argument("--title", default="", help="报告标题，默认按行业名生成")
    parser.add_argument("--companies", required=True, help="名|代码|赛道[, 名|代码|赛道…]")
    parser.add_argument(
        "--segment-types",
        default="",
        help="赛道=细类型[, …]，写入新 profile 的 segments（需覆盖 companies 里全部赛道）",
    )
    parser.add_argument("--apply", action="store_true", help="真正写文件（默认只打印）")
    parser.add_argument("--root", type=Path, default=_PROJECT_ROOT, help="仅供测试：替换项目根")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()

    try:
        changes = build_changes(args)
    except (ValueError, OSError) as exc:
        print(f"生成失败：{exc}", file=sys.stderr)
        return 1

    for path, content in changes.items():
        if args.apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"已写入 {path.relative_to(args.root)}")
        else:
            print(f"--- 将写入 {path.relative_to(args.root)} ---")
            if path.name == "SKILL.md":
                print(content[:400] + "...\n")
            else:
                print("（YAML 片段已追加，dry-run 不展开全文）\n")

    if not args.apply:
        print("确认无误后加 --apply 写入；随后运行：")
        print("  uv run python scripts/sync_indicators_snapshot.py")
        print("  uv run python scripts/lint_report_config.py")
        return 0

    if args.root == _PROJECT_ROOT:
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts/sync_indicators_snapshot.py")],
            cwd=_PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            print("降级快照同步失败，请检查配置后重跑", file=sys.stderr)
            return result.returncode
        print("下一步：uv run python scripts/lint_report_config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
