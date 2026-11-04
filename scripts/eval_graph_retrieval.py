"""图谱检索评测：问句 → 能否查到正确的边。

与 scripts/eval_graph.py 的分工：那个测「图谱构建质量」（单文件、金标实体/别名
是否对齐），这个测「图谱检索质量」（全库、问句级、端到端）。两者互补，不能互相替代。

## 为什么必须同时测两条路径

- `prefetch_neighborhood`：关系问句在模型决策**之前**抢跑的那条（agents/loop.py:395），
  结果直接塞进 system message。它只在问句命中 `_RELATION_QUERY` 正则时才跑。
- `_kg_search`：模型主动调工具的路径。

诺安 case 暴露的正是两者的分裂：明说年份时工具路径查得到，不提年份时预检索静默
跑空（返回空串、不注入）。只测其中一条会漏掉一半的失败——这也是本脚本存在的理由。

## 判定口径

正例：输出里存在一条边同时命中 (source, relation, target)。
负例：输出里一条边都解析不出来（不依赖具体文案，避免提示词改动就误判）。
干扰：输出里出现了 `forbid` 列表里的实体名，记为 false_positive（张冠李戴）。

预检索路径只对 `wants_graph_neighborhood(query)` 为真的用例统计——其余用例返回空串
是**设计行为**，计进分母会把真实问题稀释掉。

用法：
    uv run python scripts/eval_graph_retrieval.py --validate      # 校验用例与库内数据对齐
    uv run python scripts/eval_graph_retrieval.py                 # 跑全量
    uv run python scripts/eval_graph_retrieval.py --group historical
    uv run python scripts/eval_graph_retrieval.py --out data/eval/graph_retrieval.json

依赖：项目自有模块，无新增第三方依赖。需要库内已有图谱与向量服务可用。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.memory.store import store  # noqa: E402
from ai_agent.tools import kg_search as KS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "scripts" / "cases" / "graph_eval_cases.json"

# 输出行形如：  - 诺安成长混合型证券投资基金 —持股→ 深圳中科飞测科技股份有限公司：持股13,453,255股
_EDGE_RE = re.compile(r"^- (?P<src>.+?) —(?P<rel>.+?)→ (?P<dst>.+?)(?:：.*)?$")
_SEED_RE = re.compile(r"^命中实体：(?P<names>.+)$")


def parse_edges(text: str) -> list[tuple[str, str, str]]:
    """从格式化输出里抽出所有关系边。空输出 = 没有边 = 图谱没给出任何关系。"""
    edges: list[tuple[str, str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        m = _EDGE_RE.match(line)
        if m:
            edges.append((m["src"].strip(), m["rel"].strip(), m["dst"].strip()))
    return edges


def parse_seeds(text: str) -> list[str]:
    """「命中实体：」后面列出的种子名，用来看 top1 是不是问的那个。"""
    for line in (text or "").splitlines():
        m = _SEED_RE.match(line.strip())
        if m:
            names = []
            for item in m["names"].split("、"):
                name = re.sub(r"\([^)]*\)\s*$", "", item).strip()
                if name:
                    names.append(name)
            return names
    return []


def edge_matches(edges, expect: dict) -> bool:
    src, rel, dst = expect["source"], expect["relation"], expect["target"]
    for s, r, d in edges:
        if s == src and r == rel and d == dst:
            return True
    return False


def run_path(case: dict, path: str) -> dict:
    """跑一条路径。异常不算失败用例，但要如实报出来。"""
    query = case["query"]
    try:
        if path == "prefetch":
            triggered = KS.wants_graph_neighborhood(query)
            text = KS.prefetch_neighborhood(query) if triggered else ""
        else:
            triggered = True
            text = KS._kg_search(query)
    except Exception as exc:  # 检索链路自身炸了，必须暴露而不是静默计 0
        return {"error": f"{type(exc).__name__}: {exc}", "triggered": True}

    edges = parse_edges(text)
    seeds = parse_seeds(text)
    expect = case.get("expect")
    forbid = case.get("forbid") or []

    if expect is None:
        # 负例：没有边就是对的
        passed = not edges
    else:
        passed = edge_matches(edges, expect)

    hit_forbidden = [name for name in forbid if any(name in e for e in ("".join(x) for x in edges))]

    return {
        "triggered": triggered,
        "passed": passed,
        "edges": len(edges),
        "seed_top1": seeds[0] if seeds else "",
        "forbidden_hit": hit_forbidden,
        "error": "",
    }


def validate(cases: list[dict]) -> list[str]:
    """校验用例里的三元组在库里真实存在——防止评测集自己写错名字。"""
    problems: list[str] = []
    with store._get_pool().connection() as conn:
        for case in cases:
            expect = case.get("expect")
            if expect is None:
                continue
            row = conn.execute(
                """SELECT 1 FROM agent_kg_relation r
                   JOIN agent_kg_entity s ON s.id = r.source_id
                   JOIN agent_kg_entity t ON t.id = r.target_id
                   WHERE s.name = %(s)s AND r.relation = %(rel)s AND t.name = %(t)s
                   LIMIT 1""",
                {"s": expect["source"], "rel": expect["relation"], "t": expect["target"]},
            ).fetchone()
            if row is None:
                problems.append(
                    f"库里没有这条边：{case['query'][:24]} → "
                    f"{expect['source']} -{expect['relation']}-> {expect['target']}"
                )
            for name in case.get("forbid") or []:
                n = conn.execute(
                    "SELECT 1 FROM agent_kg_entity WHERE name = %(n)s LIMIT 1", {"n": name}
                ).fetchone()
                if n is None:
                    problems.append(f"forbid 里的实体不在库内：{case['query'][:24]} → {name}")
    return problems


def render(cases: list[dict], results: dict[str, dict]) -> str:
    lines: list[str] = []
    by_group: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_group[case["group"]].append(case)

    lines.append("| group | 条数 | 工具路径通过 | 预检索通过（仅触发的） | 干扰误召回 |")
    lines.append("|---|---|---|---|---|")
    total = {"tool": 0, "tool_pass": 0, "pre": 0, "pre_pass": 0, "fp": 0}
    for group, group_cases in by_group.items():
        tool_pass = sum(1 for c in group_cases if results[c["query"]]["tool"]["passed"])
        triggered = [c for c in group_cases if results[c["query"]]["prefetch"]["triggered"]]
        pre_pass = sum(1 for c in triggered if results[c["query"]]["prefetch"]["passed"])
        fp = sum(1 for c in group_cases if results[c["query"]]["tool"]["forbidden_hit"])
        lines.append(
            f"| {group} | {len(group_cases)} | {tool_pass}/{len(group_cases)} | "
            f"{pre_pass}/{len(triggered)} | {fp} |"
        )
        total["tool"] += len(group_cases)
        total["tool_pass"] += tool_pass
        total["pre"] += len(triggered)
        total["pre_pass"] += pre_pass
        total["fp"] += fp
    lines.append(
        f"| **合计** | {total['tool']} | **{total['tool_pass']}/{total['tool']}** | "
        f"**{total['pre_pass']}/{total['pre']}** | {total['fp']} |"
    )

    for path, label in (("tool", "工具路径 _kg_search"), ("prefetch", "预检索 prefetch_neighborhood")):
        failed = [
            c for c in cases
            if results[c["query"]][path]["triggered"] and not results[c["query"]][path]["passed"]
        ]
        errs = [c for c in cases if results[c["query"]][path].get("error")]
        if errs:
            lines += ["", f"## {label} 执行异常（{len(errs)}）", ""]
            for c in errs:
                lines.append(f"- {c['query'][:40]} → {results[c['query']][path]['error']}")
        if failed:
            lines += ["", f"## {label} 失败明细（{len(failed)}）", ""]
            for c in failed:
                r = results[c["query"]][path]
                exp = c.get("expect")
                want = (
                    f"{exp['source']} -{exp['relation']}-> {exp['target']}" if exp else "（无边）"
                )
                lines.append(f"- `{c['query']}`")
                lines.append(f"    期望 {want}；实际 {r['edges']} 条边，top1=`{r['seed_top1']}`")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="图谱检索评测")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--group", default="", help="只跑某个分组")
    parser.add_argument("--validate", action="store_true", help="只校验用例与库内数据对齐")
    parser.add_argument("--out", default="", help="结果 json 输出路径")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))

    if args.validate:
        problems = validate(cases)
        groups = Counter(c["group"] for c in cases)
        print(f"用例 {len(cases)} 条，分组：{dict(groups)}")
        if problems:
            print(f"\n发现 {len(problems)} 处与库不一致：")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("全部用例与库内数据对齐")
        return 0

    if args.group:
        cases = [c for c in cases if c["group"] == args.group]

    results: dict[str, dict] = {}
    for case in cases:
        results[case["query"]] = {
            "prefetch": run_path(case, "prefetch"),
            "tool": run_path(case, "tool"),
        }

    print(render(cases, results))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"query": c["query"], "group": c["group"], **results[c["query"]]} for c in cases
        ]
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
