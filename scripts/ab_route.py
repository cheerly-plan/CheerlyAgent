"""文档路由 A/B：把「收口」和「剥公司名」两个因子隔离开各跑一轮。

为什么要这个脚本：路由改成「点名即收口」之后，hit@3 掉了 1 条。但收口会连带
打开「剥公司名」（`_named_scope` 判据是"范围里每份文件都被点名"），两件事
是耦合的——只看到合并后的数字，说不清掉的那条是谁的锅。这里把因子拆开：

    old      点名 + 摘要补足（改前），范围里混着别家 → 不剥公司名
    nostrip  点名即收口，但**不剥**公司名      → 隔离「收口」
    new      点名即收口 + 剥公司名（改后）      → 收口 + 剥名
    fix      收口，且只在范围里 ≥2 个主体时剥名 → 已验证：更差，保留作反例
    fix2     收口，且只在范围里 ≥2 份文件时剥名 → 已验证：更差，保留作反例

结论（72 条语料、重排关、top_k=10）：

    old      hit@3 65/72  hit@1 50  MRR 0.794
    nostrip  hit@3 64/72  hit@1 47  MRR 0.773   ← 单独收口是净损失
    new      hit@3 64/72  hit@1 51  MRR 0.807   ← 剥名是收口的补丁

两件事必须成对上。拆开任何一半都比改前差。

用法：
    uv run python scripts/ab_route.py                  # 跑 old/nostrip/new 三档
    uv run python scripts/ab_route.py --modes old new  # 指定档位
    uv run python scripts/ab_route.py --diff           # 额外打印逐条名次变化
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.eval_retrieval as ev  # noqa: E402
import ai_agent.kb.router as RT  # noqa: E402
import ai_agent.tools.kb_search as KS  # noqa: E402
from ai_agent.core.config import settings  # noqa: E402
from ai_agent.kb.meta import query_period_policy  # noqa: E402
from ai_agent.memory.store import store  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "retrieval_eval_cases_corpus.json"
TOP_K = 10

_new_named = KS._named_scope
_new_route = RT.route_files


def named_always(query, files, parsed_ids) -> bool:
    """无条件剥名——只用于验证「收益来自剥名还是来自收口」。"""
    return True


def _subjects(file_ids) -> int:
    """范围里有几个不同主体（同一家的两年报算一个）。"""
    by_id = {
        str(r.get("file_id")): str(r.get("subject") or "").strip()
        for r in store.list_kb_files()
    }
    return len(
        {by_id.get(str(f), "") for f in (file_ids or []) if by_id.get(str(f), "")}
    )


def old_route(query, *, file_ids=None):
    """改前版本：点名 + 摘要补足，只有多主体/多期对比时才收口。"""
    if not settings.kb_doc_route_enabled:
        return []
    text = str(query or "").strip()
    if not text:
        return []
    top_k = max(1, settings.kb_doc_route_top_k)
    picked = RT.mentioned_file_ids(text)
    if file_ids:
        allowed = set(file_ids)
        picked = [f for f in picked if f in allowed]
    policy, _token = query_period_policy(text)
    if picked and policy in {"all", "span_q", "span_h"}:
        return picked
    if picked and _subjects(picked) >= 2:
        return picked
    if len(picked) >= top_k:
        return picked[:top_k]
    from ai_agent.memory.embeddings import embed_query

    vector = embed_query(text)
    rows = store.search_kb_files(
        vector,
        file_ids=list(file_ids) if file_ids else None,
        top_k=top_k,
        max_distance=settings.kb_doc_route_max_distance,
    )
    seen = set(picked)
    for row in rows:
        if row["file_id"] in seen:
            continue
        seen.add(row["file_id"])
        picked.append(row["file_id"])
        if len(picked) >= top_k:
            break
    return picked


def named_never(query, files, parsed_ids) -> bool:
    """收口后完全不剥公司名——用来隔离「收口」这一个因子。"""
    return False


def named_multi_subject(query, files, parsed_ids) -> bool:
    """只有范围里不止一家主体时才剥。已验证比现状差，保留作反例。"""
    return bool(_new_named(query, files, parsed_ids)) and _subjects(parsed_ids) >= 2


def named_multi_file(query, files, parsed_ids) -> bool:
    """只有范围里不止一份文件时才剥。已验证比现状差，保留作反例。"""
    return bool(_new_named(query, files, parsed_ids)) and len(parsed_ids or []) >= 2


def apply_mode(mode: str) -> None:
    # 多档跑在同一进程时必须先复位，否则上一档的补丁会漏到下一档
    RT.route_files = _new_route
    KS._named_scope = _new_named
    if mode == "old":
        RT.route_files = old_route
    elif mode == "oldstrip":
        RT.route_files = old_route
        KS._named_scope = named_always
    elif mode == "nostrip":
        KS._named_scope = named_never
    elif mode == "fix":
        KS._named_scope = named_multi_subject
    elif mode == "fix2":
        KS._named_scope = named_multi_file
    elif mode != "new":
        raise SystemExit(f"未知档位: {mode}")


def run(mode: str) -> tuple[dict, dict]:
    apply_mode(mode)
    settings.kb_rerank_enabled = False
    settings.kb_multiquery_enabled = False
    ev._clear_retrieval_caches()
    cases = ev.load_cases(str(CASES_PATH))
    metrics, detail = ev.run_evaluation(cases, TOP_K, False, filters_on=False)
    ranks = {q: r for q, r, _atk, _f in detail}
    print(
        f"{mode:<8} hit@3={metrics['n3']}/{metrics['total']} ({metrics['hit3']:.1%})"
        f"  hit@1={metrics['n1']:<3} MRR={metrics['mrr']:.3f}"
        f"  recall@10={metrics['recall10']:.1%}"
    )
    return metrics, ranks


def show_diff(a: dict, b: dict, label: str) -> None:
    changed = [(q, a[q], b[q]) for q in a if a[q] != b[q]]
    changed.sort(key=lambda t: (t[1] or 99) - (t[2] or 99))
    print(f"\n=== {label}（{len(changed)} 条变） ===")
    for q, x, y in changed:
        tag = "变好" if (y or 99) < (x or 99) else "变差"
        print(f"  {tag} #{x} → #{y}   {q[:46]}")
    if not changed:
        print("  （无变化）")


def main() -> int:
    parser = argparse.ArgumentParser(description="文档路由收口 A/B")
    parser.add_argument("--modes", nargs="+", default=["old", "nostrip", "new"])
    parser.add_argument("--diff", action="store_true", help="打印逐条名次变化")
    args = parser.parse_args()

    results = {m: run(m) for m in args.modes}
    if args.diff:
        for i in range(len(args.modes) - 1):
            a, b = args.modes[i], args.modes[i + 1]
            show_diff(results[a][1], results[b][1], f"{a} → {b}")
    json_path = Path("/tmp/ab_route_result.json")
    json_path.write_text(
        json.dumps({m: r[1] for m, r in results.items()}, ensure_ascii=False, indent=1)
    )
    print(f"\n逐条名次已写入 {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
