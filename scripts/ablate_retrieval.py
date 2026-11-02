"""检索链路消融实验：逐个关掉一个处理阶段，用黄金集量化它值多少分。

为什么要这个脚本：检索链路有十几个阶段、七八个手调系数，每次改动都是
「某条 case 好了、另一条坏了」的口舌之争。这里给每一步贴上价格标签：

    Δhit@1 / Δhit@3 / Δhit@10 / ΔMRR / 变差条数 / 变好条数

读法：
- Δhit@3 为负 → 关掉后掉分，这步有用，别删；
- Δhit@3 为 0 且变差=变好=0 → 这步在当前用例上**没有信号**。
  注意"没信号"不等于"没用"：可能是用例覆盖不到它的触发条件
  （多轮改写就是典型——单轮黄金集永远测不出多轮能力），
  结论应该是「补用例」而不是「删代码」。

关阶段的方式：
- 后处理函数：替换成恒等函数（返回第 N 个位置参数）；
- 配置开关：直接改 settings，跑完还原。
都不落盘、不改数据库，跑坏了对仓库零影响。

用法：
    uv run python scripts/ablate_retrieval.py
    uv run python scripts/ablate_retrieval.py --cases scripts/retrieval_eval_cases.json
    uv run python scripts/ablate_retrieval.py --subject-bucket   # 只跑 subject 分桶对照
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.eval_retrieval as ev  # noqa: E402
import ai_agent.kb.retrieval as R  # noqa: E402
import ai_agent.kb.scope as S  # noqa: E402
import ai_agent.tools.kb_search as KS  # noqa: E402
from ai_agent.core.config import settings  # noqa: E402
from ai_agent.kb import store as KBS  # noqa: E402

DEFAULT_CASES = Path(__file__).resolve().parent / "retrieval_eval_cases_corpus.json"

# 名字 -> (模块, 函数名, 恒等函数要返回第几个位置参数)
PATCHES: dict[str, tuple] = {
    "短词合并": (R, "_apply_short_phrases", 1),
    "df 稀有词过滤": (R, "_keep_distinctive_keywords", 0),
    "口径偏好 1.06/0.94": (R, "_apply_scope_preference", 1),
    "指标名加成 1.12": (R, "_boost_indicator_hits", 1),
    "去冗余 dedupe": (R, "_dedupe", 0),
    "二次配额 diversify": (R, "_diversify", 1),
    "主体名剥离": (KS, "_strip_scope_subject", 0),
    "报告期收窄": (S, "_narrow_by_period", 1),
}

# 名字 -> (settings 字段名, 关掉时设成什么)
SETTINGS: dict[str, tuple[str, object]] = {
    "混合检索(关键词路)": ("kb_hybrid_enabled", False),
    "文档路由": ("kb_doc_route_enabled", False),
    "多轮改写": ("kb_rewrite_enabled", False),
}


def _identity(param_index: int):
    """返回「返回第 param_index 个位置参数」的恒等函数工厂。"""
    def make(orig):
        def f(*a, **kw):
            return a[param_index]
        f.__name__ = orig.__name__
        return f
    return make


def run_once(cases: list[dict], top_k: int) -> tuple[dict, list]:
    """跑一轮评测。清缓存是为了不让上一轮的进程内缓存影响下一轮。"""
    ev._clear_retrieval_caches()
    return ev.run_evaluation(cases, top_k, False, filters_on=False)


def _delta_lines(base_rank: dict[str, int | None], detail: list) -> tuple[int, int]:
    """相对基线统计 (变差条数, 变好条数)。rank 变大或变 miss 算变差。"""
    worse = better = 0
    for query, rank, _atk, _f in detail:
        old = base_rank.get(query)
        if rank == old:
            continue
        if rank is None or (old is not None and rank > old):
            worse += 1
        else:
            better += 1
    return worse, better


def _print_row(label: str, m: dict, base: dict | None) -> None:
    if base is None:
        print(
            f"{label:<22} {m['hit1']:>7.0%} {m['hit3']:>7.0%} {m['hitk']:>7.0%} "
            f"{m['mrr']:>7.3f} {'-':>8} {'-':>8} {'-':>5} {'-':>5}"
        )
        return
    print(
        f"{label:<22} {m['hit1']:>7.0%} {m['hit3']:>7.0%} {m['hitk']:>7.0%} "
        f"{m['mrr']:>7.3f} {m['hit3'] - base['hit3']:>+8.1%} "
        f"{m['mrr'] - base['mrr']:>+8.3f}"
    )


def run_ablation(cases: list[dict], top_k: int) -> int:
    base_m, base_d = run_once(cases, top_k)
    base_rank = {q: r for q, r, _, _ in base_d}
    rows: list[tuple[str, dict, int, int]] = []

    for label, (module, name, idx) in PATCHES.items():
        orig = getattr(module, name)
        setattr(module, name, _identity(idx)(orig))
        try:
            m, d = run_once(cases, top_k)
        finally:
            setattr(module, name, orig)
        worse, better = _delta_lines(base_rank, d)
        rows.append((label, m, worse, better))

    for label, (key, value) in SETTINGS.items():
        orig = getattr(settings, key)
        try:
            setattr(settings, key, value)
            m, d = run_once(cases, top_k)
        except Exception as exc:  # settings 不可写时不让整轮崩掉
            print(f"{label:<22} 设置失败：{exc}")
            continue
        finally:
            setattr(settings, key, orig)
        worse, better = _delta_lines(base_rank, d)
        rows.append((label, m, worse, better))

    print(f"\n用例 {len(cases)} 条，重排关闭（测纯检索质量）")
    print("-" * 86)
    print(
        f"{'关掉的步骤':<22} {'hit@1':>7} {'hit@3':>7} {'hit@10':>7} {'MRR':>7} "
        f"{'Δhit@3':>8} {'ΔMRR':>8} {'变差':>5} {'变好':>5}"
    )
    print("-" * 86)
    _print_row("基线（全开）", base_m, None)
    for label, m, worse, better in rows:
        print(
            f"{'− ' + label:<22} {m['hit1']:>7.0%} {m['hit3']:>7.0%} {m['hitk']:>7.0%} "
            f"{m['mrr']:>7.3f} {m['hit3'] - base_m['hit3']:>+8.1%} "
            f"{m['mrr'] - base_m['mrr']:>+8.3f} {worse:>5} {better:>5}"
        )
    print("-" * 86)
    print("Δhit@3 为负 = 关掉后掉分（这步有用）；为 0 且变差=变好=0 = 当前用例上无信号")
    return 0


def run_subject_bucket(cases: list[dict], top_k: int) -> int:
    """subject 分桶对照：被污染的 c.subject vs 干净的文件主体。

    只在内存里换掉 SQL 的分桶表达式与二次配额入口，不动数据库。
    当前语料全是单主体文件，file_id 分桶 ≈ 正确的主体分桶。
    """
    print(f"\nsubject 分桶对照（band={settings.kb_subject_quota_band}，重排关闭）")
    print("-" * 86)
    base_m, base_d = run_once(cases, top_k)
    base_rank = {q: r for q, r, _, _ in base_d}
    _print_row("A 现状(按 c.subject)", base_m, None)

    orig_bucket = KBS._SUBJECT_BUCKET
    # CTE ranked 里没有 c 别名，只能用 hit 输出的列名
    KBS._SUBJECT_BUCKET = "file_id::text"
    try:
        b_m, b_d = run_once(cases, top_k)
    finally:
        KBS._SUBJECT_BUCKET = orig_bucket
    _print_row("B SQL 分桶改 file_id", b_m, base_m)

    orig_div = R._diversify
    def by_file(query, rows, top_k_, *, per_file=False):
        return R._diversify_by_file(query, rows, top_k_)
    KBS._SUBJECT_BUCKET = "file_id::text"
    R._diversify = by_file
    try:
        c_m, c_d = run_once(cases, top_k)
    finally:
        KBS._SUBJECT_BUCKET = orig_bucket
        R._diversify = orig_div
    _print_row("C SQL+二次配额按 file", c_m, base_m)

    print("-" * 86)
    for label, d in (("B SQL 分桶改 file_id", b_d), ("C SQL+二次配额按 file", c_d)):
        worse, better = _delta_lines(base_rank, d)
        print(f"{label}: 变差 {worse} 条 / 变好 {better} 条")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索链路消融实验")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="用例集路径")
    parser.add_argument("--top-k", type=int, default=10, help="评测窗口（至少 10）")
    parser.add_argument(
        "--subject-bucket", action="store_true", help="只跑 subject 分桶对照"
    )
    args = parser.parse_args()

    cases = ev.load_cases(args.cases)
    if args.subject_bucket:
        return run_subject_bucket(cases, args.top_k)
    run_ablation(cases, args.top_k)
    return run_subject_bucket(cases, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
