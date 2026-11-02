"""检索质量回归评测：黄金问答 → hit@k。

背景：embedding 底座变更、检索策略调整（混合检索、重排开关、图谱融合）
都可能悄悄改变召回排序。人眼抽查靠不住，这里用一组「query → 必须命中的
原文子串」当回归用例：每次改检索相关代码后跑一遍，hit@3 掉了就拦住。

用例存放在 scripts/retrieval_eval_cases.json，与库内真实切片对应；
知识库内容变化（增删文件、重新切片）后需要同步维护用例集——用例失效
不再伪装成"检索 miss"，由校验环节直接指出。

用例的证据写法（expect 两种形态）：
    {"expect": {"filename": "某文档", "text": "原文子串"}}          单段证据
    {"expect": {"filename": "某文档", "text_all": ["片段A", "片段B"]}} 同一 slice 内多段
    {"expect": {"sources": [{"filename": "文档A", "text_all": [...]},
                            {"filename": "文档B", "text": "..."}]}}   跨文件多段证据

注意：text 必须是**切片原文的子串**，按 markdown 原文照抄，不要自己
用「；」把两句拼起来——拼接串永远不会出现在单个切片里，那条用例从
写下的那一刻就不可能命中。

指标：hit@1 / hit@3 / hit@k 看"能不能查到"，recall@10 看"证据够不够
模型答全"，MRR@10 看"排得多靠前"——三档都命中时靠 MRR 分辨优劣。

评测走对话同一条检索（kb_search.retrieve_for_question）：问句先圈定文件，
去掉公司名，再在这些文件里查。不再把原句直接丢进全库。

用法：
    uv run python scripts/eval_retrieval.py            # 全量评测（默认关重排，纯检索质量）
    uv run python scripts/eval_retrieval.py --validate # 只校验用例与语料是否对齐
    uv run python scripts/eval_retrieval.py --rerank   # 带上 LLM 重排（需模型可用，较慢）
    uv run python scripts/eval_retrieval.py --dump     # 列出库内切片，便于编写用例
    uv run python scripts/eval_retrieval.py --top-k 10 # 调大观察窗口
    uv run python scripts/eval_retrieval.py --cases 另一份.json   # 指定用例集（默认黄金集）
    uv run python scripts/eval_retrieval.py --filters  # 开启 chunk 级元数据过滤（模拟生产传参）后评测
    uv run python scripts/eval_retrieval.py --compare-filters     # 过滤关/开各跑一轮，对照 hit@k/MRR@10
    uv run python scripts/eval_retrieval.py --stability --rerank   # 同一 query 跑 N 次，量化同问稳定性
    uv run python scripts/eval_retrieval.py --stability --rerank --no-cache   # 每次清缓存，逼出重排/嵌入真实抖动


退出码：0 通过；1 hit@3 低于阈值（默认 0.85）；2 用例失效（语料已变）。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.core.config import settings  # noqa: E402
from ai_agent.memory.store import store  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "retrieval_eval_cases.json"
# hit@3 低于这条线视为回归：黄金用例都是"应该能查到"的问题，
# 偶发一两条边缘 miss 可以接受，批量掉分一定是策略或向量出了问题
HIT3_THRESHOLD = 0.85


def load_cases(path: str | None = None) -> list[dict]:
    p = Path(path) if path else CASES_PATH
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def fetch_chunks() -> list[dict]:
    """库里全部切片（filename + text），校验与命中判定都基于它。"""
    with store._get_pool().connection() as conn:
        return conn.execute(
            """
            SELECT f.filename, c.seq, c.page, c.kind, c.text
            FROM agent_kb_chunk c JOIN agent_kb_file f ON f.id = c.file_id
            ORDER BY f.filename, c.seq
            """
        ).fetchall()


def normalize_expect(expect: dict) -> list[dict]:
    """把用例的 expect 规整成 [{"filename", "texts"}]，兼容三种写法。

    统一成列表后，命中判定只需处理"多条证据是否都出现"这一种情况：
    单段证据就是长度为 1 的列表，跨文件证据就是多项。
    """
    if "sources" in expect:
        sources = []
        for src in expect["sources"]:
            texts = src.get("text_all") or [src["text"]]
            sources.append({"filename": src["filename"], "texts": texts})
        return sources
    texts = expect.get("text_all") or [expect["text"]]
    return [{"filename": expect["filename"], "texts": texts}]


def validate_cases(cases: list[dict], rows: list[dict]) -> list[tuple[str, str]]:
    """校验用例与当前语料是否对齐，返回 (query, 原因) 列表。

    两类失效：期望的文件已经不在库里、期望的片段不在该文件的任何切片中。
    后者几乎都是同一类错误——拿两句话拼成一个期望串，而切片里没有这个拼接。
    """
    problems: list[tuple[str, str]] = []
    for case in cases:
        for src in normalize_expect(case["expect"]):
            same_file = [r for r in rows if src["filename"] in (r["filename"] or "")]
            if not same_file:
                problems.append((case["query"], f"文件不在库里：{src['filename']}"))
                continue
            for text in src["texts"]:
                if not any(text in (r["text"] or "") for r in same_file):
                    hint = "（拼接串？）" if "；" in text else ""
                    problems.append(
                        (
                            case["query"],
                            f"片段不在《{src['filename']}》任何切片里{hint}：{text[:40]}",
                        )
                    )
    return problems


def report_invalid(problems: list[tuple[str, str]]) -> None:
    print(f"⚠️  {len(problems)} 处用例与语料对不上：")
    seen: set[str] = set()
    for query, why in problems:
        label = query if query not in seen else ""
        seen.add(query)
        print(f"  {'✗ ' + label[:26] if label else ' ' * 28} {why}")


def dump_chunks() -> None:
    """列出库内切片（文件名 / seq / 位置 / 开头），编写用例时对照用。"""
    rows = fetch_chunks()
    print(f"共 {len(rows)} 个切片")
    for row in rows:
        head = (row["text"] or "").replace("\n", " ")
        print(f"{row['filename']} #{row['seq']} ({row['kind']}, p{row['page']}) | {head[:90]}")


def hit_rank(hits: list, expect: dict) -> int | None:
    """返回证据凑齐的名次；凑不齐返回 None。

    跨多条证据时，"凑齐"的名次取最后一条命中的位置——比如第一段在第 1 条、
    第二段在第 3 条，就算 hit@3 命中，这与模型实际拿到的上下文一致。
    """
    sources = normalize_expect(expect)
    done = [False] * len(sources)
    for rank, hit in enumerate(hits, start=1):
        text = hit.text or ""
        for i, src in enumerate(sources):
            if done[i] or src["filename"] not in (hit.filename or ""):
                continue
            if all(piece in text for piece in src["texts"]):
                done[i] = True
        if all(done):
            return rank
    return None


def infer_filters(query: str, case: dict) -> dict | None:
    """镜像生产 kb_search._kb_search 的「元数据过滤」推断（不含 file/query 改写）。

    生产里过滤由「模型显式传参」+「query 意图推断」两部分构成；这里用 golden
    用例自带的 ``filters`` 字段模拟「模型对该问法正确传参」（最贴近真实调用），
    再叠加 query 命中 ``_PARENT_INTENT``（母公司/单体口径）时的 scopes 硬过滤，
    与 retrieval/kb_search 的实际行为一致。返回 None 即「不加任何过滤」，与
    「过滤关」等价。

    黄金集按当前语料维护；filters 字段只写在「答案确实落在对应类型切片」的用例上
    （如表格指标 → kinds=["table"]），避免把无关过滤强加给检索、伪造精度提升。
    """
    from ai_agent.kb.retrieval import _PARENT_INTENT, SCOPE_PARENT

    filters: dict = {}
    explicit = case.get("filters")
    if isinstance(explicit, dict):
        for key in ("kinds", "sections", "scopes", "subjects", "periods"):
            val = explicit.get(key)
            if val:
                filters[key] = list(val)
    if "scopes" not in filters and _PARENT_INTENT.search(query or ""):
        filters["scopes"] = [SCOPE_PARENT]
    return filters or None


def run_evaluation(
    cases: list[dict],
    top_k: int,
    rerank: bool,
    filters_on: bool = False,
) -> tuple[dict, list[tuple]]:
    """执行一轮评测，返回 (指标字典, 逐条明细)。

    抽成独立函数以便「过滤关/开」各跑一轮做对照。关重排时关掉 rerank 评纯
    检索；打开时预热避免首例冷启动。窗口至少 10 以计算 recall@10/MRR@10。
    查询向量走进程内 lru_cache，对照时「开」轮不会重复付 embedding。
    """
    from ai_agent.memory.embeddings import embed_texts
    from ai_agent.tools.kb_search import retrieve_for_question

    if not rerank:
        settings.kb_rerank_enabled = False
    else:
        embed_texts(["预热"])

    window = max(top_k, 10)
    n1 = n3 = nk = n10 = 0
    rr = 0.0
    applied = 0
    detail: list[tuple] = []
    for case in cases:
        filters = infer_filters(case["query"], case) if filters_on else None
        if filters:
            applied += 1
        _ids, hits = retrieve_for_question(
            case["query"], top_k=window, extra_filters=filters
        )
        rank = hit_rank(hits, case["expect"])
        at1 = rank == 1
        at3 = rank is not None and rank <= 3
        atk = rank is not None and rank <= top_k
        at10 = rank is not None and rank <= 10
        n1 += at1
        n3 += at3
        nk += atk
        n10 += at10
        rr += 1.0 / rank if rank else 0.0
        detail.append((case["query"], rank, atk, filters))
    total = len(cases)
    metrics = {
        "total": total,
        "n1": n1, "n3": n3, "nk": nk, "n10": n10,
        "hit1": n1 / total, "hit3": n3 / total, "hitk": nk / total,
        "recall10": n10 / total, "mrr": rr / total, "applied": applied,
    }
    return metrics, detail


def evaluate(
    cases: list[dict],
    top_k: int,
    rerank: bool,
    filters_on: bool = False,
) -> int:
    # 延迟导入：--validate / --dump 不需要碰 embedding/LLM
    metrics, detail = run_evaluation(
        cases, top_k, rerank, filters_on=filters_on
    )
    n1 = metrics["n1"]; n3 = metrics["n3"]; nk = metrics["nk"]; n10 = metrics["n10"]
    total = metrics["total"]
    failures: list[str] = []
    print(f"{'query':<30} hit@  命中位置")
    print("-" * 70)
    for query, rank, atk, _ in detail:
        shown = f"#{rank}" if rank is not None else "miss"
        print(f"{query[:28]:<30} {'✓' if atk else '✗'}   {shown}")
        if rank is None or rank > 3:
            failures.append(query)
    print("-" * 70)
    print(
        f"用例 {total} 条 | hit@1 {n1}/{total} ({n1 / total:.0%}) | "
        f"hit@3 {n3}/{total} ({n3 / total:.0%}) | hit@{top_k} {nk}/{total} ({nk / total:.0%})"
        f" | recall@10 {n10}/{total} ({n10 / total:.0%})"
    )
    # MRR 比 hit@k 更细：同样"都命中了"，排第 1 和第 5 的差别要能看出来，
    # 否则调参到最后几名来回挪，hit@k 一动不动
    print(f"MRR@10 {metrics['mrr']:.3f}")
    if failures:
        print(f"hit@3 未通过：{'、'.join(failures)}")
    ok = metrics["hit3"] >= HIT3_THRESHOLD
    print(f"阈值 hit@3 ≥ {HIT3_THRESHOLD:.0%}：{'通过' if ok else '未通过'}")
    return 0 if ok else 1


def compare_filters(cases: list[dict], top_k: int, rerank: bool) -> int:
    """过滤关 vs 过滤开 对照：hit@k / MRR@10 与逐条名次差。

    用同一批用例各跑一轮（查询向量进程内缓存，开轮不重复付 embedding），输出
    关/开的指标对照，并列出名次发生变化的用例，便于定位「精度提升来自哪条 / 是否
    丢了召回」。过滤开应用的是各用例的 filters 字段（模拟模型正确传参）+ 母公司
    口径自动推断，与生产行为一致。
    """
    off_m, off_d = run_evaluation(cases, top_k, rerank, filters_on=False)
    on_m, on_d = run_evaluation(cases, top_k, rerank, filters_on=True)
    t = off_m["total"]
    print(f"过滤对照（top_k={top_k}, 用例={t}, 过滤开命中 {on_m['applied']}/{t} 条应用了元数据过滤）")
    print(f"{'指标':<12}{'关':>12}{'开':>12}{'Δ':>10}")
    print("-" * 46)
    rows = [
        ("hit@1", off_m["hit1"], on_m["hit1"]),
        ("hit@3", off_m["hit3"], on_m["hit3"]),
        (f"hit@{top_k}", off_m["hitk"], on_m["hitk"]),
        ("recall@10", off_m["recall10"], on_m["recall10"]),
        ("MRR@10", off_m["mrr"], on_m["mrr"]),
    ]
    for name, a, b in rows:
        delta = b - a
        dstr = f"{delta:+.0%}" if name != "MRR@10" else f"{delta:+.3f}"
        print(f"{name:<12}{a:>11.0%}{b:>11.0%}{dstr:>10}")
    # 逐条名次变化（仅列变了的，避免刷屏）
    changed = []
    for (qo, ro, _, _), (q1, r1, _, f1) in zip(off_d, on_d):
        if ro != r1:
            changed.append((qo, ro, r1, f1))
    print("-" * 46)
    if changed:
        print(f"名次变化的用例（{len(changed)}/{t}）：")
        for q, ro, r1, f1 in changed:
            so = f"#{ro}" if ro is not None else "miss"
            s1 = f"#{r1}" if r1 is not None else "miss"
            ftag = f"  filter={f1}" if f1 else ""
            print(f"  {q[:26]:<28} {so:>6} → {s1:<6}{ftag}")
    else:
        print("过滤开/关名次完全一致（过滤未改变任何召回排序）")
    # 结论提示：开不低于关即「不丢召回」；精度看 MRR / hit@k 是否提升
    if on_m["hit3"] >= off_m["hit3"] and on_m["recall10"] >= off_m["recall10"]:
        lift = "；MRR/hit@k 有提升" if (on_m["mrr"] > off_m["mrr"] or on_m["hit1"] > off_m["hit1"]) else "；精度持平"
        print(f"结论：过滤开未降低 hit@3 / recall@10（不丢召回）{lift}")
    else:
        print("⚠️ 过滤开导致 hit@3 / recall@10 下降，需排查过滤是否过窄")
    return 0


def set_jaccard(a: list[str], b: list[str]) -> float:
    """两个 top-k 序列（filename#seq）的集合重合度，1.0 表示完全一致。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def order_agreement(a: list[str], b: list[str]) -> float:
    """以 a 为基准的顺序一致性：同序=1.0，完全错位=0.0。

    只比较交集元素的位置差，比 Kendall tau 便宜，足够分辨「同问不同序」。
    """
    if not a:
        return 1.0 if not b else 0.0
    rank = {k: i for i, k in enumerate(a)}
    if not b:
        return 0.0
    common = [k for k in b if k in rank]
    if len(common) < 2:
        return 1.0 if common else 0.0
    diffs = [abs(rank[common[i]] - i) for i in range(len(common))]
    max_diff = len(common) - 1
    return 1.0 - sum(diffs) / (max_diff * len(common))


def _clear_retrieval_caches() -> None:
    """测真实抖动时，每次运行前清掉缓存，让 embedding / 重排都走冷调用。

    否则进程内 lru_cache（embed_query）与 _rerank_cache 会在同进程重复运行
    时掩盖 LLM / 向量底座的非确定输出，使分数恒为 1.0、测不出真实抖动。
    """
    from ai_agent.kb import retrieval
    from ai_agent.memory.embeddings import _cached_query_vector

    try:
        retrieval._rerank_cache.clear()
    except Exception:
        pass
    try:
        _cached_query_vector.cache_clear()
    except Exception:
        pass


def evaluate_stability(
    cases: list[dict], top_k: int, rerank: bool, runs: int, no_cache: bool = False
) -> int:
    """稳定性评测：同一 query 重复运行 runs 次，量化 top-k 结果是否一致。

    对应「同问每次结果不同」这一不稳定表现。关重排时纯 SQL 链路确定，
    分数应≈1.0；开 --rerank 后<1.0 即暴露重排/嵌入抖动的来源。

    no_cache=True 时每次运行前清掉 embed / rerank 缓存，逼出真实冷调用抖动；
    默认 False 则复现「同进程重复同问」的生产缓存表现——两者要分开看。
    """
    from ai_agent.tools.kb_search import retrieve_for_question

    if not rerank:
        settings.kb_rerank_enabled = False
    else:
        from ai_agent.memory.embeddings import embed_texts

        embed_texts(["预热"])

    window = max(top_k, 10)
    print(f"{'query':<26} 稳定(集) 稳定(序)  hit@3")
    print("-" * 60)
    sset_sum = sord_sum = 0.0
    n3 = 0
    for case in cases:
        runs_keys: list[list[str]] = []
        last_hits = None
        for _ in range(runs):
            if no_cache:
                # 每次走冷调用：embed / 重排都重新请求，暴露真实抖动
                _clear_retrieval_caches()
            _ids, last_hits = retrieve_for_question(case["query"], top_k=window)
            runs_keys.append([f"{h.filename}#{h.seq}" for h in last_hits])
        base = runs_keys[0]
        jaccards = [set_jaccard(base, ks) for ks in runs_keys[1:]]
        orders = [order_agreement(base, ks) for ks in runs_keys[1:]]
        sset = sum(jaccards) / len(jaccards) if jaccards else 1.0
        sord = sum(orders) / len(orders) if orders else 1.0
        rank = hit_rank(last_hits, case["expect"])
        at3 = rank is not None and rank <= 3
        n3 += at3
        sset_sum += sset
        sord_sum += sord
        print(
            f"{case['query'][:24]:<26} {sset:.2f}     {sord:.2f}     "
            f"{'✓' if at3 else '✗'}"
        )
    total = len(cases)
    print("-" * 60)
    print(
        f"平均稳定性(集合) {sset_sum / total:.3f} | 平均稳定性(顺序) "
        f"{sord_sum / total:.3f} | hit@3 {n3}/{total} ({n3 / total:.0%})"
    )
    if no_cache:
        print("解读(--no-cache)：每次清缓存走冷调用，分数<1.0 即真实暴露重排/嵌入抖动")
    else:
        print("解读：关重排应≈1.0（纯SQL确定）；开 --rerank 后<1.0 即暴露重排/嵌入抖动。"
              "注：未加 --no-cache 时同进程重复命中 _rerank 缓存，可能掩盖抖动")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库检索回归评测")
    parser.add_argument("--dump", action="store_true", help="列出库内切片后退出")
    parser.add_argument("--validate", action="store_true", help="只校验用例与语料是否对齐")
    parser.add_argument("--skip-invalid", action="store_true", help="跳过失效用例继续评测")
    parser.add_argument("--rerank", action="store_true", help="启用 LLM 重排（默认关闭）")
    parser.add_argument("--top-k", type=int, default=5, help="观察窗口大小（默认 5）")
    parser.add_argument(
        "--cases",
        type=str,
        default=str(CASES_PATH),
        help="用例集路径（默认黄金集 retrieval_eval_cases.json）",
    )
    parser.add_argument(
        "--filters",
        action="store_true",
        help="开启 chunk 级元数据过滤（模拟生产传参）后再评测",
    )
    parser.add_argument(
        "--compare-filters",
        action="store_true",
        help="同时跑过滤关/开两轮，输出 hit@k/MRR@10 对照与逐条名次差",
    )
    parser.add_argument(
        "--stability",
        action="store_true",
        help="同一 query 重复运行度量 top-k 稳定性（配合 --rerank 暴露抖动）",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="--stability 下每次运行前清掉 embed/rerank 缓存，逼出真实冷调用抖动",
    )
    parser.add_argument("--runs", type=int, default=3, help="--stability 重复运行次数")
    args = parser.parse_args()

    if args.dump:
        dump_chunks()
        return 0

    cases = load_cases(args.cases)
    if not cases:
        print("用例集为空", file=sys.stderr)
        return 1

    # 每次评测前先对齐语料：语料变了（换文档、重新切片）用例会静默失效，
    # 那种 miss 不是检索的问题，必须先修用例，否则分数没有意义
    problems = validate_cases(cases, fetch_chunks())
    if problems:
        report_invalid(problems)
        if not args.skip_invalid:
            print("请先修正用例：`--dump` 可查看切片原文；或加 --skip-invalid 暂时跳过")
            return 2
        bad = {query for query, _ in problems}
        cases = [c for c in cases if c["query"] not in bad]
        print(f"已跳过 {len(bad)} 条失效用例，剩余 {len(cases)} 条参与评测\n")

    if args.validate:
        print(f"用例校验通过：{len(cases)} 条全部与语料对齐")
        return 0

    if args.compare_filters:
        return compare_filters(cases, args.top_k, args.rerank)

    if args.stability:
        return evaluate_stability(cases, args.top_k, args.rerank, args.runs, args.no_cache)

    return evaluate(
        cases, args.top_k, args.rerank, filters_on=args.filters
    )


if __name__ == "__main__":
    # 显式关连接池：留着让 GC 收尾会在退出时打印
    # "couldn't stop thread 'pool-1-scheduler' within 5.0 seconds"
    try:
        code = main()
    finally:
        store.close()
    raise SystemExit(code)
