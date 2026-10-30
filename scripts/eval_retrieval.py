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

用法：
    uv run python scripts/eval_retrieval.py            # 全量评测（默认关重排，纯检索质量）
    uv run python scripts/eval_retrieval.py --validate # 只校验用例与语料是否对齐
    uv run python scripts/eval_retrieval.py --rerank   # 带上 LLM 重排（需模型可用，较慢）
    uv run python scripts/eval_retrieval.py --dump     # 列出库内切片，便于编写用例
    uv run python scripts/eval_retrieval.py --top-k 10 # 调大观察窗口

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


def load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as fh:
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


def evaluate(cases: list[dict], top_k: int, rerank: bool) -> int:
    # 延迟导入：--validate / --dump 不需要碰 embedding/LLM
    from ai_agent.kb import retrieval
    from ai_agent.memory.embeddings import embed_texts

    if not rerank:
        # 重排引入模型输出的随机性，默认关掉，评的是纯检索策略质量
        settings.kb_rerank_enabled = False
    else:
        # 预热连接与模型路径，避免首例冷启动拉高波动
        embed_texts(["预热"])

    # 观察窗口至少 10：Recall@10 / MRR@10 需要看到第 10 名，而 hit@1/3/5
    # 只取前几条。一次检索拿满窗口，指标在本地切片算，不多花调用
    window = max(top_k, 10)
    n1 = n3 = nk = n10 = 0
    rr = 0.0
    failures: list[str] = []
    print(f"{'query':<30} hit@  命中位置")
    print("-" * 70)
    for case in cases:
        hits = retrieval.search_knowledge(case["query"], top_k=window)
        rank = hit_rank(hits, case["expect"])

        at1 = rank == 1
        at3 = rank is not None and rank <= 3
        atk = rank is not None and rank <= top_k
        at10 = rank is not None and rank <= 10
        n1 += at1
        n3 += at3
        nk += atk
        n10 += at10
        # 跨多条证据时 rank 是"最后一条凑齐"的位置，用它算 MRR 与
        # "证据全部可见"的体验一致（只要有片段没进窗口，模型就答不全）
        rr += 1.0 / rank if rank else 0.0
        shown = f"#{rank}" if rank is not None else "miss"
        print(f"{case['query'][:28]:<30} {'✓' if atk else '✗'}   {shown}/{len(hits)}")
        if not at3:
            failures.append(case["query"])

    total = len(cases)
    print("-" * 70)
    print(
        f"用例 {total} 条 | hit@1 {n1}/{total} ({n1 / total:.0%}) | "
        f"hit@3 {n3}/{total} ({n3 / total:.0%}) | hit@{top_k} {nk}/{total} ({nk / total:.0%})"
        f" | recall@10 {n10}/{total} ({n10 / total:.0%})"
    )
    # MRR 比 hit@k 更细：同样"都命中了"，排第 1 和第 5 的差别要能看出来，
    # 否则调参到最后几名来回挪，hit@k 一动不动
    print(f"MRR@10 {rr / total:.3f}")
    if failures:
        print(f"hit@3 未通过：{'、'.join(failures)}")
    ok = n3 / total >= HIT3_THRESHOLD
    print(f"阈值 hit@3 ≥ {HIT3_THRESHOLD:.0%}：{'通过' if ok else '未通过'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库检索回归评测")
    parser.add_argument("--dump", action="store_true", help="列出库内切片后退出")
    parser.add_argument("--validate", action="store_true", help="只校验用例与语料是否对齐")
    parser.add_argument("--skip-invalid", action="store_true", help="跳过失效用例继续评测")
    parser.add_argument("--rerank", action="store_true", help="启用 LLM 重排（默认关闭）")
    parser.add_argument("--top-k", type=int, default=5, help="观察窗口大小（默认 5）")
    args = parser.parse_args()

    if args.dump:
        dump_chunks()
        return 0

    cases = load_cases()
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

    return evaluate(cases, args.top_k, args.rerank)


if __name__ == "__main__":
    # 显式关连接池：留着让 GC 收尾会在退出时打印
    # "couldn't stop thread 'pool-1-scheduler' within 5.0 seconds"
    try:
        code = main()
    finally:
        store.close()
    raise SystemExit(code)
