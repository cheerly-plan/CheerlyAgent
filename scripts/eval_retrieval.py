"""检索质量回归评测：黄金问答 → hit@k。

背景：embedding 底座变更、检索策略调整（混合检索、重排开关、图谱融合）
都可能悄悄改变召回排序。人眼抽查靠不住，这里用一组「query → 必须命中
的原文子串」当回归用例：每次改检索相关代码后跑一遍，hit@3 掉了就拦住。

用例存放在 scripts/retrieval_eval_cases.json，与库内真实切片对应；
知识库内容变化（增删文件）后需要同步维护用例集。

用法：
    uv run python scripts/eval_retrieval.py            # 全量评测（默认关重排，纯检索质量）
    uv run python scripts/eval_retrieval.py --rerank   # 带上 LLM 重排（需模型可用，较慢）
    uv run python scripts/eval_retrieval.py --dump     # 列出库内切片，便于编写用例
    uv run python scripts/eval_retrieval.py --top-k 10 # 调大观察窗口

退出码：hit@3 低于阈值（默认 0.85）返回 1，可接入 CI。
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


def dump_chunks() -> None:
    """列出库内切片（文件名 / seq / 位置 / 开头），编写用例时对照用。"""
    with store._get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT f.filename, c.seq, c.page, c.kind, left(c.text, 90) AS head
            FROM agent_kb_chunk c JOIN agent_kb_file f ON f.id = c.file_id
            ORDER BY f.filename, c.seq
            """
        ).fetchall()
    print(f"共 {len(rows)} 个切片")
    for row in rows:
        head = (row["head"] or "").replace("\n", " ")
        print(f"{row['filename']} #{row['seq']} ({row['kind']}, p{row['page']}) | {head}")


def evaluate(cases: list[dict], top_k: int, rerank: bool) -> int:
    # 延迟导入：--dump 不需要碰 embedding/LLM
    from ai_agent.kb import retrieval
    from ai_agent.memory.embeddings import embed_texts

    if not rerank:
        # 重排引入模型输出的随机性，默认关掉，评的是纯检索策略质量
        settings.kb_rerank_enabled = False
    else:
        # 预热连接与模型路径，避免首例冷启动拉高波动
        embed_texts(["预热"])

    n1 = n3 = nk = 0
    failures: list[str] = []
    print(f"{'query':<28} hit@  命中位置")
    print("-" * 64)
    for case in cases:
        expect = case["expect"]
        hits = retrieval.search_knowledge(case["query"], top_k=top_k)

        rank: int | None = None
        for i, hit in enumerate(hits, start=1):
            text_ok = expect["text"] in (hit.text or "")
            file_ok = expect["filename"] in (hit.filename or "")
            if text_ok and file_ok:
                rank = i
                break

        at1 = rank == 1
        at3 = rank is not None and rank <= 3
        atk = rank is not None
        n1 += at1
        n3 += at3
        nk += atk
        shown = f"#{rank}" if rank is not None else "miss"
        print(f"{case['query']:<28} {'✓' if atk else '✗'}   {shown}/{len(hits)}")
        if not at3:
            failures.append(case["query"])

    total = len(cases)
    print("-" * 64)
    print(
        f"用例 {total} 条 | hit@1 {n1}/{total} ({n1 / total:.0%}) | "
        f"hit@3 {n3}/{total} ({n3 / total:.0%}) | hit@{top_k} {nk}/{total} ({nk / total:.0%})"
    )
    if failures:
        print(f"hit@3 未通过：{'、'.join(failures)}")
    ok = n3 / total >= HIT3_THRESHOLD
    print(f"阈值 hit@3 ≥ {HIT3_THRESHOLD:.0%}：{'通过' if ok else '未通过'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库检索回归评测")
    parser.add_argument("--dump", action="store_true", help="列出库内切片后退出")
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
    return evaluate(cases, args.top_k, args.rerank)


if __name__ == "__main__":
    raise SystemExit(main())
