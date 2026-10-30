"""按当前策略重算全部切片的向量（幂等，可反复运行）。

用途：切片正文没变、但嵌入策略升级后（例如加入「文档名 + 页码」语境前缀），
存量切片的向量仍是旧策略算出来的，与新建切片不同源，检索会偏心。
重跑本脚本即可对齐。

用法：
    uv run python scripts/reembed_chunks.py --dry-run   # 只看会处理多少条
    uv run python scripts/reembed_chunks.py             # 真正写库
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.core.config import settings  # noqa: E402
from ai_agent.core.logging import get_logger  # noqa: E402
from ai_agent.kb.chunking import embedding_input  # noqa: E402
from ai_agent.memory.embeddings import embed_texts  # noqa: E402
from ai_agent.memory.store import store  # noqa: E402

logger = get_logger("reembed")


def main() -> int:
    # os._exit 不会 flush 缓冲区，这里改成行缓冲
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with store._get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.text, c.page, c.kind, c.section, f.filename
            FROM agent_kb_chunk c
            JOIN agent_kb_file f ON f.id = c.file_id
            WHERE f.status = 'ready'
            ORDER BY c.file_id, c.seq
            """
        ).fetchall()

    print(f"待重嵌切片：{len(rows)} 条")
    if args.dry_run or not rows:
        return 0

    batch = max(1, settings.kb_embedding_batch)
    done = 0
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        vectors = embed_texts(
            [
                embedding_input(
                    row["text"],
                    filename=row["filename"] or "",
                    page=row["page"],
                    kind=row["kind"],
                    # 章节必须一起传：写入路径（pipeline）带了它，
                    # 这里漏传就等于同一片文本算出两套向量
                    section=row["section"],
                )
                for row in part
            ]
        )
        with store._get_pool().connection() as conn:
            for row, vector in zip(part, vectors):
                store.update_kb_chunk_text(row["id"], row["text"], vector)
        done += len(part)
        print(f"  {done}/{len(rows)}")
    print("完成")
    return 0


if __name__ == "__main__":
    import os

    code = main()
    # 线程池/连接池的关闭日志噪音大，直接退出
    os._exit(code)
