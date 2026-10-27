"""全量重算三类向量（幂等，可反复运行）。

覆盖：agent_kb_chunk（知识库切片）、agent_episode（长期记忆）、
agent_kg_entity（图谱实体）。

用途：embedding 服务更换底座（同名模型、不同向量空间）后，存量向量
全部失效——同文本重嵌与库存向量余弦接近 0，检索完全失灵。跑一遍
本脚本即可整体对齐到当前模型。

各表的嵌入输入必须与写入路径严格一致：
  agent_kb_chunk  → embedding_input(text, filename, page, kind)  （kb/pipeline）
  agent_episode   → 原文 text                                   （episode_store）
  agent_kg_entity → "name（type）：description"                  （kb/graph）

用法：
    uv run python scripts/reembed_vectors.py --dry-run   # 只统计数量
    uv run python scripts/reembed_vectors.py             # 真正写库
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
BATCH = max(1, settings.kb_embedding_batch)


def _reembed_table(title: str, rows: list[dict], text_of, dry_run: bool) -> int:
    if not rows:
        print(f"{title}: 0 条，跳过")
        return 0
    if dry_run:
        print(f"{title}: {len(rows)} 条待重嵌")
        return 0
    done = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        vectors = embed_texts([text_of(r) for r in batch])
        with store._get_pool().connection() as conn:
            with conn.cursor() as cur:
                for row, vec in zip(batch, vectors):
                    cur.execute(
                        f"UPDATE {title} SET embedding = %s WHERE id = %s",
                        (vec, row["id"]),
                    )
        done += len(batch)
        print(f"{title}: {done}/{len(rows)}")
    print(f"{title}: 完成 {done} 条")
    return done


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with store._get_pool().connection() as conn:
        chunks = conn.execute(
            """
            SELECT c.id, c.text, c.page, c.kind, f.filename
            FROM agent_kb_chunk c
            JOIN agent_kb_file f ON f.id = c.file_id
            WHERE f.status = 'ready'
            """
        ).fetchall()
        episodes = conn.execute(
            "SELECT id, text FROM agent_episode"
        ).fetchall()
        entities = conn.execute(
            "SELECT id, name, type, description FROM agent_kg_entity"
        ).fetchall()

    total = 0
    total += _reembed_table(
        "agent_kb_chunk",
        chunks,
        lambda r: embedding_input(
            r["text"], filename=r["filename"], page=r["page"], kind=r["kind"]
        ),
        args.dry_run,
    )
    total += _reembed_table(
        "agent_episode", episodes, lambda r: r["text"], args.dry_run
    )
    total += _reembed_table(
        "agent_kg_entity",
        entities,
        lambda r: f"{r['name']}（{r['type']}）：{r['description'] or ''}",
        args.dry_run,
    )
    print(f"合计: {total} 条" + ("（dry-run，未写库）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
