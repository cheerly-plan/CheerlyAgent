"""知识库日常维护：一个入口，按子命令做事。

改了解析 / 切片之后，存量文件不会自己变。日常就这几步：

    uv run python scripts/kb.py list --name 中行
    uv run python scripts/kb.py reparse --name 中行          # 整份重解析+嵌入，图谱清掉
    uv run python scripts/kb.py reparse --name 中行 --with-graph
    uv run python scripts/kb.py graph --name 中行            # 切片已核对，只重建图谱
    uv run python scripts/kb.py reembed                     # 正文没变，只重算切片向量
    uv run python scripts/kb.py reembed --target all        # 切片 + 情景记忆 + 图谱实体
    uv run python scripts/kb.py coverage --name 中行        # 哪些页没有切片

--file / --name / --all 选文件；--dry-run 只看会动谁；--all 要加 --yes。

评测和一次性工具不在这里：eval_retrieval.py / eval_graph.py / check_kg_alignment.py。
开发启动仍是 ./scripts/dev.sh。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.core.config import settings  # noqa: E402
from ai_agent.core.logging import get_logger  # noqa: E402

logger = get_logger("kb")
_BUSY = {"parsing", "embedding"}
_WRITE_CMDS = {"reparse", "graph", "reembed"}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = _parser()
    args = parser.parse_args(argv)
    if args.handler is None:
        parser.print_help()
        return 2
    model = getattr(args, "embedding_model", "") or ""
    if model and model != settings.embedding_model:
        print(
            f"[embedding] 本次覆盖：{settings.embedding_model} → {model}"
            f"（维度需 = {settings.embedding_dim}）"
        )
        settings.embedding_model = model
    if args.cmd in _WRITE_CMDS:
        graph = settings.kb_graph_model or settings.model
        graph_url = (settings.kb_graph_base_url or "").strip() or settings.base_url
        print(
            f"[config] embedding={settings.embedding_model} "
            f"dim={settings.embedding_dim}  llm={settings.model} @ {settings.base_url}  "
            f"graph={graph} @ {graph_url}"
        )
    from ai_agent.memory.store import store  # noqa: E402

    store.ensure_schema()
    return args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/kb.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.set_defaults(handler=None, embedding_model="")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="列出知识库文件")
    p_list.add_argument("--name", help="文件名模糊匹配")
    p_list.set_defaults(handler=_cmd_list)

    p_reparse = sub.add_parser(
        "reparse", help="按当前策略重新解析+嵌入（改了表格切片后用）"
    )
    _add_select(p_reparse)
    p_reparse.add_argument(
        "--with-graph",
        action="store_true",
        help="解析完成后接着建图；默认只清图谱，方便先看切片",
    )
    p_reparse.set_defaults(handler=_cmd_reparse)

    p_graph = sub.add_parser("graph", help="只重建图谱，不重新解析")
    _add_select(p_graph)
    p_graph.set_defaults(handler=_cmd_graph)

    p_reembed = sub.add_parser("reembed", help="正文没变，只按当前策略重算向量")
    p_reembed.add_argument(
        "--target",
        choices=("chunks", "episodes", "entities", "all"),
        default="chunks",
        help="chunks=切片（默认）；all=切片+情景记忆+图谱实体",
    )
    p_reembed.add_argument("--dry-run", action="store_true")
    _add_embed_override(p_reembed)
    p_reembed.set_defaults(handler=_cmd_reembed)

    p_cov = sub.add_parser("coverage", help="看哪些页没有切片")
    p_cov.add_argument("--file", help="文件 id（可只给前缀）")
    p_cov.add_argument("--name", help="文件名模糊匹配")
    p_cov.add_argument("--full", action="store_true", help="缺页全打，不折成区间")
    p_cov.set_defaults(handler=_cmd_coverage)
    return parser


def _add_select(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file", help="文件 id（可只给前缀）")
    parser.add_argument("--name", help="文件名模糊匹配")
    parser.add_argument("--all", action="store_true", help="全部文件")
    parser.add_argument("--yes", action="store_true", help="--all 时不再确认")
    parser.add_argument("--dry-run", action="store_true")
    _add_embed_override(parser)


def _add_embed_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-model",
        default="",
        help="覆盖 embedding 模型（默认用 .env 的 EMBEDDING_MODEL，不改文件）",
    )


def _pick_files(name: str | None, file_id: str | None, all_files: bool) -> list[dict]:
    from ai_agent.memory.store import store

    with store._get_pool().connection() as conn:
        if file_id:
            rows = conn.execute(
                "SELECT id, filename, status, chunk_count, graph_status, mime, user_id "
                "FROM agent_kb_file WHERE id::text LIKE %s",
                (f"{file_id}%",),
            ).fetchall()
        elif name:
            rows = conn.execute(
                "SELECT id, filename, status, chunk_count, graph_status, mime, user_id "
                "FROM agent_kb_file WHERE filename ILIKE %s ORDER BY filename",
                (f"%{name}%",),
            ).fetchall()
        elif all_files:
            rows = conn.execute(
                "SELECT id, filename, status, chunk_count, graph_status, mime, user_id "
                "FROM agent_kb_file ORDER BY filename"
            ).fetchall()
        else:
            return []
    return [dict(r) for r in rows if r["status"] not in _BUSY]


def _require_files(args: argparse.Namespace) -> list[dict] | None:
    if not (args.file or args.name or args.all):
        print("需要 --file / --name / --all 之一")
        return None
    files = _pick_files(args.name, args.file, args.all)
    if not files:
        print("没有匹配到可处理的文件（正在解析/向量化的会跳过）")
        return None
    print(f"将处理 {len(files)} 个文件：")
    for rec in files:
        print(
            f"  {rec['id']}  {rec['filename']}  "
            f"({rec['chunk_count']} 片, 图谱 {rec['graph_status']})"
        )
    if args.all and not args.yes and not args.dry_run:
        answer = input("确认？[y/N] ").strip().lower()
        if answer != "y":
            print("已取消")
            return []
    return files


def _upload_path(rec: dict) -> Path | None:
    suffix = (rec.get("mime") or "").lower()
    directory = settings.kb_upload_dir / str(rec["id"])
    if not directory.is_dir():
        return None
    candidates = [p for p in directory.iterdir() if p.is_file() and p.name != ".gitkeep"]
    if not candidates:
        return None
    if suffix:
        for path in candidates:
            if suffix in path.name.lower() or suffix.split("/")[-1] in path.name.lower():
                return path
    return max(candidates, key=lambda p: p.stat().st_size)


def _clear_chunks_and_graph(file_id) -> None:
    from ai_agent.memory.store import store

    with store._get_pool().connection() as conn:
        conn.execute("DELETE FROM agent_kb_chunk WHERE file_id = %s", (file_id,))
        conn.execute("DELETE FROM agent_kg_relation WHERE file_id = %s", (file_id,))
        conn.execute("DELETE FROM agent_kg_entity WHERE file_id = %s", (file_id,))
        conn.execute("DELETE FROM agent_kg_mention WHERE file_id = %s", (file_id,))
    store.update_graph_status(file_id, "none", entity_count=0, batch_done=0, batch_total=0)


def _reparse_one(rec: dict, dry_run: bool) -> bool:
    from ai_agent.kb.pipeline import kb_pipeline
    from ai_agent.memory.store import store

    path = _upload_path(rec)
    print(f"  {rec['filename']}  (原件 {path.name if path else '缺失'})")
    if path is None:
        print("    ✗ 找不到原始文件，跳过")
        return False
    if dry_run:
        print("    (dry-run) 未改动")
        return False
    _clear_chunks_and_graph(rec["id"])
    print("    旧切片/旧图已清除，重新解析+嵌入中…")
    kb_pipeline._process(rec["id"], str(rec.get("user_id") or ""), path)
    with store._get_pool().connection() as conn:
        row = conn.execute(
            "SELECT status, chunk_count, error FROM agent_kb_file WHERE id = %s",
            (rec["id"],),
        ).fetchone()
        kinds = conn.execute(
            "SELECT kind, count(*) AS n FROM agent_kb_chunk "
            "WHERE file_id = %s GROUP BY kind ORDER BY n DESC",
            (rec["id"],),
        ).fetchall()
    extra = " ".join(f"{k['kind']}={k['n']}" for k in kinds)
    print(
        f"    ✓ 解析完成：status={row['status']} 切片={row['chunk_count']}"
        + (f" {extra}" if extra else "")
        + (f" err={row['error']}" if row["error"] else "")
    )
    return row["status"] == "ready"


def _graph_one(rec: dict, dry_run: bool) -> None:
    from ai_agent.kb.pipeline import kb_pipeline
    from ai_agent.memory.store import store

    print(f"  {rec['filename']}  切片 {rec['chunk_count']}  图谱 {rec['graph_status']}")
    if rec.get("graph_status") == "building":
        print("    正在建图，跳过")
        return
    if dry_run:
        print("    (dry-run) 未改动")
        return
    kb_pipeline._rebuild_graph(rec["id"])
    with store._get_pool().connection() as conn:
        row = conn.execute(
            "SELECT graph_status, graph_entity_count FROM agent_kb_file WHERE id = %s",
            (rec["id"],),
        ).fetchone()
    print(f"    ✓ 图谱 {row['graph_status']}，实体 {row['graph_entity_count']}")


def _cmd_list(args: argparse.Namespace) -> int:
    from ai_agent.memory.store import store

    with store._get_pool().connection() as conn:
        sql = (
            "SELECT f.id, f.filename, f.status, f.graph_status, "
            "count(c.seq) AS chunks, "
            "coalesce(max(coalesce(c.page_end, c.page)), 0) AS pages "
            "FROM agent_kb_file f LEFT JOIN agent_kb_chunk c ON c.file_id = f.id"
        )
        params: tuple = ()
        if args.name:
            sql += " WHERE f.filename ILIKE %s"
            params = (f"%{args.name}%",)
        sql += " GROUP BY f.id, f.filename, f.status, f.graph_status ORDER BY f.filename"
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("库里没有文件" + (f"（匹配「{args.name}」）" if args.name else ""))
        return 1
    print(f"{'file_id':<38}{'页':>6}{'切片':>7}  {'状态':<10}{'图谱':<10}文件名")
    for rec in rows:
        print(
            f"{str(rec['id']):<38}{rec['pages']:>6}{rec['chunks']:>7}  "
            f"{rec['status']:<10}{rec['graph_status']:<10}{rec['filename']}"
        )
    return 0


def _cmd_reparse(args: argparse.Namespace) -> int:
    files = _require_files(args)
    if files is None:
        return 1
    if not files:
        return 0
    for rec in files:
        ok = _reparse_one(rec, args.dry_run)
        if ok and args.with_graph:
            from ai_agent.memory.store import store

            with store._get_pool().connection() as conn:
                row = conn.execute(
                    "SELECT chunk_count, graph_status FROM agent_kb_file WHERE id = %s",
                    (rec["id"],),
                ).fetchone()
            _graph_one({**rec, **dict(row)}, args.dry_run)
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    files = _require_files(args)
    if files is None:
        return 1
    if not files:
        return 0
    for rec in files:
        _graph_one(rec, args.dry_run)
    return 0


def _cmd_reembed(args: argparse.Namespace) -> int:
    from ai_agent.kb.chunking import embedding_input
    from ai_agent.memory.embeddings import embed_texts
    from ai_agent.memory.store import store

    targets = (
        ("chunks", "episodes", "entities")
        if args.target == "all"
        else (args.target,)
    )
    batch = max(1, settings.kb_embedding_batch)
    total = 0
    with store._get_pool().connection() as conn:
        chunks = conn.execute(
            """
            SELECT c.id, c.text, c.page, c.kind, c.section, f.filename
            FROM agent_kb_chunk c
            JOIN agent_kb_file f ON f.id = c.file_id
            WHERE f.status = 'ready'
            ORDER BY c.file_id, c.seq
            """
        ).fetchall() if "chunks" in targets else []
        episodes = (
            conn.execute("SELECT id, text FROM agent_episode").fetchall()
            if "episodes" in targets
            else []
        )
        entities = (
            conn.execute(
                "SELECT id, name, type, description FROM agent_kg_entity"
            ).fetchall()
            if "entities" in targets
            else []
        )

    jobs = [
        (
            "agent_kb_chunk",
            chunks,
            lambda r: embedding_input(
                r["text"],
                filename=r["filename"] or "",
                page=r["page"],
                kind=r["kind"],
                section=r["section"],
            ),
        ),
        ("agent_episode", episodes, lambda r: r["text"]),
        (
            "agent_kg_entity",
            entities,
            lambda r: f"{r['name']}（{r['type']}）：{r['description'] or ''}",
        ),
    ]
    for table, rows, text_of in jobs:
        if not rows:
            if (table == "agent_kb_chunk" and "chunks" in targets) or (
                table == "agent_episode" and "episodes" in targets
            ) or (table == "agent_kg_entity" and "entities" in targets):
                print(f"{table}: 0 条，跳过")
            continue
        if args.dry_run:
            print(f"{table}: {len(rows)} 条待重嵌")
            continue
        done = 0
        for start in range(0, len(rows), batch):
            part = rows[start : start + batch]
            vectors = embed_texts([text_of(r) for r in part])
            with store._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    for row, vec in zip(part, vectors):
                        cur.execute(
                            f"UPDATE {table} SET embedding = %s WHERE id = %s",
                            (vec, row["id"]),
                        )
            done += len(part)
            print(f"{table}: {done}/{len(rows)}")
        print(f"{table}: 完成 {done} 条")
        total += done
    print(f"合计: {total} 条" + ("（dry-run，未写库）" if args.dry_run else ""))
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    from ai_agent.memory.store import store

    if not args.file and not args.name:
        return _cmd_list(argparse.Namespace(name=None))
    files = _pick_files(args.name, args.file, all_files=False)
    if not files:
        print("找不到文件")
        return 1
    if args.name and len(files) > 1 and not args.file:
        for rec in files:
            print(f"{rec['id']}  {rec['filename']}")
        print("匹配到多份，请用 --file 指定一份再看缺页")
        return 0
    rec = files[0]
    fid = rec["id"]
    with store._get_pool().connection() as conn:
        per_page = conn.execute(
            """
            SELECT p, count(*) AS n
            FROM (
                SELECT generate_series(page, coalesce(page_end, page)) AS p
                FROM agent_kb_chunk
                WHERE file_id = %s AND page IS NOT NULL
            ) t
            GROUP BY p
            ORDER BY p
            """,
            (fid,),
        ).fetchall()
        kinds = conn.execute(
            "SELECT kind, count(*) AS n FROM agent_kb_chunk "
            "WHERE file_id = %s GROUP BY kind ORDER BY n DESC",
            (fid,),
        ).fetchall()
        total = conn.execute(
            "SELECT count(*) AS n FROM agent_kb_chunk WHERE file_id = %s", (fid,)
        ).fetchone()["n"]
        no_page = conn.execute(
            "SELECT count(*) AS n FROM agent_kb_chunk "
            "WHERE file_id = %s AND page IS NULL",
            (fid,),
        ).fetchone()["n"]
    counts = {int(r["p"]): int(r["n"]) for r in per_page}
    max_page = max(counts) if counts else 0
    print(f"\n文件：{rec['filename']}  ({fid})")
    print(f"状态：{rec['status']}    切片 {total} 条（无页码 {no_page} 条）")
    print("类型：" + "  ".join(f"{k['kind']}={k['n']}" for k in kinds))
    print(f"页码范围：1 ~ {max_page}" if max_page else "无页码")
    missing = [p for p in range(1, max_page + 1) if p not in counts]
    covered = max_page - len(missing)
    if max_page:
        print(f"有切片：{covered} 页    缺：{len(missing)} 页（{len(missing) / max_page * 100:.1f}%）")
    if not missing:
        return 0
    if args.full:
        print("\n缺页明细：" + ", ".join(str(p) for p in missing))
        return 0
    ranges: list[tuple[int, int]] = []
    start = prev = missing[0]
    for page in missing[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append((start, prev))
        start = prev = page
    ranges.append((start, prev))
    print("\n缺页区间：")
    for a, b in ranges:
        print(f"  {a}" if a == b else f"  {a}-{b}  ({b - a + 1} 页)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
