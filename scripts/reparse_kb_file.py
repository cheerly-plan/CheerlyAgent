"""按当前解析策略重新解析已入库的文件（切片整份重来）。

改了解析层（比如新增 PDF 表格抽取）之后，存量文件的切片还是老策略的
产物，不重跑就看不到效果。本脚本做的是「整份重来」：

    删旧切片 → 重新解析 → 重新嵌入 → 写库 → 清掉旧图谱

图谱必须一起清：mention 是按 (file_id, chunk_seq) 指回原文的，切片重排
之后 seq 全变，旧 mention 会整片指向错内容——留着比没有更糟。清完
graph_status 回到 none，由管理员在知识库页重新触发建图。

用法：
    uv run python scripts/reparse_kb_file.py --name 南岭        # 按名字匹配
    uv run python scripts/reparse_kb_file.py --file <file_id>   # 精确指定
    uv run python scripts/reparse_kb_file.py --all --yes        # 全部重来
    uv run python scripts/reparse_kb_file.py --name 南岭 --dry-run
"""

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.core.logging import get_logger  # noqa: E402
from ai_agent.kb.pipeline import kb_pipeline  # noqa: E402
from ai_agent.memory.store import store  # noqa: E402

logger = get_logger("reparse")


def _pick_files(name: str | None, file_id: str | None, all_files: bool) -> list[dict]:
    with store._get_pool().connection() as conn:
        if file_id:
            rows = conn.execute(
                """
                SELECT id, filename, status, chunk_count, graph_status, mime,
                       user_id
                FROM agent_kb_file
                WHERE id::text LIKE %s
                """,
                (f"{file_id}%",),
            ).fetchall()
        elif name:
            rows = conn.execute(
                """
                SELECT id, filename, status, chunk_count, graph_status, mime,
                       user_id
                FROM agent_kb_file
                WHERE filename ILIKE %s
                ORDER BY filename
                """,
                (f"%{name}%",),
            ).fetchall()
        elif all_files:
            rows = conn.execute(
                """
                SELECT id, filename, status, chunk_count, graph_status, mime,
                       user_id
                FROM agent_kb_file
                ORDER BY filename
                """
            ).fetchall()
        else:
            return []
    return [r for r in rows if r["status"] == "ready"]


def _upload_path(rec: dict) -> Path | None:
    from ai_agent.core.config import settings

    suffix = (rec.get("mime") or "").lower()
    # mime 是 image/pdf 之类，文件本体按 id 目录 + 文件名存，直接 glob
    directory = settings.kb_upload_dir / str(rec["id"])
    if not directory.is_dir():
        return None
    candidates = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.name != ".gitkeep"
    ]
    if not candidates:
        return None
    if suffix:
        for p in candidates:
            if suffix in p.name.lower() or suffix.split("/")[-1] in p.name.lower():
                return p
    return max(candidates, key=lambda p: p.stat().st_size)


def _reparse(rec: dict, dry_run: bool) -> None:
    fid = rec["id"]
    path = _upload_path(rec)
    print(
        f"  文件：{rec['filename']}  现切片 {rec['chunk_count']}  图谱 {rec['graph_status']}"
    )
    if path is None:
        print("  ✗ 找不到原始文件，跳过（可能已清理）")
        return
    print(f"  原件：{path.name}")
    if dry_run:
        print("  (dry-run) 未改动")
        return

    with store._get_pool().connection() as conn:
        conn.execute("DELETE FROM agent_kb_chunk WHERE file_id = %s", (fid,))
        # 切片重排后旧 mention 全指向错位置，图谱整份作废
        conn.execute("DELETE FROM agent_kg_relation WHERE file_id = %s", (fid,))
        conn.execute("DELETE FROM agent_kg_entity WHERE file_id = %s", (fid,))
        conn.execute("DELETE FROM agent_kg_mention WHERE file_id = %s", (fid,))
    store.update_graph_status(fid, "none", entity_count=0)
    print("  旧切片与图谱已清除，开始重新解析（含重新嵌入，需要几分钟）…")

    # 直接跑处理流程（不入线程池）：脚本本身就是一次性的批处理，
    # 同步跑才能在结束时打印结果
    kb_pipeline._process(fid, str(rec.get("user_id") or ""), path)

    with store._get_pool().connection() as conn:
        row = conn.execute(
            "SELECT status, chunk_count, error FROM agent_kb_file WHERE id = %s", (fid,)
        ).fetchone()
        kinds = conn.execute(
            """
            SELECT kind, count(*) AS n
            FROM agent_kb_chunk
            WHERE file_id = %s
            GROUP BY kind
            ORDER BY n DESC
            """,
            (fid,),
        ).fetchall()
    print(
        f"  ✓ 完成：status={row['status']} 切片={row['chunk_count']} "
        + " ".join(f"{k['kind']}={k['n']}" for k in kinds)
    )
    if row["error"]:
        print(f"  error={row['error']}")


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--name")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--yes", action="store_true", help="--all 时不再确认")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.file or args.name or args.all):
        parser.error("需要 --file / --name / --all 之一")
    files = _pick_files(args.name, args.file, args.all)
    if not files:
        print("没有匹配到 ready 状态的文件")
        return 1
    print(f"将处理 {len(files)} 个文件：")
    for rec in files:
        print(f"  {rec['id']}  {rec['filename']}  ({rec['chunk_count']} 片)")
    if args.all and not args.yes and not args.dry_run:
        answer = input("确认全部重新解析？[y/N] ").strip().lower()
        if answer != "y":
            print("已取消")
            return 0
    for rec in files:
        _reparse(rec, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
