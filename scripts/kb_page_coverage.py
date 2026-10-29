"""检查知识库文件的「页码覆盖」：哪些页一条切片都没有。

跳页时提示「第 N 页没有切片，已跳到第 M 页」，本脚本就是回答「到底缺哪些页」。

缺页通常是这几种（都属于预期行为，不是 bug）：
- 整页无文本且没过视觉识别（扫描图、纯图页、空白页）；
- 页内只有页眉页脚/目录残留/「无图表」描述，被解析层过滤掉了；
- 视觉识别被 KB_VISION_MAX_PAGES 截断或熔断，后面的图页没转成描述。

用法：
    uv run python scripts/kb_page_coverage.py                    # 列出全部文件概览
    uv run python scripts/kb_page_coverage.py --file <file_id>   # 逐页看
    uv run python scripts/kb_page_coverage.py --file <id> --all  # 逐页全打（不折叠）
    uv run python scripts/kb_page_coverage.py --name 招股        # 按文件名模糊匹配
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.memory.store import store  # noqa: E402


def _overview() -> None:
    with store._get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.filename, f.status,
                   count(c.seq) AS chunks,
                   coalesce(max(coalesce(c.page_end, c.page)), 0) AS pages
            FROM agent_kb_file f
            LEFT JOIN agent_kb_chunk c ON c.file_id = f.id
            GROUP BY f.id, f.filename, f.status
            ORDER BY pages DESC, f.filename
            """
        ).fetchall()
    print(f"{'file_id':<38}{'页':>6}{'切片':>7}  {'状态':<10}文件名")
    for r in rows:
        print(
            f"{str(r['id']):<38}{r['pages']:>6}{r['chunks']:>7}  {r['status']:<10}{r['filename']}"
        )


def _resolve(name: str) -> list:
    with store._get_pool().connection() as conn:
        return conn.execute(
            """
            SELECT id, filename, status
            FROM agent_kb_file
            WHERE filename ILIKE %s
            ORDER BY filename
            """,
            (f"%{name}%",),
        ).fetchall()


def _coverage(file_id: str, show_all: bool) -> None:
    with store._get_pool().connection() as conn:
        rec = conn.execute(
            "SELECT id, filename, status FROM agent_kb_file WHERE id::text LIKE %s",
            (f"{file_id}%",),
        ).fetchone()
        if rec is None:
            print(f"找不到文件 {file_id}")
            return
        fid = rec["id"]
        # 每页的切片数：跨页切片（page..page_end）计入它覆盖的每一页
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
            """
            SELECT kind, count(*) AS n
            FROM agent_kb_chunk
            WHERE file_id = %s
            GROUP BY kind
            ORDER BY n DESC
            """,
            (fid,),
        ).fetchall()
        total_chunks = conn.execute(
            "SELECT count(*) AS n FROM agent_kb_chunk WHERE file_id = %s", (fid,)
        ).fetchone()["n"]
        no_page = conn.execute(
            "SELECT count(*) AS n FROM agent_kb_chunk WHERE file_id = %s AND page IS NULL",
            (fid,),
        ).fetchone()["n"]

    counts: dict[int, int] = {int(r["p"]): int(r["n"]) for r in per_page}
    max_page = max(counts) if counts else 0
    print(f"\n文件：{rec['filename']}  ({fid})")
    print(f"状态：{rec['status']}    切片 {total_chunks} 条（无页码 {no_page} 条）")
    print("类型：" + "  ".join(f"{k['kind']}={k['n']}" for k in kinds))
    print(f"页码范围：1 ~ {max_page}")

    missing = [p for p in range(1, max_page + 1) if p not in counts]
    covered = max_page - len(missing)
    print(
        f"有切片：{covered} 页    缺：{len(missing)} 页"
        f"（{len(missing) / max_page * 100:.1f}%）" if max_page else "无页码"
    )

    if not missing:
        return
    if show_all:
        print("\n缺页明细：" + ", ".join(str(p) for p in missing))
        return
    # 折叠成连续区间，读起来快
    ranges: list[tuple[int, int]] = []
    start = prev = missing[0]
    for p in missing[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append((start, prev))
        start = prev = p
    ranges.append((start, prev))
    print("\n缺页区间：")
    for a, b in ranges:
        print(f"  {a}" if a == b else f"  {a}-{b}  ({b - a + 1} 页)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="文件 id（可只给前缀）")
    parser.add_argument("--name", help="文件名模糊匹配")
    parser.add_argument("--all", action="store_true", help="缺页全打，不折叠区间")
    args = parser.parse_args()

    if args.name:
        rows = _resolve(args.name)
        if not rows:
            print(f"没有匹配「{args.name}」的文件")
            return 1
        for r in rows:
            print(f"{r['id']}  {r['filename']}")
        if len(rows) == 1:
            _coverage(str(rows[0]["id"]), args.all)
        return 0
    if args.file:
        _coverage(args.file, args.all)
        return 0
    _overview()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
