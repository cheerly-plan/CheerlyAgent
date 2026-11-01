#!/usr/bin/env python3
"""回填切片元数据（section / scope / kind）。

问题背景：现有切片是用旧版入库代码生成的，导致：
  - ``section`` / ``scope`` 两列 100% 为空；
  - ``kind`` 列是旧的 ``'c'`` 单字符枚举而非 ``text/table/figure``。
后果：检索的元数据过滤（sections/scopes）是 no-op；合并/母公司报表口径
偏好（retrieval._apply_scope_preference）因 scope 永远为 None 而**从不触发**。

做法：用当前代码重新解析 + 切片源文件，仅把 ``section`` / ``scope`` / ``kind``
三列按 ``seq`` 回填到现有切片行。**不动 text 与 embedding**（已验证当前
切片文本与库里逐片对齐），因此无需重嵌、也无需重建图谱。

安全闸门：仅对「切片数量一致且每片文本逐字对齐」的文件执行回填；不对齐
的文件跳过并报告，交人工整份重入，避免把元数据写错行。
"""
from __future__ import annotations

import argparse
import time

from ai_agent.core.config import settings
from ai_agent.kb.chunking import chunk_segments
from ai_agent.kb.parsing import parse_segments
from ai_agent.memory.store import store


def _backfill_one(fid: str, filename: str, dry_run: bool) -> dict:
    """回填单个文件。返回状态字典。"""
    d = settings.kb_upload_dir / fid
    cands = list(d.glob("*")) if d.is_dir() else []
    if not cands:
        return {"fid": fid, "name": filename, "status": "no_source", "n": None, "m": None}
    src = cands[0]

    segs = parse_segments(src)
    chunks = chunk_segments(
        segs,
        max_chars=settings.kb_chunk_size,
        overlap=settings.kb_chunk_overlap,
    )

    with store._get_pool().connection() as conn:
        db = conn.execute(
            "SELECT seq, text, kind, section, scope FROM agent_kb_chunk "
            "WHERE file_id = %s ORDER BY seq",
            (fid,),
        ).fetchall()
        n, m = len(chunks), len(db)
        if n != m:
            return {"fid": fid, "name": filename, "status": "count_mismatch", "n": n, "m": m}
        # 逐片文本对齐检查：确保 seq 与库里同一片是同一段文字
        for i in range(n):
            if chunks[i].text != (db[i]["text"] or ""):
                return {
                    "fid": fid,
                    "name": filename,
                    "status": "text_mismatch",
                    "n": n,
                    "m": m,
                    "at": i,
                }
        if dry_run:
            return {"fid": fid, "name": filename, "status": "dry_ok", "n": n, "m": m}

        cur = conn.cursor()
        for i, ch in enumerate(chunks):
            cur.execute(
                "UPDATE agent_kb_chunk SET section = %s, scope = %s, kind = %s "
                "WHERE file_id = %s AND seq = %s",
                (ch.section, ch.scope, ch.kind, fid, i),
            )
        conn.commit()

        sc = conn.execute(
            "SELECT COUNT(*) c FROM agent_kb_chunk WHERE file_id = %s "
            "AND section IS NOT NULL AND section <> ''",
            (fid,),
        ).fetchone()["c"]
        cp = conn.execute(
            "SELECT COUNT(*) c FROM agent_kb_chunk WHERE file_id = %s "
            "AND scope IS NOT NULL AND scope <> ''",
            (fid,),
        ).fetchone()["c"]
        kinds = conn.execute(
            "SELECT kind, COUNT(*) c FROM agent_kb_chunk WHERE file_id = %s GROUP BY kind",
            (fid,),
        ).fetchall()
        return {
            "fid": fid,
            "name": filename,
            "status": "updated",
            "n": n,
            "m": m,
            "section": sc,
            "scope": cp,
            "kinds": dict(kinds),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="回填切片 section/scope/kind 元数据")
    ap.add_argument("--dry-run", action="store_true", help="只检查对齐，不写库")
    ap.add_argument("--name", default=None, help="只处理文件名包含此串的文件")
    args = ap.parse_args()

    store.ensure_schema()
    with store._get_pool().connection() as conn:
        files = conn.execute(
            "SELECT id, filename FROM agent_kb_file WHERE status = 'ready' ORDER BY filename"
        ).fetchall()

    ok = skipped = 0
    for f in files:
        fid, fn = str(f["id"]), f["filename"]
        if args.name and args.name not in fn:
            continue
        t0 = time.time()
        r = _backfill_one(fid, fn, args.dry_run)
        dt = time.time() - t0
        st = r["status"]
        if st in ("updated", "dry_ok"):
            ok += 1
            extra = ""
            if st == "updated":
                extra = (
                    f" → section非空={r.get('section')} scope非空={r.get('scope')} "
                    f"kind={r.get('kinds')}"
                )
            print(f"  ✓ {fn[:30]:<32} {st} 片={r['n']} ({dt:.1f}s){extra}")
        else:
            skipped += 1
            print(f"  ✗ {fn[:30]:<32} 跳过: {st} n={r.get('n')} m={r.get('m')} at={r.get('at')}")
    print(f"\n完成：成功/检查 {ok}，跳过 {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
