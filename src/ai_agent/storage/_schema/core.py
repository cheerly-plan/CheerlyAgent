"""pgvector 扩展与基础表。

由 storage/db.py 的 ensure_schema 按表拆出，SQL 逐字不变。
"""

from __future__ import annotations


def ensure_vector_extension(conn, *, dim: int) -> None:
    """pgvector 扩展。"""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def commit_schema(conn, *, dim: int) -> None:
    """提交事务。"""
    conn.commit()
