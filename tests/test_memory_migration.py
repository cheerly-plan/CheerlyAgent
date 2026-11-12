"""记忆 schema 必须由 Alembic 建出来，不能只留在 ensure_schema()。

回归背景：渐进式记忆第 4/5 步的列与表（episode 证据列、候选表、归档表、
画像 expires_at / superseded_at）原先只在 ``storage/db.py::ensure_schema()``
里创建，而应用启动已经只跑 Alembic。全新库上情景记忆、候选与归档会静默
失效，且被各处的 fail-open except 吞掉，很难从日志发现。

这里用两层保护：

1. 离线 SQL 断言：``alembic upgrade head --sql`` 的产物必须包含这些 DDL，
   零依赖、每次跑单测都执行；
2. 空库在线校验：真正建一个临时库跑 ``alembic upgrade head``，确认列/表/
   索引可查询。没有建库权限或 Postgres 不可用时跳过。
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

from ai_agent.core.config import settings

ROOT = Path(__file__).resolve().parents[1]

_MEMORY_SQL_MARKERS = (
    "agent_user_profile_active_idx",
    "expires_at",
    "superseded_at",
    "hit_count",
    "last_seen_at",
    "last_decay_at",
    "sessions_seen",
    "agent_memory_candidate",
    "agent_memory_candidate_user_status_idx",
    "reviewed_by",
    "review_note",
    "reviewed_at",
    "agent_memory_archive",
    "agent_memory_archive_user_idx",
    "archived_at",
)


def _alembic_sql() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.lower()


def test_offline_upgrade_sql_contains_memory_schema():
    sql = _alembic_sql()
    for marker in _MEMORY_SQL_MARKERS:
        assert marker in sql, f"Alembic 迁移缺少记忆 schema: {marker}"


def _temp_database_url() -> tuple[str, str]:
    import psycopg

    parts = urlsplit(settings.database_url)
    name = f"ai_agent_mem_mig_{uuid.uuid4().hex[:10]}"
    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            conn.execute(f'CREATE DATABASE "{name}"')
    except Exception as exc:  # pragma: no cover - 无库/无权限时跳过
        pytest.skip(f"无法创建临时库，跳过空库迁移校验: {exc}")
    return name, urlunsplit(parts._replace(path=f"/{name}"))


def _drop_database(name: str) -> None:
    import psycopg

    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
    except Exception:  # pragma: no cover - 清理失败不掩盖主断言
        try:
            with psycopg.connect(settings.database_url, autocommit=True) as conn:
                conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        except Exception:
            pass


def _columns(conn, table: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    ).fetchall()
    return {str(row[0]) for row in rows}


def test_upgrade_head_on_empty_database_creates_memory_schema():
    """全新库跑一次 head 迁移，记忆相关的列、表、索引都必须存在。"""
    import psycopg

    name, url = _temp_database_url()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT,
            env={**os.environ, "DATABASE_URL": url},
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr

        with psycopg.connect(url) as conn:
            assert {"expires_at", "superseded_at"} <= _columns(
                conn, "agent_user_profile"
            )
            assert {
                "hit_count",
                "last_seen_at",
                "last_decay_at",
                "sessions_seen",
            } <= _columns(conn, "agent_episode")
            assert {
                "text",
                "facet",
                "source",
                "status",
                "reviewed_by",
                "review_note",
                "reviewed_at",
            } <= _columns(conn, "agent_memory_candidate")
            assert {
                "kind",
                "origin_id",
                "text",
                "source",
                "archived_at",
            } <= _columns(conn, "agent_memory_archive")
            indexes = {
                str(row[0])
                for row in conn.execute(
                    "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
                ).fetchall()
            }
            assert {
                "agent_user_profile_active_idx",
                "agent_memory_candidate_user_status_idx",
                "agent_memory_archive_user_idx",
            } <= indexes
    finally:
        _drop_database(name)