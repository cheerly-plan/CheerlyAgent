"""Alembic 运行环境。

连接串从 ai_agent.core.config.settings.database_url 读取（与运行时同源），
不写进 alembic.ini，避免凭据入库。迁移脚本全部手写裸 SQL（op.execute），
不用 autogenerate（本项目无 SQLAlchemy ORM 模型）。
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# 项目配置同源：alembic.ini 的 prepend_sys_path = src 已把 src 加入 sys.path
from ai_agent.core.config import settings

config = context.config

# uvicorn --reload 在 macOS 上用 fork 拉起 worker；父进程若已持有 logging 锁，
# fileConfig 再配一遍 handler 容易把子进程卡死在启动期（日志停在
# "Will assume transactional DDL"，/docs 与登录一直超时）。应用内调用迁移时
# logging 已由 setup_logging 配好，跳过 fileConfig；命令行 `alembic` 仍走配置。
if config.config_file_name is not None:
    import logging

    if not logging.getLogger().handlers:
        fileConfig(config.config_file_name)

# 无 ORM 模型，纯手写 SQL 迁移；target_metadata 留空禁用 autogenerate
target_metadata = None


def run_migrations_offline() -> None:
    """离线模式：只产出 SQL，不连库。url 来自 settings。"""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连库执行迁移。engine 用 settings.database_url。"""
    connectable = engine_from_config(
        {"sqlalchemy.url": settings.database_url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=False,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
