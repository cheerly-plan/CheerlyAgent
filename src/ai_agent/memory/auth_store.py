from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from ai_agent.memory.db import UserRecord
from psycopg import errors as pg_errors

# pbkdf2 迭代次数；写入哈希串头部，未来可平滑升级
_PBKDF2_ITERATIONS = 240_000
_TOKEN_TTL = timedelta(days=30)
_TOKEN_BYTES = 32


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return (
        f"pbkdf2_sha256${_PBKDF2_ITERATIONS}"
        f"${salt.hex()}${digest.hex()}"
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(digest.hex(), digest_hex)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    """用户账号与会话令牌存储。

    令牌只在签发时返回原文，库里存 SHA-256 摘要，泄露数据库也无法伪造有效令牌。
    """

    def create_user(
        self, username: str, password: str, display_name: str = ""
    ) -> UserRecord | None:
        """创建用户，user_id 由服务端生成（UUID）；用户名已存在时返回 None。

        并发注册同名用户时，唯一索引兜底，捕获冲突同样返回 None（路由转 409）。
        """
        with self._get_pool().connection() as conn:
            exists = conn.execute(
                "SELECT 1 FROM agent_user WHERE username = %s", (username,)
            ).fetchone()
            if exists:
                return None
            try:
                row = conn.execute(
                    """
                    INSERT INTO agent_user (id, username, display_name, password_hash)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, username, display_name, created_at
                    """,
                    (
                        uuid.uuid4(),
                        username,
                        display_name or username,
                        hash_password(password),
                    ),
                ).fetchone()
                conn.commit()
            except pg_errors.UniqueViolation:
                conn.rollback()
                return None
        return UserRecord(
            user_id=str(row["id"]),
            username=row["username"],
            display_name=row["display_name"],
            created_at=row["created_at"].isoformat(),
        )

    def authenticate(self, username: str, password: str) -> UserRecord | None:
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT id, username, display_name, password_hash, created_at
                FROM agent_user WHERE username = %s
                """,
                (username,),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return UserRecord(
            user_id=str(row["id"]),
            username=row["username"],
            display_name=row["display_name"],
            created_at=row["created_at"].isoformat(),
        )

    def issue_token(self, user_id: str) -> tuple[str, datetime]:
        """签新令牌，返回 (原文令牌, 过期时间)。user_id 为用户 UUID 字符串。"""
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        expires_at = datetime.now(UTC) + _TOKEN_TTL
        with self._get_pool().connection() as conn:
            # 顺手清理过期令牌，避免表无限增长
            conn.execute(
                "DELETE FROM agent_auth_token WHERE expires_at < now()"
            )
            conn.execute(
                """
                INSERT INTO agent_auth_token (token_hash, user_id, expires_at)
                VALUES (%s, %s, %s)
                """,
                (hash_token(token), uuid.UUID(user_id), expires_at),
            )
            conn.commit()
        return token, expires_at

    def resolve_token(self, token: str) -> UserRecord | None:
        if not token:
            return None
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.display_name, u.created_at
                FROM agent_auth_token t
                JOIN agent_user u ON u.id = t.user_id
                WHERE t.token_hash = %s AND t.expires_at > now()
                """,
                (hash_token(token),),
            ).fetchone()
        if row is None:
            return None
        return UserRecord(
            user_id=str(row["id"]),
            username=row["username"],
            display_name=row["display_name"],
            created_at=row["created_at"].isoformat(),
        )

    def revoke_token(self, token: str) -> None:
        with self._get_pool().connection() as conn:
            conn.execute(
                "DELETE FROM agent_auth_token WHERE token_hash = %s",
                (hash_token(token),),
            )
            conn.commit()
