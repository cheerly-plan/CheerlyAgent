from ai_agent.memory.auth_store import hash_password, hash_token, verify_password


def test_password_hash_roundtrip() -> None:
    stored = hash_password("s3cret-密码")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("s3cret-密码", stored)


def test_password_rejects_wrong_and_tampered() -> None:
    stored = hash_password("correct-password")
    assert not verify_password("wrong-password", stored)
    # 相同密码两次哈希盐不同，密文不同
    assert stored != hash_password("correct-password")
    # 哈希串被篡改时校验失败而不是抛异常
    assert not verify_password("correct-password", "garbage")
    assert not verify_password("correct-password", stored[:-2] + "ff")


def test_token_hash_is_sha256_hex() -> None:
    digest = hash_token("some-token")
    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert hash_token("some-token") == digest
