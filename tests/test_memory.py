import pytest

from ai_agent.core.config import settings
from ai_agent.memory.store import MemoryStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    dim = settings.embedding_dim
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * dim
        for word in text.lower().split():
            v[hash(word) % dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vecs.append([x / norm for x in v])
    return vecs


_TEST_USERS = ("a", "b")


def _clear_test_users(store: MemoryStore) -> None:
    for uid in _TEST_USERS:
        store.clear(user_id=uid)


@pytest.fixture
def mem_store(monkeypatch):
    monkeypatch.setattr("ai_agent.memory.store.embed_texts", fake_embed)
    monkeypatch.setattr(
        "ai_agent.memory.store.embed_query", lambda q: fake_embed([q])[0]
    )
    store = MemoryStore()
    try:
        store.ensure_schema()
    except Exception as e:
        pytest.skip(f"Postgres 不可用: {e}")
    _clear_test_users(store)
    yield store
    _clear_test_users(store)
    store.close()


def test_vector_search_is_semantic(mem_store):
    mem_store.write("我日常用 pytest 写单测", user_id="a")
    hits = mem_store.search("pytest 单测", user_id="a")
    assert hits
    assert "pytest" in hits[0].text


def test_store_isolates_users(mem_store):
    mem_store.write("喜欢 pytest", user_id="a")
    mem_store.write("喜欢 jest", user_id="b")
    hits = mem_store.search("pytest", user_id="a")
    assert all(item.user_id == "a" for item in hits)
    assert all("jest" not in item.text for item in hits)


def test_write_dedup(mem_store):
    first = mem_store.write("喜欢 uv", user_id="a")
    second = mem_store.write("喜欢 uv", user_id="a")
    assert first.id == second.id


def test_profile_is_listed_without_embedding(mem_store):
    mem_store.write_profile("喜欢 pytest", user_id="a")
    mem_store.write("上周用 pytest 修过登录", user_id="a")
    profile = mem_store.list_profile("a")
    assert [item.text for item in profile] == ["喜欢 pytest"]


def test_search_does_not_return_profile(mem_store):
    mem_store.write_profile("喜欢 pytest", user_id="a")
    hits = mem_store.search("pytest", user_id="a")
    assert hits == []


def test_profile_isolates_users(mem_store):
    mem_store.write_profile("喜欢 pytest", user_id="a")
    mem_store.write_profile("喜欢 jest", user_id="b")
    profile = mem_store.list_profile("a")
    assert [item.text for item in profile] == ["喜欢 pytest"]


def test_profile_write_dedup(mem_store):
    first = mem_store.write_profile("喜欢 uv", user_id="a")
    second = mem_store.write_profile("喜欢 uv", user_id="a")
    assert first.id == second.id
