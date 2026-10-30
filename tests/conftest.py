import os

# 在导入 ai_agent 之前备好配置，CI 没有 .env 也能跑单测
os.environ.setdefault("BASE_URL", "https://example.com/v1")
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("MODEL", "test-model")


import pytest

from ai_agent.core.config import settings


@pytest.fixture(autouse=True)
def _disable_kb_catalog(monkeypatch):
    """文件目录会查库，run_loop 类单测默认关掉；专门测目录的用例再打开。"""
    monkeypatch.setattr(settings, "kb_catalog_enabled", False)
    monkeypatch.setattr(settings, "kb_force_search", False)
