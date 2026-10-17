import os

# 在导入 ai_agent 之前备好配置，CI 没有 .env 也能跑单测
os.environ.setdefault("BASE_URL", "https://example.com/v1")
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("MODEL", "test-model")
