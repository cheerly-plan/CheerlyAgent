import os

# 在导入 ai_agent 之前备好配置，CI 没有 .env 也能跑单测
os.environ.setdefault("BASE_URL", "https://example.com/v1")
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("MODEL", "test-model")
# 本地 .env 可能开了整页 OCR；单测默认关掉，避免解析用例打到真实模型
os.environ.setdefault("KB_OCR_MODEL", "")
