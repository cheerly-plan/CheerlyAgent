import os

# 在导入 ai_agent 之前备好配置，CI 没有 .env 也能跑单测
os.environ.setdefault("BASE_URL", "https://example.com/v1")
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("MODEL", "test-model")
# 本地 .env 可能开了整页 OCR；单测默认关掉，避免解析用例打到真实模型
os.environ.setdefault("KB_OCR_MODEL", "")
# 经营解读默认关：无模型环境 / 既有回归不触达受约束 LLM；专用用例再显式打开
os.environ.setdefault("REPORT_ANALYSIS_ENABLED", "false")
# 巨潮自动补齐默认关：回归不打外网；专用用例再 monkeypatch 打开
os.environ.setdefault("ENABLE_CNINFO_FETCH", "false")

# 单测不依赖 Qwen tokenizer 文件/网络：注入一个确定性的计数器。
# 语义：1 个字符 ≈ 1 token，足够验证预算裁剪逻辑；tokenizer 本身由
# tests/test_tokenizer.py 单独验证（有本地文件时）。
from ai_agent.llm.tokenizer import configure_token_counter


class _TestTokenCounter:
    def count(self, text: str) -> int:
        return max(1, len(text or ""))


configure_token_counter(_TestTokenCounter())

# 单测不加载真实工具注册表去测 schema token：用一个固定值代替，
# 真实测量在服务启动时由 api.app.lifespan 完成。
from ai_agent.core.config import (  # noqa: E402
    set_measured_tool_schema_tokens,
)

set_measured_tool_schema_tokens(int(os.environ.get("TEST_SCHEMA_TOKENS", "5000")))
