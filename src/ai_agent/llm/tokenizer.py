"""Qwen tokenizer：上下文预算的唯一计数口径。

之前上下文按字符估算，但 Qwen 中文大致 1 字符 ≈ 1 token，40k 字符可能
接近 40k token，和真实窗口差得很远。这里统一用 Qwen tokenizer 计数：

- 首次使用时从本地文件加载；没有则从镜像下载到缓存，再没有则退回
  保守的启发式计数并告警（保证服务可用，但预算会有偏差）。
- ``count_tokens`` 带 LRU 缓存：同一段工具原文/参数会被反复计入
  ``fit_turns`` 的循环，避免每轮重新 tokenize。
- 测试可通过 ``configure_token_counter`` 注入轻量计数器，不依赖网络。
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Protocol

from ai_agent.core.config import settings

logger = logging.getLogger(__name__)

# 每条消息的 role/分隔符开销；Qwen chat template 还包含 tool 标记等，
# 这里留固定 overhead 作为保守估计，具体正文/参数用 tokenizer 精确算。
_MESSAGE_OVERHEAD = 4
_TOOL_CALL_OVERHEAD = 4


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class QwenTokenCounter:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(
            self._tokenizer.encode(text, add_special_tokens=False).ids
        )


class HeuristicTokenCounter:
    """tokenizer 不可用时的兜底：CJK 按 1 字/token，ASCII 按 4 字符/token。

    只用于保证服务不中断；正常部署应让 Qwen tokenizer 文件可用。
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        other = len(text) - ascii_chars
        return max(1, (ascii_chars + 3) // 4 + other)


_counter: TokenCounter | None = None
_counter_lock = Lock()


def configure_token_counter(counter: TokenCounter | None) -> None:
    """测试/宿主替换计数器；传 None 恢复默认懒加载。"""
    global _counter
    with _counter_lock:
        _counter = counter
        count_tokens.cache_clear()


def _default_cache_path() -> Path:
    configured = (settings.qwen_tokenizer_path or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "cheely" / "qwen3" / "tokenizer.json"


def _candidate_paths() -> Iterable[Path]:
    configured = (settings.qwen_tokenizer_path or "").strip()
    if configured:
        yield Path(configured).expanduser()
    # 本地开发缓存（仓库内，不参与提交）
    yield Path(".cache/qwen3/tokenizer.json")
    yield _default_cache_path()


def _download_tokenizer(dest: Path) -> bool:
    url = (settings.qwen_tokenizer_url or "").strip()
    if not url:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            url, timeout=settings.qwen_tokenizer_timeout_seconds
        ) as response:
            data = response.read()
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        logger.info("Qwen tokenizer 已下载 path=%s bytes=%d", dest, len(data))
        return True
    except Exception:
        logger.warning("Qwen tokenizer 下载失败 url=%s", url, exc_info=True)
        return False


def _load_counter() -> TokenCounter:
    global _counter
    with _counter_lock:
        if _counter is not None:
            return _counter
        try:
            from tokenizers import Tokenizer
        except ImportError:
            logger.warning("tokenizers 未安装，退回字符启发式计数")
            _counter = HeuristicTokenCounter()
            return _counter

        for path in _candidate_paths():
            if path.is_file():
                try:
                    _counter = QwenTokenCounter(Tokenizer.from_file(str(path)))
                    logger.info("Qwen tokenizer 已加载 path=%s", path)
                    return _counter
                except Exception:
                    logger.exception("Qwen tokenizer 加载失败 path=%s", path)
        if _download_tokenizer(_default_cache_path()):
            try:
                _counter = QwenTokenCounter(
                    Tokenizer.from_file(str(_default_cache_path()))
                )
                return _counter
            except Exception:
                logger.exception("下载后的 Qwen tokenizer 加载失败")
        logger.warning(
            "Qwen tokenizer 不可用，退回字符启发式计数；token 预算会有偏差"
        )
        _counter = HeuristicTokenCounter()
        return _counter


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    return _load_counter().count(text)


def count_json_tokens(value: object) -> int:
    try:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        raw = str(value)
    return count_tokens(raw)


def count_message_tokens(message: dict) -> int:
    total = _MESSAGE_OVERHEAD
    total += count_tokens(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        total += _TOOL_CALL_OVERHEAD
        total += count_tokens(str(call.get("name") or ""))
        total += count_json_tokens(call.get("arguments") or {})
    return total


def count_messages_tokens(messages: list[dict]) -> int:
    return sum(count_message_tokens(message) for message in messages)


def warmup_tokenizer() -> None:
    """启动时可调用，避免首个请求承担加载/下载延迟。"""
    try:
        count_tokens("预热")
    except Exception:
        logger.exception("Qwen tokenizer 预热失败")


__all__ = [
    "HeuristicTokenCounter",
    "QwenTokenCounter",
    "TokenCounter",
    "configure_token_counter",
    "count_json_tokens",
    "count_message_tokens",
    "count_messages_tokens",
    "count_tokens",
    "warmup_tokenizer",
]


def clip_text_to_tokens(text: str, max_tokens: int, marker: str = "") -> str:
    """按 token 预算裁一段文本（保头部）；marker 也计入预算。"""
    body = text or ""
    if max_tokens <= 0:
        return marker if count_tokens(marker) <= 0 else ""
    mark_tokens = count_tokens(marker)
    if mark_tokens > max_tokens:
        return ""
    available = max(0, max_tokens - mark_tokens)
    if count_tokens(body) <= available:
        return body + marker
    # tokenizer 对前缀单调：二分最大可容纳的字符前缀
    lo, hi = 0, len(body)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(body[:mid]) <= available:
            lo = mid
        else:
            hi = mid - 1
    return body[:lo] + marker
