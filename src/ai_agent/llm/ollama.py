"""本地 Ollama 的原生接口：把「思考模式」真正关掉。

背景：`qwen3` 这类思考型模型在 Ollama 的 OpenAI 兼容接口（``/v1/chat/completions``）
上不认 ``think`` / ``chat_template_kwargs`` / ``enable_thinking`` 三个参数，
会先吐几百到几千字的 ``reasoning``，正文要等思考跑完才出现。实测同一份记忆
抽取提示词：

* 兼容接口，``think: false``：``completion_tokens=384`` 全用在 reasoning 上，
  ``content`` 空，``finish_reason=length``；
* 兼容接口，``max_tokens=4096``：思考 4837 字后终于给出 162 字 JSON，28.8 秒；
* 原生 ``/api/chat`` + ``think: false`` + ``format: json``：思考 0 字，
  256 字 JSON，**1.1 秒**。

所以轻量任务（记忆抽取、会话摘要、标题、检索改写）在本地网关上一律走原生
接口。只有连不上或报错时才回退到 OpenAI 兼容路径——回退不是为了兼容，
而是为了保证「换一台网关/换个模型」时不会整条链路挂掉。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from functools import lru_cache

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.llm import metrics

logger = get_logger(__name__)

# 探测过的网关地址：True = 确认是 Ollama 且支持原生接口
_PROBE_CACHE: dict[str, bool] = {}

# 本地 Ollama 一次只服务一个请求（模型只有一个，并发会触发 KV 缓存切换）。
# 实测标题生成与记忆抽取撞在一起时，Ollama 会抛
# `prediction aborted, token repeat limit reached`，串行化后消失。
_NATIVE_LOCK = threading.Lock()


def _api_base(base_url: str) -> str:
    """``http://host:11434/v1`` → ``http://host:11434``。"""
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


@lru_cache(maxsize=8)
def _probe(api_base: str) -> bool:
    """打一次 ``/api/tags``：能返回模型列表就是 Ollama 原生接口可用。"""
    try:
        with urllib.request.urlopen(f"{api_base}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return isinstance(data.get("models"), list)
    except Exception:
        return False


def is_ollama(base_url: str) -> bool:
    """该网关是不是本地 Ollama。

    只缓存**成功**的探测结果：失败可能是网关刚好在加载模型/重启，缓存成
    False 会让整个进程后续都退回兼容口，必须等重启才恢复。
    """
    api_base = _api_base(base_url)
    if not api_base:
        return False
    if _PROBE_CACHE.get(api_base):
        return True
    ok = _probe(api_base)
    if ok:
        _PROBE_CACHE[api_base] = True
        _probe.cache_clear()
    return ok


def _plain_messages(messages: list[dict]) -> list[dict]:
    """OpenAI 风格消息 → Ollama 原生消息（只留 role/content 两个字段）。"""
    out: list[dict] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if isinstance(content, list):
            # 多模态内容原样传给原生接口也能识别，这里只保证是字符串或列表
            kept = content
        else:
            kept = "" if content is None else str(content)
        out.append({"role": role, "content": kept})
    return out


# 思考型模型要先写完推理才轮到正文。实测 qwen3:4b 一次摘要的推理约 1200 字，
# 换算成 token 要 1500 上下；给小了只会拿到空正文（finish=length）。
_FREETEXT_MIN_PREDICT = 1536


def chat(
    messages: list[dict],
    *,
    base_url: str,
    model: str,
    timeout: float | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """调用原生 ``/api/chat``。失败抛异常，由调用方回退。

    两种任务的取舍不一样，分开处理：

    * **JSON 任务**（``json_mode``）：``think=False`` + ``format="json"``。
      语法约束让模型没法闲扯，实测 1.1 秒直接吐合法 JSON，思考 0 字。
    * **自由文本**（摘要、标题、改写）：**不能**关思考——这个版本的 Ollama
      关掉 ``think`` 只是把推理从 ``thinking`` 字段挪进 ``content``，摘要会
      变成「首先，用户的问题是……」的思考过程。改为保持思考、把
      ``num_predict`` 放大到够它想完，正文自然出现在 ``content``（实测
      1536 token / 10 秒）。
    """
    options: dict[str, object] = {}
    budget = max_tokens
    if not json_mode:
        budget = max(max_tokens or 0, _FREETEXT_MIN_PREDICT)
    if budget is not None:
        options["num_predict"] = budget
    if temperature is not None:
        options["temperature"] = temperature
    if json_mode:
        # 只给一点点重复惩罚：实测 1.3 会把小模型压成退化输出（标题直接给空串），
        # 真正治「token repeat limit」的是下面的串行化
        options.setdefault("repeat_penalty", 1.1)
    payload: dict[str, object] = {
        "model": model,
        "messages": _plain_messages(messages),
        "stream": False,
    }
    if json_mode:
        # 只有结构化任务才关思考（关掉的同时用 format 约束，避免退化成
        # 「把推理写进正文」）
        payload["think"] = False
        payload["format"] = "json"
    if options:
        payload["options"] = options

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{_api_base(base_url)}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    data = None
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            serialized = bool(getattr(settings, "local_llm_serialize", True))
            if serialized:
                with _NATIVE_LOCK:
                    with urllib.request.urlopen(request, timeout=timeout or 60) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
            else:
                with urllib.request.urlopen(request, timeout=timeout or 60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            # 把响应体带出来：只报「HTTP Error 500」无从判断是模型没加载、
            # 显存不够还是请求非法
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                detail = ""
            metrics.incr(metrics.NATIVE_ERROR)
            last_error = RuntimeError(f"HTTP {e.code}: {detail or e.reason}")
            if e.code < 500 or attempt == 1:
                raise last_error from e
            logger.warning(
                "Ollama 原生接口 %s，重试一次：%s", e.code, detail or e.reason
            )
        except Exception as e:
            raise e
    if data is None:
        raise last_error or RuntimeError("Ollama 原生接口无响应")
    message = data.get("message") or {}
    content = str(message.get("content") or "").strip()
    thinking = str(message.get("thinking") or "")
    if not content and thinking:
        metrics.incr(metrics.NATIVE_EMPTY)
        logger.warning(
            "Ollama 原生接口返回空正文（thinking=%d 字），可能未关闭思考：model=%s",
            len(thinking),
            model,
        )
    return content