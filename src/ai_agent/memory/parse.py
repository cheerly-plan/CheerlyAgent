from __future__ import annotations

import json
import re
from typing import Any

from ai_agent.core.config import settings
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage

_MEMORY_EXTRA_BODY: dict[str, Any] = {
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}

_NUMBER = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_JSON_KEYWORDS = {"true", "false", "null"}


def complete_memory_chat(
    messages: list[ChatMessage],
    *,
    max_tokens: int,
) -> str:
    return complete_chat(
        messages,
        model=settings.llm_for_memory(),
        timeout=settings.memory_merge_timeout_seconds,
        max_retries=0,
        max_tokens=max_tokens,
        extra_body=_MEMORY_EXTRA_BODY,
    )


def load_llm_json(raw: str, *, repair: bool = False) -> object | None:
    """从模型输出里取出 JSON 对象。

    repair=False 只做「剥代码块 + 截到最外层括号」，解析不了就是 None。
    repair=True 额外做两件事（见 `repair_json`）：给裸词补引号、给截断的
    输出补括号。仅适用于「结构简单、坏了也比没有强」的场景——图谱抽取是
    一批一个结果，坏一批就丢一批实体；而实体对齐的裁决结果会被直接拿去
    合并实体，猜错代价大，那边必须保持 repair=False。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        blob = text[start : end + 1]
    elif start >= 0:
        # 有 "{" 却收不到 "}"，就是被截断了。不能落到下面的 [...] 分支：
        # 内层数组先闭口，会被整段当成结果返回，外层对象和后面的字段全丢
        return repair_json(text) if repair else None
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return None
        blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        if not repair:
            return None
    return repair_json(text)


def _quote_bare_tokens(blob: str) -> str:
    """给字符串外那些「既不是数字也不是关键字」的裸词补上引号。

    本地小模型偶发把中文名裸着写：
        ["深圳市玮顺企业管理有限公司","由",深圳明德控股发展有限公司,121436506,1]
    这一行会让整个 JSON 作废，丢掉整批实体。只处理字符串外的连续非结构
    字符，碰到结构符号就断开，因此不会动已经正确加引号的内容。
    """
    out: list[str] = []
    i, n = 0, len(blob)
    in_str = False
    while i < n:
        ch = blob[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(blob[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch in "{}[],:" or ch.isspace():
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and blob[j] not in "{}[],:" and not blob[j].isspace():
            j += 1
        token = blob[i:j]
        if _NUMBER.match(token) or token in _JSON_KEYWORDS:
            out.append(token)
        else:
            out.append('"' + token.replace('"', "") + '"')
        i = j
    return "".join(out)


def _close_open(blob: str) -> str:
    """截断的输出：补上未闭合的括号，并丢掉尾部不完整的那个元素。

    max_tokens 截断是最常见的一类坏输出——括号不配平，json.loads 直接
    报错。截到「最后一个完整元素」再补齐括号，能救回前面所有条目。
    """
    stack: list[str] = []
    in_str = False
    last_safe = -1
    i, n = 0, len(blob)
    while i < n:
        ch = blob[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
                last_safe = i + 1
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            last_safe = i + 1
        elif ch == ",":
            last_safe = i
        i += 1

    if not stack:
        return blob  # 括号本来就配平，不是截断问题
    head = blob[: last_safe + 1] if last_safe >= 0 else blob
    head = head.rstrip().rstrip(",")
    # 丢掉悬空的键（"...":  后面什么都没了）
    head = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", head)
    return head + "".join(reversed(stack))


def repair_json(raw: str) -> object | None:
    """尽力把坏掉的 JSON 解析出来；救不回来仍然返回 None。

    依次尝试：原样 → 补裸词引号 → 补裸词引号 + 补括号 → 补括号 + 补裸词
    引号。顺序有讲究：先补引号再补括号，因为裸词可能正好卡在截断处，
    先补括号会把它当成不完整元素丢掉。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    if start >= 0:
        # 有外层对象就以它为准。截断时 rfind("}") 是 -1，保留到结尾交给
        # _close_open；不能退化成取 [...]，那样会把内层数组当成整体返回
        blob = text[start:]
        end = blob.rfind("}")
        if end >= 0:
            blob = blob[: end + 1]
    else:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return None
        blob = text[start : end + 1]
    blob = blob.strip()
    if not blob:
        return None
    quoted = _quote_bare_tokens(blob)
    for candidate in (quoted, _close_open(quoted), _quote_bare_tokens(_close_open(blob))):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
