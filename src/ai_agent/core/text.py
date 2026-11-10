"""文本裁剪的公共实现：上下文预算与工具输出共用一套规则。

原先两处都是纯前缀硬截断（``text[:3000]``）：下游拿到的是没有标记的残片，
下载链接被切断、表格行被砍在中间、JSON 被截半截，模型还以为自己看到了
完整结果。这里统一改成「保头保尾 + 显式省略标记」，并保证返回值长度不
超过调用方给的预算——标记本身也要算进预算，不能裁完又超。
"""

from __future__ import annotations

CLIP_MARKER = "\n[已省略 {n} 字符；需要完整内容请收窄查询或重新调用]\n"


def clip_text(text: str, limit: int, *, marker: str = CLIP_MARKER) -> str:
    """把 text 裁到不超过 limit 个字符，按行保头保尾，中间插入省略标记。

    limit <= 0 表示不限制。head 拿 2/3 预算、tail 拿 1/3：工具输出通常
    「结论在前、链接/落款在后」，两头都比中间值钱。切点尽量对齐换行，
    避免留下半行表格或半个链接。
    """
    if limit <= 0 or len(text) <= limit:
        return text
    # 先用最大可能的省略量占位预留空间；实际标记只会更短，结果必然 <= limit
    probe = marker.format(n=len(text))
    budget = max(0, limit - len(probe))
    head_budget = budget * 2 // 3
    tail_budget = budget - head_budget
    head = _align_head(text[:head_budget])
    tail = _align_tail(text[max(len(head), len(text) - tail_budget) :])
    result = head + marker.format(n=len(text) - len(head) - len(tail)) + tail
    return result[:limit] if len(result) > limit else result


def _align_head(chunk: str) -> str:
    """切点落在行中间时退回上一个换行，不留下半行。"""
    cut = chunk.rfind("\n")
    return chunk[: cut + 1] if cut > 0 else chunk


def _align_tail(chunk: str) -> str:
    """尾部切点落在行中间时从下一个换行开始。"""
    cut = chunk.find("\n")
    return chunk[cut + 1 :] if cut >= 0 else chunk
