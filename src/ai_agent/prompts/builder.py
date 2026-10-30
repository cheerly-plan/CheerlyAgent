"""提示词拼装的最小工具集。

设计上刻意保持"零框架"：项目现在没有 langchain / jinja，也不打算引入。
这里只提供三件事——分层拼装、槽位替换、残留检查，够用就停手。

为什么槽位替换用 ``.replace`` 而不是 ``str.format`` / f-string：
底座类提示词的正文本身就含裸花括号（图谱抽取的 JSON 示例
``{"e":[["某某公司","org","",0]]}``），``format`` 会把它们当成字段而报错。
``doc_profiles.py`` 早在迁移前就是这么处理的，这里沿用同一套约定。
"""

from __future__ import annotations

import re

from ai_agent.llm.types import ChatMessage

# 槽位形如 {{DOC_CONFIG}}。只认大写字母 + 下划线，避免误伤正文里的 {{ 组合。
_SLOT_PATTERN = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


class PromptBuilder:
    """分层拼装 system prompt。

    顺序即分层顺序：先 ``section`` 的排前面。空块和 ``when=False`` 的块
    自动跳过，调用方不用写一堆 ``if``——这一点在主对话上很实用：没挂知识库
    时整块检索纪律不该出现（省 token，也避免模型去追一个不存在的工具）。

    用法::

        text = (
            PromptBuilder()
            .section("identity", IDENTITY)
            .section("kb", KB_DISCIPLINE, when=use_kb)
            .build()
        )
    """

    def __init__(self, separator: str = "\n\n") -> None:
        self._separator = separator
        self._sections: list[tuple[str, str]] = []

    def section(self, name: str, text: str, *, when: bool = True) -> PromptBuilder:
        """追加一层。``name`` 只用于日志定位和测试断言，不进正文。"""
        stripped = text.strip() if text else ""
        if when and stripped:
            self._sections.append((name, stripped))
        return self

    def build(self) -> str:
        return self._separator.join(text for _, text in self._sections)

    def messages(self) -> list[ChatMessage]:
        """包成单条 system 消息，供需要 ``ChatMessage`` 的调用点直接用。"""
        return [{"role": "system", "content": self.build()}]

    def names(self) -> list[str]:
        """已生效的分层名（被跳过的不会出现）。测试用来断言分层顺序。"""
        return [name for name, _ in self._sections]


def render(template: str, **slots: str) -> str:
    """把 ``{{KEY}}`` 替换成对应值。

    不主动调 ``assert_rendered``：有些模板会分多次渲染（先填配置块、
    再填片段），中途残留是合法的。是否需要断言由调用方决定。
    """
    out = template
    for key, value in slots.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def assert_rendered(text: str) -> str:
    """断言没有残留槽位。

    忘替换的槽位不会报错，只会让模型看到一句 ``{{DOC_CONFIG}}`` 然后把
    占位符本身当指令——这类 bug 极难从外部现象定位。宁可在渲染时就炸。
    """
    leftover = _SLOT_PATTERN.findall(text)
    if leftover:
        raise ValueError(
            f"提示词存在未替换的槽位：{sorted(set(leftover))}。"
            "检查 render() 的传参是否覆盖了模板里的全部占位符。"
        )
    return text
