"""全站 LLM 提示词的统一存放处。

以前提示词散在 15 个文件里（agents/memory/kb/tasks/tools 各写各的），
拼装方式有 f-string 内插、``.replace`` 槽位、直接字符串相加、函数内联
四种，改一处很容易漏掉另一处。这里按链路集中，文案与拼装逻辑分离：

    builder.py  分层构造器与槽位工具（唯一的"机制"）
    chat.py     主对话：人格、风格、输出结构、引用规范、工具纪律
    memory.py   长期记忆抽取、画像写入、情景合并、会话摘要
    kb.py       知识库目录、重排、图谱抽取底座、实体对齐、视觉描述
    tasks.py    会话标题、检索词改写
    tools.py    工具 description（会随 function schema 一起进模型）

约定：文案常量用大写，拼装入口用 ``build_*`` 函数。
"""

from ai_agent.prompts import builder, chat, kb, memory, tasks, tools

__all__ = ["builder", "chat", "kb", "memory", "tasks", "tools"]
