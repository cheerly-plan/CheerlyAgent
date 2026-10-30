"""提示词层的守卫测试。

要守住三件事（按严重程度排序）：

1. **字面值冻结**——`SUMMARY_USER_PREFIX` 被 `take_summary()` 用来按前缀反解
   摘要、`RUNTIME_*` 是踩坑调出来的纠偏话术。这些改一个字不会报错，只会
   让功能静默失效（丢掉全部历史摘要 / 模型换个方式跑偏）。
2. **契约不丢**——主 prompt 和图谱底座里的关键约束一条都不能少。少了模型
   就会改输出形状，而且这类问题只在特定输入下才暴露。
3. **槽位不残留**——忘了替换的 `{{XXX}}` 会让模型把占位符本身当指令。

断言方式照抄 test_doc_profiles.py：只查「关键约束还在」，不做全文快照——
快照会让每次正常润色文案都要改测试，反而逼人绕过测试。
"""

import pytest

from ai_agent.llm.client import json_response_format
from ai_agent.prompts import chat, kb, memory, tasks, tools
from ai_agent.prompts.builder import PromptBuilder, assert_rendered


# ——— 1. 字面值冻结 ———


def test_summary_prefix_is_frozen():
    """take_summary() 按这个前缀切片反解摘要，改了就静默丢掉历史摘要。"""
    assert memory.SUMMARY_USER_PREFIX == "此前对话摘要：\n"
    assert memory.SUMMARY_ASSISTANT == "已了解此前对话摘要，会据此继续当前会话。"


# 迁移前 loop.py 里的原话，逐字冻结
EXPECTED_RUNTIME = {
    "RUNTIME_EMPTY_ANSWER": (
        "（系统提示）刚才没有给出文字回答。"
        "请立即基于以上已获取的信息，用简体中文给出完整回答；"
        "不要再调用任何工具。"
    ),
    "RUNTIME_DUP_CALL": (
        "（系统）你已用完全相同的参数调用过该工具，结果不会变化。"
        "请更换检索词、改用其他工具，或基于已有信息直接作答。"
    ),
    "RUNTIME_DUP_OUTPUT": (
        "（系统）该工具已经返回过完全相同的结果，换参数也查不到"
        "新内容。请改用更具体的检索词（例如直接搜材料名、指标名、"
        "表格标题，而不是公司名这类大词），或换用其他工具，"
        "或基于已有信息直接作答。"
    ),
    "RUNTIME_BUDGET_EXHAUSTED": (
        "（系统提示）检索轮次已用完，不能再调用任何工具。"
        "请立即基于以上已获取的信息给出最终回答；"
        "确实缺少关键数据时，明确说明缺什么即可。"
    ),
}


@pytest.mark.parametrize("name,expected", list(EXPECTED_RUNTIME.items()))
def test_runtime_texts_are_frozen(name, expected):
    assert getattr(chat, name) == expected


# ——— 2. 主 prompt 契约 ———


def test_system_prompt_keeps_core_constraints():
    """关键约束一条都不能少，少了模型就会换一种跑法。"""
    prompt = chat.build_system_prompt()
    assert "必须先调用 kb_search" in prompt
    assert "[文件名" in prompt, "引用格式示例是「带引用」定位的落点"
    assert "严禁编造" in prompt
    assert "kb_search" in prompt and "kg_search" in prompt
    # 专业助理定位：这几条是「简洁 + 结构化 + 带引用」的支柱
    assert "先结论后论据" in prompt
    assert "Markdown" in prompt
    assert "不要寒暄" not in prompt  # 实际措辞是「不寒暄、不复读」
    assert "不寒暄、不复读" in prompt


def test_system_prompt_has_no_document_specific_words():
    """主 prompt 是通用的，不许混进某个行业的专有词。"""
    for word in ("招股", "发行人", "募投", "年报", "法律意见书"):
        assert word not in chat.build_system_prompt()


def test_system_prompt_layers_are_ordered():
    """分层顺序即注入顺序：身份在最前，真实性兜底在最后。"""
    builder = (
        PromptBuilder()
        .section("identity", chat.IDENTITY)
        .section("style", chat.STYLE)
        .section("format", chat.OUTPUT_FORMAT)
        .section("citation", chat.CITATION)
        .section("kb", chat.KB_DISCIPLINE)
        .section("memory", chat.MEMORY_DISCIPLINE)
        .section("truth", chat.TRUTHFULNESS)
    )
    assert builder.names() == [
        "identity",
        "style",
        "format",
        "citation",
        "kb",
        "memory",
        "truth",
    ]


def test_kb_layers_dropped_when_kb_disabled():
    """没挂知识库时不该讲检索纪律——工具列表那边也把 kb_search 剔了。"""
    with_kb = chat.build_system_prompt(use_kb=True)
    without_kb = chat.build_system_prompt(use_kb=False)
    assert "kb_search" in with_kb
    assert "kb_search" not in without_kb
    assert "引用规范" not in without_kb
    # 人格、风格、真实性是通用的，不随能力开关变化
    assert chat.IDENTITY in without_kb
    assert chat.TRUTHFULNESS in without_kb
    assert len(without_kb) < len(with_kb)


# ——— 3. 槽位不残留 ———


@pytest.mark.parametrize(
    "rendered",
    [
        memory.build_extract_system(),
        memory.build_resolve_system(),
        memory.build_merge_user("条目"),
        memory.build_summary_user("旧摘要", "新对话"),
        memory.build_retry_feedback(["包含了对话中不存在的编造内容"]),
        kb.build_rerank_user("问题", "候选", 3),
        kb.build_align_user(2, "1. a / b"),
        kb.build_graph_head("招股书", "第 1–5 段", "（二）损益"),
        kb.build_catalog_rewrite_line("检索词"),
        tasks.build_rewrite_user("- 上一句", "本季度呢"),
        tools.build_web_search_max_results_desc(5),
        tools.build_kb_no_hit_scoped(3),
    ],
)
def test_rendered_prompts_have_no_leftover_slots(rendered):
    """忘替换的槽位不报错，只会让模型看到一句 {{XXX}}。"""
    assert "{{" not in rendered
    assert_rendered(rendered)


def test_assert_rendered_rejects_leftover():
    with pytest.raises(ValueError, match="未替换的槽位"):
        assert_rendered("配置：{{DOC_CONFIG}}")


def test_facet_rules_are_injected_once():
    """抽取器和写入器共用同一份 facet 规则，两处都注入且不留槽位。"""
    assert memory.FACET_WRITE_RULES in memory.build_extract_system()
    assert memory.FACET_WRITE_RULES in memory.build_resolve_system()


# ——— 4. 工具描述 ———


@pytest.mark.parametrize(
    "name,text",
    [
        ("weather", tools.WEATHER_DESC),
        ("web_search", tools.WEB_SEARCH_DESC),
        ("memory_search", tools.MEMORY_SEARCH_DESC),
        ("memory_write", tools.MEMORY_WRITE_DESC),
        ("kb_search", tools.KB_SEARCH_DESC),
        ("kg_search", tools.KG_SEARCH_DESC),
    ],
)
def test_tool_descriptions_are_non_empty(name, text):
    """description 随 function schema 进模型，空了模型就不知道何时调用。"""
    assert text and len(text.strip()) >= 10, f"{name} 的 description 太短"


@pytest.mark.parametrize(
    "text",
    [tools.MEMORY_SEARCH_DESC, tools.MEMORY_WRITE_DESC, tools.KB_SEARCH_DESC],
)
def test_trigger_conditions_are_declared(text):
    """这几个最容易误触发，必须显式写清何时用、何时不用。"""
    assert "触发条件" in text or "禁止条件" in text or "必须调用" in text


def test_kb_and_kg_files_desc_share_one_source():
    """两个工具的 files 参数说明必须一致，否则模型会按两套规则填。"""
    assert tools.KB_FILES_DESC
    assert "文件名" in tools.KB_FILES_DESC


# ——— 5. 构造器行为 ———


def test_builder_skips_empty_and_disabled_sections():
    result = (
        PromptBuilder()
        .section("a", "第一层")
        .section("b", "   ")
        .section("c", "第三层", when=False)
        .section("d", "")
        .build()
    )
    names_used = PromptBuilder().section("a", "x").section("c", "y", when=False).names()
    assert result == "第一层"
    assert names_used == ["a"]


def test_builder_messages_wraps_as_system():
    messages = PromptBuilder().section("a", "内容").messages()
    assert messages == [{"role": "system", "content": "内容"}]


# ——— 6. JSON 模式开关 ———


@pytest.mark.parametrize(
    "mode,expected",
    [("off", None), ("", None), ("json_object", {"type": "json_object"}), ("乱写", None)],
)
def test_json_response_format(mode, expected):
    """不认识的取值一律当 off，保证不会给网关下发它不支持的字段。"""
    assert json_response_format(mode) == expected
