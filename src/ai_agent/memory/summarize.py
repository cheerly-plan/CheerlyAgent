from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.gate import jev_noul_batch
from ai_agent.memory.short_term import crude_summary, flatten, iter_turns

logger = get_logger(__name__)

SUMMARY_SYSTEM = """
你是会话摘要器。把对话压缩成后续回复仍需要的上下文摘要。
规则：
1. 只输出摘要正文，不要标题、前缀或解释；
2. 保留关键事实、用户偏好、已做约定、未完成事项和重要结论；
3. 去掉寒暄、重复内容和工具调用细节；
4. 一次性时效信息（天气、汇率、股价、临时查询结果）不写入摘要——
   它们很快过时，留在摘要里会误导后续回复；
5. 不要编造对话中没有的信息；
6. 使用简洁中文陈述句，控制在 400 字以内。
"""

# 摘要质量校验（Jev 双 Noul 并行）。摘要会每轮注入对话开头直到会话结束，
# 一条编造内容会持续污染后续上下文，所以生成后加一道独立第二意见。
# 阈值先硬编码，按日志观察误报率后再考虑提升为配置项。
_FABRICATION_INSTRUCTIONS = "摘要是否包含对话中不存在、也无法从对话合理推断的信息（编造）"
_OMISSION_INSTRUCTIONS = "摘要是否遗漏了对话中已确认的约定、偏好或未完成事项"
# 编造危害大，阈值放低些；遗漏相对可容忍且判断模糊，要求高置信才动手
_FABRICATION_THRESHOLD = 0.6
_OMISSION_THRESHOLD = 0.85
# 最多生成 2 次：首次 + 带落选原因的重试
_MAX_SUMMARY_ATTEMPTS = 2

# 摘要预过滤（Jev 逐轮 Noul）：溢出轮次先判断有无保留价值，
# 天气/寒暄这类轮次直接剔除，不送精摘——精摘输入更短更聚焦。
# 阈值与长期记忆门控一致（偏低）：漏保留比多保留代价大
_TURN_KEEP_INSTRUCTIONS = (
    "这一轮对话是否包含值得保留进会话摘要的信息？"
    "值得保留：关键事实、用户偏好、已做约定、未完成事项、重要结论。"
    "不值得保留：寒暄、闲聊、一次性时效信息（如天气、汇率、临时查询结果）、"
    "工具调用的中间过程。"
)
_TURN_KEEP_THRESHOLD = 0.4


def summarize_session(previous: str, overflow: list[ChatMessage]) -> str:
    overflow = _filter_overflow_turns(overflow)
    source = _format_overflow(overflow)
    if not source.strip():
        # 全部轮次被过滤：旧摘要保持原样，不调精摘
        return previous.strip()
    if previous.strip():
        base_content = f"已有摘要：\n{previous.strip()}\n\n新增对话：\n{source}"
    else:
        base_content = f"对话内容：\n{source}"
    feedback = ""
    for _ in range(_MAX_SUMMARY_ATTEMPTS):
        user_content = (
            f"{base_content}\n\n{feedback}" if feedback else base_content
        )
        model = settings.resolve_llm(settings.short_term_summary_model)
        try:
            raw = complete_chat(
                [
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                model=model,
                timeout=settings.short_term_summary_timeout_seconds,
                max_retries=0,
                max_tokens=512,
                extra_body={
                    "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except LLMError as e:
            logger.warning("生成会话摘要失败，回退为摘录: %s", e)
            return crude_summary(previous, overflow)
        text = " ".join((raw or "").split()).strip()
        if not text:
            return crude_summary(previous, overflow)
        verdict = _jev_check(previous, overflow, text)
        if verdict is None:
            # 校验不可用（未配 key / Jev 失败）：fail-open 直接接受
            logger.info("会话摘要已更新 chars=%s（未校验）", len(text))
            return text
        fabrication, omission = verdict
        if (
            fabrication < _FABRICATION_THRESHOLD
            and omission < _OMISSION_THRESHOLD
        ):
            logger.info(
                "会话摘要已更新 chars=%s 校验通过 fabrication=%.2f omission=%.2f",
                len(text),
                fabrication,
                omission,
            )
            return text
        logger.warning(
            "摘要校验不合格 fabrication=%.2f omission=%.2f", fabrication, omission
        )
        if feedback:
            # 已是重试结果，按错误类型分类处置：
            # 仍疑似编造 → 摘要宁可保真截断也不能带错信息进上下文，回退摘录；
            # 仅遗漏 → 两版里挑较好的接受，crude 不压缩反而更糟
            if fabrication >= _FABRICATION_THRESHOLD:
                logger.warning("重试后仍疑似编造，回退为摘录")
                return crude_summary(previous, overflow)
            logger.warning("重试后仅遗漏，接受重试版")
            return text
        feedback = _retry_feedback(fabrication, omission)
    return crude_summary(previous, overflow)


def _filter_overflow_turns(overflow: list[ChatMessage]) -> list[ChatMessage]:
    """Jev 逐轮过滤溢出对话：无保留价值的轮次不送精摘。

    fail-open——Jev 不可用时原样返回全部溢出内容，行为与旧版一致。
    只有一轮时过滤没有意义（要么整段保留要么整段丢），直接跳过。
    """
    turns = iter_turns(overflow)
    if len(turns) < 2:
        return overflow
    answers = jev_noul_batch(
        {f"turn_{i}": _TURN_KEEP_INSTRUCTIONS for i in range(len(turns))},
        _turn_filter_state(turns),
    )
    if answers is None:
        return overflow
    kept = [
        turn
        for i, turn in enumerate(turns)
        if answers[f"turn_{i}"] >= _TURN_KEEP_THRESHOLD
    ]
    if len(kept) < len(turns):
        logger.info(
            "Jev 摘要预过滤：剔除 %s/%s 个低价值轮次",
            len(turns) - len(kept),
            len(turns),
        )
    return flatten(kept)


def _turn_filter_state(turns: list[list[ChatMessage]]) -> str:
    parts: list[str] = []
    for i, turn in enumerate(turns):
        parts.append(f"轮次{i}：\n{_format_overflow(turn)}")
    return "\n\n".join(parts)[:8000]


def _jev_check(
    previous: str, overflow: list[ChatMessage], text: str
) -> tuple[float, float] | None:
    """并行问编造/遗漏两个 Noul，返回 (fabrication, omission)。

    None 表示校验不可用（未配 key / 调用失败 / 响应异常）。
    """
    answers = jev_noul_batch(
        {
            "fabrication": _FABRICATION_INSTRUCTIONS,
            "omission": _OMISSION_INSTRUCTIONS,
        },
        _check_state(previous, overflow, text),
    )
    if answers is None:
        return None
    return answers["fabrication"], answers["omission"]


def _check_state(
    previous: str, overflow: list[ChatMessage], text: str
) -> str:
    """校验输入：ground truth 必须是「旧摘要 + 溢出对话」的并集——
    精摘会保留旧摘要内容，只给溢出部分会把合法继承误判为编造。"""
    parts: list[str] = []
    if previous.strip():
        parts.append("已有摘要（其内容并入新摘要属正常继承，不算编造）：")
        parts.append(previous.strip()[:3000])
    parts.append("对话原文：")
    parts.append(_format_overflow(overflow)[:6000])
    parts.append("待检摘要：")
    parts.append(text)
    return "\n".join(parts)


def _retry_feedback(fabrication: float, omission: float) -> str:
    problems: list[str] = []
    if fabrication >= _FABRICATION_THRESHOLD:
        problems.append("包含了对话中不存在的编造内容")
    if omission >= _OMISSION_THRESHOLD:
        problems.append("遗漏了已确认的约定或未完成事项")
    return (
        "质量校验发现上一版摘要" + "、".join(problems)
        + "。请严格基于对话原文重新生成摘要，不得添加原文没有的信息。"
    )


def _format_overflow(overflow: list[ChatMessage]) -> str:
    budget = max(int(settings.short_term_memory_max_tokens), 256)
    lines: list[str] = []
    used = 0
    for msg in overflow:
        line = _line(msg)
        if not line:
            continue
        remain = budget - used
        if remain <= 0:
            lines.append("…")
            break
        if len(line) > remain:
            lines.append(line[:remain] + "…")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _line(msg: ChatMessage) -> str:
    role = msg.get("role")
    content = (msg.get("content") or "").strip()
    if role == "user":
        return f"用户：{content}" if content else ""
    if role == "assistant":
        calls = msg.get("tool_calls") or []
        names = ", ".join(call.get("name") or "tool" for call in calls)
        if names and content:
            return f"助手：{content}（调用工具 {names}）"
        if names:
            return f"助手：调用工具 {names}"
        return f"助手：{content}" if content else ""
    if role == "tool":
        name = msg.get("name") or "tool"
        snippet = content[:200]
        return f"工具[{name}]：{snippet}" if snippet else ""
    return ""
