from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm import metrics
from ai_agent.llm.client import json_response_format
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.facets import (
    coalesce_by_facet,
    format_profile_line,
    normalize_facet,
)
from ai_agent.memory.gate import jev_noul
from ai_agent.llm.json_utils import load_llm_json
from ai_agent.memory.llm_calls import complete_memory_chat
from ai_agent.memory.profile import ProfileAction, apply_profile_action
from ai_agent.memory.promotion import needs_source, observe_episode
from ai_agent.storage.store import store
from ai_agent.prompts.memory import (
    EXTRACT_SYSTEM,
    GATE_INSTRUCTIONS,
    LABEL_CURRENT_TURN,
    LABEL_EXISTING_EPISODES,
    LABEL_EXISTING_EPISODES_EMPTY,
    LABEL_EXISTING_PROFILE,
    LABEL_EXISTING_PROFILE_EMPTY,
    LABEL_SESSION_SUMMARY,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExtractedMemory:
    kind: Literal["profile", "episode"]
    content: str
    op: Literal["add", "update", "delete"] = "add"
    facet: str = "other"


def capture_long_term(
    user_id: str,
    turn: list[ChatMessage],
    summary: str,
    user_input: str,
    chat_session_id: str = "",
    memory_enabled: bool = True,
) -> None:
    if not memory_enabled:
        return
    if not any(msg.get("role") == "assistant" for msg in turn):
        return
    # Jev 门控：命中阈值以下直接跳过，省掉向量检索 + 小模型抽取；
    # 未配 key 或门控失败时放行，行为与旧版一致
    if _gate_blocks(turn, summary):
        return
    existing_episodes: list[str] = []
    try:
        existing_episodes = [
            item.text for item in store.search_episode(user_input, k=3, user_id=user_id)
        ]
    except Exception:
        logger.exception("相近情景记忆检索失败")
    existing_profile: list[str] = []
    try:
        existing_profile = [
            format_profile_line(item) for item in store.list_profile(user_id)
        ]
    except Exception:
        logger.exception("用户画像读取失败")
    items = extract_turn_memories(
        turn,
        summary,
        existing_episodes=existing_episodes,
        existing_profile=existing_profile,
    )
    for item in items:
        _write_extracted(
            user_id,
            item,
            chat_session_id=chat_session_id,
            user_input=user_input,
        )


def extract_turn_memories(
    turn: list[ChatMessage],
    summary: str = "",
    existing_episodes: list[str] | None = None,
    existing_profile: list[str] | None = None,
) -> list[ExtractedMemory]:
    source = _format_turn(turn)
    if not source.strip():
        return []
    if not any(msg.get("role") == "assistant" for msg in turn):
        return []
    user_content = _build_user_content(
        source, summary, existing_episodes, existing_profile
    )
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    # JSON 模式（MEMORY_EXTRACT_JSON_MODE=json_object）：网关不支持会直接 400，
    # 这时退回「prompt 约束 + load_llm_json 兜底」再跑一次，不让开关整条链路
    # 挂掉。默认 off，行为与开启前一致。
    fmt = json_response_format(settings.memory_extract_json_mode)
    try:
        # 768 而不是 384：记忆是 JSON 数组，一次抽两三条就要 250+ token，
        # 384 会把数组从中间截断（实测只留下第一条），解析出来静默少记
        raw = complete_memory_chat(messages, max_tokens=768, response_format=fmt)
    except LLMError as e:
        if fmt is None:
            logger.warning("抽取长期记忆失败: %s", e)
            return []
        logger.warning("JSON 模式抽取失败（网关可能不支持），回退 prompt 约束: %s", e)
        try:
            # 回退只是「不用网关的 response_format」，本地 Ollama 仍要
            # format=json + think=False：否则会退到开思考的自由文本分支，
            # 推理把预算烧光、正文空串，抽取静默返回空（曾导致用户自报姓名丢失）。
            raw = complete_memory_chat(messages, max_tokens=768, force_json=True)
        except LLMError as e2:
            logger.warning("抽取长期记忆失败: %s", e2)
            return []
    items = _parse_memories(raw, _user_text_of(turn))
    if not items and not (raw or "").strip():
        # 空正文：思考型模型把 num_predict 全用在 reasoning 上。
        # 单独计数——它和「本轮确实没有可记内容」是两回事。
        metrics.incr(metrics.MEMORY_EMPTY_REPLY)
        logger.warning("长期记忆抽取返回空正文（可能是思考耗尽预算）")
    self_intro = _self_intro_claim(turn)
    if self_intro and not any(
        item.kind == "profile" and item.facet in {"name", "occupation"} for item in items
    ):
        metrics.incr(metrics.MEMORY_MISSED_SELF_INTRO)
        logger.warning(
            "用户自报身份但未入库 self_intro=%s extracted=%s",
            self_intro[:40],
            [(item.kind, item.facet) for item in items],
        )
    if items:
        metrics.incr(metrics.MEMORY_OK)
        logger.info("长期记忆已抽取 count=%s", len(items))
    else:
        # 抽到 0 条是最难发现的降级：可能这轮真没什么可记，也可能模型/网关坏了。
        # 计数挂在 /health 上，持续为 0 而用户聊了很多就该查了
        metrics.incr(metrics.MEMORY_ZERO)
        logger.info("长期记忆本轮无抽取结果 raw_chars=%d", len(raw or ""))
    return items


def _write_extracted(
    user_id: str,
    item: ExtractedMemory,
    *,
    chat_session_id: str = "",
    user_input: str = "",
) -> None:
    try:
        if item.kind == "profile":
            # 出处门禁（与 episode→candidate 晋升路径同一纪律）：
            # 含数字/量化词的 profile 事实不能无外部出处直写画像——否则一句
            # 口误（「营收 12.5 亿」）就绕过晋升梯变成长期事实。source 只能
            # 来自外部可核验文档（年报/公告/台账/链接），普通对话派生不出，
            # 因此这里 source 恒为空时一律降级为 episode 观察，交给
            # observe_episode 的 missing_source hold 决定去留。
            if needs_source(item.content):
                written = store.write_episode(item.content, user_id=user_id)
                observe_episode(
                    written,
                    user_id=user_id,
                    chat_session_id=chat_session_id,
                    user_input=user_input,
                    source="",
                    facet=item.facet,
                )
                logger.info(
                    "数字 profile 无外部出处，降级为 episode 观察 facet=%s text=%s",
                    item.facet,
                    item.content[:60],
                )
                return
            apply_profile_action(
                ProfileAction(op=item.op, facet=item.facet, content=item.content),
                user_id=user_id,
            )
        else:
            written = store.write_episode(item.content, user_id=user_id)
            # 第 4 步：抽取落库后进观察态。开关关闭时 observe_episode 直接
            # 返回，既有写入语义逐字不变；开启时异常也由它自己 fail-open。
            observe_episode(
                written,
                user_id=user_id,
                chat_session_id=chat_session_id,
                user_input=user_input,
                facet=item.facet,
            )
    except Exception:
        logger.exception("长期记忆写入失败 kind=%s", item.kind)


def _gate_blocks(turn: list[ChatMessage], summary: str) -> bool:
    """Jev 门控：True = 本轮不值得抽取，直接跳过。

    默认不启用（settings.memory_gate_enabled=False）：端到端评测显示门控
    放行正例只有 36/40（最佳措辞 + 低阈值），而不启用是 40/40；抽取模型
    本身已能拒掉 43/44 的噪声，门控是净负收益。

    fail-open——门控不可用（未配 key / 调用失败）时返回 False 放行，
    行为退化为「每轮都调小模型抽取」。
    """
    if not settings.memory_gate_enabled:
        return False
    state = _build_gate_state(turn, summary)
    if not state:
        return False
    probability = jev_noul(GATE_INSTRUCTIONS, state)
    if probability is None:
        logger.info("Jev 门控不可用（未配 key 或调用失败），放行抽取")
        return False
    threshold = settings.memory_gate_threshold
    passed = probability >= threshold
    logger.info(
        "Jev 门控判定 prob=%.2f threshold=%.2f → %s",
        probability,
        threshold,
        "放行，继续抽取" if passed else "拦截，跳过本轮",
    )
    return not passed


def _build_gate_state(turn: list[ChatMessage], summary: str) -> str:
    """门控输入只送「摘要 + 本轮对话」，已有画像/相近记忆不喂给 Jev——
    去重交给抽取阶段看完整上下文处理，门控只做语义门槛判断。"""
    parts: list[str] = []
    text = summary.strip()
    if text:
        parts.append(f"{LABEL_SESSION_SUMMARY}{text[:600]}")
    turn_text = _format_turn(turn)[:6000]
    if turn_text:
        parts.append(f"{LABEL_CURRENT_TURN}\n{turn_text}")
    return "\n\n".join(parts)


def _build_user_content(
    source: str,
    summary: str,
    existing_episodes: list[str] | None,
    existing_profile: list[str] | None,
) -> str:
    parts: list[str] = []
    text = summary.strip()
    if text:
        parts.append(f"{LABEL_SESSION_SUMMARY}\n{text}")
    if existing_profile:
        parts.append(LABEL_EXISTING_PROFILE + "\n".join(existing_profile))
    else:
        parts.append(LABEL_EXISTING_PROFILE_EMPTY)
    if existing_episodes:
        lines = "\n".join(f"- {item}" for item in existing_episodes)
        parts.append(f"{LABEL_EXISTING_EPISODES}\n{lines}")
    else:
        parts.append(LABEL_EXISTING_EPISODES_EMPTY)
    parts.append(f"{LABEL_CURRENT_TURN}\n{source}")
    return "\n\n".join(parts)


# 用户自报姓名/身份的说法。用来在抽取结果为「空」时留下可排查的告警，
# 而不是静默丢弃。
_SELF_INTRO = re.compile(
    r"我叫|我是(?!指|说|在问)|我的名字|叫我|我是做|我从事|我做的是|我的职业"
)


def _self_intro_claim(turn: list[ChatMessage]) -> str:
    for msg in turn:
        if msg.get("role") != "user":
            continue
        text = str(msg.get("content") or "").strip()
        if text and _SELF_INTRO.search(text):
            return text
    return ""


def _format_turn(turn: list[ChatMessage]) -> str:
    lines: list[str] = []
    for msg in turn:
        line = _line(msg)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _line(msg: ChatMessage) -> str:
    role = msg.get("role")
    content = (msg.get("content") or "").strip()
    if role == "user":
        return f"用户：{content}" if content else ""
    if role == "assistant":
        calls = msg.get("tool_calls") or []
        names = ", ".join(call.get("name") or "tool" for call in calls)
        body = content[:800]
        if names and body:
            return f"助手：{body}（调用工具 {names}）"
        if names:
            return f"助手：调用工具 {names}"
        return f"助手：{body}" if body else ""
    if role == "tool":
        # 工具结果不是用户事实。以前把前 400 字喂给抽取器，模型会把
        # 「已确认中国银行总资产 411,084.49 亿元」这类检索结论写成用户记忆。
        # 这里只留工具名，用来判断「本轮查了库」，不给正文。
        name = msg.get("name") or "tool"
        return f"工具[{name}]：已调用（结果不写入记忆）"
    return ""


# 抽取模型常把本轮问答写成记忆。这些开头几乎一定是对话纪要，不是用户近况。
_RECAP_HEAD = re.compile(
    r"^(用户(询问|问了|问的是|讨论|要求|请求|查询|查了|了解到|分析|"
    r"正在分析|正在查询|确认了|提到|被[告知]|希望获取)"
    r"|助手(建议|回答|决定|通过|告知)"
    r"|讨论了|近期讨论)"
)
_COMPANY_MARK = re.compile(r"股份有限公司|有限责任公司")
# 工具名是通用的：任何「检索/查询工具」的输出都不该写成用户事实。
# 工具名本身：出现即说明这句在描述「查来的」内容。
_TOOL_MARK = ("kb_search", "kg_search", "search_history", "read_tool_result")
# 「助手建议/助手回答/助手告知」是对话纪要；而「希望助手协助」「让助手」
# 是用户对助手的长期要求，必须保留——之前把裸的「助手」当噪声，
# 误杀了「希望助手协助解读 JD」这类合法 directive。
_ASSISTANT_RECAP = re.compile(r"助手(建议|回答|决定|通过|告知|回复|提到|说)")
# 行业专用词（默认金融）。换业务时由 MEMORY_DOMAIN_NOISE_ENABLED 关闭，
# 通用规则（_ONGOING / _JOB_STATE / _TEMPORAL / _RECAP_HEAD）仍然生效。
_DOC_QA = ("招股", "年报披露")
_TOPIC_PREF = ("关注公司", "财务数据", "股东信息", "重大工程")

# 本轮动作 / 任务状态：都是「这一次在做什么」，不是用户长期事实。
# 一次性查询/任务动作。注意「用户正在用 X 做 Y」是自己持续的工作，不能算，
# 所以「正在」后面必须跟观察/查询动词，而不是任意动词。
# 本轮动作：句首是「用户 + 动作动词」，说明这句在描述这一次在做什么。
#
# 早期版本枚举动词（查询|确认|对比|分析|生成…），评测里 18/29 条 forget 用例
# 直接漏放——写、翻译、整理、统计、排查、算、定位 全不在表里，而枚举本身
# 永远列不全。改成结构判定：句首主体 + 动词 + 宾语。
#
# 排除两类会被误杀的合法记忆：
#   「用户正在做」后面若接的是项目/工作（「用户正在做 CheerlyAgent」）→ 该记
#   「用户是/用户有」是状态陈述，不是动作
_ONGOING = re.compile(
    r"^(用户|他|她)?(正在|已|刚刚|刚|刚才|在)"
    # 「在做/正在做一个…项目」是用户自己的长期项目，属于该记的 episode；
    # 「正在做一道题」仍是任务动作
    r"(?!做(?:一个|一[个套])?(?!一道))"
    # 允许「正在本地运行」「正在线上部署」这类方位/方式副词
    r"(?:本地|线上|后台|手动|自动|按|根据)?"
    # 动词表只收「单次任务」类动作。刻意不收「做/用/搭/建」——那些多数是
    # 用户自己的长期项目（「正在用 Koa 做 BFF」），收进来会误杀。
    r"(查询|查|查看|查了|确认|对比|分析|生成|想了解|了解|寻找|找|写|翻译|整理|"
    r"统计|排查|算|计算|定位|修改|改|转|提取|转换|解读|总结|搜索|搜|画|检查|"
    r"提交|测试|调试|阅读|看|跑|运行|执行|启动|部署|安装|"
    r"起草|撰写|批改|处理|出|做|按|统计|核对)"
    r"|(正在|在)把.{0,40}?(翻译|转成|改成|提取|整理|统计)"
)
# 后台任务状态：job_id、预计耗时、进度。
_JOB_STATE = re.compile(
    r"job_id|后台任务|预计\s*\d+\s*[-~至到]\s*\d+\s*分钟|任务已提交|进度"
)
# 股票代码 / 行情这类一次性标识（行业专用，可关）。
_MARKET_CODE = re.compile(
    r"\b\d{6}\b|融资融券|涨停|领涨|板块走势|成交量|换手率|总市值"
)
# 任务对象的「主语句」：句子以被处理对象开头时，几乎一定是任务纪要。
# 「患者于…入院」「该案已…开庭」「订单…已发货」都不含「用户」二字，
# 之前只能靠「用户正在…」拦，换个领域就漏。
_OBJECT_SUBJECT = re.compile(
    r"^(患者|病人|该患者|该病患|该案|本案|该单|订单|该订单|工单|"
    r"客户|该客户|该公司|该公司|本文件|该文件|此文件|该合同|本合同|"
    r"该商品|该产品|该词条|该学生)"
    # 中文没有词边界，\b 在「本合同第四条」处不成立，不能用来收尾
    r"|^[\u4e00-\u9fa5]{2,4}(本次|本期|这次|这一|于\d)"
)


def _domain_noise_enabled() -> bool:
    from ai_agent.core.config import settings

    return bool(getattr(settings, "memory_domain_noise_enabled", True))
# 短期行程：下周/明天出差这类不能当居住地。
_TEMPORAL = re.compile(
    r"^(用户)?(我)?(下周|本周|这周|明天|后天|今天|月底|下个月|近期)"
    r"|(下周|明天|后天|本周).{0,6}(出差|开会|出发|回)"
)
# 祈使句任务请求：「帮我写/把X翻译成/给标准答案」——用户在指派一次性任务，
# 不是陈述自己的长期事实。之前 _ONGOING 只认得少数动词（查询/确认/对比…），
# 写、翻译、整理、统计、排查全漏，评测里 15 条 forget 用例直接放行。
# 必须排除「帮我记住」这类真正要求记忆的说法。
_IMPERATIVE_TASK = re.compile(
    r"^(帮我|请帮|帮忙|替我|麻烦你?)(写|翻译|整理|统计|排查|算|生成|找|定位|修改|改|转|"
    r"提取|转换|解读|总结|对比|分析|查|搜|做|画|检查)"
    r"|^(把|将).{0,40}?(翻译|转成|转成|改成|提取|整理|统计)"
)
# 明确要求记住的祈使句不能算任务请求；「希望/想要 + 帮我…」是长期偏好，也不算任务
_REMEMBER_REQUEST = re.compile(r"记住|记一下|记下来|别忘|记住我|帮我记|希望|想要|倾向于|偏好")

# 当前进行中的动作 / 刚刚发生的观测：「我现在正在跑…」「我等下要去…」
# 与 _TEMPORAL 的区别：_TEMPORAL 只看句首时间词，这里看人 + 时间副词 + 动作。
_NOW_ACTING = re.compile(
    r"(我|用户)(现在|刚刚|刚才|等下|等会|一会儿|今天|明天|后天|这周|这阵)"
    r".{0,12}(要|去|在跑|在看|在写|正在|提交了|已经提交)"
)

# 抽取器输出常把第一人称转述成「用户…」，两种写法都算本人陈述。
_OWNER = r"(?:我|用户)"

# 用户明确陈述居住地的句式；只有命中才允许写 locale。
_LOCALE_STATEMENT = re.compile(
    _OWNER + r"(住在|住|常驻|长期在|在.{0,10}(工作|生活|执业|执业|上班)|是.{0,10}人)"
    # 「我在北京执业」也是工作地陈述
    + r"|^在[\u4e00-\u9fa5]{2,6}(执业|工作|生活|上班)"
)
# 用户明确陈述职业的句式；只有命中才允许写 occupation。
_OCCUPATION_STATEMENT = re.compile(
    _OWNER
    + r"(是|做|的职业|的工作|的岗位|身份)|"
    + _OWNER
    + r"在.{0,15}(工作|任职|上班)|职业是|岗位是"
    # 「从事半导体行业研究」「做后端开发」这类动词结构同样是职业陈述：
    # 抽取器常把「我是做 X 的」压缩成「从事 X」，之前会被误判为非本人陈述
    + r"|^(从事|做|负责|担任)"
)
# 系统默认值：写进去没有信息量。
_DEFAULT_LANGUAGE = {"中文", "汉语", "简体中文", "普通话", "简体"}
# 职业变更的历史→现在转述（「曾是X，现为Y」「之前是X，现在是Y」）
_OCCUPATION_TRANSITION = re.compile(
    r"(曾|以前|之前|原来)(是|为|做).{0,20}(现|现在|目前|如今)(是|为|做)"
)
# 「中大力德财务分析师」这类：公司简称 + 职业名词 = 由查询推断的身份。
# 必须排除行业名修饰（「半导体行业研究员」是正当职业描述，不是某家公司
# 的职位）：行业词与通用修饰词出现在职业名词前时不算公司名。
_INDUSTRY_WORD = re.compile(
    r"(行业|领域|方向|业务|产业|板块)"
)
_IN_COMPANY_AS_ROLE = re.compile(
    r"[\u4e00-\u9fa5]{2,8}(财务|数据|投资|证券)(分析师|研究员|经理|总监)"
)
# 职业名词兜底：句子里出现职业名，且没有把公司写成身份
_JOB_NOUN = re.compile(
    r"工程师|分析师|研究员|产品经理|设计师|运营|销售|律师|医生|教师|学生|"
    r"高管|总监|经理|程序员|开发者|架构师|顾问|会计|审计"
)


def _is_own_occupation(text: str) -> bool:
    """职业是否来自用户本人的陈述（而非问过哪家公司）。"""
    if _OCCUPATION_STATEMENT.search(text):
        return True
    if _OCCUPATION_TRANSITION.search(text):
        return True
    # 「前端开发工程师」这类裸职业名词也接受，但必须简短、不含公司名，
    # 且不是「在某公司做…分析师」这种由查询推断的身份。
    if _COMPANY_MARK.search(text) or _IN_COMPANY_AS_ROLE.search(text):
        return False
    # 行业名修饰的职业描述是合法的：「半导体行业研究员」不等于「某公司的分析师」
    if _INDUSTRY_WORD.search(text):
        return True
    return bool(_JOB_NOUN.search(text)) and len(text) <= 20


def is_transient_text(text: str) -> bool:
    """句子是否是「这一次」的内容（查询、任务状态、行情、短期行程）。

    抽取阶段用它决定是否入库；召回阶段也用它给陈旧噪声降权——历史库里
    已经写进去的行，不能只在写入侧拦住。
    """
    body = (text or "").strip()
    if not body:
        return True
    if _ONGOING.search(body) or _JOB_STATE.search(body):
        return True
    if _domain_noise_enabled() and _MARKET_CODE.search(body):
        return True
    if _RECAP_HEAD.match(body) or _TEMPORAL.match(body):
        return True
    if _OBJECT_SUBJECT.match(body):
        return True
    if _IMPERATIVE_TASK.search(body) and not _REMEMBER_REQUEST.search(body):
        return True
    if _NOW_ACTING.search(body):
        return True
    if any(mark in body for mark in _TOOL_MARK):
        return True
    if _ASSISTANT_RECAP.search(body):
        return True
    return _domain_noise_enabled() and any(mark in body for mark in _DOC_QA)


def _keep_extracted(item: ExtractedMemory, user_text: str = "") -> bool:
    """小模型常把招股书问答写成记忆，这里挡一层确定性噪声。

    规则都写成确定性正则，不做语义判断：记忆一旦写错会跟着画像长期注入，
    宁可在抽取阶段多丢一条，也不让「用户正在查询…」变成长期事实。

    ``user_text`` 是本轮用户原话。locale / occupation 的「必须第一人称陈述」
    检查**看原话，不看抽出值**：抽取器正常输出的是规范化后的值
    （``locale="深圳"``、``occupation="5年前端开发…"``），要求抽出值里带
    「我住在」「我是」等于把合格输出判成噪声——实测线上就是这么丢的。
    """
    text = item.content.strip()
    if item.op == "delete":
        # delete 是「撤回」意图：内容常为空或不完整，不做事实性过滤，
        # 否则用户说「别再记深圳」会被前置规则吞掉，旧记忆清不掉。
        return not _JOB_STATE.search(text)
    # 与召回层共用同一套判定：任务状态 / 本轮动作 / 任务对象主语句 /
    # 短期安排 / 工具输出 / 行业一次性信息。统一入口避免两处规则漂移
    # （曾经就因为这里少了一条，医疗、法律类的任务纪要能混进记忆）。
    if is_transient_text(text):
        return False
    if item.facet == "language" and text.strip(" 。.") in _DEFAULT_LANGUAGE:
        return False
    # 归因文本：有原话就看原话（抽取器的输出是规范化值，不是原话转述），
    # 没有原话才退回看抽出值本身。
    claim = (user_text or "").strip() or text
    if item.facet == "locale" and not _LOCALE_STATEMENT.search(claim):
        # 居住地必须来自用户明确陈述；「下周去深圳出差」不是 locale
        return False
    if item.facet == "occupation" and not _OCCUPATION_STATEMENT.search(claim):
        # 职业必须来自第一人称陈述；问过某公司 ≠ 用户是该公司分析师。
        # 但「曾是画家，现为前端工程师」这类历史→现在的转述是合法更新，
        # 由 is_own_occupation_statement 一并放行。
        if not _is_own_occupation(text):
            return False
    if item.facet in {"preference", "other", "constraint"}:
        if any(mark in text for mark in _TOPIC_PREF):
            return False
        if any(mark in text for mark in ("招股", "kb_search", "kg_search")):
            return False
    if item.facet == "occupation" and len(_COMPANY_MARK.findall(text)) >= 2:
        return False
    return True


def _user_text_of(turn: list[ChatMessage]) -> str:
    """本轮用户原话（多条用户消息拼起来），供 facet 陈述校验回看。"""
    parts = [
        str(msg.get("content") or "").strip()
        for msg in turn
        if msg.get("role") == "user"
    ]
    return "\n".join(part for part in parts if part)


def _parse_memories(raw: str, user_text: str = "") -> list[ExtractedMemory]:
    data = load_llm_json(raw)
    if data is None:
        logger.warning("抽取长期记忆 JSON 无法解析")
        return []
    if isinstance(data, dict):
        rows = data.get("memories")
        if rows is None and ("kind" in data or "content" in data):
            # 模型返回裸数组 ``[{...}]`` 时 load_llm_json 会剥掉外层，
            # 解成单个对象。之前这里直接 return []，整条记忆静默丢失——
            # 实测这是「该记没记」波动的主要来源之一。
            rows = [data]
    else:
        rows = data
    if not isinstance(rows, list):
        return []
    items: list[ExtractedMemory] = []
    for row in rows[:2]:
        item = _memory_from_row(row)
        if item is None:
            continue
        if not _keep_extracted(item, user_text):
            logger.info(
                "丢弃非跨会话记忆 kind=%s facet=%s text=%s",
                item.kind,
                item.facet,
                item.content[:80],
            )
            continue
        items.append(item)
    return _coalesce_memories(items)


def _coalesce_memories(items: list[ExtractedMemory]) -> list[ExtractedMemory]:
    profiles = [item for item in items if item.kind == "profile"]
    others = [item for item in items if item.kind != "profile"]
    return (
        coalesce_by_facet(
            profiles,
            facet_of=lambda item: item.facet,
            op_of=lambda item: item.op,
        )
        + others
    )


# kind 位置被写成 facet 名时，仍然按 profile 收下（见 _memory_from_row）
_PROFILE_KIND_ALIASES = frozenset(
    {"directive", "name", "occupation", "locale", "language", "preference", "constraint", "other"}
)


def _memory_from_row(row: object) -> ExtractedMemory | None:
    if not isinstance(row, dict):
        return None
    kind = row.get("kind")
    content = " ".join(str(row.get("content") or "").split()).strip()
    if kind == "episode":
        if len(content) < 4 or len(content) > 200:
            return None
        # 模型若给了 facet 就透传；空串让 observe_episode 用确定性推断，
        # 不要在这一层直接写死成 other
        facet = normalize_facet(row.get("facet"))
        return ExtractedMemory(
            kind="episode",
            content=content,
            facet="" if facet == "other" else facet,
        )
    if kind != "profile":
        # 模型偶尔把 facet 写进 kind（``{"kind":"directive","facet":"directive"}``）：
        # prompt 里 facet 清单和 kind 清单挨着，它把两者混了。这种情况不丢，
        # 按 profile + 对应 facet 收下——实测这条容错救回了 preference/
        # directive 类记忆的偶发丢失。
        if kind in _PROFILE_KIND_ALIASES:
            row = {**row, "kind": "profile", "facet": row.get("facet") or kind}
        else:
            return None
    op = str(row.get("op") or "add").strip()
    if op not in ("add", "update", "delete"):
        return None
    facet = normalize_facet(row.get("facet"))
    if op == "delete":
        return ExtractedMemory(kind="profile", content=content, op=op, facet=facet)
    if len(content) < 2 or len(content) > 200:
        return None
    return ExtractedMemory(kind="profile", content=content, op=op, facet=facet)
