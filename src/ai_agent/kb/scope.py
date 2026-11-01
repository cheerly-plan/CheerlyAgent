"""会话级知识库检索范围与查询变换。

检索范围（kb_scope）：前端在输入框里选定「只查这几份文件」后，范围随
ChatRequest 传进来。工具调用发生在同一个生成器上下文里
（chat_sse_events → run_stream → execute_tool），用 contextvar 传递比改
Tool.func 的签名干净。模型用文件名圈定范围（tools 的 files 参数），服务端
再解析成 id；用户圈定的会话范围只在模型没指定文件时生效。

查询变换（query rewrite）：
- effective_kb_query：多轮对话改写（补全 / 指代消解），把当前追问结合上一轮
  对话改写成自包含检索词；无上下文或降级时退回原话。
- expand_queries：检索增强改写（multi-query），把一条查询扩写成多个同义 /
  不同角度的变体，分别检索后由检索层做 RRF 融合，提升对比 / 口语长问句召回。
- decompose_queries：子问题分解（sub-question），把「A 和 B 的 X」这类多主体
  对比问句拆成「A 的 X」「B 的 X」各自独立检索，保证每个主体至少一条命中
  （multi-query 的同义改写对这类问句无效，拆主体才是正解）。
三路都带完整降级链，失败一律退回原 query，不挡主流程。
"""

import re
import uuid
from collections.abc import Iterable
from contextvars import ContextVar, Token

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import LLMError, complete_chat

logger = get_logger(__name__)

_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("kb_scope", default=())
# 查询改写上下文：上一轮对话（user 问 + assistant 答）的文本串，供
# effective_kb_query 补全多轮追问里省略的主体 / 指代。
_REWRITE_CTX: ContextVar[list[str]] = ContextVar("kb_rewrite_ctx", default=[])


def set_kb_scope(file_ids: Iterable[str] | None) -> Token:
    cleaned = tuple(str(fid).strip() for fid in (file_ids or []) if str(fid).strip())
    return _SCOPE.set(cleaned)


def reset_kb_scope(token: Token) -> None:
    _SCOPE.reset(token)


def kb_scope() -> tuple[str, ...]:
    return _SCOPE.get()


def set_rewrite_context(turns: Iterable[str] | None) -> Token:
    """设置查询改写的对话上下文（上一轮 user 问 + assistant 答）。"""
    cleaned = [s.strip() for s in (turns or []) if s and str(s).strip()]
    return _REWRITE_CTX.set(cleaned)


def reset_rewrite_context(token: Token) -> None:
    _REWRITE_CTX.reset(token)


def rewrite_context() -> list[str]:
    return _REWRITE_CTX.get()


def effective_kb_query(query: str) -> str:
    """检索词先做多轮对话改写（补全 / 指代消解），返回自包含检索词。

    结合上一轮对话把当前追问里省略的主体、指代补全（如「那它去年呢？」→
    补全公司名）。改写关闭 / 无上下文 / 失败 / 结果等于原句时退回原话。
    """
    q = (query or "").strip()
    if not q:
        return q
    return _rewrite_query(q, rewrite_context())


def parse_scope_uuids(raw_ids: Iterable[str] | None) -> list[uuid.UUID] | None:
    parsed: list[uuid.UUID] = []
    for raw in raw_ids or []:
        try:
            parsed.append(uuid.UUID(str(raw).strip()))
        except ValueError:
            continue
    return parsed or None


def resolve_tool_file_ids(
    files: list[str] | None, query: str | None = None
) -> list[uuid.UUID] | None:
    """模型给的文件名/主体名优先；没给就先用文档路由，再退回会话范围 / 全库。

    返回语义（工具层据此区分）：
    - 非空 list：检索范围已确定（模型点名解析成功，或文档路由选出候选）
    - None：既没点名、路由也没结果、会话范围为空 → 调用方按全库处理
    - 空 list []：模型指定了 files 但一个都解析不到（编造的文件名/主体名）
      → 调用方应给出「未找到」提示，而不是静默退回全库查无关内容

    文档路由放在这条链路上而不是提示词里：files 是否填、填得准不准本来
    就不该由模型负责——摘要向量（见 kb/router.py）在检索层就能选出候选
    文档，文件清单因此不必再占用上下文。
    """
    from ai_agent.kb.retrieval import resolve_files_by_name

    if files:
        # resolve_files_by_name 已区分三种语义：
        #   非空 list → 圈定成功；[] → 点名解析不到（未找到）；
        #   None → 命中覆盖全部、无缩小意义（退回全库）。
        # 直接透传，不再把 None 误判成「未找到」。
        return resolve_files_by_name(files)
    routed = _route(query)
    if routed:
        return routed
    return parse_scope_uuids(kb_scope())


def _route(query: str | None) -> list[uuid.UUID]:
    """按文档摘要做语义路由。任何异常都退回空列表（等价于路由没结果）。

    路由失败不能影响对话：底层接口 TP99 抖动一次，上层还有会话范围和
    全库两条路可走，抛出去只会让 kb_search 直接报 ERROR。
    """
    text = (query or "").strip()
    if not text:
        return []
    try:
        from ai_agent.kb.router import route_files

        return route_files(text)
    except Exception:
        logger.warning("文档路由失败，退回会话范围 / 全库", exc_info=True)
        return []


# ——— 多轮对话改写（补全 + 指代消解）———

_REWRITE_SYSTEM = (
    "你是检索词改写助手。用户在多轮对话里追问时，常常省略上一轮已经说过的"
    "主体或语境（例如上一轮在问「中国能源建设股份有限公司」，这一轮只说"
    "「股东持股情况」）。你的任务：结合上一轮对话，把当前这句改写/扩写成"
    "一条自包含、能独立完成检索的查询词。\n"
    "要求：\n"
    "1. 必须结合上一轮对话的语境，把省略的主体、指代补全，必要时可适当扩写"
    "补充同义词、相关要点，让检索更充分；\n"
    "2. 补全的主体应来自上一轮对话里出现过的事实，不要凭空编造与上下文"
    "无关的专有名词；\n"
    "3. 如果当前这句本身已经自包含、无需改写，原样输出当前这句即可；\n"
    "4. 只输出改写后的那一条查询词，不要任何解释、标点引导或换行。"
)


def _rewrite_query(user_input: str, prev_turn: list[str]) -> str:
    """结合上一轮对话，把当前追问改写成自包含的检索词（补全 + 扩写）。

    上下文取「上一轮 user 问 + assistant 答」两段——主体常出现在最早一问或
    助手回答里，据此能把「本季度」这类指代补回正确的公司。失败 / 超时 / 无上文
    一律退回原话（非阻塞）。
    """
    if not settings.kb_rewrite_enabled:
        return user_input
    prev = [s.strip() for s in prev_turn if s and s.strip()]
    if not prev:
        return user_input
    model = settings.kb_rewrite_model or settings.resolve_llm()
    context = "\n".join(f"- {s}" for s in prev)
    try:
        out = complete_chat(
            [
                {"role": "system", "content": _REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"上一轮对话：\n{context}\n\n"
                        f"当前问题：{user_input}\n"
                        "请结合上一轮对话，输出改写后的检索词。"
                    ),
                },
            ],
            model=model,
            timeout=settings.kb_rewrite_timeout_seconds,
            max_retries=0,
            max_tokens=256,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except LLMError as e:
        logger.warning("kb 对话改写失败，退回原话：%s", e)
        return user_input
    except Exception:
        logger.exception("kb 对话改写异常，退回原话")
        return user_input
    rewritten = (out or "").strip().strip("\"'“”‘’")
    if not rewritten or rewritten == user_input.strip():
        return user_input
    logger.info("kb 对话改写：%r -> %r", user_input, rewritten)
    return rewritten


# ——— 检索增强改写（multi-query）———

_MULTIQUERY_SYSTEM = (
    "你是检索查询扩展助手。给定一条知识库检索词，请生成若干条语义相近但"
    "表述不同的检索词变体，从不同角度覆盖同一信息需求，以提升向量检索的召回。\n"
    "要求：\n"
    "1. 变体必须保留原查询的关键实体、数字、专有名词，不可丢失或替换；\n"
    "2. 每条变体是一个独立、完整的检索词，不要疑问句、不要解释；\n"
    "3. 覆盖角度：同义词替换、近义表述、拆开复合概念、补全隐含主语；\n"
    "4. 返回恰好 N 条，每条一行，不要编号、不要任何额外文字。"
)


def expand_queries(query: str, count: int | None = None) -> list[str]:
    """检索增强改写：把一条查询扩写成多条变体（multi-query）。

    返回包含原 query 在内的去重变体列表；关闭 / 失败 / 超时一律退回 [query]。
    各变体由检索层分别检索后做 RRF 融合。
    """
    q = (query or "").strip()
    if not q:
        return [q] if q else []
    if not settings.kb_multiquery_enabled:
        return [q]
    n = count or settings.kb_multiquery_count
    try:
        out = complete_chat(
            [
                {"role": "system", "content": _MULTIQUERY_SYSTEM},
                {
                    "role": "user",
                    "content": f"请生成 {n} 条检索词变体，每条一行：\n{q}",
                },
            ],
            model=settings.kb_multiquery_model or settings.resolve_llm(),
            timeout=settings.kb_multiquery_timeout_seconds,
            max_retries=0,
            max_tokens=256,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except Exception:
        logger.warning("kb multi-query 扩展失败，退回原 query：%s", q[:60])
        return [q]
    variants = [v.strip().strip("\"'“”‘’") for v in (out or "").splitlines()]
    variants = [v for v in variants if v and v != q]
    if not variants:
        return [q]
    seen: set[str] = {q}
    result = [q]
    for v in variants:
        if v not in seen:
            seen.add(v)
            result.append(v)
        if len(result) >= n:
            break
    return result


# ——— 子问题分解（sub-question）———

# 并列 / 对比信号词：命中才考虑拆。多主体检测（mentioned_subjects）会再卡一道，
# 双重前置把 LLM 调用压缩到「真的多主体对比问句」这一小撮上
_DECOMPOSE_SIGNAL = re.compile(r"和|与|及|以及|、|对比|比较|相比|分别|vs", re.IGNORECASE)

_DECOMPOSE_SYSTEM = (
    "你是检索子问题分解助手。用户一条问句里如果同时问了多个主体（多家公司/多份"
    "文件）的同一指标，比如「中国银行和长安汽车的营业收入分别是多少」，请把它拆成"
    "若干条只问单个主体的独立检索词，每条自包含、可独立完成检索。\n"
    "要求：\n"
    "1. 只拆「多主体并列/对比」的问句；单主体问句（哪怕含多个指标，如「中国银行的"
    "营收和净利润」）不要拆，原样输出；\n"
    "2. 每条子问题必须完整保留原问句里的指标、数字、时间、口径等关键约束，只把主体"
    "拆开，不要丢约束；\n"
    "3. 每个主体对应一条子问题，主体名用原问句里的写法，不要另起别称；\n"
    "4. 每条子问题输出成紧凑的检索词「主体+指标+时间」（如「中国银行前三季度营业"
    "收入」），去掉「是多少」「分别」「请问」等疑问词与语气词，不要疑问句、不要解释；\n"
    "5. 输出若干条，每条一行，不要编号、不要任何额外文字。"
)


def decompose_queries(
    query: str, file_ids: list[uuid.UUID] | None = None
) -> list[str]:
    """多主体对比问句 → 单主体子问题列表（sub-question decomposition）。

    只对「A 和 B 的 X」这类多主体并列/对比问句生效：拆成「A 的 X」「B 的 X」
    分别独立检索，再由检索层轮转合并，保证每个主体至少一条命中（multi-query
    的同义改写对这类问句无效，实测反而放大第一主体优势）。关闭 / 无并列信号 /
    非多主体 / 失败 / 拆不出多条时一律退回 [query]。
    """
    q = (query or "").strip()
    if not q:
        return [q] if q else []
    if not settings.kb_decompose_enabled:
        return [q]
    if not _DECOMPOSE_SIGNAL.search(q):
        return [q]
    # 确定性前置：query 得真的点名了 ≥2 个不同文件主体，才值得多付一次 LLM 调用
    from ai_agent.kb.retrieval import mentioned_subjects

    if len(mentioned_subjects(q, file_ids=file_ids)) < 2:
        return [q]
    try:
        out = complete_chat(
            [
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": f"请拆解下面这条问句：\n{q}"},
            ],
            model=settings.kb_decompose_model or settings.resolve_llm(),
            timeout=settings.kb_decompose_timeout_seconds,
            max_retries=0,
            max_tokens=256,
            # 拆解是结构化任务，温度取 0 保证同一问句每次拆出同样的子问题，
            # 否则评测结果会随采样抖动、改动好坏无法归因
            temperature=0.0,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except Exception:
        logger.warning("kb 子问题分解失败，退回原 query：%s", q[:60])
        return [q]
    subs = [s.strip().strip("\"'“”‘’") for s in (out or "").splitlines()]
    subs = [s for s in subs if s and s != q]
    if len(subs) < 2:
        return [q]
    seen: set[str] = set()
    result: list[str] = []
    for s in subs:
        if s not in seen:
            seen.add(s)
            result.append(s)
        if len(result) >= settings.kb_decompose_max:
            break
    if len(result) < 2:
        return [q]
    logger.info("kb 子问题分解：%r -> %s", q, result)
    return result


def extract_prev_turn(messages: list[dict], max_turns: int = 2) -> list[str]:
    """提取「当前 user 句」之前最近 max_turns 轮对话（每轮 user 问 + assistant 答）。

    倒序找最后一个有效 user 句作为当前句；再往前逐轮收集 user 问与紧跟的
    assistant 最终回答（跳过带 tool_calls 的中间态），按时间正序摊平。多轮
    收集能覆盖「主体出现在更早一轮、中间隔无主体追问」的情况。
    """
    if not messages or max_turns <= 0:
        return []
    cur_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "user" and str(m.get("content") or "").strip():
            cur_idx = i
            break
    if cur_idx is None or cur_idx == 0:
        return []
    turns: list[str] = []
    collected = 0
    for i in range(cur_idx - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "user" and str(m.get("content") or "").strip():
            # 每轮作为「user 问 + assistant 答」一个整体，倒序遍历时 prepend，
            # 既保持时间正序，又保证每对内 user 在前（而非末尾整体 reverse，
            # 后者会把每对内部顺序也颠倒）。
            pair = [str(m.get("content") or "").strip()]
            # 找该 user 句之后、下一句 user 之前，最近一个有正文的 assistant
            # 回答：跳过带 tool_calls 的中间态与 tool 消息，取最终正文。
            j = i + 1
            while j < len(messages) and messages[j].get("role") != "user":
                if messages[j].get("role") == "assistant":
                    content = str(messages[j].get("content") or "").strip()
                    if content:
                        pair.append(content)
                        break
                j += 1
            turns = pair + turns
            collected += 1
            if collected >= max_turns:
                break
    return turns
