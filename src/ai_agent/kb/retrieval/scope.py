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
两路都带完整降级链，失败一律退回原 query，不挡主流程。
多主体对比不在这里拆开：文件范围圈定后，在同一批文件里查一次字段。
"""

import uuid
from collections.abc import Iterable
from contextvars import ContextVar, Token

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import LLMError, complete_chat, lightweight_llm

logger = get_logger(__name__)

_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("kb_scope", default=())
# 本轮用户原句。模型常把检索词收成科目名，圈文件时还要能看到公司名。
_QUESTION: ContextVar[str] = ContextVar("kb_question", default="")
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


def set_kb_question(text: str | None) -> Token:
    return _QUESTION.set((text or "").strip())


def reset_kb_question(token: Token) -> None:
    _QUESTION.reset(token)


def current_question() -> str:
    return _QUESTION.get()


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

    explicit = bool(files)
    if files:
        # resolve_files_by_name 已区分三种语义：
        #   非空 list → 圈定成功；[] → 点名解析不到（未找到）；
        #   None → 命中覆盖全部、无缩小意义（退回全库）。
        # 直接透传，不再把 None 误判成「未找到」。
        chosen = resolve_files_by_name(files)
        if chosen == []:
            return []
    else:
        routed = _route(query)
        chosen = routed or parse_scope_uuids(kb_scope())
    return _narrow_by_period(query, chosen, explicit=explicit)


def _narrow_by_period(
    query: str | None,
    file_ids: list[uuid.UUID] | None,
    *,
    explicit: bool,
) -> list[uuid.UUID] | None:
    """按问句里的报告期收窄文件。

    指定了季度或年份就只留那一期；没提时期时，同一主体只留最新一期。
    同比、历年不收窄。点了两个年份又指定季度时，同一主体的这一季都留下，
    路由只命中其中一年也会把另一年补进来。显式点名的文件如果被收成空，保留原范围。

    没写时期时，如果问句精确点到的实体只存在于被「最新一期」砍掉的文件里，
    改查那些文件。实体在留下的文件里还有一份时，范围不动。
    """
    from ai_agent.kb.ingest.meta import period_matches, period_rank, query_period_policy
    from ai_agent.storage.store import store

    policy, token = query_period_policy(query or "")
    try:
        rows = store.list_kb_files()
    except Exception:
        logger.warning("按报告期收窄失败，保持原范围", exc_info=True)
        return file_ids
    ready_all = [row for row in rows if row.get("status") == "ready"]
    if file_ids is not None:
        want = {str(fid) for fid in file_ids}
        ready = [row for row in ready_all if str(row.get("file_id")) in want]
    else:
        ready = ready_all
    if policy in {"all", "span_q", "span_h"} and file_ids is not None:
        # 问句要对比多期，但路由或模型可能只点到其中一年。同一主体的其它期补上。
        subjects = {
            str(row.get("subject") or "").strip()
            for row in ready
            if str(row.get("subject") or "").strip()
        }
        if subjects:
            widened = [
                row
                for row in ready_all
                if str(row.get("subject") or "").strip() in subjects
                and (
                    policy == "all"
                    or period_matches(str(row.get("period") or ""), policy, token)
                )
            ]
            if widened:
                return _parse_file_ids(widened) or file_ids
        return file_ids
    if policy == "all":
        return file_ids
    # 目录里对不上这些 id（或库是空的）时无法判断报告期，保持调用方已经解析出的范围
    if not ready:
        return file_ids
    if policy == "latest":
        best: dict[str, tuple[int, int]] = {}
        for row in ready:
            subject = str(row.get("subject") or "").strip()
            rank = period_rank(str(row.get("period") or ""))
            if not subject or rank == (0, 0):
                continue
            if subject not in best or rank > best[subject]:
                best[subject] = rank
        kept = []
        for row in ready:
            subject = str(row.get("subject") or "").strip()
            rank = period_rank(str(row.get("period") or ""))
            if not subject or rank == (0, 0) or rank == best.get(subject):
                kept.append(row)
    else:
        kept = [
            row
            for row in ready
            if period_matches(str(row.get("period") or ""), policy, token)
        ]
    if not kept:
        # 显式点名：保持调用方已解析出的范围（用户点名的文件就是范围）。
        # 路由猜出来的：收窄成空说明**路由猜错了**，此时返回 [] 会被上层当成
        # 「模型点了不存在的文件」直接回「未找到」，一次检索都不跑；
        # 退回 None（全库）让向量召回还有机会命中。
        narrowed: list[uuid.UUID] | None = file_ids if explicit else None
    else:
        narrowed = _parse_file_ids(kept) or (file_ids if explicit else [])
    if policy != "latest":
        return narrowed
    return _prefer_exact_entity(
        query, narrowed, ready_all, explicit=explicit, before=file_ids
    )


def _prefer_exact_entity(
    query: str | None,
    narrowed: list[uuid.UUID] | None,
    ready_rows: list[dict],
    *,
    explicit: bool,
    before: list[uuid.UUID] | None,
) -> list[uuid.UUID] | None:
    """精确实体被「只留最新一期」砍掉时，改用它实际所在的文件。

    问句写了年份或季度时不插手。留下的文件里已经有这个实体，也不改。
    用户点名的文件里本来没有它，同样不把范围扩到别的文件。
    """
    if not query or narrowed is None:
        return narrowed
    ready_ids = {str(row.get("file_id")) for row in ready_rows}
    try:
        from ai_agent.storage.store import store

        hits = store.exact_entity_files(query)
    except Exception:
        logger.warning("精确实体查文件失败，保持期收窄结果", exc_info=True)
        return narrowed
    kept_hits: list[uuid.UUID] = []
    seen: set[str] = set()
    for fid in hits:
        key = str(fid)
        if key not in ready_ids or key in seen:
            continue
        seen.add(key)
        kept_hits.append(fid)
    if not kept_hits:
        return narrowed
    narrowed_ids = {str(fid) for fid in narrowed}
    if any(str(fid) in narrowed_ids for fid in kept_hits):
        return narrowed
    if explicit:
        before_ids = {str(fid) for fid in (before or [])}
        kept_hits = [fid for fid in kept_hits if str(fid) in before_ids]
        if not kept_hits:
            return narrowed
    logger.info("精确实体不在最新一期，改查它所在的文件")
    return kept_hits


def _parse_file_ids(rows: list[dict]) -> list[uuid.UUID]:
    parsed: list[uuid.UUID] = []
    for row in rows:
        raw = row.get("file_id")
        if isinstance(raw, uuid.UUID):
            parsed.append(raw)
        else:
            try:
                parsed.append(uuid.UUID(str(raw)))
            except ValueError:
                continue
    return parsed


def _route(query: str | None) -> list[uuid.UUID]:
    """按文档摘要做语义路由。任何异常都退回空列表（等价于路由没结果）。

    路由失败不能影响对话：底层接口 TP99 抖动一次，上层还有会话范围和
    全库两条路可走，抛出去只会让 kb_search 直接报 ERROR。
    """
    text = (query or "").strip()
    if not text:
        return []
    try:
        from ai_agent.kb.retrieval.router import route_files

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
    model, client = lightweight_llm(settings.kb_rewrite_model)
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
            client=client,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "think": False,
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
    model, client = lightweight_llm(settings.kb_multiquery_model)
    try:
        out = complete_chat(
            [
                {"role": "system", "content": _MULTIQUERY_SYSTEM},
                {
                    "role": "user",
                    "content": f"请生成 {n} 条检索词变体，每条一行：\n{q}",
                },
            ],
            model=model,
            timeout=settings.kb_multiquery_timeout_seconds,
            max_retries=0,
            max_tokens=256,
            client=client,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "think": False,
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
