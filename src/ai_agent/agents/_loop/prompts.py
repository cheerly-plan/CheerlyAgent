"""系统提示词拼装：画像/回忆/知识库能力/报告期清单等区块。

由原 agents/loop.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

from ai_agent.core.exceptions import LLMError
from ai_agent.core.text import clip_text
from ai_agent.memory.facets import format_profile_block
from ai_agent.kb.ingest.doc_profiles import resolve_profile
from ai_agent.core.config import settings
from ai_agent.memory.progressive import full_trace, rerank_items
from ai_agent.prompts.chat import build_system_prompt
from ai_agent.storage.store import store

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)


# 测试 monkeypatch 打在 ``ai_agent.agents.loop`` 上；内部调用一律经
# _public() 取当前属性，否则补丁会被本地绑定绕过（静默失效）。
def _public():
    import ai_agent.agents.loop as public

    return public


def _llm_error_message(error: LLMError) -> str:
    """把上游错误翻成用户能照着做的一句话。

    403/401 重试永远不会好（模型未开通、key 无权限、账户欠费），
    对用户说「请稍后重试」只会让他反复点；必须指向真因。
    """
    if error.is_permission_denied:
        return "模型服务拒绝了本次调用（模型未开通或无权限），请联系管理员检查模型账户"
    if error.is_rate_limited:
        return "模型服务繁忙，请稍后再试"
    return "模型服务暂时不可用，请稍后重试"

def _profile_prompt(user_id: str, *, user_input: str = "") -> str:
    try:
        items = store.list_profile(user_id)
        _public().logger.info("list_profile user_id=%s count=%s", user_id, len(items))
    except Exception:
        _public().logger.exception("用户画像读取失败")
        return ""

    # 渐进式记忆恒定启用：注入「索引 + 与当前问句相关的条目」，被省略的
    # 画像由 expand_memory 按需展开。任何异常 fail-open 回退到全量画像。
    # 画像少于 list_profile 的 20 条上限时零额外查询、输出逐字不变；触及
    # 上限才数一次库内真实总数，让索引说明不再把「本次载入」当成库内总数。
    total: int | None = None
    if len(items) >= 20:
        try:
            total = store.count_active_profile(user_id)
        except Exception:
            total = None
    try:
        block, trace = _public().build_memory_index(
            items,
            user_input=user_input,
            top_k=settings.memory_index_top_k,
            max_chars=settings.memory_index_max_chars,
            expand_hint=True,
            total=total,
        )
    except Exception:
        _public().logger.exception("渐进式记忆块构建失败，回退全量画像")
        block = format_profile_block(items)
        trace = full_trace(items, block, reason="fallback_error", mode="fallback")
    if settings.memory_trace_enabled:
        trace.emit(_public().logger)
    return block

def _recall_block(
    user_id: str, user_input: str, chat_session_id: str = ""
) -> str:
    """第 3 步：按当前问句自动召回情景记忆摘要，不依赖模型主动调工具。

    没有问句时不召回（空查询的向量结果没有意义）。召回会多一次本地
    embedding 与向量查询，任何异常都 fail-open 跳过，不让记忆链路拖垮对话。
    """
    if not user_input.strip():
        return ""
    try:
        # 向量先取宽 3 倍，再按「相关性 + 新鲜度 + 重要性」收窄到 top_k：
        # 纯向量排序只看语义相似，刚发生、被反复用到的近况会被淹没。
        search_kwargs: dict[str, object] = {
            "k": max(settings.memory_auto_recall_top_k * 3, 3),
            "user_id": user_id,
        }
        # 默认不传新关键字，调用形式与改动前逐字一致；只有拿到真实会话 id
        # （run_stream 正常链路）才把跨会话复现证据带下去。
        if chat_session_id:
            search_kwargs["chat_session_id"] = chat_session_id
        hits = store.search_episode(user_input, **search_kwargs)
        hits = rerank_items(
            hits, user_input, top_k=settings.memory_auto_recall_top_k
        )
        block = _public().format_recall_block(
            hits, max_chars=settings.memory_auto_recall_max_chars
        )
        _public().logger.info("memory_recall hits=%d chars=%d", len(hits), len(block))
        return block
    except Exception:
        _public().logger.exception("情景记忆自动召回失败，本轮跳过")
        return ""

def _kb_capability_block(kb_file_ids: tuple[str, ...] = ()) -> str:
    """拼接知识库能力声明，注入系统提示词。

    静态提示词只讲「怎么选工具」（关系→kg_search、数字→kb_search），具体
    图谱覆盖哪些实体/关系由本块按库内实际文档类型动态生成——加公告、年报
    等新类型时无需改提示词，只要 kb/skills 或 doc_profiles 覆盖里配好即可，提示词不再只针对季报。
    这样模型按问题选对应类型的关系，而不是被硬编码的季报语义带偏。
    """
    try:
        rows = store.list_kb_files()
    except Exception:
        _public().logger.exception("知识库文件清单读取失败")
        return ""
    ready = [r for r in rows if r.get("status") == "ready"]
    if kb_file_ids:
        want = {str(f) for f in kb_file_ids}
        ready = [r for r in ready if str(r.get("file_id")) in want]
    if not ready:
        return ""

    # 按 doc_type 分组；空 doc_type 回落到 general（恰好能暴露漏标类型的问题）
    groups: dict[str, list[dict]] = {}
    for r in ready:
        dt = str(r.get("doc_type") or "general").strip() or "general"
        groups.setdefault(dt, []).append(r)

    lines = [
        "【知识库能力】知识库图谱存的是实体之间的关系（「谁和谁有什么关系」）。"
        "各文档类型在图谱中覆盖的实体与关系如下，不在其中的关系请走 kb_search 读原文：",
    ]
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if len(ranked) > _KB_CAP_MAX_DOC_TYPES:
        lines.append(
            f"（文档类型共 {len(ranked)} 类，此处只列文档数最多的 "
            f"{_KB_CAP_MAX_DOC_TYPES} 类；完整清单用 kb_list_files 查）"
        )
        ranked = ranked[:_KB_CAP_MAX_DOC_TYPES]
    for dt, members in ranked:
        profile = resolve_profile(dt)
        scope = "、".join(profile.entity_scope) if profile.entity_scope else "（通用抽取，无限定范围）"
        preds = "、".join(profile.predicates) if profile.predicates else "（通用抽取，无限定谓词）"
        count = len(members)
        lines.append(f"- {profile.label}（{count} 份）")
        lines.append(f"  实体范围={scope}")
        lines.append(f"  关系={preds}")
    lines.append(
        "注：具体数字、报表科目、原文表述不在图谱里，请用 kb_search 查；"
        "关系题同时需要某个科目金额时，kg_search 与 kb_search 两个都调。"
    )
    period_line = _public()._period_inventory(ready)
    if period_line:
        lines.append(period_line)
    return "\n".join(lines)

def _period_inventory(ready: list[dict]) -> str:
    """库里实际有哪些报告期。模型容易按训练截止日判断某年财报还没披露。"""
    from ai_agent.kb.ingest.meta import period_rank

    counts: dict[str, int] = {}
    for row in ready:
        period = str(row.get("period") or "").strip()
        if period:
            counts[period] = counts.get(period, 0) + 1
    if not counts:
        return ""
    ordered = sorted(counts, key=period_rank, reverse=True)
    parts = [f"{period}（{counts[period]} 份）" for period in ordered]
    return (
        "库内已有报告期：" + "、".join(parts) + "。"
        "这些是已经入库的文件。某年有没有资料，以这份清单、kb_list_files 和检索结果抬头为准，"
        "不要按训练知识里的披露进度推断。"
        "表格中更早的年份是对比列，不是文件的报告期。"
    )

def _system_prompt(
    user_id: str,
    *,
    use_kb: bool = True,
    use_memory: bool = True,
    use_report: bool = False,
    kb_file_ids: tuple[str, ...] = (),
    user_input: str = "",
    chat_session_id: str = "",
) -> str:
    # 提示词里的检索纪律/引用规范按本轮是否挂了知识库同步裁剪：工具列表
    # 已经在下面按开关剔掉了 kb_search/kg_search，提示词再讲一遍「必须先
    # 调用 kb_search」只会让模型去调一个不存在的工具。
    # 研报纪律节随 use_report 开关（研报抽数走服务端流水线直读知识库，不
    # 依赖模型调用 kb_search），与 use_kb 无关。
    core = build_system_prompt(
        use_kb=use_kb,
        use_memory=use_memory,
        use_report=use_report,
        report_catalog=_public().skill_catalog(),
    )
    # 只注入文档类型能覆盖的实体和关系。文件清单不进提示词，模型用
    # kb_list_files 现查；检索范围由问句里的主体和文档路由决定。
    # 无问句时保持原来的调用形式：存量测试会把 _profile_prompt 换成只收
    # 位置参数的桩，多传一个关键字参数会把它们打挂；顺带让这条路径与
    # 改动前逐字一致。只有真的拿到问句才把相关性查询传下去。
    block = (
        _public()._profile_prompt(user_id, user_input=user_input)
        if use_memory and user_input
        else (_public()._profile_prompt(user_id) if use_memory else "")
    )
    cap_block = _public()._kb_capability_block(kb_file_ids) if use_kb else ""
    recall = (
        _public()._recall_block(user_id, user_input, chat_session_id)
        if use_memory
        else ""
    )
    tail = "\n\n".join(p for p in (_public()._today_line(), block, recall, cap_block) if p)
    if not tail:
        return core
    # 动态块原先完全不受上下文预算约束：画像条数、库内文档类型数、报告类型
    # 数都会让它线性增长。这里按行保头保尾裁到硬上限，并留一句显式提示，
    # 让模型知道清单不完整、需要时自己查，而不是拿着半截清单硬答。
    tail = clip_text(tail, settings.system_dynamic_max_chars, marker=_DYNAMIC_CLIP_MARK)
    return f"{core}\n\n{tail}"

def _today_line() -> str:
    from datetime import date

    return (
        f"【当前日期】今天是 {date.today().isoformat()}。"
        "此日之前的报告期都是已披露文件，不要当成尚未发布的未来数据。"
    )

# 知识库能力清单按 doc_type 分组，库里类型多了会线性撑大 system；只列文档数
# 最多的几类，其余让模型用 kb_list_files 现查。整体还有 system_dynamic_max_chars 兜底。
_KB_CAP_MAX_DOC_TYPES = 8

_DYNAMIC_CLIP_MARK = "\n[系统动态信息已省略 {n} 字符，完整清单请用 kb_list_files 查询]\n"
