"""知识库检索工具。

检索编排见 kb/retrieval.py：向量召回 → 去冗余 → LLM 重排 → 邻接扩展。
知识库是全站共享的（管理员上传、所有人可查），与 memory_search
的语义区分：记忆是当前用户的对话沉淀，这里是共享文档原文。
"""

import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.ingest.chunking import html_tables_to_markdown
from ai_agent.kb.retrieval.citations import Citation, add_citations
from ai_agent.kb.retrieval.retrieval import (
    Hit,
    format_where,
    search_knowledge_multi,
    _PARENT_INTENT,
    SCOPE_PARENT,
)
from ai_agent.kb.retrieval.scope import (
    current_question,
    effective_kb_query,
    expand_queries,
    kb_scope,
    resolve_tool_file_ids,
)
from ai_agent.storage.store import store
from ai_agent.prompts.tools import (
    KB_EMPTY_HINT,
    KB_ERROR_HINT,
    KB_FILES_DESC,
    KB_FILES_NOT_FOUND_HINT,
    KB_NO_HIT_HINT,
    KB_SEARCH_DESC,
    KB_SEARCH_QUERY_DESC,
    KB_SEARCH_TOPK_DESC,
    build_kb_no_hit_scoped,
)
from ai_agent.tools.base import Tool

logger = get_logger(__name__)

# 邻接片段只给开头/结尾一小段：够接上下文，又不至于把命中片淹掉
_NEIGHBOR_CLIP = 200

# 剥离主体名后若只剩这些语气/疑问助词，视为「问的就是全称本身」，回退原句
_PARTICLE_RE = re.compile(r"^[的是]+|[的是]+$")
# 「表现」「表示」里的「表」不是在要表格
_TABLE_ASK = re.compile(r"表格|表中|附表|这张表|(?<![表现态])表(?![现示态])")


def _question_names_files_query_lost(tool_query: str, asked: str) -> bool:
    """用户点了公司，检索词里却没有这些文件。说明公司名被收掉了。"""
    from ai_agent.kb.retrieval.router import mentioned_file_ids

    named = {str(fid) for fid in mentioned_file_ids(asked)}
    if not named:
        return False
    kept = {str(fid) for fid in mentioned_file_ids(tool_query)}
    return not named <= kept


def _kinds_user_asked(asked: str, kinds: list[str] | None) -> list[str] | None:
    """用户没要表格时，丢掉模型自己加的 table。没有原句就保持原参数。"""
    if not kinds or not asked:
        return kinds
    if "table" in kinds and not _TABLE_ASK.search(asked):
        kinds = [kind for kind in kinds if kind != "table"]
    return kinds or None


def _sections_user_asked(asked: str, sections: list[str] | None) -> list[str] | None:
    """章节词没有出现在用户原句里，就不当成用户要的范围。"""
    if not sections or not asked:
        return sections
    kept = [section for section in sections if section and section in asked]
    return kept or None


def _strip_scope_subject(query: str, file_ids: list) -> str:
    """files 已指定时，从检索 query 里确定性地剥掉被点名文件的主体名/文件名。

    query 在 files 提供时有双重职责：选文件（已由 resolve_tool_file_ids 用
    filenames 完成）+ 检索信号（向量嵌入 + 关键词抽词）。主体名对检索是纯噪声——
    它把嵌入拉向反复出现公司名的页眉页、并占掉关键词 6 个名额挤出真正要找的字段
    词（如「证券代码」「营业收入」）。这里用解析后的文件自身元数据
    （subject / 别名 / 文件名）做剥离，不靠模型原文硬删（模型可能用缩写）。

    剥完若只剩不到 2 个有效字（问的就是「全称是什么」），回退原句，避免把真正的
    问题也删没了。files 未提供（路由模式）时调用方不应走到这里。
    """
    names: list[str] = []
    for fid in file_ids:
        f = store.get_kb_file(fid)
        if not f:
            continue
        if f.get("subject"):
            names.append(f["subject"])
        for alias in (f.get("subject_aliases") or []):
            if alias:
                names.append(alias)
        if f.get("filename"):
            # 去掉扩展名再剥，避免「.pdf」这类被当成分隔残留
            names.append(os.path.splitext(f["filename"])[0])
    if not names:
        return query
    # 长名先剥，避免短别名先把长名拆断后长名匹配不上
    names.sort(key=len, reverse=True)
    stripped = query
    for name in names:
        if len(name) >= 2:  # 太短的词不剥，避免误删问题里的通用片段
            stripped = stripped.replace(name, "")
    stripped = _PARTICLE_RE.sub("", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip()
    # 剥完几乎没剩（问的就是全称本身），回退原句，别把真问题删没了
    if len(stripped) < 2:
        return query
    if stripped != query:
        logger.debug("kb_search 检索 query 去主体化: %r → %r", query, stripped)
    return stripped


class KbSearchArgs(BaseModel):
    query: str = Field(..., description=KB_SEARCH_QUERY_DESC)
    # files 可选：不填就由文档路由（按摘要向量）自己选定范围。以前靠
    # 模型对着文件清单填这一项，文件一多既选不准又占上下文
    files: list[str] | None = Field(default=None, description=KB_FILES_DESC)
    top_k: int = Field(default=3, ge=1, le=10, description=KB_SEARCH_TOPK_DESC)
    # 元数据过滤（双轨）：kinds/sections 由模型显式指定；scope 由 query 意图推断
    kinds: list[Literal["text", "table", "figure"]] | None = Field(
        default=None,
        description=(
            "选填。只有用户明确只要其中一种时才填：table=表格、figure=图表、text=正文。"
            "增长、原因、情况这类问题要同时看正文和表格，留空。"
        ),
    )
    sections: list[str] | None = Field(
        default=None,
        description=(
            "选填。只有用户点名某一章时才填章节关键词。"
            "不要自行填章节。"
        ),
    )


def retrieve_for_question(
    query: str,
    files: list[str] | None = None,
    top_k: int = 3,
    kinds: list[str] | None = None,
    sections: list[str] | None = None,
    extra_filters: dict | None = None,
) -> tuple[list | None, list[Hit]]:
    """对话和评测共用的检索：先圈文件，再去掉公司名，在这些文件里查。

    返回 (parsed_ids, hits)。parsed_ids 为 [] 表示点了文件但一个都没解析到；
    为 None 表示没圈住，按全库查。检索失败抛异常，由调用方转文案。
    """
    asked = current_question().strip()
    # 模型把「中国黄金的营收增长情况」收成「营业收入」后，公司名和「增长」都没了，
    # 路由会按科目去找别的文件。原句点了名而检索词没点到时，改回原句再圈文件。
    if asked and not files and _question_names_files_query_lost(query, asked):
        query = asked
    else:
        query = effective_kb_query(query)
    if asked:
        kinds = _kinds_user_asked(asked, kinds)
        sections = _sections_user_asked(asked, sections)
    parsed_ids = resolve_tool_file_ids(files, query=query)
    if parsed_ids == []:
        return [], []

    # 范围里全是点名的文件时，问句里的公司名只剩噪声。摘要补进来的文件不剥：
    # 「证券代码」这种词每份报告封面都有，剥掉公司名就分不出是哪一家。
    named_only = _named_scope(query, files, parsed_ids)
    retrieval_query = (
        _strip_scope_subject(query, parsed_ids) if named_only else query
    )

    filters: dict = dict(extra_filters or {})
    if kinds:
        filters["kinds"] = kinds
    if sections:
        filters["sections"] = sections
    if "scopes" not in filters and _PARENT_INTENT.search(retrieval_query):
        filters["scopes"] = [SCOPE_PARENT]

    queries = (
        expand_queries(retrieval_query)
        if settings.kb_multiquery_enabled
        else [retrieval_query]
    )
    search_kwargs: dict = {
        "file_ids": parsed_ids,
        "top_k": top_k,
        "filters": filters or None,
    }
    # 范围里有多家（或同一家的多份）时按文件留名额。股东表的主体会被改成股东名，
    # 按主体留名额会让一家占满 top_k。
    if named_only and parsed_ids and len(parsed_ids) >= 2:
        search_kwargs["per_file"] = True
    if retrieval_query != query:
        # 公司名剥掉之后，分档还要靠原句知道点了哪几家
        search_kwargs["diversify_query"] = query
    hits = search_knowledge_multi(queries, **search_kwargs)
    return parsed_ids, hits


def _named_scope(query: str, files: list[str] | None, parsed_ids: list | None) -> bool:
    """范围里每一份都是问句或 files 点到的，没有摘要补进来的其他公司。"""
    if not parsed_ids:
        return False
    if files:
        return True
    from ai_agent.kb.retrieval.router import mentioned_file_ids

    named = {str(fid) for fid in mentioned_file_ids(query)}
    return {str(fid) for fid in parsed_ids} <= named


def _kb_search(
    query: str,
    files: list[str] | None = None,
    top_k: int = 3,
    kinds: list[str] | None = None,
    sections: list[str] | None = None,
) -> str:
    if not store.has_ready_kb_file():
        return KB_EMPTY_HINT

    try:
        parsed_ids, hits = retrieve_for_question(
            query, files=files, top_k=top_k, kinds=kinds, sections=sections
        )
    except Exception:
        logger.exception("知识库检索失败 query=%s", query[:80])
        return KB_ERROR_HINT

    if parsed_ids == []:
        # 模型点了不存在的文件，和「问句里的报告期库里没有」不是同一件事
        if files:
            return KB_FILES_NOT_FOUND_HINT
        return KB_NO_HIT_HINT

    if not hits:
        if _user_limited_scope(files, parsed_ids):
            # 用户或模型圈了文件，空结果很可能是范围外，提示放宽而不是换词。
            # 系统按报告期收窄不算用户限定，不能说成「你只查了这几份」。
            return build_kb_no_hit_scoped(len(parsed_ids or []))
        return KB_NO_HIT_HINT
    # 向量距离超阈值的命中与主题无关：直接当未命中，不下发引用，避免把不
    # 相关片段当依据、又挂着"中国银行季报·第9页"这类矛盾来源（见截图问题）。
    # distance=None 是关键词独有命中（确实含 query 词），保留作相关。
    relevant = _keep_relevant(hits)
    if not relevant:
        if _user_limited_scope(files, parsed_ids):
            return build_kb_no_hit_scoped(len(parsed_ids or []))
        return KB_NO_HIT_HINT
    # 引用走旁路交给上层（SSE 下发 + 随消息落库），文本里仍保留
    # 「文件名 · 位置」抬头，模型据此在回答里交代出处
    add_citations(_citations_of(relevant))
    body = "\n\n".join(_format_hit_with_context(hit) for hit in relevant)
    periods: list[str] = []
    for hit in relevant:
        period = (hit.period or "").strip()
        if period and period not in periods:
            periods.append(period)
    if not periods:
        return body
    # 表内常有上年同期。先写明这次命中的是哪几期，避免把对比列当成库的覆盖范围。
    note = (
        "本次命中的报告期：" + "、".join(periods) + "。"
        "文中更早的年份是对比列，不是文件的报告期，也不表示库里没有这些年份。"
    )
    return f"{note}\n\n{body}"


def _user_limited_scope(files: list[str] | None, parsed_ids: list | None) -> bool:
    """只有用户在输入框里圈了文件，或模型明确点了 files，才算范围限制。"""
    return bool(parsed_ids) and (bool(files) or bool(kb_scope()))


def _keep_relevant(hits: list[Hit]) -> list[Hit]:
    """剔除向量距离超阈值的命中。

    距离上限与检索召回共用 settings.kb_search_max_distance：向量路已经用
    它过滤过一次，但这里再卡一遍是兜底——关键词路（retrieval._fuse_with_keywords）
    不参与距离过滤，RRF 融合后仍可能把无关片段顶进 top_k。distance 为 None 表示
    只走了关键词路（确实含 query 关键词），不按向量距离判死，保留。
    """
    return [
        h
        for h in hits
        if h.distance is None or h.distance <= settings.kb_search_max_distance
    ]


def _citations_of(hits: list[Hit]) -> list[Citation]:
    return [
        Citation(
            file_id=hit.file_id,
            filename=hit.filename,
            seq=hit.seq,
            page=hit.page,
            kind=hit.kind,
            where=hit.where,
            text=hit.text,
            page_end=hit.page_end,
            section=hit.section,
        )
        for hit in hits
    ]


def _format_hit(row: dict) -> str:
    """按内容类型给出引用位置：
    - text：PDF 有页码（跨页切片给区间），无页文档退化为段号
    - table：PDF/Word 用页码，Excel 用 sheet 序号（表N）
    - figure：读图结果必须标注「图像识别」，避免模型当成原文数据引用

    有章节时追加「· 第十节 管理层讨论与分析」：模型回答时能直接说清
    「哪一节里的哪一页」，用户核对时不用再靠页码猜章节。
    """
    kind = row.get("kind") or "text"
    where = format_where(
        kind=kind,
        page=row.get("page"),
        page_end=row.get("page_end"),
        seq=int(row.get("seq") or 0),
        filename=row.get("filename") or "",
        mime=row.get("mime") or "",
    )
    section = row.get("section")
    if section and kind != "figure":
        where = f"{where} · {section}"
    # 表格正文经常只有「本报告期」，年份写在文件上。抬头带上报告期，两年才分得开。
    period = str(row.get("period") or "").strip()
    if period:
        where = f"{period} · {where}"
    # 引用芯片仍用库存的 HTML（前端要 colspan）。这里只改模型看到的正文。
    body = html_tables_to_markdown(row.get("text") or "")
    if kind == "figure":
        return f"[{row['filename']} · {where}（图像识别，可能有误差）]\n{body}"
    return f"[{row['filename']} · {where}]\n{body}"


def _format_hit_with_context(hit: Hit) -> str:
    """命中片 + 前后邻接片段。

    切片是按字数切的，结论经常跨片（"原因有三：" 接下一页）。只给命中片
    会让模型答半截，所以把相邻片也带上，并明确标注哪段才是命中内容。
    """
    head = _format_hit(
        {
            "filename": hit.filename,
            "kind": hit.kind,
            "page": hit.page,
            "page_end": hit.page_end,
            "section": hit.section,
            "seq": hit.seq,
            "period": hit.period,
            "text": "",
        }
    ).rstrip("\n")
    lines = [head]
    for text in hit.before:
        lines.append(f"〔上文〕{_clip_tail(html_tables_to_markdown(text))}")
    lines.append(html_tables_to_markdown(hit.text))
    for text in hit.after:
        lines.append(f"〔下文〕{_clip_head(html_tables_to_markdown(text))}")
    return "\n".join(lines)


def _clip_tail(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _NEIGHBOR_CLIP else f"…{text[-_NEIGHBOR_CLIP:]}"


def _clip_head(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _NEIGHBOR_CLIP else f"{text[:_NEIGHBOR_CLIP]}…"


kb_search_tool = Tool(
    name="kb_search",
    description=KB_SEARCH_DESC,
    func=_kb_search,
    args_model=KbSearchArgs,
)
