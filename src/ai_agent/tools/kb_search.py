"""知识库检索工具。

检索编排见 kb/retrieval.py：向量召回 → 去冗余 → LLM 重排 → 邻接扩展。
知识库是全站共享的（管理员上传、所有人可查），与 memory_search
的语义区分：记忆是当前用户的对话沉淀，这里是共享文档原文。
"""

import uuid

from pydantic import BaseModel, Field

from ai_agent.core.logging import get_logger
from ai_agent.kb.citations import Citation, add_citations
from ai_agent.kb.retrieval import Hit, format_where, search_knowledge, tag_file_scope
from ai_agent.kb.scope import kb_scope
from ai_agent.memory.store import store
from ai_agent.tools.base import Tool

logger = get_logger(__name__)

# 邻接片段只给开头/结尾一小段：够接上下文，又不至于把命中片淹掉
_NEIGHBOR_CLIP = 200


class KbSearchArgs(BaseModel):
    query: str = Field(..., description="检索问题或关键词，用自然语言描述要找的内容")
    file_ids: list[str] | None = Field(
        default=None, description="限定检索的文件 id 列表，缺省检索全部文件"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="返回的片段数")


def _kb_search(
    query: str, file_ids: list[str] | None = None, top_k: int = 3
) -> str:
    if not store.has_ready_kb_file():
        return "知识库为空或还没有处理完成的文件，请提示用户联系管理员上传资料。"

    # 模型没显式指定文件时，跟随用户在输入框里选的会话范围（可能为空 = 全库）
    raw_ids = file_ids if file_ids else list(kb_scope())
    parsed_ids: list | None = None
    if raw_ids:
        parsed_ids = []
        for raw in raw_ids:
            try:
                parsed_ids.append(uuid.UUID(raw))
            except ValueError:
                return f"ERROR: 非法的文件 id {raw}"

    try:
        if parsed_ids is None:
            hits = _search_with_tag_scope(query, top_k)
        else:
            hits = search_knowledge(query, file_ids=parsed_ids, top_k=top_k)
    except Exception:
        logger.exception("知识库检索失败 query=%s", query[:80])
        return "ERROR: 知识库暂时不可用，请稍后重试"

    if not hits:
        if parsed_ids:
            # 有范围限制时，空结果很可能是"范围外"，提示用户放宽而不是换词
            return (
                f"在用户限定的 {len(parsed_ids)} 份文档里没有找到相关内容。"
                "可以提示用户清除知识库范围限制后再查，或换个说法。"
            )
        return "没有在知识库中找到相关内容，可以换个说法再试。"
    # 引用走旁路交给上层（SSE 下发 + 随消息落库），文本里仍保留
    # 「文件名 · 位置」抬头，模型据此在回答里交代出处
    add_citations(_citations_of(hits))
    return "\n\n".join(_format_hit_with_context(hit) for hit in hits)


def _search_with_tag_scope(query: str, top_k: int) -> list[Hit]:
    """标签预过滤 + 弱召回兜底。

    用户没圈定范围时先按标签收窄：查询关键词命中文件标签 → 只搜这批
    文件，别的文件（尤其是 FAQ 这类干扰项）出不了候选，重排输入更干净。
    收窄是"猜"出来的，可能判窄，所以召回不足 top_k 就用全库结果补位，
    收窄结果排前面——宁可多花一次向量调用，不拿召回率换精度。
    """
    try:
        scope_ids = tag_file_scope(query)
    except Exception:
        logger.warning("kb 标签范围推导失败，走全库检索", exc_info=True)
        scope_ids = None
    if not scope_ids:
        return search_knowledge(query, top_k=top_k)
    logger.info("kb 标签预过滤：收窄到 %d 份文件", len(scope_ids))
    hits = search_knowledge(query, file_ids=scope_ids, top_k=top_k)
    if len(hits) >= top_k:
        return hits
    seen = {(h.file_id, h.seq) for h in hits}
    for extra in search_knowledge(query, top_k=top_k):
        if (extra.file_id, extra.seq) not in seen:
            hits.append(extra)
            seen.add((extra.file_id, extra.seq))
    return hits[:top_k]


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
    if kind == "figure":
        return f"[{row['filename']} · {where}（图像识别，可能有误差）]\n{row['text']}"
    section = row.get("section")
    if section:
        where = f"{where} · {section}"
    return f"[{row['filename']} · {where}]\n{row['text']}"


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
            "text": "",
        }
    ).rstrip("\n")
    lines = [head]
    for text in hit.before:
        lines.append(f"〔上文〕{_clip_tail(text)}")
    lines.append(hit.text)
    for text in hit.after:
        lines.append(f"〔下文〕{_clip_head(text)}")
    return "\n".join(lines)


def _clip_tail(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _NEIGHBOR_CLIP else f"…{text[-_NEIGHBOR_CLIP:]}"


def _clip_head(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _NEIGHBOR_CLIP else f"{text[:_NEIGHBOR_CLIP]}…"


kb_search_tool = Tool(
    name="kb_search",
    display_name="知识库检索",
    description=(
        "检索知识库里的共享文档。当问题可能与团队资料/文档有关，"
        "或用户要求查文档、查资料时调用；query 用自然语言描述要找的内容。"
    ),
    func=_kb_search,
    args_model=KbSearchArgs,
)
