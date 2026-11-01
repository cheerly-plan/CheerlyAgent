"""知识库检索工具。

检索编排见 kb/retrieval.py：向量召回 → 去冗余 → LLM 重排 → 邻接扩展。
知识库是全站共享的（管理员上传、所有人可查），与 memory_search
的语义区分：记忆是当前用户的对话沉淀，这里是共享文档原文。
"""

import os
import re

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.citations import Citation, add_citations
from ai_agent.kb.retrieval import Hit, format_where, search_knowledge
from ai_agent.kb.scope import effective_kb_query, resolve_tool_file_ids
from ai_agent.memory.store import store
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


def _kb_search(
    query: str, files: list[str] | None = None, top_k: int = 3
) -> str:
    if not store.has_ready_kb_file():
        return KB_EMPTY_HINT

    # files 没填时由文档路由按摘要向量选文件（files 是模型的可选项，不是
    # 必答题）；路由也没结果时 file_ids=None，退化为搜全库
    query = effective_kb_query(query)
    parsed_ids = resolve_tool_file_ids(files, query=query)
    # files 明确指定但解析不到（编造的文件名/主体名）→ 明确提示，而非静默退回全库
    if parsed_ids == []:
        return KB_FILES_NOT_FOUND_HINT

    # files 已指定时，选文件这步已完成，检索 query 里再带主体名纯属噪声：会把嵌入
    # 拉向反复出现公司名的页眉页、并挤占关键词名额。确定性剥掉（用解析后文件自身的
    # subject/别名/文件名），向量与关键词两路共用同一句剥完的话。files 未提供
    # （路由模式）则原句不动——那时主体还得用来选文件。
    retrieval_query = (
        _strip_scope_subject(query, parsed_ids)
        if files is not None and parsed_ids
        else query
    )

    try:
        hits = search_knowledge(retrieval_query, file_ids=parsed_ids, top_k=top_k)
    except Exception:
        logger.exception("知识库检索失败 query=%s", query[:80])
        return KB_ERROR_HINT

    if not hits:
        if parsed_ids:
            # 有范围限制时，空结果很可能是"范围外"，提示用户放宽而不是换词
            return build_kb_no_hit_scoped(len(parsed_ids))
        return KB_NO_HIT_HINT
    # 向量距离超阈值的命中与主题无关：直接当未命中，不下发引用，避免把不
    # 相关片段当依据、又挂着"中国银行季报·第9页"这类矛盾来源（见截图问题）。
    # distance=None 是关键词独有命中（确实含 query 词），保留作相关。
    relevant = _keep_relevant(hits)
    if not relevant:
        if parsed_ids:
            return build_kb_no_hit_scoped(len(parsed_ids))
        return KB_NO_HIT_HINT
    # 引用走旁路交给上层（SSE 下发 + 随消息落库），文本里仍保留
    # 「文件名 · 位置」抬头，模型据此在回答里交代出处
    add_citations(_citations_of(relevant))
    return "\n\n".join(_format_hit_with_context(hit) for hit in relevant)


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
    description=KB_SEARCH_DESC,
    func=_kb_search,
    args_model=KbSearchArgs,
)
