"""文档级摘要：把一份文件压成一段「它讲的是什么」的文本，单独 embedding。

为什么要有这一层索引：切片向量回答「哪一段相关」，摘要向量回答
**「哪一份文件相关」**。检索先用摘要做文档路由，再在候选文档里用切片向量
找原文——模型就不必对着注入的文件清单挑 files 了，文件数也不再进提示词。

拼装规则，不走 LLM：入库路上每多一次模型调用就多一个失败点、多一笔成本，
而文件名、主体、报告期、文档类型、图谱实体这些字段本来就在库里，拼起来
足够把「中国能源建设 2026 三季报」和「顺丰控股 2025 年报」区分开。需要
更准的描述时（比如行业研报的主题）用 store.set_kb_file_summary() 覆盖，
流程本身不用改。
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Sequence

from ai_agent.core.config import settings

# 断句：中文句末标点、英文句点与换行。摘录按它裁整句，不留半句话
_SENTENCE_END = re.compile(r"[。！？；!\?;\n]")

# 摘录的最小长度：财报前几片里混着「单位：千元币种：人民币」「3 / 19」这类
# 表头与页脚，它们既不代表主题，还会让不同僚公司的摘要彼此靠近
_MIN_HEAD_CHARS = 24


def doc_type_label(doc_type: str) -> str:
    """文档类型取展示名（季报 → 上市公司季报），模型看中文更稳。"""
    from ai_agent.kb.doc_profiles import resolve_profile

    key = (doc_type or "").strip()
    return resolve_profile(key).label if key else ""


def first_sentences(text: str, limit: int) -> str:
    """摘取正文开头若干完整句，总长不超过 limit。

    宁可短一点也不截断半句：半句话里常常只剩一个介词短语，进向量后
    既不能表达主题，还会掺进一堆无关词。
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit]
    ends = list(_SENTENCE_END.finditer(cut))
    # 最后一句至少留够六成篇幅，否则宁可直接按字符截断（"本公司报告期内"
    # 这种断句比截断更糟）
    if ends and ends[-1].end() >= limit * 0.6:
        return cut[: ends[-1].end()].strip()
    return cut.strip()


def _dedup(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        value = (name or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_file_summary(
    *,
    filename: str,
    subject: str = "",
    aliases: Sequence[str] = (),
    period: str = "",
    doc_type: str = "",
    description: str = "",
    entities: Sequence[str] = (),
    chunks: Sequence[str] = (),
) -> str:
    """拼文档摘要。空字段一律不出现，避免出现「主体：」这种空行。

    各段的权重靠位置表达：文件名与主体在最前（路由最依赖它们区分
    「谁家的哪一期报告」），正文摘录在最后（只用于补充「讲了什么」）。
    """
    lines: list[str] = []
    name = (filename or "").strip()
    if name:
        lines.append(f"文件名：{name}")

    subject = (subject or "").strip()
    alias_list = _dedup(a for a in aliases if a and a != subject)
    if subject:
        tail = f"（又称：{'、'.join(alias_list)}）" if alias_list else ""
        lines.append(f"主体：{subject}{tail}")
    elif alias_list:
        lines.append(f"主体：{'、'.join(alias_list)}")

    period = (period or "").strip()
    if period:
        lines.append(f"报告期：{period}")

    doc_type = (doc_type or "").strip()
    if doc_type:
        lines.append(f"文档类型：{doc_type}")

    description = (description or "").strip()
    if description:
        lines.append(f"内容描述：{description}")

    # 实体按提及频次取最靠前的少数几个：图谱里排前面的通常是文档真正讲的
    # 主体，再往下就是会计科目、流水科目——把它们塞进摘要只会让两份不同
    # 公司的财报摘要彼此靠近
    entities = [
        name for name in _dedup(entities) if name != subject
    ][: settings.kb_summary_entities]
    if entities:
        lines.append(f"涉及主体：{'、'.join(entities)}")

    heads = [
        head
        for head in (
            first_sentences(chunk, settings.kb_summary_head_chars)
            for chunk in chunks[: settings.kb_summary_head_chunks]
        )
        if len(head) >= _MIN_HEAD_CHARS
    ]
    if heads:
        lines.append("内容摘录：")
        lines.extend(f"- {head}" for head in heads)

    text = "\n".join(lines).strip()
    limit = settings.kb_summary_max_chars
    return text[:limit].rstrip() if len(text) > limit else text


def index_file_summary(file_id: uuid.UUID) -> str:
    """为一个已入库文件生成摘要并写回摘要向量，返回摘要文本。

    返回空串表示没写成——素材不足（文件还没切片）或向量化失败，都已经在
    库里记了 summary_status，调用方不必再 try/except。摘要失败不该影响
    文件本身可用性：文件照常 ready，只是没法被语义路由选中，退回到
    「模型指定 files」的老路径即可。
    """
    from ai_agent.core.logging import get_logger
    from ai_agent.memory.embeddings import embed_texts
    from ai_agent.memory.store import store

    logger = get_logger(__name__)
    record = store.get_kb_file(file_id) or {}
    # 表格片段首句是「单位：万元」这类表头，对表达主题没有帮助
    chunks = [
        row["text"]
        for row in store.list_kb_chunk_inputs(file_id)
        if row.get("kind") == "text"
    ]
    summary = build_file_summary(
        filename=record.get("filename") or "",
        subject=record.get("subject") or "",
        aliases=list(record.get("subject_aliases") or []),
        period=record.get("period") or "",
        doc_type=doc_type_label(record.get("doc_type") or ""),
        description=record.get("description") or "",
        entities=store.list_kb_file_entity_names(file_id, settings.kb_summary_entities),
        chunks=chunks,
    )
    if not summary:
        store.set_kb_file_summary(file_id, "", None, status="empty")
        logger.info("kb 文件 %s 没有可用于生成摘要的素材", file_id)
        return ""
    try:
        vector = embed_texts([summary])[0]
    except Exception:
        logger.exception("kb 摘要向量化失败 file_id=%s", file_id)
        store.set_kb_file_summary(file_id, summary, None, status="failed")
        return ""
    store.set_kb_file_summary(file_id, summary, vector, status="ready")
    return summary
