"""主体与文件解析：问句里的主体/实体识别、文件名路由。

由原 retrieval.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

import uuid

from ai_agent.kb.ingest.meta import FILENAME_NOISE as _FILENAME_NOISE
from ai_agent.kb.ingest.meta import EXT_RE as _EXT_RE
from ai_agent.kb.ingest.meta import ENTITY_SUFFIX as _ENTITY_SUFFIX
from ai_agent.storage.store import store
from ai_agent.kb.ingest.meta import NAME_PUNCT as _NAME_PUNCT

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)


# 测试用 monkeypatch.setattr(retrieval, NAME, ...) 打补丁的知识点：
# 直接 NAME(...) 走的是本模块 globals，绕过模块属性，补丁会失效；
# 统一经 _public() 取「对外模块」的当前属性再调用，补丁才生效。
def _public():
    import ai_agent.kb.retrieval.retrieval as public

    return public


def resolve_files_by_name(names: list[str]) -> list[uuid.UUID] | None:
    """把模型给的文件名/简称解析成就绪文件 id。

    不要求与库里文件名一字不差：精确全名、去扩展名、去标点子串、主体简称
    都能命中。解析失败返回 None，调用方应退回会话范围或全库，不要报错。
    一个提示词命中同一家公司的多份报告时全部纳入。命中了全部就绪文件则
    等于没圈范围，也返回 None。
    """
    hints = [_clean_filename_hint(name) for name in names]
    hints = [h for h in hints if h]
    if not hints:
        return None
    try:
        files = store.list_kb_files()
    except Exception:
        logger.warning("按文件名解析范围失败，跳过", exc_info=True)
        return None
    ready = [item for item in files if item.get("status") == "ready"]
    if not ready:
        return None
    matched: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for hint in hints:
        hint_hit = False
        for item in ready:
            fid = item["file_id"]
            if fid in seen:
                continue
            if _filename_hint_matches(hint, item.get("filename") or ""):
                seen.add(fid)
                matched.append(fid)
                hint_hit = True
        # 文件名没命中这个提示词，再按元数据维度（subject 列 + 别名）匹配：
        # 多主体文档（行业研报）文件名里没有「宁德时代」，但它确实是正文主体，
        # 只有走 subject/别名才能把这种提示词路由到对应文件。
        # 「中科飞测」不在别名表里，但是全称里独有的一段，同样算点名。
        if not hint_hit:
            from ai_agent.kb.ingest.meta import fragment_subjects

            catalog = [
                (str(item.get("subject") or ""), list(item.get("subject_aliases") or []))
                for item in ready
            ]
            fragments = fragment_subjects(hint, catalog)
            for item in ready:
                fid = item["file_id"]
                if fid in seen:
                    continue
                subject = str(item.get("subject") or "")
                if subject in fragments or _subject_token_matches(
                    hint,
                    subject,
                    item.get("subject_aliases") or [],
                ):
                    seen.add(fid)
                    matched.append(fid)
                    hint_hit = True
    if not matched:
        return []  # 点名但一个都解析不到 → 上层按「未找到」处理
    # 仅在就绪文件多于 1 份时，「命中覆盖全部」才等于没缩小范围；
    # 单文件库（ready==1）命中那唯一一份就是成功圈定，不可短路。
    if len(ready) > 1 and len(matched) == len(ready):
        return None  # 点名无缩小意义 → 上层退回全库
    return matched

def _subject_token_matches(tok: str, subject: str, aliases: list[str]) -> bool:
    """提示词里的主体是否整段出现在文件 subject 或别名里。

    「中国能建」只在别名表有这四个字时命中，不再和全称做最长公共子串。
    """
    for cand in [subject, *list(aliases or [])]:
        if cand and _public().name_in_text(tok, cand):
            return True
    return False

def _clean_filename_hint(name: str) -> str:
    """模型可能把目录行整段贴进来（含「｜标签：…」），只留文件名部分。"""
    text = str(name or "").strip()
    if not text:
        return ""
    for sep in ("｜", " | ", "|"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    return text

def _filename_hint_matches(hint: str, filename: str) -> bool:
    if not hint or not filename:
        return False
    hint_l = hint.lower().strip()
    name_l = filename.lower().strip()
    if hint_l == name_l:
        return True
    hint_stem = _EXT_RE.sub("", hint_l)
    name_stem = _EXT_RE.sub("", name_l)
    if hint_stem == name_stem:
        return True
    hint_norm = _NAME_PUNCT.sub("", hint_stem)
    name_norm = _NAME_PUNCT.sub("", name_stem)
    if hint_norm and name_norm:
        if hint_norm == name_norm:
            return True
        if len(hint_norm) >= 4 and (
            hint_norm in name_norm or name_norm in hint_norm
        ):
            return True
    return _public().name_in_text(hint, _file_subject(filename))

def _file_subject(filename: str) -> str:
    """文件名去掉年份、报告类型、扩展名后的主体。

    「顺丰控股股份有限公司2026 年第三季度报告」→「顺丰控股股份有限公司」。
    年份和「季度报告」这类字样每份文件都有，留着只会制造假重合。
    """
    stem = _EXT_RE.sub("", filename or "")
    stem = _FILENAME_NOISE.sub("", stem)
    return _NAME_PUNCT.sub("", stem).lower()

def _subject_core(subject: str) -> str:
    """剥掉「股份有限公司」这类组织形式后缀，留下真正的主体名。"""
    return _ENTITY_SUFFIX.sub("", subject)

def query_looks_on_topic(
    query: str, file_ids: list[uuid.UUID] | None = None
) -> bool:
    """问句是否点名了库里某份文件的主体或标签。

    用来挡「深圳带伞」这类出行问题：字面能撞上「深圳市顺丰投资」，
    但问句并没有在问顺丰。点名了公司/标签才算在查库。
    """
    probe = (query or "").strip()
    if not probe:
        return False
    try:
        files = store.list_kb_files()
    except Exception:
        logger.warning("kb 主题判断拿文件列表失败", exc_info=True)
        return False
    ready = [item for item in files if item.get("status") == "ready"]
    if file_ids:
        want = {str(fid) for fid in file_ids}
        ready = [item for item in ready if str(item.get("file_id")) in want]
    for item in ready:
        if _public().name_in_text(probe, _file_subject(item.get("filename") or "")):
            return True
        for tag in item.get("tags") or []:
            text = str(tag).strip()
            if len(text) >= 2 and text in probe:
                return True
    return False

def mentioned_subjects(
    query: str, file_ids: list[uuid.UUID] | None = None
) -> set[str]:
    """query 里点名的文件主体（subject 核心名）集合。

    主体、别名、文件名去噪后的名字整段出现才算。别名表没有的简称，
    只要是某一家主体名里独有的一段（「中科飞测」），也算点名——
    对话靠这个决定要不要强制调用知识库工具。
    """
    text = (query or "").strip()
    if not text:
        return set()
    try:
        files = store.list_kb_files()
    except Exception:
        logger.warning("kb 主体提及检测拿文件列表失败", exc_info=True)
        return set()
    ready = [item for item in files if item.get("status") == "ready"]
    if file_ids:
        want = {str(fid) for fid in file_ids}
        ready = [item for item in ready if str(item.get("file_id")) in want]
    subjects: set[str] = set()
    for item in ready:
        names = [item.get("subject") or ""]
        names.extend(alias for alias in (item.get("subject_aliases") or []) if alias)
        names.append(_file_subject(item.get("filename") or ""))
        for name in names:
            if name and _public().name_in_text(text, name):
                core = _subject_core(name)
                subjects.add(core or name)
                break
    from ai_agent.kb.ingest.meta import fragment_subjects

    catalog = [
        (str(item.get("subject") or ""), list(item.get("subject_aliases") or []))
        for item in ready
    ]
    for subject in fragment_subjects(text, catalog):
        core = _subject_core(subject)
        subjects.add(core or subject)
    return subjects

def mentioned_entities(
    query: str, file_ids: list[uuid.UUID] | None = None
) -> list[str]:
    """问句里按字面出现的图谱实体名或别名。

    文件主体的简称由 mentioned_subjects 负责。这里补人名、子公司、产品
    这类不在文件 subject 上的名字：实体名或别名整段出现在问句里才算命中。
    读库失败返回空，调用方退回「不强制检索」。
    """
    text = (query or "").strip()
    if len(text) < 2:
        return []
    try:
        names = store.query_mentions_entity(text, file_ids=file_ids)
    except Exception:
        logger.warning("kb 实体提及检测失败", exc_info=True)
        return []
    return [name for name in names if name]
