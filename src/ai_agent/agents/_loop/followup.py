"""追问与预检索判定：主体延续、图谱预取、强制检索规则。

由原 agents/loop.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

import re
from ai_agent.core.config import settings
import uuid
from ai_agent.kb.retrieval.scope import reset_kb_scope, set_kb_scope
from ai_agent.llm.types import ChatMessage

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)


# 测试 monkeypatch 打在 ``ai_agent.agents.loop`` 上；内部调用一律经
# _public() 取当前属性，否则补丁会被本地绑定绕过（静默失效）。
def _public():
    import ai_agent.agents.loop as public

    return public


def _latest_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""

def _question_already_retrieved(messages: list[ChatMessage]) -> bool:
    """当前这句问话之后已经有工具结果，就不再强制检索。"""
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "tool":
            return True
        if role == "user":
            return False
    return False

def _followup_subjects(
    messages: list[ChatMessage],
    file_ids: list[uuid.UUID] | None,
) -> set[str]:
    """当前句省略主体时，回看此前几轮用户问话点过的库内主体。

    追问常把主体留在上一轮（「那毛利率呢」「去年呢」）。强制检索门只看
    当前句字面会把这类问句放过去，模型不查库、拿上一轮的检索片段或训练
    记忆直接作答。回看窗口与查询改写共用 kb_rewrite_history_turns，命中
    任意一轮即认定这句仍是知识库追问。读库失败 fail-open 返回空（不强制，
    与字面检测的放行语义一致）。只回看主体；人名/子公司等实体追问更少见，
    不在这次兜底范围里。
    """
    from ai_agent.kb.retrieval import mentioned_subjects

    cur: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            cur = i
            break
    if cur is None:
        return set()
    turns = 0
    for i in range(cur - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        text = str(messages[i].get("content") or "").strip()
        if not text:
            continue
        turns += 1
        if turns > settings.kb_rewrite_history_turns:
            break
        try:
            hits = mentioned_subjects(text, file_ids=file_ids)
        except Exception:
            _public().logger.warning("追问主体回看检测失败，按不强制处理", exc_info=True)
            return set()
        if hits:
            _public().logger.info(
                "追问未点名主体，回看此前问句命中 subjects=%s",
                "、".join(sorted(hits)),
            )
            return hits
    return set()

def _kg_prefetch_present(messages: list[ChatMessage]) -> bool:
    """这句问话之后已经注入过邻域，续跑时不要再查一次。"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return False
        if msg.get("role") == "system" and str(msg.get("content") or "").startswith(
            _KG_PREFETCH_MARK
        ):
            return True
    return False

def _attach_kg_prefetch(
    messages: list[ChatMessage], kb_file_ids: tuple[str, ...]
) -> None:
    """关系问句在选工具前放入图谱邻域。科目金额问句不放。"""
    if _public()._kg_prefetch_present(messages):
        return
    text = _public()._latest_user_text(messages).strip()
    if not text:
        return
    from ai_agent.tools.kb.kg_search import prefetch_neighborhood

    scope_token = set_kb_scope(kb_file_ids)
    try:
        block = prefetch_neighborhood(text)
    finally:
        reset_kb_scope(scope_token)
    if not block:
        return
    _public().logger.info("图谱预检索已注入 chars=%d", len(block))
    messages.append({"role": "system", "content": block})

def _must_call_kb_tool(
    messages: list[ChatMessage],
    tools: list[dict],
    kb_file_ids: tuple[str, ...] = (),
) -> bool:
    """问句点名了文件主体或图谱实体时，这一步必须调用知识库工具。

    主体（公司名、简称）和图谱实体（人名、子公司、产品）都会触发。
    只强制「必须查」，选 kb_search 还是 kg_search 仍由模型按【知识库能力】决定。
    当前句字面点名主体/实体就强制；追问省略了主体时，回看前几轮用户问句
    是否点过主体（见 _followup_subjects，可用 kb_followup_force_enabled 关闭）。
    这句之后已经有工具结果（续跑、检索完准备作答）不再强制。读库失败则放行。
    """
    names = {t.get("function", {}).get("name") for t in tools}
    if not (_KB_RETRIEVAL_TOOLS & names):
        return False
    if _public()._question_already_retrieved(messages):
        return False
    text = _public()._latest_user_text(messages).strip()
    if not text:
        return False
    if _REPORT_VERB.search(text) and _REPORT_NOUN.search(text):
        return False
    from ai_agent.kb.retrieval import mentioned_entities, mentioned_subjects

    file_ids: list[uuid.UUID] | None = None
    if kb_file_ids:
        parsed: list[uuid.UUID] = []
        for raw in kb_file_ids:
            try:
                parsed.append(uuid.UUID(str(raw)))
            except ValueError:
                continue
        file_ids = parsed or None
    try:
        subjects = mentioned_subjects(text, file_ids=file_ids)
    except Exception:
        _public().logger.exception("主体提及检测失败")
        subjects = set()
    # 主体命中已经足够强制检索，但实体仍要查：问的是公司 A 里的实体 B 时，
    # 日志里得看得到 B，不能因为 A 命中就把实体检测跳过。
    entities = mentioned_entities(text, file_ids=file_ids)
    if not subjects and not entities and settings.kb_followup_force_enabled:
        # 追问兜底：这句没点名主体，但前几轮问句点过库内主体——主体多半被
        # 省略或写成了指代。字面检测会放行，模型便不查库直接作答；这里把
        # 这句仍按知识库追问强制检索（具体主体回看逻辑见 _followup_subjects）。
        subjects = _public()._followup_subjects(messages, file_ids)
    if subjects or entities:
        _public().logger.info(
            "问句命中知识库，强制调用工具 subjects=%s entities=%s",
            "、".join(sorted(subjects)),
            "、".join(entities),
        )
    return bool(subjects or entities)

# 问句点名了库内主体或图谱实体时，第一步只留这两个工具并强制调用。
# 天气、记忆、联网留在后面的轮次，避免模型用它们应付「必须调一个工具」。
_KB_RETRIEVAL_TOOLS = frozenset({"kb_search", "kg_search", "kb_read"})

_KG_PREFETCH_MARK = "【图谱邻域】"

# 「生成一份研报」走独立流水线：工具自己会检索、抽取、计算、校验。
# 这类请求不必先被强制查一轮 kb_search，否则既多烧一次调用，又把原文片段
# 塞进上下文、诱导模型自己撰写正文而不去调 generate_report。
# 必须同时有「生成类动词」和「研报类名词」，避免误伤「研报里说了什么」这种检索题。
_REPORT_VERB = re.compile(r"生成|出一份|做一份|写一份|制作|导出|来一份|给我一份")

_REPORT_NOUN = re.compile(r"研报|财报深度分析|分析报告|PDF", re.IGNORECASE)
