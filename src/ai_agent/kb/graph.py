"""实体关系抽取：切片 → 实体 / 关系 / 提及 → 图谱表。

为什么用 LLM 而不是 NER 模型：
- 文档里的关键实体多是行业概念、公司、指标（"HBM3E""长鑫存储""资本开支"），
  通用 NER 模型识别不了，且拿不到关系；
- 一次调用同时产出实体与关系，关系还带一句说明，前端展示和检索都用得上。

成本控制：
- 按 settings.kb_graph_batch_chunks 个切片打包成一次调用（默认 8）；
- 抽取失败不抛给上层——图谱是增强能力，失败只是没有图，不能拖垮已经可检索的文件。
"""

import re
import uuid

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import Chunk
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.embeddings import embed_texts
from ai_agent.memory.parse import load_llm_json
from ai_agent.memory.store import store

logger = get_logger(__name__)

# 与前端图例一一对应；模型返回值不在集合内的一律归为 other
ENTITY_TYPES = (
    "person",
    "org",
    "product",
    "technology",
    "metric",
    "event",
    "location",
    "concept",
    "other",
)

# 出现在几乎每篇文档里、对区分内容毫无帮助的词，抽到也只会污染图谱
_STOP_NAMES = {
    "文档",
    "本文",
    "报告",
    "研究",
    "分析",
    "数据",
    "内容",
    "章节",
    "表格",
    "图表",
    "公司",
    "行业",
    "市场",
    "产品",
    "技术",
}

# 首尾需要剥掉的包裹符号。刻意不含圆括号：括号常是名字的一部分
# （「长鑫存储（合肥）」剥成「长鑫存储（合肥」就毁名了）
_EDGE_CHARS = r"\s\"'“”‘’《》〈〉<>,.;:!?\-—–_/\\|\[\]【】{}·"
# 归一化键用：全角括号/引号统一成半角，避免「（合肥）」与「(合肥)」匹配不上
_FULLWIDTH = str.maketrans(
    {"（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
     "“": '"', "”": '"', "‘": "'", "’": "'"}
)
_KEY_TRIM = re.compile(rf"^[{_EDGE_CHARS}]+|[{_EDGE_CHARS}]+$")
# 展示名只去首尾引号/书名号，圆括号按原样保留（中文里全角括号更自然）
_DISPLAY_TRIM = re.compile(
    r"^[\s\"'“”‘’《》〈〉<>,.;:!?\-—–_/\\|\[\]【】{}·]+|"
    r"[\s\"'“”‘’《》〈〉<>,.;:!?\-—–_/\\|\[\]【】{}·]+$"
)
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

_SYSTEM = """你是知识图谱抽取器。从给定文档片段中抽取实体与它们之间的明确关系。

要求：
1. 实体是专有名词或关键概念：公司、机构、人物、产品/型号、技术、指标、事件、地点、业务概念。
2. 不要抽取泛化词（如"文档""行业""数据""报告"）、纯数字、代词。
3. 关系必须是片段中明确陈述的，relation 用 2-8 个汉字的动词或名词短语（如"供应商""属于""投资""量产""依赖""同比增长"）。
4. 每个实体给一句不超过 40 字的 description，说明它在片段里的身份或作用；片段信息不足时留空字符串。
5. type 只能取：person / org / product / technology / metric / event / location / concept / other。
6. 每个实体和关系都要带 chunk 字段，表示它出现在输入的第几个片段（从 0 开始）——
   用于回溯原文，必须是实际出现的那一片，不要瞎填。
7. 只输出 JSON，不要解释、不要 Markdown 代码块。格式：
{"entities":[{"name":"长鑫存储","type":"org","description":"国产 DRAM 制造商","chunk":0}],
 "relations":[{"source":"长鑫存储","target":"DRAM","relation":"生产","description":"量产 DRAM 颗粒","chunk":0}]}"""


class _Entity(BaseModel):
    name: str
    type: str = "other"
    description: str = ""
    chunk: int = 0


class _Relation(BaseModel):
    source: str
    target: str
    relation: str = "相关"
    description: str = ""
    chunk: int = 0


class _Extraction(BaseModel):
    entities: list[_Entity] = Field(default_factory=list)
    relations: list[_Relation] = Field(default_factory=list)


def normalize_name(name: str) -> str:
    """用于去重与关系连边的归一化键。

    全角括号/引号先转半角，再去首尾包裹符号、压缩空白、转小写，
    这样「长鑫存储（合肥）」和「长鑫存储(合肥)」会落到同一个键上。
    """
    text = (name or "").translate(_FULLWIDTH).strip()
    cleaned = _KEY_TRIM.sub("", text).strip()
    return re.sub(r"\s+", "", cleaned).lower()


def display_name(name: str) -> str:
    """展示名：只去掉首尾的引号、书名号、空白，圆括号原样保留。"""
    return _DISPLAY_TRIM.sub("", (name or "").strip()).strip()


def _clean_type(raw: str) -> str:
    value = (raw or "").strip().lower()
    return value if value in ENTITY_TYPES else "other"


def _usable_name(name: str, norm: str) -> bool:
    if not norm or norm in _STOP_NAMES:
        return False
    if len(norm) > 40 or _PUNCT_ONLY.match(norm):
        return False
    # 纯数字（含千分位、百分号）不当实体
    return not re.fullmatch(r"[\d.,%+-]+", norm)


def _build_prompt(chunks: list[Chunk], start_seq: int) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        where = f"第{chunk.page}页" if chunk.page is not None else "无页码"
        parts.append(f"[{i}] （{chunk.kind} · {where}）\n{chunk.text}")
    return (
        f"以下是文档的第 {start_seq} 段起的连续片段：\n\n"
        + "\n\n".join(parts)
        + "\n\n请抽取这些片段中的实体与关系，按约定 JSON 输出。"
    )


def _extract_batch(chunks: list[Chunk], start_seq: int) -> _Extraction:
    raw = complete_chat(
        [
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(role="user", content=_build_prompt(chunks, start_seq)),
        ],
        # 默认用主模型而不是 small_model：抽取是长 JSON 结构化任务，
        # 小模型容易漏关系、把 JSON 写坏（部署方的 SMALL_MODEL 也可能并未开通）
        model=settings.kb_graph_model or settings.model,
        timeout=settings.kb_graph_timeout_seconds,
        max_retries=1,
        # 一批 8 个切片、几十个实体时 2k token 会截断，截断的 JSON 直接解析失败
        max_tokens=4096,
        extra_body={"enable_thinking": False},
    )
    data = load_llm_json(raw)
    if not isinstance(data, dict):
        logger.warning(
            "kb 图谱抽取返回非 JSON，已跳过该批（片段 %d 起，返回长度 %d）",
            start_seq,
            len(raw or ""),
        )
        return _Extraction()
    try:
        return _Extraction.model_validate(data)
    except Exception:
        logger.warning("kb 图谱抽取 JSON 结构不符，已跳过该批")
        return _Extraction()


def build_file_graph(file_id: uuid.UUID, chunks: list[Chunk]) -> dict:
    """抽取并写入某个文件的图谱，返回计数。

    跨批合并策略：同名（归一化后）实体合并，type 取首次出现的值，
    mention_count 累计，描述保留更长的那个。
    """
    batch_size = max(1, settings.kb_graph_batch_chunks)
    entities: dict[str, dict] = {}
    # (source_norm, target_norm, relation) -> relation dict，天然去重
    relations: dict[tuple[str, str, str], dict] = {}
    mentions: dict[tuple[str, int], int] = {}

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        result = _extract_batch(batch, start)
        if not result.entities:
            continue

        # 本批先建立「归一化名 → 实体」映射，供关系连边
        batch_index: dict[str, dict] = {}
        for item in result.entities:
            norm = normalize_name(item.name)
            if not _usable_name(item.name, norm):
                continue
            entity = entities.get(norm)
            if entity is None:
                entity = {
                    "name": display_name(item.name),
                    "norm": norm,
                    "type": _clean_type(item.type),
                    "description": (item.description or "").strip()[:200],
                }
                entities[norm] = entity
            else:
                if len(item.description or "") > len(entity["description"]):
                    entity["description"] = item.description.strip()[:200]
            batch_index[norm] = entity
            # 提及记录模型自己指认的片段（越界就夹回本批范围），
            # 这样 mention_count 才有区分度，实体→原文也能精确定位
            seq = start + min(max(item.chunk, 0), len(batch) - 1)
            key = (norm, seq)
            mentions[key] = mentions.get(key, 0) + 1

        for rel in result.relations:
            src = normalize_name(rel.source)
            dst = normalize_name(rel.target)
            if src == dst or src not in batch_index or dst not in batch_index:
                continue
            rel_name = (rel.relation or "相关").strip()[:20] or "相关"
            key = (src, dst, rel_name)
            if key in relations:
                continue
            relations[key] = {
                "source": src,
                "target": dst,
                "relation": rel_name,
                "description": (rel.description or "").strip()[:200],
                "chunk_seq": start + min(max(rel.chunk, 0), len(batch) - 1),
            }

    if not entities:
        logger.info("kb 图谱抽取完成但无有效实体 file_id=%s", file_id)
        return {"entities": 0, "relations": 0}

    # 实体名+描述一起向量化：只向量化名字的话，"长鑫存储" 和 "DRAM 制造" 这类
    # 语义相关的实体在向量空间里挨不上
    norms = list(entities.keys())
    texts = [
        f"{entities[n]['name']}（{entities[n]['type']}）：{entities[n]['description']}"
        for n in norms
    ]
    vectors: list[list[float]] = []
    batch = max(1, settings.kb_embedding_batch)
    for i in range(0, len(texts), batch):
        vectors.extend(embed_texts(texts[i : i + batch]))

    entity_rows: list[dict] = []
    id_by_norm: dict[str, uuid.UUID] = {}
    mention_count: dict[str, int] = {}
    for norm, _seq in mentions:
        mention_count[norm] = mention_count.get(norm, 0) + 1
    for norm, vector in zip(norms, vectors):
        entity = entities[norm]
        eid = uuid.uuid5(uuid.NAMESPACE_URL, f"kg:{file_id}:{norm}:{entity['type']}")
        id_by_norm[norm] = eid
        entity_rows.append(
            {
                "id": eid,
                "name": entity["name"],
                "norm_name": norm,
                "type": entity["type"],
                "description": entity["description"],
                "embedding": vector,
                "mention_count": mention_count.get(norm, 1),
            }
        )

    relation_rows = [
        {
            "id": uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"kgr:{file_id}:{rel['source']}:{rel['target']}:{rel['relation']}",
            ),
            "source_id": id_by_norm[rel["source"]],
            "target_id": id_by_norm[rel["target"]],
            "relation": rel["relation"],
            "description": rel["description"],
            "weight": 1.0,
            "chunk_seq": rel["chunk_seq"],
        }
        for rel in relations.values()
    ]
    mention_rows = [
        {"entity_id": id_by_norm[norm], "chunk_seq": seq, "hits": hits}
        for (norm, seq), hits in mentions.items()
        if norm in id_by_norm
    ]

    store.replace_file_graph(file_id, entity_rows, relation_rows, mention_rows)
    logger.info(
        "kb 图谱写入 file_id=%s entities=%d relations=%d",
        file_id,
        len(entity_rows),
        len(relation_rows),
    )
    return {"entities": len(entity_rows), "relations": len(relation_rows)}
