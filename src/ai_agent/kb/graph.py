"""实体关系抽取：切片 → 实体 / 关系 / 提及 → 图谱表。

为什么用 LLM 而不是 NER 模型：
- 文档里的关键实体多是行业概念、公司、指标（"HBM3E""长鑫存储""资本开支"），
  通用 NER 模型识别不了，且拿不到关系；
- 一次调用同时产出实体与关系，关系还带一句说明，前端展示和检索都用得上。

成本控制（这个任务是全链路最慢的一环，700 多个切片 ≈ 94 次调用）：
- 按 settings.kb_graph_batch_chunks 个切片打包成一次调用（默认 8）。
  批大小受**模型上下文**限制而不是算术：本地 Ollama 的 qwen2.5:7b
  context_length 只有 4096，16 片以上输入就会挤掉输出、模型开始返回
  别的 schema（实测 32 片时直接返回 {"公司":…}）；
- 输出用紧凑数组格式（e/r），不重复写 name/type/description 这些键。
  这类任务是**输出 token 决定耗时**，输出从 2476 字压到 524 字，
  单批 35s → 23s；
- 输出结构与约定不符时把批拆半重试一次。上下文吃紧时模型会悄悄换个
  schema，直接丢弃等于静默丢一批人（旧实现就吃过这个亏）；
- 短切片（目录行、页码、页眉）在进批之前就滤掉；
- 批间按 settings.kb_graph_concurrency 并发（本地单实例模型并发不涨吞吐）；
- 每 kb_graph_flush_batches 批落一次库，中断也留着部分图，不用从头再抽。

抽取失败不抛给上层：图谱是增强能力，失败只是没有图，不能拖垮已经可检索的文件。
"""

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import Chunk
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.embeddings import embed_texts_parallel
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
4. type 只能取：person / org / product / technology / metric / event / location / concept / other。
5. e 的每项是 [名称, 类型, 说明, 片段序号]；r 的每项是 [主体, 客体, 关系, 说明, 片段序号]。
   说明是不超过 20 字的身份或作用，片段里信息不足就留空字符串；
   片段序号表示它出现在输入的第几个片段（从 0 开始，必须是实际出现的那一片，不要瞎填）。
6. 只输出一行 JSON，不要解释、不要 Markdown 代码块。格式：
{"e":[["某某公司","org","",0]],"r":[["某某公司","某某产品","生产","",0]]}

注意：上面的 JSON 只是格式示意，"某某公司""某某产品"是占位符。
实体和关系的名称一律取自片段原文；片段里没出现的词严禁输出，
也不要把格式示例里的占位符当成实体。"""


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


def _parse_extraction(data: object) -> _Extraction | None:
    """把模型返回的 dict 解成 _Extraction，认不出结构返回 None。

    紧凑格式（e / r 数组）是主线：同样的内容，省掉逐项重复的
    name/type/description/chunk 键，输出 token 直接砍一半，而生成 token
    就是这类任务的全部耗时。冗长格式仍然认——模型偶尔会自己换回去，
    老版本输出、别家模型也未必听话，多认一种不亏。
    """
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("e"), list) or isinstance(data.get("r"), list):
        return _from_compact(data)
    if isinstance(data.get("entities"), list) or isinstance(data.get("relations"), list):
        try:
            return _Extraction.model_validate(data)
        except Exception:
            return None
    return None


def _from_compact(data: dict) -> _Extraction:
    out = _Extraction()
    for item in data.get("e") or []:
        row = _compact_row(item, 4, ("name", "type", "description", "chunk"))
        if row is None:
            continue
        name = str(row[0] or "").strip()
        if not name:
            continue
        out.entities.append(
            _Entity(
                name=name,
                type=str(row[1] or "other"),
                description=str(row[2] or ""),
                chunk=_as_int(row[3]),
            )
        )
    for item in data.get("r") or []:
        row = _compact_row(
            item, 5, ("source", "target", "relation", "description", "chunk")
        )
        if row is None:
            continue
        source = str(row[0] or "").strip()
        target = str(row[1] or "").strip()
        if not source or not target:
            continue
        out.relations.append(
            _Relation(
                source=source,
                target=target,
                relation=str(row[2] or "相关"),
                description=str(row[3] or ""),
                chunk=_as_int(row[4]),
            )
        )
    return out


def _compact_row(item: object, size: int, keys: tuple[str, ...]) -> list | None:
    """把一行数组/对象规整成定长 list；模型混着给也接。"""
    if isinstance(item, (list, tuple)):
        row = list(item)
    elif isinstance(item, dict):
        row = [item.get(k) for k in keys]
    else:
        return None
    if len(row) < 2:
        return None
    return (row + [None] * size)[:size]


def _as_int(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


# 纯数字/标点切片（页码行、目录点线、"……"）：抽不出实体，白付一次调用
_DIGITISH = re.compile(r"^[\d\s.,%+\-—–~至/、:：()（）\[\]【】]+$")


def _worth_extracting(text: str) -> bool:
    """这个切片值不值得付一次 LLM 调用。

    招股书/年报里目录行、页码、"（此页无正文）"这类碎片不少，
    它们既抽不出实体，又会占用批里的位置（批大小受上下文限制，
    塞几个碎片就等于少抽一段真内容）。
    """
    body = (text or "").strip()
    if len(body) < max(1, settings.kb_graph_min_chunk_chars):
        return False
    return not _DIGITISH.match(body)


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
        if chunk.page and chunk.page_end and chunk.page_end != chunk.page:
            where = f"第{chunk.page}–{chunk.page_end}页"
        tag = f"{chunk.kind} · {where}"
        if chunk.section:
            tag += f" · {chunk.section}"
        parts.append(f"[{i}] （{tag}）\n{chunk.text}")
    # 整批同属一节时，把节名提到开头：模型据此判断"这些片段是同一个业务
    # 主题的连续论述"，关系更容易连上；跨节的批不写，免得它硬凑跨节关系
    sections = {c.section for c in chunks}
    head = f"以下是文档的第 {start_seq} 段起的连续片段"
    if len(sections) == 1 and chunks[0].section:
        head += f"，全部来自【{chunks[0].section}】"
    return (
        head
        + "：\n\n"
        + "\n\n".join(parts)
        + "\n\n请抽取这些片段中的实体与关系，按约定 JSON 输出。"
    )


def _extract_batch(chunks: list[Chunk], start_seq: int, _depth: int = 0) -> _Extraction:
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
    parsed = _parse_extraction(load_llm_json(raw))
    if parsed is not None:
        return parsed

    # 结构与约定不符：多半是上下文吃紧，模型把输出挤成了别的形状。
    # 直接当"这批没实体"丢掉就是静默丢人——拆半重试一次，半批的输入
    # 只有原来一半，同一模型通常就能正常输出（再失败就真放弃，别无限递归）
    if _depth < 1 and len(chunks) > 2:
        mid = len(chunks) // 2
        logger.warning(
            "kb 图谱抽取输出不可用（片段 %d 起 %d 片，返回 %d 字），拆半重试",
            start_seq,
            len(chunks),
            len(raw or ""),
        )
        return _merge_extractions(
            _extract_batch(chunks[:mid], start_seq, _depth + 1),
            _extract_batch(chunks[mid:], start_seq + mid, _depth + 1),
            offset=mid,
        )
    logger.warning(
        "kb 图谱抽取返回不可解析，跳过该批（片段 %d 起 %d 片，返回 %d 字）",
        start_seq,
        len(chunks),
        len(raw or ""),
    )
    return _Extraction()


def _merge_extractions(first: _Extraction, second: _Extraction, *, offset: int) -> _Extraction:
    """合并拆半重试的两半结果，把后半的片段序号偏移回原批的位置。"""
    for item in second.entities:
        item.chunk += offset
    for rel in second.relations:
        rel.chunk += offset
    return _Extraction(
        entities=first.entities + second.entities,
        relations=first.relations + second.relations,
    )


def _batch_bounds(
    chunks: list[Chunk], batch_size: int
) -> list[tuple[int, list[Chunk]]]:
    """按章节组批：一节的切片不被批边界切开，小章节合批减少调用次数。

    纯按 seq 每 batch_size 片切会把一节拦腰截断（"第十节"的 33 片可能被
    切进两批，后半批看不到前半批刚定义的实体），关系就更容易抽漏——这
    正是「597 个实体只有 299 条关系」的一个来源。这里以章节为装箱单位：
    整节能塞进当前批就塞，塞不下换批；单节大于一批时才在节内均分。

    合批只会在相邻小节之间发生（一节的尾巴 + 下一节的头），每片都带
    section 标签，模型分得清；拆批则绝不发生在节内。
    """
    groups: list[tuple[str | None, int, int]] = []  # (section, start, end)
    for i, chunk in enumerate(chunks):
        if groups and groups[-1][0] == chunk.section:
            groups[-1] = (chunk.section, groups[-1][1], i + 1)
        else:
            groups.append((chunk.section, i, i + 1))

    bounds: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for _section, start, end in groups:
        size = end - start
        if size > batch_size:
            # 单节就是超一批：只能按批大小顺序切（不超配置的批大小，保证
            # 单次请求 token 可预估）。切剩的尾部不立刻封批——留给下一节
            # 合批，避免出现只有 1~2 片的空转调用
            if current:
                bounds.append(current)
            offset = start
            while offset + batch_size < end:
                bounds.append((offset, offset + batch_size))
                offset += batch_size
            current = (offset, end)
            continue
        if current is None:
            current = (start, end)
        elif current[1] - current[0] + size <= batch_size:
            current = (current[0], end)
        else:
            bounds.append(current)
            current = (start, end)
    if current:
        bounds.append(current)
    return [(start, chunks[start:end]) for start, end in bounds]


def _extract_all(
    chunks: list[Chunk],
    batch_size: int,
    sink: Callable[[int, int, list[Chunk], _Extraction], None] | None = None,
) -> list[tuple[int, list[Chunk], _Extraction]]:
    """并行抽取所有批，返回 (起始 seq, 本批切片, 抽取结果)，按批序排列。

    批与批互相独立——合并按归一化名在后置阶段做，不依赖抽取顺序。
    串行是图谱构建慢的主因：94 批 × 每批一次 LLM 调用逐个等，
    并发后墙钟时间约缩到 1/并发数（并发上限受部署形态约束，
    本地单实例模型并发不涨吞吐，见 KB_GRAPH_CONCURRENCY）。

    sink 让调用方「边抽边合并」：回调严格按批序发生，合并结果与串行一致，
    因此可以每 N 批落一次库（见 _GraphAccumulator）。

    单批异常只跳过该批（图谱是增强能力，部分图好过没有）；
    但全部批都异常说明是配置类错误（模型名/key），照旧抛给上层标 failed。
    """
    bounds = _batch_bounds(chunks, batch_size)
    results: list[_Extraction | None] = [None] * len(bounds)
    concurrency = max(1, min(settings.kb_graph_concurrency, len(bounds)))

    # 并发的完成顺序是乱的，而合并语义（同名实体取首次出现的类型）依赖批序，
    # 所以用 pending + 游标把乱序结果重新排队，再按序回调 sink
    lock = threading.Lock()
    pending: dict[int, tuple[int, list[Chunk], _Extraction]] = {}
    cursor = {"next": 0}

    def deliver() -> None:
        while cursor["next"] in pending:
            index = cursor["next"]
            start, batch, result = pending.pop(index)
            if sink is not None:
                try:
                    sink(index, start, batch, result)
                except Exception:
                    # 落库/合并不该拖垮整个抽取：记全栈，继续抽其余批
                    logger.exception("kb 图谱合并写入失败（片段 %d 起）", start)
            cursor["next"] += 1

    def run(index: int, start: int, batch: list[Chunk]) -> None:
        # 异常接在 worker 里：该批结果置 None 走跳过逻辑。
        # LLMError（超时、限流、返回异常）是预期内的外部故障，只记一行——
        # 本地小模型超时时会连堆栈刷屏，日志淹没有用信息；其余异常才是 bug，留全栈
        try:
            result: _Extraction | None = _extract_batch(batch, start)
        except LLMError as e:
            logger.warning(
                "kb 图谱抽取批失败（片段 %d 起）：%s（超时上限 %.0fs）",
                start,
                e,
                settings.kb_graph_timeout_seconds,
            )
            result = None
        except Exception:
            logger.exception("kb 图谱抽取批异常（片段 %d 起）", start)
            result = None
        with lock:
            results[index] = result
            if result is not None:
                pending[index] = (start, batch, result)
            deliver()

    if concurrency <= 1:
        for i, (start, batch) in enumerate(bounds):
            run(i, start, batch)
    else:
        with ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="kg-extract"
        ) as pool:
            futures = [
                pool.submit(run, i, start, batch)
                for i, (start, batch) in enumerate(bounds)
            ]
            for future in futures:
                future.result()  # run 已兜底，这里只是等齐

    failed = [i for i, r in enumerate(results) if r is None]
    if failed:
        if len(failed) == len(bounds):
            raise RuntimeError("图谱抽取全部批次失败（模型或网络配置问题）")
        # 汇总一行：逐批打过的失败细节留给前面，这里给个规模感，
        # 让「图谱怎么这么稀疏」在日志里能直接对上原因
        logger.warning(
            "kb 图谱抽取跳过 %d/%d 批（片段 %s 起），图谱会不完整",
            len(failed),
            len(bounds),
            ",".join(str(bounds[i][0]) for i in failed[:10])
            + ("…" if len(failed) > 10 else ""),
        )
    return [
        (start, batch, results[i] or _Extraction())
        for i, (start, batch) in enumerate(bounds)
    ]


class _GraphAccumulator:
    """跨批增量合并 + 分次落库。

    为什么要增量：700 多个切片是几十批、几十分钟的活。全部抽完才写库的话，
    中途超时或被重启就一点都不剩——用户感受到的是"等了一小时，白等"。
    每 N 批覆盖写一次，进程被杀也留着部分图，前端轮询的实体数还能往上走。

    合并策略与一次性合并完全一致：同名（归一化后）实体合并，type 取首次
    出现的值，mention_count 累计，描述保留更长的那个。
    """

    def __init__(self, file_id: uuid.UUID) -> None:
        self.file_id = file_id
        self.entities: dict[str, dict] = {}
        # (source_norm, target_norm, relation) -> relation dict，天然去重
        self.relations: dict[tuple[str, str, str], dict] = {}
        self.mentions: dict[tuple[str, int], int] = {}
        # norm -> 向量：分次落库时只给新实体算向量，别整份重算
        self._vectors: dict[str, list[float]] = {}

    def add(self, start: int, batch: list[Chunk], result: _Extraction) -> None:
        if not result.entities:
            return

        for item in result.entities:
            norm = normalize_name(item.name)
            if not _usable_name(item.name, norm):
                continue
            entity = self.entities.get(norm)
            if entity is None:
                entity = {
                    "name": display_name(item.name),
                    "norm": norm,
                    "type": _clean_type(item.type),
                    "description": (item.description or "").strip()[:200],
                }
                self.entities[norm] = entity
            else:
                if len(item.description or "") > len(entity["description"]):
                    entity["description"] = item.description.strip()[:200]
                    # 描述变了，向量也得重算（它在 _vectors 里已经过期）
                    self._vectors.pop(norm, None)
            # 提及记录模型自己指认的片段（越界就夹回本批范围），
            # 这样 mention_count 才有区分度，实体→原文也能精确定位
            seq = start + min(max(item.chunk, 0), len(batch) - 1)
            key = (norm, seq)
            self.mentions[key] = self.mentions.get(key, 0) + 1

        for rel in result.relations:
            src = normalize_name(rel.source)
            dst = normalize_name(rel.target)
            # 端点查全局实体表，不查「本批」：主体常在前一批被抽出，本批只是
            # 顺带提到它（"公司向亚厦控股采购…"），只在本批内找的话这类关系
            # 会被整条丢掉，最后表现为「实体都在图上却互不相连」的孤立点
            if src == dst or src not in self.entities or dst not in self.entities:
                continue
            rel_name = (rel.relation or "相关").strip()[:20] or "相关"
            key = (src, dst, rel_name)
            if key in self.relations:
                continue
            self.relations[key] = {
                "source": src,
                "target": dst,
                "relation": rel_name,
                "description": (rel.description or "").strip()[:200],
                "chunk_seq": start + min(max(rel.chunk, 0), len(batch) - 1),
            }

    def flush(self) -> dict:
        """把当前累积结果整文件覆盖写一次，返回计数（幂等，可反复调）。"""
        if not self.entities:
            return {"entities": 0, "relations": 0}

        norms = list(self.entities.keys())
        missing = [n for n in norms if n not in self._vectors]
        if missing:
            # 实体名+描述一起向量化：只向量化名字的话，"长鑫存储" 和
            # "DRAM 制造" 这类语义相关的实体在向量空间里挨不上
            vectors = embed_texts_parallel(
                [
                    f"{self.entities[n]['name']}（{self.entities[n]['type']}）："
                    f"{self.entities[n]['description']}"
                    for n in missing
                ],
                concurrency=settings.kb_embedding_concurrency,
            )
            self._vectors.update(zip(missing, vectors))

        entity_rows: list[dict] = []
        id_by_norm: dict[str, uuid.UUID] = {}
        mention_count: dict[str, int] = {}
        for norm, _seq in self.mentions:
            mention_count[norm] = mention_count.get(norm, 0) + 1
        for norm in norms:
            entity = self.entities[norm]
            eid = uuid.uuid5(
                uuid.NAMESPACE_URL, f"kg:{self.file_id}:{norm}:{entity['type']}"
            )
            id_by_norm[norm] = eid
            entity_rows.append(
                {
                    "id": eid,
                    "name": entity["name"],
                    "norm_name": norm,
                    "type": entity["type"],
                    "description": entity["description"],
                    "embedding": self._vectors[norm],
                    "mention_count": mention_count.get(norm, 1),
                }
            )

        relation_rows = [
            {
                "id": uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"kgr:{self.file_id}:{rel['source']}:{rel['target']}:"
                    f"{rel['relation']}",
                ),
                "source_id": id_by_norm[rel["source"]],
                "target_id": id_by_norm[rel["target"]],
                "relation": rel["relation"],
                "description": rel["description"],
                "weight": 1.0,
                "chunk_seq": rel["chunk_seq"],
            }
            for rel in self.relations.values()
        ]
        mention_rows = [
            {"entity_id": id_by_norm[norm], "chunk_seq": seq, "hits": hits}
            for (norm, seq), hits in self.mentions.items()
            if norm in id_by_norm
        ]

        store.replace_file_graph(
            self.file_id, entity_rows, relation_rows, mention_rows
        )
        return {"entities": len(entity_rows), "relations": len(relation_rows)}


def build_file_graph(
    file_id: uuid.UUID,
    chunks: list[Chunk],
    *,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """抽取并写入某个文件的图谱，返回计数。

    progress 在每次阶段落库后回调（带当前累计计数），上层拿它更新
    graph_status 的 entity_count，前端就能看到"建到一半"的进度。
    """
    kept = [c for c in chunks if _worth_extracting(c.text)]
    if len(kept) < len(chunks):
        logger.info(
            "kb 图谱跳过 %d/%d 个低信息切片（短于 %d 字或纯数字）",
            len(chunks) - len(kept),
            len(chunks),
            settings.kb_graph_min_chunk_chars,
        )
    if not kept:
        logger.info("kb 图谱无可用切片 file_id=%s", file_id)
        return {"entities": 0, "relations": 0}

    batch_size = max(1, settings.kb_graph_batch_chunks)
    flush_every = max(1, settings.kb_graph_flush_batches)
    # 批数要按「章节组批」后的实际批数算：进度百分比和分次落库的节奏都
    # 依赖它，用 ceil(片数/批大小) 会在小章节合批后偏高
    total_batches = len(_batch_bounds(kept, batch_size))
    accumulator = _GraphAccumulator(file_id)
    started = time.perf_counter()
    state = {"batches": 0}

    def sink(index: int, start: int, batch: list[Chunk], result: _Extraction) -> None:
        accumulator.add(start, batch, result)
        state["batches"] = index + 1
        if (index + 1) % flush_every == 0 and index + 1 < total_batches:
            counts = accumulator.flush()
            logger.info(
                "kb 图谱阶段写入 file_id=%s %d/%d 批 entities=%d relations=%d %.0fs",
                file_id,
                index + 1,
                total_batches,
                counts["entities"],
                counts["relations"],
                time.perf_counter() - started,
            )
            if progress is not None:
                progress(counts)

    _extract_all(kept, batch_size, sink=sink)
    counts = accumulator.flush()
    logger.info(
        "kb 图谱抽取完成 file_id=%s batches=%d chunks=%d(过滤前 %d) 实体=%d 关系=%d %.0fs",
        file_id,
        total_batches,
        len(kept),
        len(chunks),
        counts["entities"],
        counts["relations"],
        time.perf_counter() - started,
    )
    if progress is not None:
        progress(counts)
    return counts
