"""知识图谱抽取：切片 → LLM 抽取 → 实体/关系结构体，以及命名规范化。

从 graph.py 拆出(S5)。包含三层：
1. 输出结构与容错解析：模型在上下文吃紧时会悄悄换个 schema，直接丢弃等于
   静默丢一批人，所以这里做了紧凑数组(e/r)与完整结构的双向兼容；
2. 命名规范化：normalize_name / display_name / is_metric_like 等，把模型
   吐出的各种写法收敛成统一的实体名；
3. 批量抽取：批大小受**模型上下文**限制而不是算术(见 _extract_all)，
   失败时把批拆半重试一次。

这是全链路最慢的一环(700 多个切片 ≈ 94 次调用)，成本控制的关键点都在
_extract_all 与 _extract_batch 的注释里。
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import Chunk
from ai_agent.llm.client import complete_chat, graph_client, graph_model
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.parse import load_llm_json

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

# 谓词误抽成实体。prompt 第一句已经要求「实体必须是名词或名词短语」，模型
# 仍会在个别片段把「搭载/发布/任职」写成点（三份季报各 1–2 个）。这里只拦
# **完全等同**于常见谓词的短名字：按后缀猜词性会误伤「持股比例」「研发投入」
# 这类真名词，而名单里的词本身没有一个是可检索的实体
_PREDICATE_NAMES = {
    "搭载",
    "发布",
    "任职",
    "支持",
    "持股",
    "研发",
    "提供",
    "实现",
    "采用",
    "推出",
    "生产",
    "销售",
    "增长",
    "下降",
    "包括",
    "进行",
    "完成",
    # 季报里常见的动宾被拆开：原文「对接阿联酋」「建设氢能平台」，
    # 模型把动词单独建成 other 点，边还叫同一个词
    "对接",
    "建设",
    "签约",
    "布局",
    "拓展",
    "开展",
    "推进",
    "加快",
    "加强",
    "提升",
    "优化",
    "合作",
    "产生",
    "上升",
    "互动",
}

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
# 文件名 → 主体全称：「安克创新科技股份有限公司2026 年第三季度报告」→
# 「安克创新科技股份有限公司」。招股书文件名常带城市前缀，取第一个公司
# 后缀更稳：宁可退回 None（不做兜底），也不猜错主体
_SUBJECT_RE = re.compile(
    r"^(.{2,30}?(?:股份有限公司|有限责任公司|有限公司|集团公司|集团|银行|研究院|学院|医院|大学))"
)

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
# 数字、空白、常见标点、时间词和计量用字。占比 ≥0.6 且含数字 → 数值/日期型名字。
# 「2004年」「62,200万千瓦」「1,017,077.00 元」会命中；
# 「2017年重大资产重组」「HBM3E」数字占比不够，当正常实体。
_METRIC_CHARS = re.compile(r"[\d\s,，.．·、+\-/%年月日度万亿千百十瓦时吨元]")

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
        row = _fix_short_relation(item, row)
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


_NUMERIC_ROW = re.compile(r"^[-\d,.\s%]+$")


def _fix_short_relation(item: object, row: list) -> list | None:
    """只有 4 项的 r 行：分清是「漏了说明」还是「漏了客体」。

    `_compact_row` 会把短行补到 5 项，补出来的位置全错。实测本地模型对
    「持股」这类关系常写成 [主体, 关系, 空或数值, 序号]——客体被吞掉，
    硬补就变成「公司 --相关--> 持有股份」这种假边，还会把谓语当成实体。
    两种形态靠第 3 项区分：是空/纯数值 = 漏了客体（整条丢掉，不猜）；
    是文字 = [主体, 客体, 关系, 序号]（照常用，说明补空）。
    """
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        return row
    third = item[2]
    text = "" if third is None else str(third).strip()
    if not text or _NUMERIC_ROW.match(text):
        return None
    return [item[0], item[1], item[2], "", item[3]]


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
# 勾选结果（parsing 归一化后的写法）：「……是否经过审计：☑（已勾选）否」
_SELECTED = re.compile(r"[☑☐]（[已未]勾选）")


def _worth_extracting(text: str) -> bool:
    """这个切片值不值得付一次 LLM 调用。

    招股书/年报里目录行、页码、"（此页无正文）"这类碎片不少，
    它们既抽不出实体，又会占用批里的位置（批大小受上下文限制，
    塞几个碎片就等于少抽一段真内容）。

    但「短」不等于「没信息」：勾选行天生就在一句话以内（实测 27~36 字），
    却是全文最确定的结论——是否经过审计、是否需要追溯调整、某节是否适用。
    按长度把它们当碎片扫掉，图谱里就永远问不到这些是否题。
    """
    body = (text or "").strip()
    if _SELECTED.search(body):
        return True
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


def _claimed_offset(claimed: int, start: int, count: int) -> int:
    """把模型报的片段编号换算成批内下标。

    它可能报批内下标，也可能报 prompt 里写的绝对段号（批起点不为 0 时
    两者不等），两种都接住；都不像就夹到批内。
    """
    if count <= 0:
        return 0
    if start <= claimed <= start + count - 1:
        return claimed - start
    return min(max(claimed, 0), count - 1)


def _locate_chunk(name: str, haystack: list[str], fallback: int) -> int:
    """这个实体/关系出现在本批的第几片。

    优先按名字做包含匹配：名字是我们手上的确定信息，比模型自报的位置
    可靠得多——实测 821 条提及里只有 5% 指对了片子，其余几乎全落在批首
    那一片。匹配不到（模型改写了名字、原文只有简称或别名）才退回它报的
    位置，至少还落在同一批里。
    """
    norm = normalize_name(name)
    if norm:
        for i, text in enumerate(haystack):
            if norm in text:
                return i
    return fallback


def _clean_type(raw: str) -> str:
    value = (raw or "").strip().lower()
    return value if value in ENTITY_TYPES else "other"


def is_metric_like(name: str) -> bool:
    """整条名字基本由数字和计量单位构成（时间点、金额、比例、指标值）。"""
    text = (name or "").strip()
    if not text:
        return False
    hit = len(_METRIC_CHARS.findall(text))
    return hit / len(text) >= 0.6 and any(ch.isdigit() for ch in text)


# prompt 示例里的占位词。示例一旦用真实公司名（曾经写成「深圳明德控股」
# 「顺丰控股」），模型在片段缺主语时会把它当模板抄进结果——安克文件的图里
# 长出过一个挂着 14 条关系的「顺丰控股」，就是这么来的。示例改成占位词后
# 再加这道过滤：万一模型仍然照抄，也进不了图谱
# 只拦「某某」式占位词，不拦「某公司」——原文匿名化时真的会这么写，
# 拿它当过滤条件会把正常实体一并丢掉
_PLACEHOLDER_RE = re.compile(r"某某|××|X{2,}")


def _usable_name(name: str, norm: str) -> bool:
    if not norm or norm in _STOP_NAMES or norm in _PREDICATE_NAMES:
        return False
    if len(norm) > 40 or _PUNCT_ONLY.match(norm):
        return False
    if _PLACEHOLDER_RE.search(norm):
        return False
    # 纯数字、时间点（「2004年」）、带单位的数值（「62,200万千瓦」）
    # 是取值或限定语，不是实体。指向它们的关系会因端点缺失被丢掉。
    if is_metric_like(name) or is_metric_like(norm):
        return False
    return True


def _doc_title(name: str | None) -> str:
    """文件名去掉扩展名，用作「这段文字属于哪份文档」的标识。"""
    return re.sub(r"\.[A-Za-z0-9]{1,5}$", "", (name or "").strip())


def _subject_from_filename(name: str | None) -> str | None:
    """从文件名推断文档主体全称，推断不出返回 None（不猜）。"""
    m = _SUBJECT_RE.match(_doc_title(name))
    return m.group(1) if m else None


def _build_prompt(chunks: list[Chunk], start_seq: int, doc_name: str | None = None) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        where = f"第{chunk.page}页" if chunk.page is not None else "无页码"
        if chunk.page and chunk.page_end and chunk.page_end != chunk.page:
            where = f"第{chunk.page}–{chunk.page_end}页"
        tag = f"{chunk.kind} · {where}"
        if chunk.section:
            # 章节名单独标出来：模型靠它判断「这段在讲哪块业务」，比页码有用得多
            # （「（二）非经常性损益项目和金额」里的数字，绝不会被当成主营收入）
            tag += f" · 章节：{chunk.section}"
        # 标号必须是**绝对**段号：写批内下标 [0] 时，模型引用只能靠数位置，
        # 而开头又写着「第 28–35 段」，两套编号并存，它报回来的数字两边都不像
        seq = chunk.seq if chunk.seq is not None else start_seq + i
        parts.append(f"[{seq}] （{tag}）\n{chunk.text}")
    # 整批同属一节时，把节名提到开头：模型据此判断"这些片段是同一个业务
    # 主题的连续论述"，关系更容易连上；跨节的批不写，免得它硬凑跨节关系
    sections = {c.section for c in chunks}
    # 批次定位写全「文档 / 段 / 页」：报表类文档的表格片段往往通篇没有公司
    # 全称，只有文件名能告诉模型这些科目和股东属于哪个主体。缺了它，模型
    # 会给"持股/每股收益"硬找一个主语（实测把 prompt 示例里的公司名抄了进来）
    span = (
        f"第 {start_seq} 段"
        if len(chunks) == 1
        else f"第 {start_seq}–{start_seq + len(chunks) - 1} 段"
    )
    pages = [c.page for c in chunks if c.page is not None]
    if pages:
        low, high = min(pages), max(pages)
        span += f"（第 {low} 页）" if low == high else f"（第 {low}–{high} 页）"
    title = _doc_title(doc_name)
    head = f"以下是《{title}》的{span}起的连续片段" if title else f"以下是文档的{span}起的连续片段"
    if len(sections) == 1 and chunks[0].section:
        head += f"，全部来自【{chunks[0].section}】"
    return (
        head
        + "：\n\n"
        + "\n\n".join(parts)
        + "\n\n请抽取这些片段中的实体与关系，按约定 JSON 输出。"
    )


def _extract_batch(
    chunks: list[Chunk],
    start_seq: int,
    system_prompt: str,
    _depth: int = 0,
    _reasked: bool = False,
    doc_name: str | None = None,
) -> _Extraction:
    raw = complete_chat(
        [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=_build_prompt(chunks, start_seq, doc_name)),
        ],
        model=graph_model(),
        client=graph_client(),
        timeout=settings.kb_graph_timeout_seconds,
        max_retries=1,
        # 一批 8 个切片、几十个实体时 2k token 会截断，截断的 JSON 直接解析失败
        max_tokens=4096,
        # 0 温：同一批切片两次跑出不同实体是「图谱不稳定」的主因
        temperature=settings.kb_graph_temperature,
        extra_body={"enable_thinking": False},
    )
    # repair=True：坏输出里能救的部分先救回来（裸词补引号、截断补括号）。
    # 这里和实体对齐不同——对齐结果是直接拿去合并实体的，猜错代价大；
    # 图谱抽取坏一批就是丢一批实体，救回来永远比丢掉强
    parsed = _parse_extraction(load_llm_json(raw, repair=True))
    if parsed is not None:
        return parsed

    # 第一次失败：同批原样重问一次。实测同一批「一次成功、一次失败」很常见
    # （本地 7b 的输出稳定性是随机的），重问只有 1 次调用的代价，而且不切分
    # 输入，跨片段的关系不会因为拆批而断掉——比直接拆半更划算
    if _depth == 0 and not _reasked:
        logger.warning(
            "kb 图谱抽取输出不可用（片段 %d 起 %d 片，返回 %d 字），原样重问一次",
            start_seq,
            len(chunks),
            len(raw or ""),
        )
        return _extract_batch(chunks, start_seq, system_prompt, _depth, True, doc_name)

    # 重问仍失败：多半是上下文吃紧，模型把输出挤成了别的形状。直接当
    # "这批没实体"丢掉就是静默丢人——拆半重试一次，半批的输入只有原来
    # 一半，同一模型通常就能正常输出（再失败就真放弃，别无限递归）
    if _depth < 1 and len(chunks) > 2:
        mid = len(chunks) // 2
        logger.warning(
            "kb 图谱抽取输出仍不可用（片段 %d 起 %d 片，返回 %d 字），拆半重试",
            start_seq,
            len(chunks),
            len(raw or ""),
        )
        # 后半批的段号取它自己首片的真实 seq：过滤过的批里 seq 不连续，
        # start_seq + mid 只是下标偏移，写进段号会和标号对不上
        mid_seq = chunks[mid].seq if chunks[mid].seq is not None else start_seq + mid
        return _merge_extractions(
            _extract_batch(chunks[:mid], start_seq, system_prompt, _depth + 1, False, doc_name),
            _extract_batch(chunks[mid:], mid_seq, system_prompt, _depth + 1, False, doc_name),
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
    return [
        (chunks[start].seq if chunks[start].seq is not None else start, chunks[start:end])
        for start, end in bounds
    ]


def _extract_all(
    chunks: list[Chunk],
    batch_size: int,
    system_prompt: str,
    sink: Callable[[int, int, list[Chunk], _Extraction], None] | None = None,
    doc_name: str | None = None,
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
            result: _Extraction | None = _extract_batch(
                batch, start, system_prompt, doc_name=doc_name
            )
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
