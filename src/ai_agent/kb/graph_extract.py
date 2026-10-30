"""知识图谱抽取的批组织与调用：切片 → 分批 → LLM → 结构化结果。

从 graph.py 拆出(S5)，命名规范化与可用性判据在 graph_names.py，输出结构与
容错解析在 graph_schema.py，本模块只管「怎么把切片喂给模型」：

1. 进批前过滤不值得调用的碎片（_worth_extracting）；
2. 按章节组批（_batch_bounds）：一节不被批边界切开，关系才不容易抽漏；
3. 单批容错（_extract_batch）：原样重问一次 → 拆半重试，坏一批不静默丢人；
4. 并发编排（_extract_all）：批间独立，回调严格按批序，可边抽边合并。

这是全链路最慢的一环（700 多个切片 ≈ 94 次调用），成本控制的关键点都在
_extract_all 与 _extract_batch 的注释里。
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import Chunk
from ai_agent.kb.graph_names import _doc_title
from ai_agent.kb.graph_schema import _Extraction, _parse_extraction
from ai_agent.llm.client import complete_chat, graph_client, graph_model
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.parse import load_llm_json

logger = get_logger(__name__)


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
