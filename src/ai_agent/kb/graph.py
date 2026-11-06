"""知识图谱构建入口：把抽取结果累加成图并落库。

从原 graph.py 拆出(S5)。切片分批与 LLM 调用在 graph_extract.py，实体名的
归一化/可用性判据在 graph_names.py，输出结构与容错解析在 graph_schema.py；
本模块只负责两件事：
- _GraphAccumulator：把各批次结果归并成实体表、关系表、提及表，
  跨批的同名实体合并、关系去重；
- build_file_graph：编排批次划分、并发调用、分批落库与进度上报。

抽取失败不抛给上层：图谱是增强能力，失败只是没有图，不能拖垮已经可检索
的文件。每 kb_graph_flush_batches 批落一次库，中断也留着部分图。
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Callable

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import (
    Chunk,
    _HTML_CELL,
    _HTML_TABLE,
    _MD_SEPARATOR,
    _cell_plain,
    _table_rows,
)
from ai_agent.kb.meta import _legal_names, subject_core
from ai_agent.kb.doc_profiles import (
    DEFAULT_LEVEL,
    build_system_prompt,
    normalize_level,
    resolve_profile,
)
from ai_agent.kb.graph_extract import (
    _batch_bounds,
    _extract_all,
    _worth_extracting,
)
from ai_agent.kb.graph_names import (
    _claimed_offset,
    _clean_type,
    _locate_chunk,
    resolve_graph_subject,
    _usable_name,
    display_name,
    normalize_name,
)
from ai_agent.kb.graph_names import _subject_from_filename  # noqa: F401  # test_kg 按旧路径访问
from ai_agent.kb.graph_schema import _Extraction, _Relation
from ai_agent.memory.embeddings import embed_texts_parallel
from ai_agent.memory.store import store

logger = get_logger(__name__)


# 泛称自称（「本公司」「本基金」「股东」…）不单独拦字典：模型把它写在
# 关系端点上时由 _resolve 改写成本文档主体全称，写在实体里时按 _usable_name
# 的通用判据处理。曾经这里维护过一份端点噪声名单，但它只服务于补桩
# （已移除，见 finalize），留着就是一份没人看的字典。

# 文档自指名。模型在关系里大量写「本公司持股…」「公司实现营业收入…」：
# 这些**既不能建实体**（一个泛称会把所有子图连成一片），**也不能直接丢**
# （丢的是整条关系）。单文件图谱里它们几乎总是指这份文件的主体，所以按
# 封面上的上市公司全称落地。封面没有文字时才回退到文件名。
_SELF_REF = {
    "本公司",
    "公司",
    "该公司",
    "本集团",
    "该集团",
    "发行人",
    "上市公司",
    "本行",
    "本基金",
    "本单位",
    "本企业",
    "上述公司",
    "本公司股份",
}


class _GraphAccumulator:
    """跨批增量合并 + 分次落库。

    为什么要增量：700 多个切片是几十批、几十分钟的活。全部抽完才写库的话，
    中途超时或被重启就一点都不剩——用户感受到的是"等了一小时，白等"。
    每 N 批覆盖写一次，进程被杀也留着部分图，前端轮询的实体数还能往上走。

    合并策略与一次性合并完全一致：同名（归一化后）实体合并，type 取首次
    出现的值，mention_count 累计，描述保留更长的那个。
    """

    def __init__(
        self,
        file_id: uuid.UUID,
        subject: str | None = None,
        *,
        trim_holders: bool = False,
    ) -> None:
        self.file_id = file_id
        # 文件主体全称（由文件名推断）：自指名端点落到它身上，见 _resolve
        self.subject_name = subject
        self.subject_norm = normalize_name(subject) if subject else None
        # 季报才收持股长尾。中间的分次落库不收，最后一次 flush 前再收，
        # 否则前几批的前 10 名会把后来才出现的大股东挤掉。
        self.trim_holders = trim_holders
        # 被改写的端点数与「没有主体可落、只能丢」的端点数，收尾时打一行日志
        self.self_ref_hits = 0
        self.entities: dict[str, dict] = {}
        # (source_norm, target_norm, relation) -> relation dict，天然去重
        self.relations: dict[tuple[str, str, str], dict] = {}
        self.mentions: dict[tuple[str, int], int] = {}
        # norm -> 向量：分次落库时只给新实体算向量，别整份重算
        self._vectors: dict[str, list[float]] = {}
        # 端点尚未出现的待定关系：(关系, 批起始, 批长度)
        self._pending: list[tuple[_Relation, int, int]] = []

    def add(self, start: int, batch: list[Chunk], result: _Extraction) -> None:
        # 只有整批「实体和关系都没有」才算空批。曾经这里只判 entities，
        # 于是「本批没抽到新实体、只有指向别批实体的关系」的批会被整批丢掉
        if not result.entities and not result.relations:
            return

        # 真实段号：建图前会过滤掉低信息切片，批内下标和库里的 seq 不是一回事，
        # 拿下标当 seq 写 mention，实体详情里的「原文片段」就会指到同批别的片上
        seqs = [c.seq if c.seq is not None else start + i for i, c in enumerate(batch)]
        count = len(batch)
        # 每批只归一化一次，供「这个名字出现在哪一片」的定位复用
        haystack = [normalize_name(c.text) for c in batch]

        for item in result.entities:
            norm = normalize_name(item.name)
            # 自指名不是实体：它指的就是主体，留着只会生成一个把整张图
            # 串成一片的万能节点
            if norm in _SELF_REF and self.subject_norm:
                continue
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
            # 提及记录该实体真正落在哪一片：先按名字在本批里找，找不到才信
            # 模型自报的位置（越界就夹回本批范围）
            offset = _locate_chunk(
                item.name, haystack, _claimed_offset(item.chunk, start, count)
            )
            key = (norm, seqs[offset])
            self.mentions[key] = self.mentions.get(key, 0) + 1

        for rel in result.relations:
            # 关系同样按端点名定位：用它报的位置，引用会指到无关的片上去
            rel.chunk = _locate_chunk(
                rel.source, haystack, _claimed_offset(rel.chunk, start, count)
            )
            if not self._add_relation(rel, seqs):
                # 端点还没出现：先挂起，等后面某一批把它抽出来再补上。
                # 直接丢会让「关系指向后文才出现的实体」整条消失——这正
                # 是「实体都在图上却互不相连」的另一个来源
                self._pending.append((rel, seqs))
        if self._pending:
            still_pending: list[tuple[_Relation, list[int]]] = []
            for item in self._pending:
                if not self._add_relation(*item):
                    still_pending.append(item)
            self._pending = still_pending

    def finalize(self) -> None:
        """收尾：把仍挂起的关系再整体试一次，端点仍不是实体的直接丢边。

        挂起队列在每批 add() 里重试，但**最后一批之后没有 add 了**——指向
        "整份文档都没抽成实体、只在关系里露过名字"的那些关系，到收尾时
        还躺在 _pending 里，flush() 不会碰它，随进程一起消失。表现是
        "关系数总比实体关系对不上"，且很难从日志里看出原因。所以收尾要
        再整体重试一次：最后几批可能刚把端点补出来。

        重试完仍然缺端点的，**丢弃整条关系，不给它补桩**。这里原来会给
        缺失端点补一个 type=other 的桩节点（描述写"仅作为关系端点出现"），
        实测三份季报补出 145 个这样的点，代价大于收益：
        - 桩没有正文依据，也没有 mention，前端点开是一片空白，检索命中后
          引用不出任何原文；
        - 名字来自模型在关系里随手写的一串字，模型笔误（截断、带标点、
          抄 prompt 示例）会被固化成图上节点，且永远不会被修正；
        - 它们撑起来的边是不可验证的——"某实体与某个查不到出处的名字之间
          存在某关系"，问一句"依据在哪"就答不上来。
        缺一条没有实体支撑的边，比留一个凭空捏出来的点更容易判断图的可信度。

        收尾是终态：连不上的就到此为止，别留在队列里假装"下次还能补"。
        """
        dropped: list[tuple[_Relation, list[int]]] = []
        for item in self._pending:
            if not self._add_relation(*item):
                dropped.append(item)
        # 收尾是终态：连不上的就到此为止，别留在队列里假装"下次还能补"
        self._pending = []
        if self.trim_holders:
            _trim_minor_stakes(self.entities, self.relations, self.mentions, self.subject_norm)
            for norm in list(self._vectors):
                if norm not in self.entities:
                    self._vectors.pop(norm, None)
        if dropped or self.self_ref_hits:
            names = sorted({n for rel, _ in dropped for n in (rel.source, rel.target)})
            logger.info(
                "kb 图谱收尾：%d 条关系因端点未抽成实体被丢弃，自指名端点改写 %d 个"
                "（主体=%s）；缺失端点样例：%s",
                len(dropped),
                self.self_ref_hits,
                self.subject_name or "未识别",
                "、".join(names[:8]) or "无",
            )

    def _resolve(self, raw: str) -> str:
        """把自指名端点换成主体全称；换不了就原样返回（后面照常挡掉）。"""
        norm = normalize_name(raw)
        if norm in _SELF_REF and self.subject_norm:
            self._ensure_subject()
            return self.subject_norm
        return norm

    def _ensure_subject(self) -> None:
        """主体全称是图上必需的一个点：自指名端点指过来时按需补出来。

        不放在 __init__ 里预置：整份文档一条关系都没有时，凭空多出来的
        孤立主体只会拉高孤立率。描述留给模型——后续批抽到同名实体时，
        更长的描述会覆盖掉这里的默认值。
        """
        if not self.subject_norm or self.subject_norm in self.entities:
            return
        self.entities[self.subject_norm] = {
            "name": self.subject_name,
            "norm": self.subject_norm,
            "type": "org",
            "description": "本文档主体",
        }

    def _add_relation(self, rel: _Relation, seqs: list[int]) -> bool:
        """写入一条关系。端点缺失返回 False（调用方决定挂起还是丢弃）。"""
        src = self._resolve(rel.source)
        dst = self._resolve(rel.target)
        # 记「有多少条关系是靠自指名兜底才留下来的」：这就是要不要保留
        # 这套兜底的判断依据（重试与去重都走 return True，不会重复计数）
        resolved = src != normalize_name(rel.source) or dst != normalize_name(rel.target)
        # 端点查全局实体表，不查「本批」：主体常在前一批被抽出，本批只是
        # 顺带提到它（"公司向亚厦控股采购…"），只在本批内找的话这类关系
        # 会被整条丢掉，最后表现为「实体都在图上却互不相连」的孤立点
        if src == dst:
            return True  # 自环，没有意义，也不必再重试
        if src not in self.entities or dst not in self.entities:
            return False
        rel_name = (rel.relation or "相关").strip()[:20] or "相关"
        key = (src, dst, rel_name)
        if key in self.relations:
            return True
        self.relations[key] = {
            "source": src,
            "target": dst,
            "relation": rel_name,
            "description": (rel.description or "").strip()[:200],
            "chunk_seq": seqs[min(max(rel.chunk, 0), len(seqs) - 1)],
        }
        if resolved:
            self.self_ref_hits += 1
        return True

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


# 季报股东表只留主要持有人。低于这个比例、又不在前 10 名的，写进说明。
_MIN_STAKE_PERCENT = 1.0
_TOP_HOLDERS = 10
# 小额持有人的全称往往很长（指数基金账户）。说明要比普通实体描述宽，
# 否则写到一半就被截断，点已经删了，说明里也找不到。
_HOLDER_NOTE_LIMIT = 800
_STAKE_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
_STAKE_SHARES = re.compile(r"(\d{1,3}(?:[,，]\d{3})+|\d+)\s*股")


def _parse_stake(description: str) -> tuple[float | None, int | None]:
    """从关系说明里取出持股比例和股数。比例只认 0–100，股数认「12,345股」。"""
    percent: float | None = None
    for match in _STAKE_PERCENT.finditer(description or ""):
        value = float(match.group(1))
        if 0 < value <= 100:
            percent = value
    shares: int | None = None
    found = _STAKE_SHARES.search(description or "")
    if found:
        shares = int(found.group(1).replace(",", "").replace("，", ""))
    return percent, shares


def _trim_minor_stakes(
    entities: dict[str, dict],
    relations: dict[tuple[str, str, str], dict],
    mentions: dict[tuple[str, int], int],
    subject_norm: str | None,
) -> None:
    """收掉小额持股。方向改成持有人指向被持股公司，再按同一家被持股公司分组。

    有比例的按比例，只有股数的用「股数 / 已知名比例」反推比例后参加同一排序。
    说明里既没有比例也没有股数的留下：看不出大小，不当作长尾删掉。
    被收掉的持有人记进被持股公司的说明；不再被任何关系用到的点一并去掉。
    """
    flipped: dict[tuple[str, str, str], dict] = {}
    for key, rel in relations.items():
        src, dst, name = key
        if name == "持股" and subject_norm and src == subject_norm and dst != subject_norm:
            src, dst = dst, src
        new_key = (src, dst, name)
        current = flipped.get(new_key)
        if current is None or len(rel.get("description") or "") > len(current.get("description") or ""):
            flipped[new_key] = {**rel, "source": src, "target": dst}
    relations.clear()
    relations.update(flipped)

    grouped: dict[str, list[tuple[tuple[str, str, str], dict]]] = {}
    for key, rel in relations.items():
        if key[2] != "持股":
            continue
        grouped.setdefault(key[1], []).append((key, rel))

    dropped_names: list[str] = []
    dropped_norms: set[str] = set()
    for target, items in grouped.items():
        sized: list[tuple[tuple[str, str, str], float | None]] = []
        parsed: list[tuple[float | None, int | None]] = []
        for key, rel in items:
            parsed.append(_parse_stake(rel.get("description") or ""))
        totals = [
            shares / (percent / 100.0)
            for percent, shares in parsed
            if percent and shares
        ]
        total = sorted(totals)[len(totals) // 2] if totals else None
        for (key, _rel), (percent, shares) in zip(items, parsed):
            size = percent
            if size is None and shares and total:
                implied = shares / total * 100.0
                if 0 < implied <= 100:
                    size = implied
            sized.append((key, size))
        ranked = sorted(
            (item for item in sized if item[1] is not None),
            key=lambda item: (-item[1], item[0][0]),
        )
        top = {key for key, _size in ranked[:_TOP_HOLDERS]}
        stake_by_key = {key: parsed[i] for i, (key, _rel) in enumerate(items)}
        # 没有任何一行同时写出比例和股数时，反推不出总股本。
        # 只有股数的按股数自己排前 10，不和比例混成一个单位。
        share_only = sorted(
            (
                (key, stake_by_key[key][1])
                for key, size in sized
                if size is None and stake_by_key[key][0] is None and stake_by_key[key][1]
            ),
            key=lambda item: (-item[1], item[0][0]),
        )
        share_top = {key for key, _shares in share_only[:_TOP_HOLDERS]}
        removed: list[tuple[str, float | None, int | None]] = []
        for key, size in sized:
            percent, shares = stake_by_key[key]
            if size is None and shares is not None and percent is None:
                keep = key in share_top
            elif size is None:
                keep = True
            else:
                keep = size >= _MIN_STAKE_PERCENT or key in top
            if keep:
                continue
            relations.pop(key, None)
            holder = entities.get(key[0], {})
            removed.append((holder.get("name") or key[0], percent, shares))
        if removed and target in entities:
            entities[target]["description"] = _append_holder_note(
                entities[target].get("description") or "", removed
            )
        dropped_names.extend(name for name, _percent, _shares in removed)
        dropped_norms.update(key[0] for key, _size in sized if key not in relations)

    if not dropped_names:
        return
    for norm in dropped_norms:
        if norm == subject_norm:
            continue
        if any(norm in key[:2] for key in relations):
            continue
        entities.pop(norm, None)
    for key in list(mentions):
        if key[0] not in entities:
            mentions.pop(key, None)
    logger.info(
        "kb 图谱收掉 %d 条小额持股（低于 1%% 且不在前 10 名）：%s",
        len(dropped_names),
        "、".join(dropped_names[:8]),
    )


def _append_holder_note(
    description: str, removed: list[tuple[str, float | None, int | None]]
) -> str:
    parts: list[str] = []
    for name, percent, shares in removed:
        if percent is not None:
            parts.append(f"{name} {percent:g}%")
        elif shares is not None:
            parts.append(f"{name} {shares:,}股")
        else:
            parts.append(name)
    note = "其余持股：" + "、".join(parts)
    base = (description or "").strip()
    if not base or base == "本文档主体":
        return note[:_HOLDER_NOTE_LIMIT]
    return f"{base}；{note}"[:_HOLDER_NOTE_LIMIT]


# 三大表及权益变动表。表题通常单独成行；续表的列名格里会写成「合并利润表项目」。
_STATEMENT_TITLE = re.compile(
    r"资产负债表|利润表|现金流量表|所有者权益变动表|股东权益变动表"
)
# 科目表的金额列。股东表的「期末持股数量」不含这些词，不会被当成科目行。
_STATEMENT_HEADER = re.compile(
    r"期末余额|期初余额|本期金额|上期金额|本年累计|上年同期|前三季度|本报告期|年初至报告期末"
)
_AMOUNT_CELL = re.compile(r"^[\d.,%+\-]+$")
# 附注列：「七、1」「（三十四）」。不是金额，但仍然是科目行，不是另一家公司。
_NOTE_REF = re.compile(
    r"^[（(]?(?:[一二三四五六七八九十百]+|\d+)[）)]?(?:[、.．]\d+)?[）)]?$"
)
_BLANK_AMOUNT = {"—", "－", "-", "--", "——", "/", "不适用"}


def _chunk_table_rows(text: str) -> list[list[str]]:
    """取出表格的单元格。HTML 与 Markdown 两种切片都认，其它形态返回空。"""
    raw = text or ""
    if _HTML_TABLE.search(raw):
        rows: list[list[str]] = []
        for table in _HTML_TABLE.finditer(raw):
            for row in _table_rows(table.group(0)):
                cells = [_cell_plain(match.group(2)) for match in _HTML_CELL.finditer(row)]
                if any(cells):
                    rows.append(cells)
        return rows
    rows = []
    for line in raw.splitlines():
        stripped = line.strip()
        if "|" not in stripped or _MD_SEPARATOR.match(stripped):
            continue
        body = stripped[1:] if stripped.startswith("|") else stripped
        rows.append([part.strip() for part in re.split(r"(?<!\\)\|", body)])
    return rows


def _is_amount_cell(cell: str) -> bool:
    text = (cell or "").strip().translate(str.maketrans({"，": ",", "（": "(", "）": ")"}))
    if not text or text in _BLANK_AMOUNT:
        return True
    bare = text.replace(" ", "").strip("()")
    return bool(_AMOUNT_CELL.fullmatch(bare) or _NOTE_REF.fullmatch(text.replace(" ", "")))


def _is_header_row(cells: list[str]) -> bool:
    filled = [cell for cell in cells if cell]
    if len(filled) < 2:
        return False
    # 「货币资金 | 不适用」是科目行：除第一列外都是金额或空。
    if all(_is_amount_cell(cell) for cell in cells[1:]):
        return False
    if any(_is_amount_cell(cell) and re.search(r"\d", cell) for cell in filled):
        return False
    return any(re.search(r"[\u4e00-\u9fff]", cell) for cell in filled)


def _is_account_row(cells: list[str]) -> bool:
    if not cells or not (cells[0] or "").strip():
        return False
    return all(_is_amount_cell(cell) for cell in cells[1:])


def _looks_like_statement(text: str, rows: list[list[str]]) -> bool:
    for raw in text.splitlines():
        line = raw.strip().strip("*")
        if line and len(line) <= 40 and line[-1] not in "。；，、" and _STATEMENT_TITLE.search(line):
            return True
    for cells in rows[:4]:
        blob = "".join(cells)
        if _STATEMENT_TITLE.search(blob):
            return True
        labeled = any(cell in {"项目", "科目"} or cell.endswith("项目") for cell in cells if cell)
        if labeled and _STATEMENT_HEADER.search(blob):
            return True
    return False


def _has_other_company(text: str, subject: str | None) -> bool:
    """行里有另一家公司全称。文件主体自己出现在表题上不算。"""
    for name in _legal_names(text):
        if subject and _same_company(name, subject):
            continue
        return True
    return False


def _same_company(name: str, subject: str) -> bool:
    left, right = normalize_name(name), normalize_name(subject)
    if left and left == right:
        return True
    core_left, core_right = normalize_name(subject_core(name)), normalize_name(subject_core(subject))
    return len(core_left) >= 2 and core_left == core_right


def _is_statement_line(chunk: Chunk, subject: str | None) -> bool:
    """季报三大表的科目行：第一列是科目，其余是金额或附注，行内没有另一家公司。

    股东表、子公司表的第一列是机构全称，留下。分行业营收的列名是营业收入，
    不是「本期金额」，也留下。只在 quarterly_report 上调用。
    """
    if chunk.kind != "table":
        return False
    text = chunk.text or ""
    rows = _chunk_table_rows(text)
    if not rows or not _looks_like_statement(text, rows):
        return False
    data = [row for row in rows if not _is_header_row(row)]
    if not data or not all(_is_account_row(row) for row in data):
        return False
    return not _has_other_company(text, subject)


def _cover_page_text(chunks: list[Chunk]) -> str:
    """封面是页码最小的那一页。没有页码就当封面没有文字，调用方回退文件名。"""
    pages = [chunk.page for chunk in chunks if chunk.page]
    if not pages:
        return ""
    first = min(pages)
    return "\n".join(
        (chunk.text or "") for chunk in chunks if chunk.page == first
    ).strip()


def build_file_graph(
    file_id: uuid.UUID,
    chunks: list[Chunk],
    *,
    doc_type: str | None = None,
    level: str = DEFAULT_LEVEL,
    progress: Callable[[dict], None] | None = None,
    doc_name: str | None = None,
) -> dict:
    """抽取并写入某个文件的图谱，返回计数。

    doc_type 决定注入哪套文档配置（招股书/年报/法律意见书/通用…），
    未知或为空时走通用配置；level 是同一配置下的轻重粒度（完整/轻量）。
    两者都只影响 prompt，抽取与合并逻辑一份代码通吃。

    doc_name 是文件名。封面第一页有上市公司全称时，自指名和每批 prompt
    都用这个全称；封面没有文字、或认不出公司全称时，才回退到文件名。
    财报表格片段通篇没有公司全称，prompt 里缺了主体，模型会给持股猜一个主语。

    progress 每完成一批回调一次（带累计实体数和 已完成批/总批），
    上层拿它更新 graph_status，前端就能看到百分比。
    """
    profile = resolve_profile(doc_type)
    system_prompt = build_system_prompt(profile.key, level)
    cover = _cover_page_text(chunks)
    subject = resolve_graph_subject(doc_name, cover)
    kept = [c for c in chunks if _worth_extracting(c.text)]
    if len(kept) < len(chunks):
        logger.info(
            "kb 图谱跳过 %d/%d 个低信息切片（短于 %d 字或纯数字）",
            len(chunks) - len(kept),
            len(chunks),
            settings.kb_graph_min_chunk_chars,
        )
    if profile.key == "quarterly_report":
        before = len(kept)
        kept = [c for c in kept if not _is_statement_line(c, subject)]
        if len(kept) < before:
            logger.info(
                "kb 图谱跳过 %d 个季报科目行（资产负债表/利润表/现金流量表）",
                before - len(kept),
            )
    if not kept:
        logger.info("kb 图谱无可用切片 file_id=%s", file_id)
        return {"entities": 0, "relations": 0}

    batch_size = max(1, settings.kb_graph_batch_chunks)
    flush_every = max(1, settings.kb_graph_flush_batches)
    # 批数要按「章节组批」后的实际批数算：进度百分比和分次落库的节奏都
    # 依赖它，用 ceil(片数/批大小) 会在小章节合批后偏高
    total_batches = len(_batch_bounds(kept, batch_size))
    # 封面全称写进 prompt。文件名里的简称会让模型再抽出一个同主体的点。
    # 封面没抽出全称时，prompt 仍用原来的文件名。
    listed = resolve_graph_subject(None, cover)
    prompt_name = listed or doc_name
    logger.info("kb 图谱主体 file_id=%s subject=%s", file_id, subject or "未识别")
    accumulator = _GraphAccumulator(
        file_id, subject, trim_holders=profile.key == "quarterly_report"
    )
    started = time.perf_counter()

    def report(done: int) -> None:
        if progress is None:
            return
        progress(
            {
                "entities": len(accumulator.entities),
                "relations": len(accumulator.relations),
                "batches": done,
                "total_batches": total_batches,
            }
        )

    report(0)

    def sink(index: int, start: int, batch: list[Chunk], result: _Extraction) -> None:
        accumulator.add(start, batch, result)
        report(index + 1)
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

    _extract_all(kept, batch_size, system_prompt, sink=sink, doc_name=prompt_name)
    # 收尾必须先于 flush：挂起的关系只有在最后一批之后再试一次才可能落地，
    # 否则 flush 写出去的图里会缺掉「端点出现在更早/更晚批」的整条关系
    accumulator.finalize()
    counts = accumulator.flush()
    logger.info(
        "kb 图谱抽取完成 file_id=%s doc_type=%s level=%s batches=%d chunks=%d(过滤前 %d) "
        "实体=%d 关系=%d %.0fs",
        file_id,
        profile.key,
        normalize_level(level),
        total_batches,
        len(kept),
        len(chunks),
        counts["entities"],
        counts["relations"],
        time.perf_counter() - started,
    )
    if progress is not None:
        progress(
            {
                "entities": counts["entities"],
                "relations": counts["relations"],
                "batches": total_batches,
                "total_batches": total_batches,
            }
        )
    return counts
