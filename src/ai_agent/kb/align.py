"""实体对齐：把同一实体的不同写法合成一个点。

为什么必须有这一步（而不是指望抽取时写对）：

- 抽取是逐批做的，模型每批独立起名——同一家公司在 A 批写「亚厦股份」、
  在 B 批写「浙江亚厦装饰股份有限公司」，两批之间没有任何共享上下文；
- 招股书/年报本来就大量使用"全称 + 简称 + 曾用名"混写；
- 归一化（graph.normalize_name）只能消除括号/空格/大小写这类**字符层面**
  的差异，消不掉"简称 vs 全称"这种**语义层面**的差异。

后果不只是"图不好看"：被拆成三份的实体各挂三分之一的关系，度数都低，
前端按度数取点时全被当成孤立点裁掉，检索时也各召回一部分子图——用户看到
的是"这张图怎么什么都没连上"。

三步走：
1. **候选召回**：向量近邻（实体名+描述已向量化）+ 名称包含关系（简称/全称
   的向量未必近，但字面上是子串）。两条路互补，缺一条就会漏掉一类；
2. **裁决**：收紧的自动规则先吃掉明显对子（字符串包含、极高相似度且同类型），
   剩下的灰区交给模型逐对判"是不是同一个东西"——宁可漏合并，也不能把母公司
   和子公司并成一个；
3. **合并**：并查集成簇，取提及最多的那个当主体，把提及、关系、描述并过去，
   被合并的写法写进 agent_kg_alias 留档（用户能问"这两个怎么变一个了"）。

对齐失败不影响已有图谱：图谱本身仍可查，只是多几个重复点。
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.kb.graph import normalize_name
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.parse import load_llm_json
from ai_agent.memory.store import store

logger = get_logger(__name__)

# 类型不同但可以合并的情形：other/concept 常常是抽取时没判准的类型，
# 不该因为它们类型不一致就拆成两个点
_SOFT_TYPES = {"other", "concept"}
# 短名参与"包含即合并"的最小长度：「国电」这种 2 字词包含关系太容易误中
_MIN_CONTAIN_LEN = 4

_SYSTEM = """你在做实体消歧：判断每一对名称是否指**同一个现实对象**。

判定要点：
1. 同一个对象的常见不同写法要判 1：全称与简称（"国电电力" / "国电电力发展股份有限公司"）、
   中英文名、带或不带地名/行业后缀、曾用名、明显的错别字；
2. 只要有一方可能是**不同的对象**就判 0：母公司与子公司、同一集团下两个厂、
   型号相近但不同的产品（"iPhone 15" / "iPhone 15 Pro"）、数值型指标的不同口径；
   特别注意：**一个名字是另一个的前缀，不等于同一实体**——「国电电力」是发行人的
   简称，但「国电电力大同发电有限责任公司」是另一家法人，三者不能并成一个；
3. 拿不准时判 0——错并比漏并更糟，并把两个不相关的部分连成一片。

只输出一行 JSON：{"same":[1,0,1,...]}，个数与输入对数一致，1=同一个，0=不同。
不要解释、不要 Markdown 代码块。"""


class _Verdict(BaseModel):
    same: list[int] = Field(default_factory=list)


def type_compatible(a: str, b: str) -> bool:
    """类型是否允许合并：同类型，或一方是软类型（other/concept）。"""
    return a == b or a in _SOFT_TYPES or b in _SOFT_TYPES


def name_contained(a: str, b: str) -> bool:
    """一个名字是另一个的子串（简称/全称），且短名够长不至于误中。"""
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= _MIN_CONTAIN_LEN and short in long


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def candidate_pairs(
    entities: list[dict],
    sim_pairs: list[dict],
    *,
    min_sim: float | None = None,
) -> list[dict]:
    """合成候选对：向量近邻 + 名称包含，去重后按相似度降序。

    entities: [{id, name, norm_name, type, mention_count}]
    sim_pairs: [{src, dst, sim}]（由 pgvector 在库里算，Python 不重复算 1024 维）

    两条召回路径缺一不可：向量近邻抓"换了说法的同一实体"，名称包含抓
    "简称 vs 全称"——后者的描述常常不同（全称那份带注册地址、简称那份带业务），
    向量相似度可能只有 0.8 出头，纯靠向量会漏。
    """
    threshold = settings.kg_align_min_sim if min_sim is None else min_sim
    by_id = {str(e["id"]): e for e in entities}
    out: dict[tuple[str, str], dict] = {}

    for row in sim_pairs:
        src, dst = str(row["src"]), str(row["dst"])
        a, b = by_id.get(src), by_id.get(dst)
        if a is None or b is None or src == dst:
            continue
        sim = float(row.get("sim") or 0)
        if sim < threshold:
            continue
        if not type_compatible(a["type"], b["type"]):
            continue
        key = _pair_key(src, dst)
        if key in out:
            continue
        out[key] = {"a": a, "b": b, "score": sim, "why": "vector"}

    # 名称包含：不看相似度也要进候选，但要求类型兼容
    for i, a in enumerate(entities):
        norm_a = a.get("norm_name") or normalize_name(a["name"])
        if len(norm_a) < _MIN_CONTAIN_LEN:
            continue
        for b in entities[i + 1 :]:
            norm_b = b.get("norm_name") or normalize_name(b["name"])
            if not type_compatible(a["type"], b["type"]):
                continue
            if not name_contained(norm_a, norm_b):
                continue
            key = _pair_key(str(a["id"]), str(b["id"]))
            existing = out.get(key)
            if existing is not None:
                existing["why"] = "contain"
                # 包含关系是字面证据，相似度取不低于门槛的值，确保排序靠前
                existing["score"] = max(existing["score"], threshold)
                continue
            out[key] = {
                "a": a,
                "b": b,
                "score": threshold,
                "why": "contain",
            }

    limit = max(1, settings.kg_align_max_pairs)
    ranked = sorted(out.values(), key=lambda p: p["score"], reverse=True)
    return ranked[:limit]


def auto_accept(pair: dict) -> bool:
    """不用问模型就能定的对子：**只有归一化后完全同名**的。

    两道人工验证过的红线，都别指望用相似度兜：
    - 字面包含：「国电电力」既是发行人简称，又是「国电电力大同发电…」的前缀，
      自动合并会顺着前缀把一片独立法人串成一个点；
    - 高向量相似：「中国华电集团公司」和「中国华能集团公司」只差一个字，
      实测相似度过 0.95，但这是两家公司——这类错并没人会去翻账单发现。
    所以只有"名字一模一样、只是类型判得不一样"才自动合，其余一律交模型裁决；
    模型不可用时宁可不合并（图上多几个重复点，也比把两家公司并成一家强）。
    """
    a, b = pair["a"], pair["b"]
    if not type_compatible(a["type"], b["type"]):
        return False
    return (a.get("norm_name") or normalize_name(a["name"])) == (
        b.get("norm_name") or normalize_name(b["name"])
    )


def _format_pairs(pairs: list[dict]) -> str:
    lines = []
    for i, pair in enumerate(pairs):
        a, b = pair["a"], pair["b"]
        desc_a = (a.get("description") or "无说明")[:40]
        desc_b = (b.get("description") or "无说明")[:40]
        lines.append(
            f"{i + 1}. [{a['type']}] {a['name']}（{desc_a}）"
            f" ↔ [{b['type']}] {b['name']}（{desc_b}）"
        )
    return "\n".join(lines)


# 连续失败到这个次数就认为通道整体不可用，不再逐批空等
_MAX_CONSECUTIVE_FAILURES = 3


class AlignUnavailable(RuntimeError):
    """裁决模型不可用（连续超时/报错），整轮对齐中止。

    不中止的代价实测过：单批超时 180 秒 × 两次重试，30 批要空转一个半小时，
    期间 Ollama 一直被占着，连正常问答都排队。宁可快速失败、把状态标成
    failed 让用户重试。
    """


def judge_pairs(pairs: list[dict]) -> list[bool]:
    """把灰区对子交给模型裁决；调用失败的对子保守判 0（不合并）。

    模型连续失败（超时/返回不可解析）时抛 AlignUnavailable 快速退出——
    失败一次通常是模型偶发卡顿，连续三次就是服务有问题，继续跑只是白等。
    """
    if not pairs:
        return []
    batch = max(1, settings.kg_align_batch)
    verdicts: list[bool] = []
    failures = 0
    for start in range(0, len(pairs), batch):
        part = pairs[start : start + batch]
        try:
            raw = complete_chat(
                [
                    ChatMessage(role="system", content=_SYSTEM),
                    ChatMessage(
                        role="user",
                        content=f"共 {len(part)} 对：\n\n{_format_pairs(part)}\n\n"
                        "请输出 JSON。",
                    ),
                ],
                model=settings.kb_graph_model or settings.model,
                # 裁决的提示词只有几百字、输出不到 100 字，正常几秒就回；
                # 等 180 秒只会在服务卡死时把整轮对齐拖垮
                timeout=settings.kg_align_timeout_seconds,
                max_retries=0,
                max_tokens=256,
                extra_body={"enable_thinking": False},
            )
            data = load_llm_json(raw)
            same = _Verdict.model_validate(data).same if isinstance(data, dict) else []
            failures = 0
        except (LLMError, ValueError, TypeError) as e:
            failures += 1
            logger.warning(
                "实体对齐裁决失败（第 %d 对起 %d 对，连续 %d 次）：%s",
                start,
                len(part),
                failures,
                e,
            )
            same = []
        except Exception:
            failures += 1
            logger.exception("实体对齐裁决异常（第 %d 对起）", start)
            same = []
        if failures >= _MAX_CONSECUTIVE_FAILURES:
            raise AlignUnavailable(f"裁决模型连续 {failures} 次不可用，已中止对齐")
        # 模型少给/多给都按位置对齐，缺的判 0（保守）
        for i in range(len(part)):
            verdicts.append(bool(same[i]) if i < len(same) else False)
    return verdicts


def plan_merges(entities: list[dict], accepted: list[dict]) -> list[dict]:
    """并查集成簇，返回 [{from_id, to_id, alias_name, alias_norm, score}]。

    每个簇选一个代表（提及次数最多，其次名字更长——长的通常是全称更像正式名），
    其余成员指向代表。纯函数，便于单测。
    """
    by_id = {str(e["id"]): e for e in entities}
    parent: dict[str, str] = {sid: sid for sid in by_id}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in accepted:
        a, b = str(pair["a"]["id"]), str(pair["b"]["id"])
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    clusters: dict[str, list[str]] = {}
    for sid in by_id:
        clusters.setdefault(find(sid), []).append(sid)

    merges: list[dict] = []
    limit = max(2, settings.kg_align_max_cluster)
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        if len(members) > limit:
            # 一个簇大到这个程度，多半是"某条包含链把一片法人串起来了"，
            # 而不是真有那么多写法。宁可整簇不合并，也别把图压成一团
            logger.warning(
                "实体对齐跳过异常大簇（%d 个成员，上限 %d）：%s",
                len(members),
                limit,
                "、".join(by_id[m]["name"] for m in members[:5]),
            )
            continue
        canonical = max(
            members,
            key=lambda m: (
                int(by_id[m].get("mention_count") or 0),
                len(by_id[m].get("norm_name") or by_id[m]["name"]),
            ),
        )
        for member in members:
            if member == canonical:
                continue
            source = by_id[member]
            merges.append(
                {
                    "from_id": member,
                    "to_id": canonical,
                    "alias_name": source["name"],
                    "alias_norm": source.get("norm_name")
                    or normalize_name(source["name"]),
                    "canonical_name": by_id[canonical]["name"],
                    "score": 0.0,
                }
            )
    return merges


def _score_of(a_id: str, b_id: str, accepted: list[dict]) -> float:
    for pair in accepted:
        ids = {str(pair["a"]["id"]), str(pair["b"]["id"])}
        if ids == {a_id, b_id}:
            return float(pair["score"])
    return 0.0


def align_file_graph(
    file_id: uuid.UUID,
    *,
    progress=None,
    dry_run: bool = False,
) -> dict:
    """跑一遍实体对齐并落库，返回统计。

    dry_run=True 只算不写（调参、看候选质量时用）。
    """
    entities = store.list_file_entities(file_id)
    if len(entities) < 2:
        return {"entities": len(entities), "candidates": 0, "merged": 0, "aliases": []}

    sim_pairs = store.entity_similar_pairs(
        file_id,
        min_sim=settings.kg_align_min_sim,
        neighbors=settings.kg_align_neighbors,
    )
    pairs = candidate_pairs(entities, sim_pairs)

    auto = [p for p in pairs if auto_accept(p)]
    judged = [p for p in pairs if not auto_accept(p)]
    verdicts = judge_pairs(judged)
    accepted = auto + [p for p, ok in zip(judged, verdicts) if ok]

    merges = plan_merges(entities, accepted)
    for merge in merges:
        merge["score"] = _score_of(merge["from_id"], merge["to_id"], accepted)

    logger.info(
        "实体对齐 file_id=%s 实体=%d 候选=%d（自动 %d / 裁决 %d）合并=%d",
        file_id,
        len(entities),
        len(pairs),
        len(auto),
        len(judged),
        len(merges),
    )
    if merges and not dry_run:
        store.apply_alignment(file_id, merges)
    if progress is not None:
        progress({"merged": len(merges)})
    return {
        "entities": len(entities),
        "candidates": len(pairs),
        "auto": len(auto),
        "judged": len(judged),
        "merged": len(merges),
        "aliases": merges,
    }
