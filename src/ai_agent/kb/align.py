"""实体对齐：把同一实体的不同写法合成一个点。

为什么必须有这一步（而不是指望抽取时写对）：

- 抽取是逐批做的，模型每批独立起名——同一家公司在 A 批写「亚厦股份」、
  在 B 批写「浙江亚厦装饰股份有限公司」，两批之间没有任何共享上下文；
- 招股书/年报本来就大量使用"全称 + 简称 + 曾用名"混写；
- 归一化（graph_names.normalize_name）只能消除括号/空格/大小写这类**字符层面**
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

import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.kb.graph_names import is_metric_like, normalize_name
from ai_agent.llm.client import (
    complete_chat,
    graph_client,
    graph_model,
    json_response_format,
)
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.parse import load_llm_json
from ai_agent.memory.store import store
from ai_agent.prompts.kb import ALIGN_SYSTEM, build_align_user

logger = get_logger(__name__)

# 类型不同但可以合并的情形：other/concept 常常是抽取时没判准的类型，
# 不该因为它们类型不一致就拆成两个点
_SOFT_TYPES = {"other", "concept"}
# 短名参与"包含即合并"的最小长度：「国电」这种 2 字词包含关系太容易误中
_MIN_CONTAIN_LEN = 4

# 名字本身已经"锁死"语义的类型：差一个字就是另一个东西，不存在"同一实体的
# 不同写法"。财务科目是最典型的一类——「经营活动现金流入小计」和「流出小计」
# 的向量相似度高得离谱，「应付职工薪酬」和「长期应付职工薪酬」同理，交给模型
# 逐对裁决必然出错。实测库里 39 条合并有 32 条是 metric，其中过半是把方向
# 相反或口径不同的科目并成了一个，整条科目的关系全被串到对面去。
_RIGID_TYPES = {"metric"}

# 文档内泛指词：「本公司」和「该公司」在同一份文件里可能指向不同主体，
# 并成一个会让所有关系挂到错的公司在；这类词要归一只能靠指代消解，不是对齐。
_PRONOUN_NAMES = {
    "本公司", "该公司", "公司", "本集团", "发行人",
    "上市公司", "本行", "本基金", "本单位", "本企业",
}


def is_rigid(entity: dict) -> bool:
    """这个名字是否已精确到「只能和完全同名合并」。

    三条都指向同一件事：字面上差一点，语义上就是另一个对象。
    - metric 类型（财务科目、指标）；
    - 文档内泛指词（本公司/该公司）；
    - 名字基本由数字/日期构成（「2005年」「62,200万千瓦」）。
    """
    if entity.get("type") in _RIGID_TYPES:
        return True
    name = (entity.get("name") or "").strip()
    if name in _PRONOUN_NAMES:
        return True
    return is_metric_like(name)

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


# 数值/日期型名字（「2005年」「62,200万千瓦」）入库时就会丢掉；
# 旧图里若还留着，互相对齐时仍然不能合并——2005年 和 2005年8月 不是同一个点。
# 判定见 graph_names.is_metric_like。

# 法人的通用尾巴：简称补上这些词就是全称，中间没有加任何限定词
_GENERIC_TAIL = (
    "发展", "股份", "有限", "责任", "公司", "集团", "控股", "投资",
    "（", "）", "(", ")",
)


# 母公司名 + 这些尾巴 = 下属单位，绝不是同一个实体
_SUBUNIT_TAIL = ("厂", "部", "分公司", "支行", "车间", "项目部", "分厂", "办事处")


def looks_like_subunit(short: str, long_: str) -> bool:
    """长名是短名的前缀扩展，且尾巴落在「厂/部/分公司」上——是下属单位。

    招股书里「国电电力发展股份有限公司」和「国电电力发展股份有限公司
    朝阳发电厂」字面上是前缀关系，模型很容易判成同一实体，但一个是上市
    主体、一个是分公司，合并后母公司的所有关系都会继承电厂的噪声。
    """
    if not long_.startswith(short) or short == long_:
        return False
    rest = long_[len(short) :]
    return rest.endswith(_SUBUNIT_TAIL)


# 限定词放在短名前面 = 另一个东西：「应付职工薪酬」和「长期应付职工薪酬」
# 是两张报表行（流动/非流动），「期末现金」和「期初现金」是两个时点。
# 这类名字在字面上是「包含」关系，包含规则会一路放行到模型面前。
_QUALIFIER_PREFIX = (
    "长期", "短期", "期初", "期末", "年初", "年末", "本期", "上期",
    "非流动", "流动", "其他", "其中", "净", "总",
)


def has_qualifier_prefix(short: str, long_: str) -> bool:
    """长名 = 限定词 + 短名：限定词改变了语义，不是同一实体的两种写法。

    只处理「限定词在前、短名在后」这一侧；短名在前缀、只多出通用法人后缀的
    情形由 `pure_suffix_extension` 判定为同一实体，两者互补。
    """
    if short == long_ or not long_.endswith(short):
        return False
    head = long_[: len(long_) - len(short)]
    return any(head.startswith(q) for q in _QUALIFIER_PREFIX)


_DIGIT_RUN = re.compile(r"\d+(?:\.\d+)?")


def number_conflict(a: str, b: str) -> bool:
    """两个名字里的数字对不上 → 不是同一个东西。

    「ActiveShield™4.0 温控系统」和「5.0」只差一个数字，向量相似度 0.93、
    模型也判了同一个，但这是两代产品。同理「第1期回购」vs「第2期回购」
    是两个方案。

    判据用「数字集合互不包含」而不是「不等」：「A股回购股份」与
    「2025年第1期A股回购股份方案」一边没有数字，简称补上年份/期数是
    正常写法，放行。
    """
    na = set(_DIGIT_RUN.findall(a or ""))
    nb = set(_DIGIT_RUN.findall(b or ""))
    if not na or not nb:
        return False
    return not (na <= nb or nb <= na)


def pure_suffix_extension(short: str, long_: str) -> bool:
    """短名是全称的前缀，且剩下那截只是通用法人后缀（没有加地名/厂名）。

    「国电电力」+「发展股份有限公司」= 同一个法人，这是改名/简称的常见形态，
    可以放心判同一；而「国电电力」+「大同发电有限责任公司」多了一个地名，
    是另一家公司——同样满足"前缀"，但绝不该合。这条规则把两者分开。
    """
    if short == long_ or not long_.startswith(short):
        return False
    rest = long_[len(short) :]
    if not rest:
        return False
    i = 0
    while i < len(rest):
        for token in _GENERIC_TAIL:
            if rest.startswith(token, i):
                i += len(token)
                break
        else:
            return False
    return True


# 产品/技术类名字：同一个品牌前缀挂着一串型号，字面上高度相似，但都是不同的东西。
# 这类不能交给模型裁决——本地 7b 看到「Anker」「Anker Travel Charging Station」
# 「Anker Universal Charger」会把它们连成一个簇，品牌点吞掉全部产品。
_SKU_TYPES = {"product", "technology"}


def looks_like_different_sku(a: str, b: str) -> bool:
    """两个产品名是不是「同品牌不同型号」。

    三种形态都算（都已归一化成小写、无标点）：
    1. 一个名字是另一个的前缀，尾巴不是通用后缀（「Anker」vs「Anker Travel
       Charging Station」）——品牌挂型号；
    2. 共享一段够长的开头，但谁也不包含谁（「Anker Universal Charger」vs
       「Anker Travel Charging Station」）——同品牌的两个型号；
    3. 一个名字是另一个加了尾巴（「X 充电器」vs「X 充电器 Pro」）。
    只有「拓展坞」和「Hub」这种一个字都不共用的同义译名才放行——那种情况
    本来也过不了前缀判据。
    """
    if not a or not b or a == b:
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if name_contained(a, b):
        # 包含关系里，前缀扩展只在「尾巴是通用法人后缀」时才算同一实体；
        # 产品名没有法人后缀这一说，所以一律按不同型号处理
        return not pure_suffix_extension(short, long_)
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    return common >= _MIN_CONTAIN_LEN


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
        # 语义已锁死的名字（metric / 泛指词 / 纯数值）不进候选：
        # 它们之间「相似」几乎总是意味着「不同」，问模型只会被带偏
        if is_rigid(a) or is_rigid(b):
            continue
        # 数字对不上（4.0 vs 5.0、第1期 vs 第2期）是不同对象，问模型也会被判成
        if number_conflict(a["name"], b["name"]):
            continue
        norm_a = a.get("norm_name") or normalize_name(a["name"])
        norm_b = b.get("norm_name") or normalize_name(b["name"])
        # 同品牌不同型号：字面高度相似、向量也近，但都是独立产品。
        # 交给模型裁决时 7b 会把品牌点和整串型号连成一个簇
        if a["type"] in _SKU_TYPES and b["type"] in _SKU_TYPES:
            if looks_like_different_sku(norm_a, norm_b):
                continue
        # 概念/指标类只认字面包含：模型在"其他应付款 vs 应付账款"、
        # "东北地区 vs 华北地区"这种同领域近邻上判错过太多次，光靠向量
        # 相似度过不了这道关
        if a["type"] in _SOFT_TYPES and b["type"] in _SOFT_TYPES:
            if not name_contained(norm_a, norm_b):
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
            if is_rigid(a) or is_rigid(b):
                continue
            if number_conflict(a["name"], b["name"]):
                continue
            if not name_contained(norm_a, norm_b):
                continue
            short, long_ = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
            # 前缀 + 厂/部/分公司 = 下属单位，连候选都不进
            if looks_like_subunit(short, long_):
                continue
            # 限定词 + 短名 = 另一个科目/另一个口径（长期↔短期、期初↔期末）
            if has_qualifier_prefix(short, long_):
                continue
            # 同品牌不同型号（Anker / Anker Travel Charging Station）
            if a["type"] in _SKU_TYPES and b["type"] in _SKU_TYPES:
                if looks_like_different_sku(norm_a, norm_b):
                    continue
            key = _pair_key(str(a["id"]), str(b["id"]))
            # 纯通用后缀扩展（简称 + 发展股份有限公司）是最可靠的一类，给最高分：
            # 排序靠前，簇容量截断时先保它
            strong = pure_suffix_extension(short, long_)
            score = 0.99 if strong else threshold
            existing = out.get(key)
            if existing is not None:
                existing["why"] = "suffix" if strong else "contain"
                # 包含关系是字面证据，相似度取不低于门槛的值，确保排序靠前
                existing["score"] = max(existing["score"], score)
                continue
            out[key] = {
                "a": a,
                "b": b,
                "score": score,
                "why": "suffix" if strong else "contain",
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
    # 简称 + 纯通用后缀（"国电电力" → "国电电力发展股份有限公司"）是改名/
    # 全称的固定形态，中间没加任何地名或限定词，字面上就能定
    if pair.get("why") == "suffix":
        return True
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

    def _ask(messages: list[ChatMessage], fmt: dict[str, Any] | None) -> str:
        return complete_chat(
            messages,
            model=graph_model(),
            client=graph_client(),
            # 裁决的提示词只有几百字、输出不到 100 字，正常几秒就回；
            # 等 180 秒只会在服务卡死时把整轮对齐拖垮
            timeout=settings.kg_align_timeout_seconds,
            max_retries=0,
            max_tokens=256,
            # 裁决共用抽取温度（0）：并错一对就丢一个实体，且合并不可逆
            temperature=settings.kb_graph_temperature,
            extra_body={"enable_thinking": False},
            response_format=fmt,
        )

    fmt = json_response_format(settings.kb_align_json_mode)
    for start in range(0, len(pairs), batch):
        part = pairs[start : start + batch]
        messages = [
            ChatMessage(role="system", content=ALIGN_SYSTEM),
            ChatMessage(
                role="user",
                content=build_align_user(len(part), _format_pairs(part)),
            ),
        ]
        try:
            try:
                raw = _ask(messages, fmt)
            except LLMError as e:
                if fmt is None:
                    raise
                # 网关不支持 JSON 模式会直接 400。这是配置问题不是模型抖动，
                # 不该计入「连续失败」把整轮对齐中止掉——本轮后续批次改用
                # prompt 约束即可，输出仍由 load_llm_json 兜底解析。
                logger.warning("实体对齐 JSON 模式不受支持，回退 prompt 约束：%s", e)
                fmt = None
                raw = _ask(messages, None)
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
    """带容量上限的并查集，返回 [{from_id, to_id, alias_name, alias_norm, score}]。

    每个簇选一个代表（提及次数最多，其次名字更长——长的通常是全称更像正式名），
    其余成员指向代表。纯函数，便于单测。

    为什么要**按分数从高到低**加边、而不是先连完再看簇大小：实测一份招股书
    里「国电电力」既是发行人简称，也是「国电电力大连庄河发电…」的前缀，
    几条边一连，一个簇里躺了 5 个不同法人。整簇丢弃会把「国电电力 ↔
    国电电力发展股份有限公司」这条明显正确的合并也一起丢掉——那正是对齐
    最该做的事。所以边按置信度排序，超过容量上限的那条边跳过，簇就停在该
    停的地方：正确的高分对子先合，靠后的一层层前缀链被截断。
    """
    by_id = {str(e["id"]): e for e in entities}
    parent: dict[str, str] = {sid: sid for sid in by_id}
    size: dict[str, int] = {sid: 1 for sid in by_id}
    limit = max(2, settings.kg_align_max_cluster)
    blocked = 0

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in sorted(accepted, key=lambda p: float(p.get("score") or 0), reverse=True):
        a, b = str(pair["a"]["id"]), str(pair["b"]["id"])
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] + size[rb] > limit:
            blocked += 1
            continue
        # 小的挂到大的下面，后续 find 更浅
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    if blocked:
        logger.warning("实体对齐有 %d 条合并因超出簇容量上限（%d）被跳过", blocked, limit)

    clusters: dict[str, list[str]] = {}
    for sid in by_id:
        clusters.setdefault(find(sid), []).append(sid)

    merges: list[dict] = []
    for root, members in clusters.items():
        if len(members) < 2:
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
