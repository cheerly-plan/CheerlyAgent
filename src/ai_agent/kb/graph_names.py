"""图谱实体名的归一化、可用性判据，以及按名字在批内定位。

从 graph_extract.py 拆出(S5)。模型吐出的实体名写法五花八门——同一家公司
在三个片段里可能是「长鑫存储（合肥）」「长鑫存储(合肥)」「长鑫存储 」，
「是否经过审计」也会被写成实体。这里把三件事集中在一处：

1. 归一化键（normalize_name）：只用于去重与连边，全角括号/引号先转半角，
   再去首尾包裹符号、压空白、转小写，让上面三种写法落到同一个键；
2. 展示名（display_name）：用于入库与前端展示，刻意保留圆括号，因为中文
   里全角括号更自然，剥掉反而毁名；
3. 可用性判据（_usable_name 等）：谓词、停用词、纯数值、占位词一律拦掉，
   它们是图谱噪声的主要来源。

批内定位（_locate_chunk / _claimed_offset）跟着名字走：定位的第一判据是
「名字出现在哪一片」，只有名字匹配不上时才退回模型自报的序号。
"""

from __future__ import annotations

import re


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

# prompt 示例里的占位词。示例一旦用真实公司名（曾经写成「深圳明德控股」
# 「顺丰控股」），模型在片段缺主语时会把它当模板抄进结果——安克文件的图里
# 长出过一个挂着 14 条关系的「顺丰控股」，就是这么来的。示例改成占位词后
# 再加这道过滤：万一模型仍然照抄，也进不了图谱
# 只拦「某某」式占位词，不拦「某公司」——原文匿名化时真的会这么写，
# 拿它当过滤条件会把正常实体一并丢掉
_PLACEHOLDER_RE = re.compile(r"某某|××|X{2,}")


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


def is_metric_like(name: str) -> bool:
    """整条名字基本由数字和计量单位构成（时间点、金额、比例、指标值）。"""
    text = (name or "").strip()
    if not text:
        return False
    hit = len(_METRIC_CHARS.findall(text))
    return hit / len(text) >= 0.6 and any(ch.isdigit() for ch in text)


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
