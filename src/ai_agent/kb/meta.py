"""知识库检索元数据：主体（公司）/ 报告期 / 长尾字段。

为什么需要它：检索此前只有 ``file_id`` 一个结构维度，"A 和 B 的营业收入"
这类对比问句只能用**单一 query 向量**在 ``top_k`` 个名额里全局抢——离混合
向量更近的那家会把名额吃满，另一家一条都回不来。补上 ``subject`` 维度后，
检索可以按主体分桶、每桶独立配额（见 ``kb/store.search_kb_chunks``），
这类问题不必再靠模型「自觉拆成两次调用」来兜。

抽取走**规则**而不是 LLM：文件名「XX公司 + 年份 + 报告类型」的结构很强，
规则就能定死；规则可回放、可解释、零成本，也不会因为模型抽错而静默污染
过滤（抽错主体比抽不到更糟——它会让该主体的切片整体查不到）。

存储分两层：热维度（subject / period）成真实列，可直接索引、可
``PARTITION BY`` 分桶；长尾（报告种类等）进 ``meta`` JSONB。
"""

from __future__ import annotations

import re

# 扩展名
EXT_RE = re.compile(r"\.(pdf|docx?|xlsx?|xlsm|md|markdown|txt|csv)$", re.I)
# 每份文件标题里都有、对「是哪家主体」没有区分度的部分
FILENAME_NOISE = re.compile(
    r"\d{4}\s*年|第[一二三四五六七八九十]+季度|半年度报告|季度报告|年度报告|报告|公告"
)
# 组织形式后缀：剥掉才知道主体核心名有多长
ENTITY_SUFFIX = re.compile(r"(股份有限公司|有限责任公司|有限公司|集团公司|集团|公司)$")
# 名称里的标点与空白：比对前先抹平，避免全角/半角差异造成假不重合
NAME_PUNCT = re.compile(
    r"""[\s\-_/·・,，。.、:：;；!！?？'"“”‘’()（）\[\]【】<>《》]+"""
)

_YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*年?")
_QUARTER_RE = re.compile(r"第([一二三四])季度")
_QUARTER_MAP = {"一": 1, "二": 2, "三": 3, "四": 4}
_HALF_RE = re.compile(r"半年度|中期")
_REPORT_KIND_RE = re.compile(r"季度报告|半年度报告|年度报告")


def normalize_name(name: str) -> str:
    """抹平标点与大小写，用于比对（不用于展示）。"""
    return NAME_PUNCT.sub("", name or "").lower()


def subject_core(subject: str) -> str:
    """剥掉「股份有限公司」这类组织形式后缀，留下真正的主体名。"""
    return ENTITY_SUFFIX.sub("", subject or "")


def extract_subject(filename: str) -> str:
    """文件名去掉扩展名、年份、报告类型后的主体（保留可读原名）。

    「中国能源建设股份有限公司2026年第三季度报告.pdf」→「中国能源建设股份有限公司」。
    展示与落库都用这个可读形式；比对时再走 :func:`normalize_name`。
    """
    stem = EXT_RE.sub("", filename or "")
    stem = FILENAME_NOISE.sub("", stem)
    # 噪声剥离后尾部可能留下分隔符（「XX公司 - 2026年...」）
    return NAME_PUNCT.sub("", stem).strip()


def extract_period(filename: str) -> str:
    """报告期：2026年第三季度报告 → 2026Q3；半年度 → 2026H1；年报 → 2026。"""
    stem = EXT_RE.sub("", filename or "")
    year_m = _YEAR_RE.search(stem)
    if not year_m:
        return ""
    year = year_m.group(0)[:4]
    quarter_m = _QUARTER_RE.search(stem)
    if quarter_m:
        return f"{year}Q{_QUARTER_MAP[quarter_m.group(1)]}"
    if _HALF_RE.search(stem):
        return f"{year}H1"
    return year


def extract_aliases(subject: str) -> list[str]:
    """主体的可匹配写法：全称、去组织形式后缀的核心名，及两者的归一化形。

    简称（「中国能源建设」→「中国能建」）无法由规则推导，靠检索侧的
    n-gram 重合判定（见 ``retrieval._query_mentions``），不在这里硬凑。
    """
    if not subject:
        return []
    core = subject_core(subject)
    out: list[str] = []
    for item in (subject, core, normalize_name(subject), normalize_name(core)):
        if item and item not in out:
            out.append(item)
    return out


def extract_report_kind(filename: str) -> str:
    """报告种类，进 meta JSONB（长尾，不单独成列）。"""
    stem = EXT_RE.sub("", filename or "")
    m = _REPORT_KIND_RE.search(stem)
    return m.group(0) if m else ""


# 像「公司主体」的实体名后缀：多主体文档（行业研报）里靠它挑候选主体
COMPANY_SUFFIX = re.compile(
    r"(股份有限公司|有限责任公司|有限公司|集团公司|集团|控股|银行|证券|保险|基金)$"
)


def count_mentions(text: str, forms: list[str]) -> int:
    """统计某个主体的各种写法在切片里出现的总次数。

    中文没有词边界，直接数子串；同一主体多种写法（全称/核心名）次数相加，
    避免「中国能建」与「中国能源建设股份有限公司」被当成两家公司各算一半。
    """
    if not text:
        return 0
    return sum(text.count(form) for form in forms if form)


def pick_chunk_subject(
    text: str,
    candidates: list[tuple[str, list[str]]],
    default: str = "",
    min_hits: int = 1,
) -> str:
    """多主体文档里给单个切片挑归属主体。

    ``candidates`` 是 (主体, 该主体的可匹配写法列表)。取提及次数最多的那个；
    次数相同取先被提到的那个（位置靠前通常是这段的主语）；一次都没提到才
    退回 ``default``（文件的主体）。

    平票不退回 default 而按位置定：两段都提到两家公司的切片，归给任何一家
    都比留在「文件主体」桶里强——那个桶既不是 A 也不是 B，两家都查不到。
    """
    forms_by = dict(candidates)
    if default and default not in forms_by:
        forms_by[default] = extract_aliases(default) or [default]
    best_subject, best_hits, best_pos = default, min_hits - 1, -1
    for subject, forms in forms_by.items():
        hits = count_mentions(text, forms)
        if not hits:
            continue
        positions = [text.find(form) for form in forms if form and form in text]
        pos = min(positions) if positions else len(text)
        if hits > best_hits or (
            hits == best_hits and hits >= min_hits and (best_pos < 0 or pos < best_pos)
        ):
            best_subject, best_hits, best_pos = subject, hits, pos
    return best_subject


def extract_file_meta(filename: str) -> dict:
    """一次算齐文件的全部元数据，供入库写入与存量回填共用。

    返回 dict 键与 ``agent_kb_file`` 列名一致：subject / subject_aliases /
    period / meta。主体抽不出来时 subject 为空串——检索侧对空主体退回
    「不分桶」的原行为，不会因此查不到东西。
    """
    subject = extract_subject(filename)
    kind = extract_report_kind(filename)
    return {
        "subject": subject,
        "subject_aliases": extract_aliases(subject),
        "period": extract_period(filename),
        "meta": {"report_kind": kind} if kind else {},
    }
