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

    简称（「中国能源建设」→「中国能建」）无法由规则推导。图谱对齐产出的
    别名在建图结束后写回文件，检索按整词命中，不在这里硬凑。
    """
    if not subject:
        return []
    core = subject_core(subject)
    out: list[str] = []
    for item in (subject, core, normalize_name(subject), normalize_name(core)):
        if item and item not in out:
            out.append(item)
    return out


def looks_like_company(name: str) -> bool:
    """文件名剩下来的字是不是公司，而不是「员工考勤制度」这种标题。"""
    return bool(COMPANY_SUFFIX.search(name or ""))


def infer_doc_type(filename: str) -> str:
    """从文件名推断文档类型。推断不出返回空串，调用方保留人工指定的值。

    「半年度报告」里含「年度报告」，必须先判半年报和季报。
    """
    stem = EXT_RE.sub("", filename or "")
    if re.search(r"招股意向书|招股说明书", stem):
        return "prospectus"
    if "法律意见书" in stem:
        return "legal_opinion"
    if "尽职调查" in stem:
        return "due_diligence"
    if re.search(r"季度报告|半年度报告|中期报告", stem):
        return "quarterly_report"
    if "年度报告" in stem:
        return "annual_report"
    if "公告" in stem:
        return "announcement"
    return ""


# 封面上的公司名：后缀前最多 24 个字，避免把整段导语吞进来
_COVER_COMPANY = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{2,24}"
    r"(?:股份有限公司|有限责任公司|有限公司|集团公司|银行|证券|保险|基金)"
)


def dominant_company(text: str) -> str:
    """封面文本里出现最多的公司名。次数相同取更长的那个。"""
    counts: dict[str, int] = {}
    for match in _COVER_COMPANY.finditer(text or ""):
        name = match.group(0)
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda name: (counts[name], len(name)))


def name_in_text(text: str, name: str, *, min_len: int = 2) -> bool:
    """名称或去掉组织形式后的核心名，整段出现在文本里。

    不做最长公共子串。「中国能建」不会因为和全称共享「中国能」而命中，
    必须别名表里真有这四个字。
    """
    hay = normalize_name(text)
    if not hay:
        return False
    for token in (normalize_name(name), normalize_name(subject_core(name))):
        if len(token) >= min_len and token in hay:
            return True
    return False


_PERIOD_RE = re.compile(r"^((?:19|20)\d{2})(?:Q([1-4])|H1)?$")
_QUERY_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_QUERY_QUARTER_RE = re.compile(r"第?([一二三四1-4])季度")
_QUERY_HALF_RE = re.compile(r"半年度|半年报|中期")
_MULTI_PERIOD_RE = re.compile(r"同比|环比|历年|各季度|各期|近[一二三四五六七八九十\d]+年")
_QUARTER_DIGIT = {"一": "1", "二": "2", "三": "3", "四": "4", "1": "1", "2": "2", "3": "3", "4": "4"}


def period_rank(period: str) -> tuple[int, int]:
    """报告期的新旧。年报排在同年季报之后（年报发布更晚）；无法解析的排最后。"""
    match = _PERIOD_RE.match(period or "")
    if not match:
        return (0, 0)
    year = int(match.group(1))
    if match.group(2):
        return (year, int(match.group(2)))
    if (period or "").endswith("H1"):
        return (year, 2)
    return (year, 9)


def query_period_policy(query: str, *, today=None) -> tuple[str, str]:
    """问句要收成哪一期。

    返回 (策略, 令牌)：
    - all / ""：同比、历年、两个年份，不收窄
    - exact / 2026Q3：指定了季度或半年
    - year / 2025：只给了年份，或「去年」「今年」
    - latest / ""：没提时期，同一主体只留最新一期
    """
    from datetime import date

    text = query or ""
    if not text.strip() or _MULTI_PERIOD_RE.search(text):
        return ("all", "")
    years = list(dict.fromkeys(_QUERY_YEAR_RE.findall(text)))
    if len(years) >= 2:
        return ("all", "")
    anchor = today or date.today()
    year = ""
    if years:
        year = years[0]
    elif "前年" in text:
        year = str(anchor.year - 2)
    elif "去年" in text:
        year = str(anchor.year - 1)
    elif "今年" in text:
        year = str(anchor.year)
    quarter = _QUERY_QUARTER_RE.search(text)
    if quarter:
        # 「三季度营收」没写年份时，按当前年理解
        pinned = year or str(anchor.year)
        return ("exact", f"{pinned}Q{_QUARTER_DIGIT[quarter.group(1)]}")
    if _QUERY_HALF_RE.search(text):
        pinned = year or str(anchor.year)
        return ("exact", f"{pinned}H1")
    if year:
        return ("year", year)
    return ("latest", "")


def period_matches(period: str, policy: str, token: str) -> bool:
    if policy == "exact":
        return period == token
    if policy == "year":
        return (period or "").startswith(token)
    return True


def resolve_file_identity(filename: str, cover: str = "") -> dict:
    """文件名规则优先，封面只补文件名里没有的公司名和报告期。"""
    meta = extract_file_meta(filename)
    if not looks_like_company(meta["subject"]):
        found = dominant_company(cover)
        if found:
            meta["subject"] = found
            meta["subject_aliases"] = extract_aliases(found)
    if not meta["period"]:
        meta["period"] = extract_period(cover)
    meta["doc_type"] = infer_doc_type(filename) or infer_doc_type(cover)
    return meta


def assign_chunk_subjects(
    texts: list[str],
    default: str,
    library: list[tuple[str, list[str]]],
) -> list[str]:
    """正文里出现至少两家公司时，按提及次数给每片定主体。否则整份继承文件主体。"""
    if not texts:
        return []
    blob = "\n".join(texts)
    candidates: list[tuple[str, list[str]]] = []
    known: set[str] = set()
    for subject, aliases in library:
        subject = (subject or "").strip()
        if not subject or subject in known:
            continue
        known.add(subject)
        candidates.append((subject, list(aliases or []) or extract_aliases(subject)))
    for name in _COVER_COMPANY.findall(blob):
        if name not in known:
            known.add(name)
            candidates.append((name, extract_aliases(name)))
    appearing = [
        (subject, forms)
        for subject, forms in candidates
        if count_mentions(blob, forms) > 0
    ]
    if len(appearing) < 2:
        return [default] * len(texts)
    return [
        pick_chunk_subject(text, appearing, default=default) for text in texts
    ]


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
    return {
        "subject": subject,
        "subject_aliases": extract_aliases(subject),
        "period": extract_period(filename),
        "meta": {},
    }
