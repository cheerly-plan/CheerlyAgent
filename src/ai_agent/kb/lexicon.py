"""四字短词表：写进结巴后不再切开。

自动词表从表格行项目和公司名抽出，重新解析后整份重写。
手改文件只留例外，检索进程启动时两份一起读。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import extract_table_indicators
from ai_agent.kb.meta import NAME_PUNCT

logger = get_logger("kb.lexicon")

_DATA = Path(__file__).resolve().parent / "data"
_GENERATED = _DATA / "short_words.txt"
_EXTRA = _DATA / "short_words_extra.txt"

# 行项目开头的序号、分组和「其中 / 减 / 加」。剥掉才看得到科目本身。
_LEADING = re.compile(
    r"^(?:"
    r"[（(][一二三四五六七八九十0-9]+[）)]|"
    r"[一二三四五六七八九十0-9]+[、.．]|"
    r"其中[:：]?|"
    r"[加减][:：]|"
    r"[-－—]"
    r")\s*"
)
_TRAILING_PAREN = re.compile(r"[（(][^）)]*[）)]\s*$")
_PERIOD_SUFFIX = re.compile(r"_(年初至报告期末|本报告期)$")
_FOUR = re.compile(r"^[\u4e00-\u9fff]{4}$")
_TWO = re.compile(r"^[\u4e00-\u9fff]{2}$")
_KEEP_WORD = re.compile(r"^[\u4e00-\u9fff]{2,8}$")
_LEADING_CJK = re.compile(r"[\u4e00-\u9fff]+")

_tokenizer = None
_loaded = False


def generated_path() -> Path:
    return _GENERATED


def extra_path() -> Path:
    return _EXTRA


def core_four(name: str) -> str:
    """行项目收成恰好四个汉字。序号、括号备注和报告期后缀剥掉。收不成就空。"""
    text = (name or "").strip().strip("*").strip()
    text = _PERIOD_SUFFIX.sub("", text).strip()
    changed = True
    while changed:
        changed = False
        nxt = _LEADING.sub("", text, count=1).strip()
        if nxt != text:
            text = nxt
            changed = True
            continue
        nxt = _TRAILING_PAREN.sub("", text).strip()
        if nxt != text:
            text = nxt
            changed = True
    return text if _FOUR.fullmatch(text) else ""


def org_four(name: str) -> str:
    """公司名取开头四个汉字。全称更长时，简称通常就是这四个字。"""
    run = _LEADING_CJK.match(NAME_PUNCT.sub("", name or ""))
    if run is None or len(run.group(0)) < 4:
        return ""
    return run.group(0)[:4]


def candidate_words(line_names: Iterable[str], org_names: Iterable[str]) -> list[str]:
    """行项目和公司名里的四字候选，去重并保持先出现的顺序。"""
    seen: set[str] = set()
    words: list[str] = []
    for name in line_names:
        word = core_four(name)
        if word and word not in seen:
            seen.add(word)
            words.append(word)
    for name in org_names:
        word = org_four(name)
        if word and word not in seen:
            seen.add(word)
            words.append(word)
    return words


def parse_extra(text: str) -> tuple[list[str], set[str]]:
    """手改例外。``+词`` 保留，``-词`` 排除。``#`` 之后是注释。"""
    keeps: list[str] = []
    drops: set[str] = set()
    for line in (text or "").splitlines():
        raw = line.split("#", 1)[0].strip()
        if not raw:
            continue
        sign, word = raw[0], raw[1:].strip()
        if sign == "+":
            bucket: list[str] | None = keeps
        elif sign == "-":
            if _KEEP_WORD.fullmatch(word):
                drops.add(word)
            continue
        else:
            bucket = keeps
            word = raw
        if bucket is not None and _KEEP_WORD.fullmatch(word) and word not in keeps:
            keeps.append(word)
    return keeps, drops


def read_extra(path: Path | None = None) -> tuple[list[str], set[str]]:
    file = path or extra_path()
    if not file.is_file():
        return [], set()
    return parse_extra(file.read_text(encoding="utf-8"))


def _cutter():
    """不带用户词表的结巴。生成时要用它，避免上一份词表把词判成「本来就不拆」。"""
    global _tokenizer
    import jieba

    if _tokenizer is None:
        _tokenizer = jieba.Tokenizer()
        _tokenizer.initialize()
    return _tokenizer.cut


def is_two_plus_two(word: str, cutter: Callable[[str], Iterable[str]] | None = None) -> bool:
    """结巴会把这四个字切成两个二字。已经是一个词的不算。"""
    tokens = [token.strip() for token in (cutter or _cutter())(word) if token.strip()]
    return len(tokens) == 2 and all(_TWO.fullmatch(token) for token in tokens)


def choose_words(
    words: Iterable[str],
    boundary: dict[str, int],
    total: int,
    keeps: Iterable[str],
    drops: Iterable[str],
    cutter: Callable[[str], Iterable[str]] | None = None,
) -> list[str]:
    """留下该整段查的四字。

    自动候选要同时满足：结巴会拆成两个二字，并且库里至少有一处后面不再是汉字，
    这样的切片不超过一半。排除名单优先。保留名单是自动没抽到、仍要整段查的词。
    """
    blocked = set(drops)
    chosen: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in blocked or word in seen:
            continue
        hits = boundary.get(word, 0)
        if hits <= 0 or hits * 2 > total:
            continue
        if not is_two_plus_two(word, cutter):
            continue
        seen.add(word)
        chosen.append(word)
    for word in keeps:
        if word in blocked or word in seen or not _KEEP_WORD.fullmatch(word):
            continue
        seen.add(word)
        chosen.append(word)
    return chosen


def render_words(words: Iterable[str]) -> str:
    lines = [
        "# 自动生成，不要手改。例外写在 short_words_extra.txt。",
        "# 重新解析文档，或运行 kb.py lexicon，会整份重写本文件。",
    ]
    lines.extend(words)
    lines.append("")
    return "\n".join(lines)


def write_words(words: list[str], path: Path | None = None) -> Path:
    file = path or generated_path()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(render_words(words), encoding="utf-8")
    return file


def _corpus(store) -> tuple[list[str], list[str]]:
    with store._get_pool().connection() as conn:
        files = conn.execute(
            """
            SELECT f.filename, f.subject, f.subject_aliases
            FROM agent_kb_file f
            WHERE f.status = 'ready'
            """
        ).fetchall()
        chunks = conn.execute(
            """
            SELECT c.text
            FROM agent_kb_chunk c
            JOIN agent_kb_file f ON f.id = c.file_id
            WHERE f.status = 'ready'
            """
        ).fetchall()
    line_names: list[str] = []
    for row in chunks:
        line_names.extend(extract_table_indicators(row["text"] or ""))
    org_names: list[str] = []
    for row in files:
        org_names.append(row["filename"] or "")
        org_names.append(row["subject"] or "")
        aliases = row["subject_aliases"] or []
        if isinstance(aliases, str):
            aliases = [aliases]
        org_names.extend(str(alias) for alias in aliases if alias)
    return line_names, org_names


def rebuild_short_words(store, *, write: bool = True) -> list[str]:
    """按当前库重算短词。``write=False`` 只返回结果，不覆盖文件。"""
    line_names, org_names = _corpus(store)
    words = [word for word in candidate_words(line_names, org_names) if is_two_plus_two(word)]
    keeps, drops = read_extra()
    total, boundary = store.keyword_boundary_freq(words)
    chosen = choose_words(words, boundary, total, keeps, drops)
    if write:
        write_words(chosen)
        logger.info("kb 短词表已重写：%s 个", len(chosen))
    return chosen


def load_short_words() -> None:
    """把自动词表和手改保留词读进结巴。同一进程只读一次。"""
    global _loaded
    if _loaded:
        return
    try:
        import jieba
    except Exception:
        logger.warning("jieba 不可用，短词表未加载", exc_info=True)
        return
    try:
        generated = generated_path()
        if generated.is_file():
            for word in _words_in(generated.read_text(encoding="utf-8")):
                jieba.suggest_freq(word, tune=True)
        keeps, drops = read_extra()
        for word in drops:
            jieba.del_word(word)
        for word in keeps:
            jieba.suggest_freq(word, tune=True)
    except Exception:
        logger.warning("kb 短词表加载失败", exc_info=True)
        return
    _loaded = True


def _words_in(text: str) -> list[str]:
    words: list[str] = []
    for line in text.splitlines():
        raw = line.split("#", 1)[0].strip()
        if not raw:
            continue
        word = raw.split()[0]
        if _KEEP_WORD.fullmatch(word):
            words.append(word)
    return words
