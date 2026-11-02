"""长期记忆抽取评测：黄金对话 → 该记什么、不该记什么。

改抽取提示词、噪声过滤或 facet 规则后跑一遍。用例不连库、不写记忆，
只把一轮对话交给 extract_turn_memories，再对照期望。

用例字段：
    turn               本轮消息，至少要有一条助手回复（否则抽取器直接返回空）
    existing_profile   已有画像行，写法与线上一致，如「- 职业：画家」
    existing_episodes  已有情景原文
    summary            会话摘要，可省略
    expect             必须命中的记忆。kind=profile 时可写 facet / op；
                       contains 全部要出现在 content 里，excludes 全部不能出现
    expect_empty       本轮不该写出任何记忆
    forbid             任一抽出内容含这些子串即失败（用来抓「问董事长=用户是董事长」）
    forbid_facet       指定侧面的内容含这些子串即失败（用来抓「打算转行写成现职」）

指标：用例通过率看「该记的记了、不该记的没记」；召回看期望条数盖住多少；
精确率看抽出的条数里有多少对上了期望，多记会拉低精确率但不单独判用例失败，
除非踩中 forbid。

用法：
    uv run python scripts/eval_memory.py                 # 跑画像/抽取黄金对话（会调记忆模型）
    uv run python scripts/eval_memory.py --list          # 只列出用例，不调模型
    uv run python scripts/eval_memory.py --case weather  # 只跑一条
    uv run python scripts/eval_memory.py --cases 另一份.json
    uv run python scripts/eval_memory.py --recall        # 情景记忆召回：写入干扰项后检索，用完删除

情景召回看的是 search_episode（也就是 search_memory 那条路）：
同一用户名下先写入多条情景，再问一句，期望子串应出现在命中列表里。
前后变化的查询可写 avoid：旧事实必须落在另一条情景里，并且排在新事实后面。
expect 必须只出现在一条情景中，否则分不清命中的是哪条。

退出码：0 通过率达到阈值（默认 0.8）；1 低于阈值；2 用例集读不到或库不可用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CASES_PATH = Path(__file__).resolve().parent / "memory_eval_cases.json"
EPISODE_CASES_PATH = Path(__file__).resolve().parent / "episode_eval_cases.json"
PASS_THRESHOLD = 0.8
# 评测专用用户，跑完会删掉这个用户的情景和画像，不碰真实用户
EPISODE_EVAL_USER = "eval-episode"


def load_cases(path: str | None = None) -> list[dict]:
    target = Path(path) if path else CASES_PATH
    with target.open(encoding="utf-8") as fh:
        cases = json.load(fh)
    if not isinstance(cases, list):
        raise ValueError("用例集必须是 JSON 数组")
    return cases


def _contains_all(text: str, needles: list[str] | None) -> bool:
    return all(needle in text for needle in needles or [])


def _contains_none(text: str, needles: list[str] | None) -> bool:
    return all(needle not in text for needle in needles or [])


def item_matches(spec: dict, item: object) -> bool:
    """一条期望是否被这条抽出记忆满足。未写的 facet/op 不约束。"""
    kind = getattr(item, "kind", None)
    if spec.get("kind") and spec["kind"] != kind:
        return False
    if kind == "profile":
        if spec.get("facet") and spec["facet"] != getattr(item, "facet", None):
            return False
        if spec.get("op") and spec["op"] != getattr(item, "op", None):
            return False
    content = getattr(item, "content", "") or ""
    return _contains_all(content, spec.get("contains")) and _contains_none(
        content, spec.get("excludes")
    )


def _forbid_hits(case: dict, items: list) -> list[str]:
    reasons: list[str] = []
    for item in items:
        content = getattr(item, "content", "") or ""
        for needle in case.get("forbid") or []:
            if needle in content:
                reasons.append(f"写出了不该记的「{needle}」：{content}")
        if getattr(item, "kind", None) != "profile":
            continue
        facet = getattr(item, "facet", "")
        for rule in case.get("forbid_facet") or []:
            if rule.get("facet") == facet and rule.get("contains", "") in content:
                reasons.append(
                    f"{facet} 写成了「{rule['contains']}」：{content}"
                )
    return reasons


def score_case(case: dict, items: list) -> dict:
    """对照一条黄金对话和抽出结果。

    返回 passed / matched / missed / extras / forbid。
    expect_empty 时抽出任何一条都算未通过。
    """
    forbid = _forbid_hits(case, items)
    if case.get("expect_empty"):
        missed = ["本轮不应写出记忆"] if items else []
        return {
            "passed": not items and not forbid,
            "matched": [],
            "missed": missed,
            "extras": list(items),
            "forbid": forbid,
            "expect_count": 1 if items or not case.get("expect") else 0,
            "matched_count": 0 if items else 1,
        }

    expects = list(case.get("expect") or [])
    used: set[int] = set()
    matched: list[dict] = []
    missed: list[dict] = []
    for spec in expects:
        found = next(
            (
                index
                for index, item in enumerate(items)
                if index not in used and item_matches(spec, item)
            ),
            None,
        )
        if found is None:
            missed.append(spec)
        else:
            used.add(found)
            matched.append(spec)
    extras = [item for index, item in enumerate(items) if index not in used]
    expect_count = len(expects)
    # 只有禁写、没有必写的用例：没踩禁写即通过，召回按 1/1 计
    if expect_count == 0:
        return {
            "passed": not forbid,
            "matched": [],
            "missed": [],
            "extras": extras,
            "forbid": forbid,
            "expect_count": 1,
            "matched_count": 0 if forbid else 1,
        }
    return {
        "passed": not missed and not forbid,
        "matched": matched,
        "missed": missed,
        "extras": extras,
        "forbid": forbid,
        "expect_count": expect_count,
        "matched_count": len(matched),
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for row in results if row["passed"])
    expect_count = sum(row["expect_count"] for row in results)
    matched_count = sum(row["matched_count"] for row in results)
    predicted = sum(row["predicted"] for row in results)
    true_positive = sum(row["true_positive"] for row in results)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "recall": (matched_count / expect_count) if expect_count else 0.0,
        "precision": (true_positive / predicted) if predicted else 1.0,
    }


def _format_item(item: object) -> str:
    kind = getattr(item, "kind", "?")
    content = getattr(item, "content", "")
    if kind == "profile":
        facet = getattr(item, "facet", "")
        op = getattr(item, "op", "")
        return f"profile/{facet}/{op}：{content}"
    return f"episode：{content}"


def _format_spec(spec: dict | str) -> str:
    if isinstance(spec, str):
        return spec
    kind = spec.get("kind", "")
    facet = spec.get("facet")
    op = spec.get("op")
    contains = "、".join(spec.get("contains") or [])
    head = "/".join(part for part in (kind, facet, op) if part)
    return f"{head} 含「{contains}」" if contains else head


def render_report(rows: list[dict], summary: dict) -> str:
    lines = [
        "# 长期记忆抽取评测",
        "",
        f"用例 {summary['passed']}/{summary['total']} 通过"
        f"（{summary['pass_rate']:.0%}）",
        f"召回 {summary['recall']:.0%}    精确率 {summary['precision']:.0%}",
        "",
    ]
    failed = [row for row in rows if not row["passed"]]
    if not failed:
        lines.append("全部用例通过。")
        return "\n".join(lines)
    lines.append(f"未通过 {len(failed)} 条：")
    for row in failed:
        lines.append(f"- {row['id']}  {row.get('note') or ''}")
        for spec in row["missed"]:
            lines.append(f"    缺少 {_format_spec(spec)}")
        for reason in row["forbid"]:
            lines.append(f"    {reason}")
        if row["extracted"]:
            lines.append("    实际抽出：")
            for item in row["extracted"]:
                lines.append(f"      {_format_item(item)}")
        elif row["missed"]:
            lines.append("    实际抽出：（空）")
    return "\n".join(lines)


def run_case(case: dict) -> dict:
    from ai_agent.memory.long_term import extract_turn_memories

    items = extract_turn_memories(
        case.get("turn") or [],
        case.get("summary") or "",
        existing_episodes=case.get("existing_episodes") or [],
        existing_profile=case.get("existing_profile") or [],
    )
    scored = score_case(case, items)
    # 只有写明了 expect 的条数算真正例。禁写用例抽出的内容都算多记。
    true_positive = len(scored["matched"]) if case.get("expect") else 0
    return {
        "id": case.get("id") or "",
        "note": case.get("note") or "",
        "passed": scored["passed"],
        "missed": scored["missed"],
        "forbid": scored["forbid"],
        "extracted": items,
        "expect_count": scored["expect_count"],
        "matched_count": scored["matched_count"],
        "predicted": len(items),
        "true_positive": true_positive,
    }


def load_episode_cases(path: str | None = None) -> dict:
    target = Path(path) if path else EPISODE_CASES_PATH
    with target.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("情景用例集必须是 JSON 对象")
    return data


def validate_episode_set(data: dict) -> list[str]:
    """期望子串必须恰好落在一条情景里，否则命中无法归因。"""
    episodes = [str(text) for text in data.get("episodes") or []]
    problems: list[str] = []
    if len(episodes) < 2:
        problems.append("至少要有两条情景，否则没有干扰项")
    for query in data.get("queries") or []:
        expect = str(query.get("expect") or "")
        label = query.get("id") or query.get("query") or ""
        owners = [text for text in episodes if expect and expect in text]
        if len(owners) != 1:
            problems.append(f"{label}: 「{expect}」应只出现在一条情景里，实际 {len(owners)} 条")
            continue
        avoid = str(query.get("avoid") or "")
        if not avoid:
            continue
        avoid_owners = [text for text in episodes if avoid in text]
        if len(avoid_owners) != 1:
            problems.append(f"{label}: 「{avoid}」应只出现在一条情景里，实际 {len(avoid_owners)} 条")
        elif avoid_owners[0] == owners[0]:
            problems.append(f"{label}: 「{expect}」和「{avoid}」落在同一条情景里，分不清前后")
    return problems


def episode_hit_rank(hits: list, expect: str) -> int | None:
    """期望子串在命中列表中的名次（从 1 开始）。没出现返回 None。"""
    for index, hit in enumerate(hits, start=1):
        text = getattr(hit, "text", "") or ""
        if expect and expect in text:
            return index
    return None


def episode_recall_ok(rank: int | None, avoid_rank: int | None, *, hit_k: int) -> bool:
    """期望出现在窗口内，并且排在「以前」那条前面。

    avoid_rank 为空表示旧事实没进结果，算分清了。两者名次相同通常是被合并进
    同一条情景，现在和以前都在这段里，不算分清。
    """
    if rank is None or rank > hit_k:
        return False
    if avoid_rank is None:
        return True
    return rank < avoid_rank


def summarize_recall(rows: list[dict], *, hit_k: int) -> dict:
    total = len(rows)
    if not total:
        return {"total": 0, "hit1": 0.0, "hit_k": 0.0, "mrr": 0.0, "k": hit_k}
    hit1 = sum(
        1 for row in rows if episode_recall_ok(row["rank"], row.get("avoid_rank"), hit_k=1)
    )
    within = sum(
        1
        for row in rows
        if episode_recall_ok(row["rank"], row.get("avoid_rank"), hit_k=hit_k)
    )
    mrr = sum((1 / row["rank"]) if row["rank"] else 0.0 for row in rows) / total
    return {
        "total": total,
        "hit1": hit1 / total,
        "hit_k": within / total,
        "mrr": mrr,
        "k": hit_k,
    }


def render_recall(rows: list[dict], summary: dict) -> str:
    k = summary["k"]
    lines = [
        "# 情景记忆召回评测",
        "",
        f"hit@1 {summary['hit1']:.0%}    hit@{k} {summary['hit_k']:.0%}"
        f"    MRR {summary['mrr']:.2f}    （{summary['total']} 问）",
        "",
    ]
    missed = [
        row
        for row in rows
        if not episode_recall_ok(row["rank"], row.get("avoid_rank"), hit_k=k)
    ]
    if not missed:
        lines.append(f"全部问题都在前 {k} 条内命中。")
        return "\n".join(lines)
    lines.append("未进窗口的问题：")
    for row in missed:
        rank = "未命中" if row["rank"] is None else f"第 {row['rank']} 名"
        avoid = row.get("avoid") or ""
        avoid_rank = row.get("avoid_rank")
        extra = ""
        if avoid and avoid_rank is not None and (
            row["rank"] is None or avoid_rank <= row["rank"]
        ):
            extra = f"，旧事实「{avoid}」排第 {avoid_rank} 名"
        lines.append(
            f"- {row['id']}  {row['query']}  → {rank}，期望「{row['expect']}」{extra}"
        )
        if row["top"]:
            lines.append(f"    第 1 名：{row['top']}")
    return "\n".join(lines)


def run_episode_recall(data: dict, *, top_k: int) -> list[dict]:
    """把黄金情景写入评测用户，逐条查询，最后删掉这个用户的长期记忆。"""
    from ai_agent.memory.store import store

    store.clear_long_term(user_id=EPISODE_EVAL_USER)
    try:
        for text in data["episodes"]:
            store.write_episode(str(text), user_id=EPISODE_EVAL_USER)
        rows: list[dict] = []
        for query in data["queries"]:
            hits = store.search_episode(
                str(query["query"]),
                k=top_k,
                user_id=EPISODE_EVAL_USER,
            )
            expect = str(query["expect"])
            avoid = str(query.get("avoid") or "")
            rank = episode_hit_rank(hits, expect)
            rows.append(
                {
                    "id": query.get("id") or "",
                    "query": query.get("query") or "",
                    "expect": expect,
                    "avoid": avoid,
                    "rank": rank,
                    "avoid_rank": episode_hit_rank(hits, avoid) if avoid else None,
                    "top": hits[0].text if hits else "",
                }
            )
        return rows
    finally:
        store.clear_long_term(user_id=EPISODE_EVAL_USER)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="长期记忆评测")
    parser.add_argument("--cases", help="用例集路径。抽取默认 memory_eval_cases.json，召回默认 episode_eval_cases.json")
    parser.add_argument("--case", help="只跑指定 id")
    parser.add_argument("--list", action="store_true", help="只列出用例，不调模型、不写库")
    parser.add_argument("--recall", action="store_true", help="评情景记忆召回，而不是抽取")
    parser.add_argument("--top-k", type=int, default=5, help="召回时每问取回的条数，默认 5")
    parser.add_argument("--hit-k", type=int, default=3, help="算命中率的窗口，默认 3")
    parser.add_argument(
        "--threshold",
        type=float,
        default=PASS_THRESHOLD,
        help=f"通过率阈值。抽取看用例通过率，召回看 hit@k，默认 {PASS_THRESHOLD}",
    )
    args = parser.parse_args(argv)
    if args.recall:
        return _main_recall(args)
    return _main_extract(args)


def _main_extract(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.cases)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"用例集无法读取：{exc}", file=sys.stderr)
        return 2
    if args.case:
        cases = [case for case in cases if case.get("id") == args.case]
        if not cases:
            print(f"没有 id={args.case} 的用例", file=sys.stderr)
            return 2
    if args.list:
        for case in cases:
            print(f"{case.get('id')}\t{case.get('note') or ''}")
        return 0
    rows = [run_case(case) for case in cases]
    summary = summarize(rows)
    print(render_report(rows, summary))
    return 0 if summary["pass_rate"] >= args.threshold else 1


def _main_recall(args: argparse.Namespace) -> int:
    try:
        data = load_episode_cases(args.cases)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"用例集无法读取：{exc}", file=sys.stderr)
        return 2
    if args.case:
        data = {
            **data,
            "queries": [query for query in data.get("queries") or [] if query.get("id") == args.case],
        }
        if not data["queries"]:
            print(f"没有 id={args.case} 的查询", file=sys.stderr)
            return 2
    problems = validate_episode_set(data)
    if problems:
        print("情景用例对不上：", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2
    if args.list:
        for query in data["queries"]:
            print(f"{query.get('id')}\t{query.get('query')}")
        return 0
    try:
        rows = run_episode_recall(data, top_k=args.top_k)
    except Exception as exc:
        print(f"情景召回评测失败：{exc}", file=sys.stderr)
        return 2
    summary = summarize_recall(rows, hit_k=args.hit_k)
    print(render_recall(rows, summary))
    return 0 if summary["hit_k"] >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
