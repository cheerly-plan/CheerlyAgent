"""工具选择评测：问句 → 模型调用了哪个工具。

检索评测打的是 kb_search / kg_search 自己找不找得到，不看模型会不会选错工具。
这套只记录模型发出的工具名。工具结果用固定桩顶上，不执行检索、不联网搜索，
避免把「选错工具」和「检索没命中」混在一个分数里。

准备上下文的路径与对话一致：系统提示词、知识库能力、关系问句的图谱预检索、
点到主体时 tool_choice=required 且只留 kb_search / kg_search。预检索仍会查库，
因为线上就是先把邻域放进上下文，再让模型选工具。

用例字段：
    query    用户问句
    group    kb（只要原文）/ kg（只要关系）/ both（两个都要）
    expect   这几轮里必须出现的工具名
    forbid   出现即失败的工具名
    note     说明

模型可能先调一个、看到结果再调另一个，所以最多看 3 轮。某一轮不再调工具就停。

用法：
    uv run python scripts/eval_tool_route.py --list
    uv run python scripts/eval_tool_route.py
    uv run python scripts/eval_tool_route.py --case 注册资本

退出码：0 通过率达到阈值（默认 0.8）；1 低于阈值；2 用例读不到或模型调用失败。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.agents.loop import (  # noqa: E402
    _attach_kg_prefetch,
    _must_call_kb_tool,
    _system_prompt,
)
from ai_agent.llm.client import stream_chat  # noqa: E402
from ai_agent.tools.registry import tool_schemas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "scripts" / "cases" / "tool_route_eval_cases.json"
PASS_THRESHOLD = 0.8
MAX_STEPS = 3
_KB_TOOLS = frozenset({"kb_search", "kg_search"})
# 桩只声明「这一类内容已经给到」，不提示下一步该调哪个工具。
_STUB = {
    "kb_search": "原文切片：本题问到的科目金额、日期或条款已在表格或正文中列出。",
    "kg_search": "图谱：本题问到的实体关系已列出。",
    "web_search": "网页结果：已列出与问题对应的公开信息。",
    "kb_list_files": "知识库里有若干已就绪文件。",
    "get_weather": "天气：已查到。",
    "search_memory": "没有相关记忆。",
    "write_memory": "已记下。",
}


def load_cases(path: str | None = None) -> list[dict]:
    target = Path(path) if path else DEFAULT_CASES
    cases = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("用例集必须是 JSON 数组")
    return cases


def score_case(case: dict, called: list[str]) -> dict:
    """expect 里的工具都出现过、forbid 里的工具一次都没出现，才算通过。"""
    expect = list(case.get("expect") or [])
    forbid = list(case.get("forbid") or [])
    missing = [name for name in expect if name not in called]
    forbidden = [name for name in forbid if name in called]
    return {
        "passed": not missing and not forbidden,
        "called": called,
        "missing": missing,
        "forbidden": forbidden,
    }


def _decide(messages: list[dict], tools: list[dict], tool_choice: str) -> dict:
    result = None
    for kind, payload in stream_chat(messages, tools, tool_choice=tool_choice):
        if kind == "result":
            result = payload
    if result is None:
        raise RuntimeError("模型没有返回结果")
    return result


def run_case(case: dict) -> dict:
    """按对话的第一步准备上下文，最多看三轮工具选择。不执行真实工具。"""
    messages: list[dict] = [
        {"role": "system", "content": _system_prompt("eval-tool-route", use_kb=True)},
        {"role": "user", "content": case["query"]},
    ]
    _attach_kg_prefetch(messages, ())
    tools = tool_schemas()
    called: list[str] = []
    for step in range(MAX_STEPS):
        tool_choice = "auto"
        step_tools = tools
        if _must_call_kb_tool(messages, tools, ()):
            tool_choice = "required"
            step_tools = [
                item for item in tools if item["function"]["name"] in _KB_TOOLS
            ]
        result = _decide(messages, step_tools, tool_choice)
        calls = result.get("tool_calls") or []
        if not calls:
            break
        messages.append(
            {
                "role": "assistant",
                "content": result.get("content") or "",
                "tool_calls": calls,
            }
        )
        for index, call in enumerate(calls):
            name = call["name"]
            called.append(name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"eval-{step}-{index}",
                    "name": name,
                    "content": _STUB.get(name, f"{name} 已返回。"),
                }
            )
    return score_case(case, called)


def render(cases: list[dict], results: list[dict]) -> str:
    lines = [
        "| group | 条数 | 通过 |",
        "|---|---|---|",
    ]
    by_group: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        by_group.setdefault(case["group"], []).append(index)
    passed = 0
    for group, indexes in by_group.items():
        ok = sum(1 for i in indexes if results[i]["passed"])
        passed += ok
        lines.append(f"| {group} | {len(indexes)} | {ok}/{len(indexes)} |")
    lines.append(f"| **合计** | {len(cases)} | **{passed}/{len(cases)}** |")
    failed = [
        (case, result)
        for case, result in zip(cases, results, strict=True)
        if not result["passed"]
    ]
    if failed:
        lines += ["", f"失败 {len(failed)} 条：", ""]
        for case, result in failed:
            lines.append(f"- `{case['query']}`")
            lines.append(
                f"    期望 {case.get('expect') or []}，"
                f"禁止 {case.get('forbid') or []}，"
                f"实际 {' → '.join(result['called']) or '（未调工具）'}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="工具选择评测")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--list", action="store_true", help="只列出用例，不调模型")
    parser.add_argument("--case", default="", help="只跑问句里含这段文字的用例")
    parser.add_argument("--threshold", type=float, default=PASS_THRESHOLD)
    args = parser.parse_args()

    try:
        cases = load_cases(args.cases)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"用例集读不到：{exc}")
        return 2

    if args.case:
        cases = [case for case in cases if args.case in case["query"]]
        if not cases:
            print(f"没有问句包含「{args.case}」的用例")
            return 2

    groups = Counter(case["group"] for case in cases)
    if args.list:
        print(f"用例 {len(cases)} 条，分组：{dict(groups)}")
        for case in cases:
            print(
                f"- [{case['group']}] {case['query']}"
                f"  期望 {case.get('expect')}  禁止 {case.get('forbid') or []}"
            )
        return 0

    results: list[dict] = []
    for index, case in enumerate(cases, start=1):
        try:
            result = run_case(case)
        except Exception as exc:
            print(f"[{index}/{len(cases)}] {case['query']} 调用失败：{exc}")
            return 2
        results.append(result)
        mark = "通过" if result["passed"] else "失败"
        called = " → ".join(result["called"]) or "（未调工具）"
        print(f"[{index}/{len(cases)}] {mark} {case['query']} → {called}", flush=True)

    print()
    print(render(cases, results))
    passed = sum(1 for result in results if result["passed"])
    rate = passed / len(results) if results else 0.0
    print(f"\n通过率 {passed}/{len(results)} = {rate:.0%}，阈值 {args.threshold:.0%}")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
