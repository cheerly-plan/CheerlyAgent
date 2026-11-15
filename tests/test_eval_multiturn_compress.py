"""多轮压缩评测：确定性模式下事实必须无损保留。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "eval_multiturn_compress.py"
    spec = importlib.util.spec_from_file_location("eval_multiturn_compress", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# 确定性摘要（crude_summary）是原文照抄，压缩本身不该丢事实。
# LLM 摘要会压缩改写，事实保留率只能本地跑（需要模型）。
def test_shipped_cases_are_wellformed():
    data = json.loads(
        (ROOT / "scripts" / "cases" / "multiturn_compress_cases.json").read_text()
    )
    assert len(data) >= 25
    for case in data:
        assert case["setup"], case["id"]
        assert case["followup"], case["id"]
        assert case["must_any"], case["id"]
        # setup 必须是偶数条（成对的 user/assistant）
        assert len(case["setup"]) % 2 == 0, case["id"]


def test_facts_live_in_early_turns():
    """关键事实必须落在会被压掉的早期轮，否则这条用例测不出问题。"""
    from ai_agent.memory.short_term import compress

    data = json.loads(
        (ROOT / "scripts" / "cases" / "multiturn_compress_cases.json").read_text()
    )
    trivial = []
    for case in data:
        messages = [
            {**msg, "seq": index + 1}
            for index, msg in enumerate(case["setup"])
        ]
        recent, _summary, _cursor = compress(messages, max_tokens=1)
        kept = set(range(len(messages) - len(recent), len(messages)))
        if all(
            any(
                any(token in messages[index]["content"] for index in kept)
                for token in group
            )
            for group in case["must_any"]
        ):
            trivial.append(case["id"])
    # 允许少数天然只有一轮的用例，但不能占多数
    assert len(trivial) <= len(data) // 4, trivial


def test_deterministic_mode_preserves_all_facts():
    mod = _load()
    data = json.loads(
        (ROOT / "scripts" / "cases" / "multiturn_compress_cases.json").read_text()
    )
    report = mod.run(data, budget=0, summarize=None, keep_last=1)
    assert report["rate"] == 1.0, report["failures"]
