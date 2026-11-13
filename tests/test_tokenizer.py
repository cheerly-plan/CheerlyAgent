"""Qwen tokenizer 计数的最小回归；没有本地 tokenizer 文件时跳过。"""

from pathlib import Path

import pytest

from ai_agent.llm.tokenizer import HeuristicTokenCounter, QwenTokenCounter

_TOKENIZER = Path(".cache/qwen3/tokenizer.json")


@pytest.mark.skipif(not _TOKENIZER.is_file(), reason="本地没有 Qwen tokenizer 文件")
def test_qwen_tokenizer_counts_chinese_and_ascii():
    from tokenizers import Tokenizer

    counter = QwenTokenCounter(Tokenizer.from_file(str(_TOKENIZER)))
    assert 0 < counter.count("绿的谐波2026Q3营业收入") < 30
    assert 0 < counter.count("hello world, this is a test") < 20
    assert counter.count("") == 0


def test_heuristic_counter_is_conservative():
    counter = HeuristicTokenCounter()
    assert counter.count("") == 0
    assert counter.count("绿的谐波") >= 4
    assert counter.count("abcdefgh") >= 2
