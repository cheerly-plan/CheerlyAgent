"""load_llm_json 对数组输出的解析。

历史 bug：无条件先找 "{"，在 [{"a":1},{"b":2}] 里会从第一个 { 截到最后一个 }，
得到 "{...},{...}" 这种非法 JSON，解析失败返回 None。多元素数组因此一律
判失败——深度分析章节返回裸 claims 数组时全被当成「无法解析」。
"""

from ai_agent.llm.json_utils import load_llm_json


def test_multi_element_array_parses():
    raw = '[{"text": "a", "claim_type": "known"}, {"text": "b", "claim_type": "inferred"}]'
    payload = load_llm_json(raw, repair=True)
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["text"] == "a"


def test_single_element_array_parses():
    raw = '[{"text": "a"}]'
    assert load_llm_json(raw, repair=True) == [{"text": "a"}]


def test_object_still_parses():
    raw = '{"verdict": "高", "claims": [{"text": "a"}, {"text": "b"}]}'
    payload = load_llm_json(raw, repair=True)
    assert isinstance(payload, dict)
    assert payload["verdict"] == "高"


def test_code_fenced_array_parses():
    raw = '```json\n[{"a": 1}, {"b": 2}]\n```'
    assert load_llm_json(raw, repair=True) == [{"a": 1}, {"b": 2}]
