"""记忆抽取评测的打分。分数是否可信取决于这里，不取决于有没有调到模型。"""

from types import SimpleNamespace

from scripts.eval_memory import (
    episode_hit_rank,
    episode_recall_ok,
    item_matches,
    load_cases,
    load_episode_cases,
    score_case,
    summarize,
    summarize_recall,
    validate_episode_set,
)


def _item(kind: str, content: str, facet: str = "", op: str = "add"):
    return SimpleNamespace(kind=kind, content=content, facet=facet, op=op)


def test_gold_cases_have_a_turn_and_an_expectation():
    cases = load_cases()
    assert cases
    for case in cases:
        assert case["id"]
        assert any(msg.get("role") == "assistant" for msg in case["turn"])
        assert case.get("expect") or case.get("expect_empty") or case.get("forbid_facet")


def test_item_matches_profile_facet_and_text():
    item = _item("profile", "林晓", facet="name", op="add")
    assert item_matches(
        {"kind": "profile", "facet": "name", "contains": ["林晓"]}, item
    )
    assert not item_matches(
        {"kind": "profile", "facet": "occupation", "contains": ["林晓"]}, item
    )


def test_item_matches_rejects_excluded_text():
    item = _item("profile", "曾住北京，现住上海", facet="locale", op="update")
    assert not item_matches(
        {"kind": "profile", "facet": "locale", "contains": ["上海"], "excludes": ["北京"]},
        item,
    )


def test_expect_empty_fails_when_anything_is_written():
    case = {"expect_empty": True, "forbid": ["董事长"]}
    scored = score_case(case, [_item("profile", "用户是董事长", facet="occupation")])
    assert not scored["passed"]
    assert scored["forbid"]


def test_expect_empty_passes_when_nothing_is_written():
    scored = score_case({"expect_empty": True, "forbid": ["董事长"]}, [])
    assert scored["passed"]
    assert scored["matched_count"] == 1


def test_forbid_facet_blocks_intent_written_as_occupation():
    case = {"forbid_facet": [{"facet": "occupation", "contains": "Agent"}]}
    items = [_item("profile", "现为 Agent 开发", facet="occupation", op="update")]
    scored = score_case(case, items)
    assert not scored["passed"]
    assert scored["forbid"]


def test_extra_item_does_not_fail_when_expect_is_met():
    case = {
        "expect": [{"kind": "profile", "facet": "name", "contains": ["林晓"]}],
    }
    items = [
        _item("profile", "林晓", facet="name"),
        _item("episode", "用户正在整理通讯录"),
    ]
    scored = score_case(case, items)
    assert scored["passed"]
    assert len(scored["extras"]) == 1


def test_episode_hit_rank_skips_distractors():
    hits = [
        SimpleNamespace(text="用户家里的猫叫年糕"),
        SimpleNamespace(text="用户日常用 pytest 写后端单测"),
    ]
    assert episode_hit_rank(hits, "pytest") == 2
    assert episode_hit_rank(hits, "TypeScript") is None


def test_episode_gold_expects_are_unique():
    problems = validate_episode_set(load_episode_cases())
    assert problems == []


def test_before_after_does_not_count_when_old_fact_ranks_first():
    assert episode_recall_ok(2, 1, hit_k=3) is False
    assert episode_recall_ok(1, 2, hit_k=3) is True
    assert episode_recall_ok(1, 1, hit_k=3) is False


def test_summarize_recall_counts_window():
    summary = summarize_recall(
        [{"rank": 1}, {"rank": 2}, {"rank": None}],
        hit_k=3,
    )
    assert summary["hit1"] == 1 / 3
    assert summary["hit_k"] == 2 / 3
    assert summary["mrr"] == (1 + 0.5) / 3


def test_summarize_pass_rate_and_precision():
    summary = summarize(
        [
            {
                "passed": True,
                "expect_count": 1,
                "matched_count": 1,
                "predicted": 1,
                "true_positive": 1,
            },
            {
                "passed": False,
                "expect_count": 1,
                "matched_count": 0,
                "predicted": 1,
                "true_positive": 0,
            },
        ]
    )
    assert summary["pass_rate"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["precision"] == 0.5
