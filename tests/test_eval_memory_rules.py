"""记忆规则层离线评测脚本：不调模型、不连库。"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "eval_memory_rules.py"
    spec = importlib.util.spec_from_file_location("eval_memory_rules", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _case(cid, expect, content, kind="episode", facet=""):
    return {
        "id": cid,
        "expect": expect,
        "user": content,
        "extracted": [{"kind": kind, "facet": facet, "content": content}],
    }


def test_remember_case_passing(mod):
    report = mod.run([_case(1, "remember", "我在做一家社区诊所的慢病管理系统", facet="other")])
    assert report["remember"].get("pass", 0) == 1
    assert report["remember_rate"] == 1.0


def test_remember_case_false_drop(mod):
    report = mod.run([_case(2, "remember", "用户正在查询某公司股价")])
    # Counter 只在事件发生时才有对应键，全 0 时该键不存在
    assert report["remember"].get("pass", 0) == 0
    assert report["remember"]["false_drop"] == 1


def test_forget_case_passing(mod):
    report = mod.run([_case(3, "forget", "用户正在翻译一段英文文档")])
    assert report["forget"].get("pass", 0) == 1


def test_forget_case_false_keep_lists_uncovered(mod):
    report = mod.run([_case(4, "forget", "帮我把厨房收拾一下")])
    # 该句没有任何确定性规则覆盖，应记入 uncovered 供补规则参考
    assert report["forget"]["false_keep"] == 1
    assert report["uncovered"][0]["id"] == 4


def test_attribution_names_the_rule(mod):
    report = mod.run([_case(5, "forget", "用户正在翻译一段英文文档")])
    assert report["rule_attribution"] == {}


def test_rules_hit_reports_rule_names(mod):
    hits = mod._rules_hit("用户正在查询某公司股价")
    assert "_ONGOING" in hits


def test_long_term_project_not_dropped(mod):
    """「正在做/正在用 X 做 Y」是用户自己的长期项目，不能误杀。"""
    for text in (
        "正在做 CheerlyAgent 项目，另有 Harness 版本",
        "用户正在用 Node.js + Koa 做 BFF，把多个后端接口聚合成一层",
    ):
        report = mod.run([_case(6, "remember", text)])
        assert report["remember"]["pass"] == 1, text


def test_remember_request_not_treated_as_task(mod):
    """「帮我记住…」是要求记忆，不是一次性任务。"""
    report = mod.run([_case(7, "remember", "帮我记住我的生日是 3 月 5 日", facet="other")])
    assert report["remember"]["pass"] == 1


def test_shipped_case_file_is_valid(mod):
    data = json.loads((ROOT / "scripts" / "cases" / "memory_rule_cases.json").read_text())
    assert len(data) >= 60
    for case in data:
        assert case["expect"] in {"remember", "forget"}
        assert case["extracted"], case["id"]


def test_shipped_case_file_meets_thresholds(mod):
    """仓库内的规则用例集必须通过脚本内置通过线。"""
    data = json.loads((ROOT / "scripts" / "cases" / "memory_rule_cases.json").read_text())
    report = mod.run(data)
    assert report["remember_rate"] >= 0.95
    assert report["forget_rate"] >= 0.80


def test_rule_eval_passes_user_text_for_facet_checks(mod):
    """评测必须把 user 传给 _keep_extracted，否则 locale/occupation 会假失败。"""
    case = {
        "id": 1,
        "expect": "remember",
        "user": "我在北京执业，常驻朝阳区",
        "extracted": [
            {"kind": "profile", "facet": "locale", "content": "北京朝阳区"}
        ],
    }
    report = mod.run([case])
    assert report["remember_rate"] == 1.0


def test_crossdomain_suite_meets_threshold(mod):
    """换领域（医疗/法律/教育/客服）不能退化——防止规则过拟合到金融+技术。"""
    data = json.loads(
        (ROOT / "scripts" / "cases" / "memory_crossdomain_cases.json").read_text()
    )
    report = mod.run(data)
    assert report["remember_rate"] >= 0.95
    assert report["forget_rate"] >= 0.75


def test_crossdomain_rule_owned_cases_have_no_gaps(mod):
    """owner=rule 的漏放必须为 0；model/either 归语义判断，不计入。"""
    data = json.loads(
        (ROOT / "scripts" / "cases" / "memory_crossdomain_cases.json").read_text()
    )
    rule_forget = [c for c in data if c["expect"] == "forget" and c["owner"] == "rule"]
    report = mod.run(rule_forget)
    assert report["forget_rate"] == 1.0, report["uncovered"]
