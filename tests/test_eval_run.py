"""统一评测 runner 的门禁逻辑单测。

门禁判定写错会漏放回归，比评测脚本本身更该有测试：
diff 必须区分「新增回归」与「存量失败」，结果可信性必须优先于分数。
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from scripts.eval_run import (
    CONFIG_PATH,
    CmdResult,
    SuiteResult,
    _validate_episode,
    _validate_gold_graph,
    _validate_memory_cases,
    _validate_query_cases,
    append_trend,
    collect_external_suite,
    decide,
    diff_against_baseline,
    env_keys,
    load_json,
    suite_policy,
)

CFG = {
    "diff": {
        "metric_tolerance": 0.05,
        "metric_keys": ["pass_rate"],
        "block_on_new_failures": True,
        "block_on_invalid_cases": True,
        "block_on_env_error": True,
    }
}


def _report(cases, infra="ok", pass_rate=1.0):
    return {
        "suites": {
            "pytest_eval_selftest": {
                "infra": infra,
                "cases": cases,
                "metrics": {"pass_rate": pass_rate},
            }
        }
    }


def _baseline(cases, pass_rate=1.0):
    return {
        "suites": {
            "pytest_eval_selftest": {
                "cases": cases,
                "metrics": {"pass_rate": pass_rate},
            }
        }
    }


def _diff(current_cases, base_cases, **kwargs):
    report = _report(current_cases, **kwargs)
    return diff_against_baseline(report, _baseline(base_cases), CFG)


# ——— 逐条 diff ———


def test_new_failure_is_a_regression():
    diff = _diff({"a": "pass", "b": "fail"}, {"a": "pass", "b": "pass"})
    assert diff["new_failures"] == ["pytest_eval_selftest::b"]
    assert diff["still_failing"] == []
    assert decide(_report({"a": "pass", "b": "fail"}), diff, CFG) == ("regression", 1)


def test_still_failing_does_not_block():
    diff = _diff({"a": "pass", "b": "fail"}, {"a": "pass", "b": "fail"})
    assert diff["new_failures"] == []
    assert diff["still_failing"] == ["pytest_eval_selftest::b"]
    assert decide(_report({"a": "pass", "b": "fail"}), diff, CFG)[0] == "pass"


def test_fixed_case_is_recorded():
    diff = _diff({"b": "pass"}, {"b": "fail"})
    assert diff["fixed"] == ["pytest_eval_selftest::b"]
    assert diff["new_failures"] == []


def test_new_case_failure_counts_as_regression():
    diff = _diff({"a": "pass", "c": "fail"}, {"a": "pass"})
    assert diff["new_cases"] == ["pytest_eval_selftest::c"]
    assert diff["new_failures"] == ["pytest_eval_selftest::c"]


def test_removed_case_only_warns():
    diff = _diff({"a": "pass"}, {"a": "pass", "b": "pass"})
    assert diff["removed_cases"] == ["pytest_eval_selftest::b"]
    assert diff["new_failures"] == []
    assert decide(_report({"a": "pass"}), diff, CFG)[0] == "pass"


def test_empty_baseline_treats_everything_as_new():
    report = _report({"a": "pass", "b": "fail"})
    diff = diff_against_baseline(report, {}, CFG)
    assert len(diff["new_cases"]) == 2
    assert diff["new_failures"] == ["pytest_eval_selftest::b"]


# ——— 指标容差 ———


def test_metric_drop_beyond_tolerance_blocks():
    report = _report({"a": "pass"}, pass_rate=0.90)
    baseline = _baseline({"a": "pass"}, pass_rate=1.0)
    diff = diff_against_baseline(report, baseline, CFG)
    assert diff["metric_delta"]["pytest_eval_selftest.pass_rate"] == -0.1
    assert diff["metric_regressions"]
    assert decide(report, diff, CFG) == ("regression", 1)


def test_metric_delta_not_judged_when_case_set_changed():
    report = _report({"a": "pass", "b": "fail"}, pass_rate=0.5)
    baseline = _baseline({"a": "pass"}, pass_rate=1.0)
    diff = diff_against_baseline(report, baseline, CFG)
    assert diff["metric_regressions"] == []
    assert diff["new_failures"] == ["pytest_eval_selftest::b"]


# ——— 结果可信性优先 ———


def test_env_error_takes_priority_over_scores():
    report = _report({"a": "pass"}, infra="env_error")
    diff = diff_against_baseline(report, _baseline({"a": "pass"}), CFG)
    assert decide(report, diff, CFG) == ("env_error", 3)


def test_invalid_cases_take_priority_over_scores():
    report = _report({"a": "pass"}, infra="invalid_cases")
    diff = diff_against_baseline(report, _baseline({"a": "pass"}), CFG)
    assert decide(report, diff, CFG) == ("invalid_cases", 2)


def test_env_error_wins_over_invalid_cases_and_regression():
    report = {
        "suites": {
            "a": {"infra": "invalid_cases", "cases": {"x": "fail"}, "metrics": {"pass_rate": 0.0}},
            "b": {"infra": "env_error", "cases": {}, "metrics": {"pass_rate": 0.0}},
        }
    }
    diff = diff_against_baseline(report, {}, CFG)
    assert decide(report, diff, CFG) == ("env_error", 3)


# ——— 用例集 schema 校验 ———


def test_query_case_validation_accepts_valid():
    rows = _validate_query_cases("c.json", [{"query": "营业收入", "expect": {"text": "x"}}])
    assert rows == [("c.json::00", "pass", "")]


def test_query_case_validation_rejects_missing_expect():
    rows = _validate_query_cases("c.json", [{"query": "营业收入"}])
    assert rows[0][1] == "fail"
    assert "expect" in rows[0][2]


def test_query_case_validation_rejects_non_array():
    rows = _validate_query_cases("c.json", {"query": "x"})
    assert rows[0][0].endswith("::shape")
    assert rows[0][1] == "fail"


def test_query_case_accepts_negative_with_null_expect():
    rows = _validate_query_cases("c.json", [
        {"query": "库里没有的公司", "group": "negative", "expect": None}
    ])
    assert rows[0][1] == "pass"


def test_memory_case_allows_expect_empty():
    rows = _validate_memory_cases("m.json", [
        {"id": "m1", "turn": [{"role": "user", "content": "今天天气"}], "expect_empty": True}
    ])
    assert rows[0][1] == "pass"


def test_memory_case_requires_id_and_turn():
    rows = _validate_memory_cases("m.json", [{"expect": [{"kind": "profile"}]}])
    assert rows[0][1] == "fail"
    assert "id" in rows[0][2] and "turn" in rows[0][2]


def test_memory_case_allows_forbid_facet_as_only_assertion():
    rows = _validate_memory_cases("m.json", [
        {
            "id": "career-intent",
            "turn": [{"role": "user", "content": "我打算转行"}],
            "forbid_facet": [{"facet": "occupation", "contains": "Agent"}],
        }
    ])
    assert rows[0][1] == "pass"


def test_episode_case_requires_query_fields():
    data = {"episodes": ["一条情景"], "queries": [{"id": "q1", "query": "猫叫什么"}]}
    rows = _validate_episode("e.json", data)
    assert [row[1] for row in rows] == ["pass", "fail"]
    assert "expect" in rows[1][2]


def test_gold_graph_requires_entities_and_relations():
    rows = _validate_gold_graph("g.json", {"file": "x.pdf", "max_hops": 2, "aliases": []})
    statuses = {row[0]: row[1] for row in rows}
    assert statuses["g.json::entities"] == "fail"
    assert statuses["g.json::relations"] == "fail"
    assert statuses["g.json::file"] == "pass"


def test_env_fingerprint_change_is_reported_but_not_blocking():
    report = _report({"a": "pass"})
    report["env_fingerprint"] = {"cases_sha256": "new"}
    baseline = _baseline({"a": "pass"})
    baseline["env_fingerprint"] = {"cases_sha256": "old"}
    diff = diff_against_baseline(report, baseline, CFG)
    assert "cases_sha256" in diff["env_changed"]
    assert diff["new_failures"] == []
    assert decide(report, diff, CFG) == ("pass", 0)


def test_suite_finish_computes_metrics():
    suite = SuiteResult("demo", tier=0)
    suite.cases = {"a": "pass", "b": "fail", "c": "skip"}
    suite.finish(time.monotonic())
    assert suite.metrics == {
        "total": 3, "passed": 1, "failed": 1, "skipped": 1, "pass_rate": 0.3333,
    }


# ——— T1/T2 外部脚本调度 ———

EXT_CFG = {"timeouts_s": {"demo_eval": 10}}


def _item(tier=1, policy="blocking", exit_map=None):
    return {
        "name": "demo_eval",
        "tier": tier,
        "command": "uv run python scripts/demo_eval.py",
        "exit_map": exit_map or {"0": "pass", "1": "regression", "2": "invalid_cases"},
        "requires_env": ["DATABASE_URL"],
        "policy": policy,
    }


def test_external_suite_skips_when_dependency_missing():
    suite = collect_external_suite(_item(), EXT_CFG, set())
    assert suite.infra == "skipped"
    assert suite.cases == {}
    assert "DATABASE_URL" in suite.message


def test_external_suite_maps_exit_1_to_fail_case():
    with patch("scripts.eval_run._run", return_value=CmdResult("hit@3 0.70", "", 1)):
        suite = collect_external_suite(_item(), EXT_CFG, {"DATABASE_URL"})
    assert suite.cases == {"demo_eval::exit": "fail"}
    assert suite.infra == "ok"
    assert "hit@3 0.70" in suite.details["demo_eval::stdout"]


def test_external_suite_maps_exit_2_to_invalid_cases():
    with patch("scripts.eval_run._run", return_value=CmdResult("", "", 2)):
        suite = collect_external_suite(_item(), EXT_CFG, {"DATABASE_URL"})
    assert suite.infra == "invalid_cases"


def test_external_suite_unknown_exit_code_is_invalid_cases():
    with patch("scripts.eval_run._run", return_value=CmdResult("", "", 42)):
        suite = collect_external_suite(_item(), EXT_CFG, {"DATABASE_URL"})
    assert suite.infra == "invalid_cases"


def test_external_suite_timeout_is_env_error():
    result = CmdResult("", "", -1, "命令超时（>10s）")
    with patch("scripts.eval_run._run", return_value=result):
        suite = collect_external_suite(_item(), EXT_CFG, {"DATABASE_URL"})
    assert suite.infra == "env_error"


def test_external_suite_advisory_fail_is_a_fail_case_not_infra_error():
    item = _item(tier=2, policy="advisory",
                 exit_map={"0": "pass", "1": "advisory_fail", "2": "env_error"})
    with patch("scripts.eval_run._run", return_value=CmdResult("通过率 0.60", "", 1)):
        suite = collect_external_suite(item, EXT_CFG, {"DATABASE_URL"})
    assert suite.cases["demo_eval::exit"] == "fail"
    assert suite.infra == "ok"
    assert suite.tier == 2


def test_external_suite_never_writes_stderr_into_report():
    secret = "postgresql://user:password@host/db"
    with patch("scripts.eval_run._run", return_value=CmdResult("ok", secret, 0)):
        suite = collect_external_suite(_item(), EXT_CFG, {"DATABASE_URL"})
    assert secret not in json.dumps(suite.to_dict(), ensure_ascii=False)


def _tier_report(tier, status, infra="ok"):
    return {
        "suites": {
            "demo_eval": {
                "tier": tier, "infra": infra,
                "cases": {"demo_eval::exit": status},
                "metrics": {"pass_rate": 0.0},
            }
        }
    }


def test_t2_failure_is_advisory_and_does_not_block():
    report = _tier_report(2, "fail")
    diff = diff_against_baseline(report, {}, CFG)
    assert diff["advisory_new_failures"] == ["demo_eval::demo_eval::exit"]
    assert diff["blocking_new_failures"] == []
    assert decide(report, diff, CFG) == ("pass", 0)


def _cfg_with_policy(name="demo_eval", tier=2, policy="blocking"):
    return {
        **CFG,
        "suites": [{"name": name, "tier": tier, "policy": policy}],
    }


def test_shipped_config_marks_memory_gates_blocking():
    cfg = load_json(CONFIG_PATH)
    assert suite_policy(cfg, "eval_memory", 2) == "blocking"
    assert suite_policy(cfg, "memory_e2e", 1) == "blocking"


def test_configured_t2_blocking_policy_blocks():
    """eval_memory 这类 T2 显式标 blocking 时，不再被「T2 只做趋势」吃掉。"""
    report = _tier_report(2, "fail")
    cfg = _cfg_with_policy(tier=2, policy="blocking")
    diff = diff_against_baseline(report, {}, cfg)
    assert diff["blocking_new_failures"] == ["demo_eval::demo_eval::exit"]
    assert diff["advisory_new_failures"] == []
    assert decide(report, diff, cfg) == ("regression", 1)


def test_configured_t1_advisory_policy_does_not_block():
    report = _tier_report(1, "fail")
    cfg = _cfg_with_policy(tier=1, policy="advisory")
    diff = diff_against_baseline(report, {}, cfg)
    assert diff["advisory_new_failures"]
    assert diff["blocking_new_failures"] == []
    assert decide(report, diff, cfg) == ("pass", 0)


def test_configured_blocking_t2_env_error_blocks():
    report = _tier_report(2, "pass", infra="env_error")
    cfg = _cfg_with_policy(tier=2, policy="blocking")
    assert decide(report, {}, cfg) == ("env_error", 3)


def test_t1_failure_blocks():
    report = _tier_report(1, "fail")
    diff = diff_against_baseline(report, {}, CFG)
    assert diff["blocking_new_failures"]
    assert decide(report, diff, CFG) == ("regression", 1)


def test_t2_env_error_does_not_block():
    report = _tier_report(2, "pass", infra="env_error")
    diff = diff_against_baseline(report, {}, CFG)
    assert decide(report, diff, CFG) == ("pass", 0)


def test_t2_metric_drop_is_not_a_regression():
    report = {
        "suites": {
            "demo_eval": {
                "tier": 2, "infra": "ok", "cases": {"x": "pass"},
                "metrics": {"pass_rate": 0.5},
            }
        }
    }
    baseline = {
        "suites": {
            "demo_eval": {"cases": {"x": "pass"}, "metrics": {"pass_rate": 1.0}}
        }
    }
    diff = diff_against_baseline(report, baseline, CFG)
    assert diff["metric_regressions"] == []
    assert decide(report, diff, CFG) == ("pass", 0)


def test_env_keys_reads_dotenv_keys_without_values():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".env").write_text(
            "DATABASE_URL=postgresql://u:secret@h/db\n# comment\nAPI_KEY=abcd\n",
            encoding="utf-8",
        )
        with patch("scripts.eval_run.ROOT", root), patch.dict(os.environ, {}, clear=True):
            keys = env_keys()
        assert "DATABASE_URL" in keys and "API_KEY" in keys
        assert "postgresql://u:secret@h/db" not in keys


def test_append_trend_records_only_t2_suites():
    with tempfile.TemporaryDirectory() as tmp:
        trend = Path(tmp) / "trend.jsonl"
        report = {
            "run_id": "abc-1", "git_sha": "abc", "verdict": "pass",
            "suites": {
                "t0_eval": {"tier": 0, "infra": "ok", "metrics": {}, "message": "ok"},
                "t2_eval": {"tier": 2, "infra": "ok",
                            "metrics": {"pass_rate": 1.0}, "message": "ok"},
            },
        }
        with patch("scripts.eval_run.TREND_PATH", trend):
            append_trend(report)
            append_trend(report)
        lines = trend.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert list(json.loads(lines[0])["suites"]) == ["t2_eval"]


# ——— 公共依赖与集中阈值（eval/config.json 的 base_env / thresholds）———

def test_base_env_is_merged_with_suite_requires_env():
    cfg = {"timeouts_s": {"demo_eval": 10}, "base_env": ["BASE_URL", "MODEL"]}
    # 缺公共依赖就 skip，即使 suite 自己的 DATABASE_URL 在
    suite = collect_external_suite(_item(), cfg, {"DATABASE_URL"})
    assert suite.infra == "skipped"
    assert "BASE_URL" in suite.message

    with patch("scripts.eval_run._run", return_value=CmdResult("ok", "", 0)) as run:
        suite = collect_external_suite(
            _item(), cfg, {"BASE_URL", "MODEL", "DATABASE_URL"}
        )
    assert suite.infra == "ok"
    assert run.call_args.args[0] == ["uv", "run", "python", "scripts/demo_eval.py"]


def test_threshold_is_injected_only_when_configured():
    with patch("scripts.eval_run._run", return_value=CmdResult("ok", "", 0)) as run:
        collect_external_suite(_item(), EXT_CFG, {"DATABASE_URL"})
    assert "--threshold" not in run.call_args.args[0]

    cfg = {"timeouts_s": {"demo_eval": 10}, "thresholds": {"demo_eval": 0.9}}
    with patch("scripts.eval_run._run", return_value=CmdResult("ok", "", 0)) as run:
        collect_external_suite(_item(), cfg, {"DATABASE_URL"})
    assert run.call_args.args[0][-2:] == ["--threshold", "0.9"]
