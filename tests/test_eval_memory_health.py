"""记忆健康度脚本的自测：噪声拦截、召回、排序健康度、用例集有效性。"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_memory_health.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_noise_cases_are_all_blocked_by_rules():
    result = _run("--noise-only", "--json", "/tmp/mem_health_noise.json")
    assert result.returncode == 0, result.stderr
    report = json.loads(Path("/tmp/mem_health_noise.json").read_text(encoding="utf-8"))
    noise = report["noise"]
    assert noise["total"] >= 8
    assert noise["rule_rate"] == 1.0, f"仍有噪声漏过规则：{noise['leaks']}"


def test_recall_and_health_report_are_produced():
    result = _run("--json", "/tmp/mem_health_full.json")
    assert result.returncode == 0, result.stderr
    report = json.loads(Path("/tmp/mem_health_full.json").read_text(encoding="utf-8"))
    recall = report["recall"]
    assert recall["total"] >= 6
    # 常青事实必须能被召回
    assert recall["recall"] == 1.0
    # 一次性内容不能挤进 top_k
    assert report["ranking"]["transient_rate"] == 0.0


def test_threshold_failure_returns_regression_code():
    result = _run("--noise-only", "--threshold", "1.5")
    assert result.returncode == 1
    assert "未达标" in result.stderr


def test_health_script_has_no_db_or_model_dependency():
    """脚本必须纯本地：不 import store、不调 embed/model。"""
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("from ai_agent.storage.store import", "embed_texts", "complete_chat"):
        assert forbidden not in source
