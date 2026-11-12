#!/usr/bin/env python3
"""统一评测 runner：分层调度 → 结构化结果 → 基线逐条 diff → 放行/阻断。

设计取舍：

1. **T0 只放零依赖检查**。检索 / 图谱脚本的 ``--validate`` 要连 PostgreSQL
   （``fetch_chunks()`` 会开连接池），不属于静态层，归 T1。把它们塞进门禁
   会让无密钥的 runner 每次必挂，并把环境故障记成代码回归。
2. **不聚合解析现有脚本的退出码**。T0 的判定全部来自结构化产物（pytest
   junitxml、ruff json、内存校验结果），避开各脚本语义不一致的 0/1/2。
   T1/T2 接入前必须逐个核实退出码，注册表里先标 ``enabled=false``。
3. **门禁是相对基线，不是绝对通过率**。项目有存量红灯，绝对阈值会永远红；
   只阻断「基线 pass → 本轮 fail」的新增回归，存量失败只计数。
4. **用例集失效单独归类**。语料变更后用例静默失效会伪装成检索 miss，
   分数不可信时必须阻断，而不是记一次普通失败。
5. **确定性检查不重试**。失败原因不会因重跑改变，重试只会拖长 CI 并掩盖问题；
   外部命令一律带显式超时，缺命令 / 超时归为 env_error 而非崩溃。

用法：
    uv run python scripts/eval_run.py                         默认跑 T0 并与基线 diff
    uv run python scripts/eval_run.py --no-diff               只体检、不做回归判定
    uv run python scripts/eval_run.py --init-baseline         把本轮结果写成基线
    uv run python scripts/eval_run.py --list                  列出 suite 与依赖
    uv run python scripts/eval_run.py --out 报告.json         指定报告路径
    uv run python scripts/eval_run.py --tiers 0,1,2 --allow-external   CI 全量

T1/T2 会连 DB / 调模型，必须显式加 --allow-external 才执行；缺依赖环境变量的
suite 标为 skipped，不阻断。T2 结果只写 data/eval/trend.jsonl 做趋势观测，
永不阻断发布。

退出码：0 放行 / 1 新增回归 / 2 结果或用例集不可信 / 3 环境错误 / 4 参数或基线错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "eval" / "config.json"
BASELINE_PATH = ROOT / "eval" / "baseline.json"
RUNS_DIR = ROOT / "data" / "eval" / "runs"
TREND_PATH = ROOT / "data" / "eval" / "trend.jsonl"
CASES_DIR = ROOT / "scripts" / "cases"

SELFTEST_FILES = (
    "tests/test_eval_retrieval.py",
    "tests/test_eval_memory.py",
)
EVAL_SCRIPTS = (
    "scripts/eval_retrieval.py",
    "scripts/eval_memory.py",
    "scripts/eval_tool_route.py",
    "scripts/eval_graph.py",
    "scripts/eval_graph_retrieval.py",
    "scripts/eval_run.py",
)

# 用例集静态形状。深层语义（期望片段是否真在切片里）由各脚本自己的
# --validate 负责，那是 T1；这里只挡「文件被改坏、必填字段丢失、类型不对」。
QUERY_CASE_FILES = (
    "retrieval_eval_cases.json",
    "retrieval_eval_cases_3f.json",
    "retrieval_eval_cases_corpus.json",
    "graph_eval_cases.json",
    "tool_route_eval_cases.json",
)
MEMORY_CASE_FILE = "memory_eval_cases.json"
EPISODE_CASE_FILE = "episode_eval_cases.json"
GOLD_GRAPH_FILE = "gold_graph.json"


@dataclass
class SuiteResult:
    """一个 suite 的结果。infra 与 cases 分开：前者说结果可不可信，后者说分数。"""

    name: str
    tier: int
    infra: str = "ok"  # ok / env_error / invalid_cases
    cases: dict[str, str] = field(default_factory=dict)  # id -> pass / fail / skip
    details: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    message: str = ""

    def finish(self, started: float) -> SuiteResult:
        self.duration_s = round(time.monotonic() - started, 3)
        total = len(self.cases)
        passed = sum(1 for status in self.cases.values() if status == "pass")
        failed = sum(1 for status in self.cases.values() if status == "fail")
        skipped = sum(1 for status in self.cases.values() if status == "skip")
        self.metrics = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        }
        return self

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "infra": self.infra,
            "cases": self.cases,
            "details": self.details,
            "metrics": self.metrics,
            "duration_s": self.duration_s,
            "message": self.message,
        }


@dataclass
class CmdResult:
    stdout: str
    stderr: str
    returncode: int
    error: str | None = None


def _run(cmd: list[str], timeout_s: float) -> CmdResult:
    """外部命令统一入口：显式超时 + 异常分类，绝不静默失败。"""
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s
        )
    except FileNotFoundError:
        return CmdResult("", "", -1, f"命令不存在：{cmd[0]}")
    except subprocess.TimeoutExpired:
        return CmdResult("", "", -1, f"命令超时（>{timeout_s:g}s）：{' '.join(cmd[:3])}")
    return CmdResult(proc.stdout or "", proc.stderr or "", proc.returncode)


# ——— T0 采集器 ———


def collect_pytest_selftest(cfg: dict) -> SuiteResult:
    """评测脚本自测。逐条结果来自 junit.xml，不靠退出码。"""
    suite = SuiteResult("pytest_eval_selftest", tier=0)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        cmd = ["uv", "run", "pytest", "-q", *SELFTEST_FILES, "--junitxml", str(junit)]
        res = _run(cmd, cfg["timeouts_s"]["pytest_eval_selftest"])
        if res.error:
            suite.infra = "env_error"
            suite.message = res.error
            return suite.finish(started)
        if not junit.exists():
            suite.infra = "invalid_cases"
            suite.message = f"pytest 未产出 junit.xml（退出码 {res.returncode}）：{res.stderr[-300:]}"
            return suite.finish(started)
        try:
            root = ET.parse(junit).getroot()
        except ET.ParseError as exc:
            suite.infra = "invalid_cases"
            suite.message = f"junit.xml 解析失败：{exc}"
            return suite.finish(started)
        for node in root.iter("testcase"):
            cid = f"{node.get('classname') or 'unknown'}::{node.get('name') or 'unknown'}"
            failure = node.find("failure")
            error = node.find("error")
            if failure is not None or error is not None:
                suite.cases[cid] = "fail"
                child = failure if failure is not None else error
                suite.details[cid] = ((child.get("message") or "") if child is not None else "")[:300]
            elif node.find("skipped") is not None:
                suite.cases[cid] = "skip"
            else:
                suite.cases[cid] = "pass"
    suite.finish(started)
    if not suite.cases:
        suite.infra = "invalid_cases"
        suite.message = "junit.xml 里没有收集到任何 testcase"
    else:
        suite.message = f"评测脚本自测 {suite.metrics['passed']}/{suite.metrics['total']} passed"
    return suite


def collect_ruff(cfg: dict) -> SuiteResult:
    """评测脚本静态检查。每个被检查文件一个检查项，便于逐条 diff。"""
    suite = SuiteResult("ruff_eval_scripts", tier=0)
    started = time.monotonic()
    cmd = ["uv", "run", "ruff", "check", "--output-format", "json", *EVAL_SCRIPTS]
    res = _run(cmd, cfg["timeouts_s"]["ruff_eval_scripts"])
    if res.error:
        suite.infra = "env_error"
        suite.message = res.error
        return suite.finish(started)
    # ruff 正常退出只有 0（无诊断）和 1（有诊断）；其它值说明配置或工具本身坏了
    if res.returncode not in (0, 1):
        suite.infra = "invalid_cases"
        suite.message = f"ruff 异常退出（{res.returncode}）：{res.stderr[-300:]}"
        return suite.finish(started)
    try:
        diags = json.loads(res.stdout or "[]")
    except json.JSONDecodeError as exc:
        suite.infra = "invalid_cases"
        suite.message = f"ruff json 输出解析失败：{exc}"
        return suite.finish(started)
    if not isinstance(diags, list):
        suite.infra = "invalid_cases"
        suite.message = "ruff json 输出不是数组"
        return suite.finish(started)

    by_file: dict[str, list[dict]] = {}
    for item in diags:
        if isinstance(item, dict):
            by_file.setdefault(Path(item.get("filename") or "").name, []).append(item)
    for script in EVAL_SCRIPTS:
        filename = Path(script).name
        cid = f"ruff::{filename}"
        hits = by_file.pop(filename, [])
        if hits:
            suite.cases[cid] = "fail"
            suite.details[cid] = "; ".join(
                f"{d.get('code', '?')} {d.get('message', '')}" for d in hits[:3]
            )[:300]
        else:
            suite.cases[cid] = "pass"
    for leftover, hits in by_file.items():
        cid = f"ruff::{leftover}"
        suite.cases[cid] = "fail"
        suite.details[cid] = f"{len(hits)} 条诊断落在未登记文件"
    suite.finish(started)
    suite.message = f"ruff 诊断 {len(diags)} 条 / {len(EVAL_SCRIPTS)} 个脚本"
    return suite


def _validate_query_cases(name: str, data: object) -> list[tuple[str, str, str]]:
    """常规问句用例集：非空数组，每条要有非空 query 和非空 expect。"""
    if not isinstance(data, list) or not data:
        return [(f"{name}::shape", "fail", "用例集必须是非空 JSON 数组")]
    out: list[tuple[str, str, str]] = []
    for index, item in enumerate(data):
        cid = f"{name}::{index:02d}"
        problems: list[str] = []
        if not isinstance(item, dict):
            problems.append("条目不是对象")
        else:
            query = item.get("query")
            if not isinstance(query, str) or not query.strip():
                problems.append("query 缺失或不是非空字符串")
            # 负例（group=negative）的 expect 按设计就是 null：期望查不到。
            # 只要求键存在，避免把合法负例误判成用例被改坏。
            if item.get("group") == "negative":
                if "expect" not in item:
                    problems.append("负例缺少 expect 键（应为 null）")
            elif not item.get("expect"):
                problems.append("expect 缺失或为空；负例请写 group=negative + expect=null")
        out.append((cid, "fail" if problems else "pass", "；".join(problems)))
    return out


def _validate_memory_cases(name: str, data: object) -> list[tuple[str, str, str]]:
    """记忆抽取用例集：主键是 id + turn，expect / expect_empty 二选一。"""
    if not isinstance(data, list) or not data:
        return [(f"{name}::shape", "fail", "用例集必须是非空 JSON 数组")]
    out: list[tuple[str, str, str]] = []
    for index, item in enumerate(data):
        cid = f"{name}::{index:02d}"
        problems: list[str] = []
        if not isinstance(item, dict):
            problems.append("条目不是对象")
        else:
            ident = item.get("id")
            if not isinstance(ident, str) or not ident.strip():
                problems.append("id 缺失或不是非空字符串")
            turn = item.get("turn")
            if not isinstance(turn, list) or not turn:
                problems.append("turn 必须是非空数组")
            # 断言有四种合法形态：正向 expect / 不该记 expect_empty /
            # 禁止子串 forbid / 指定侧面禁止 forbid_facet。只写后两者也合法。
            assertions = ("expect", "expect_empty", "forbid", "forbid_facet")
            if not any(item.get(key) for key in assertions):
                problems.append(
                    "expect / expect_empty / forbid / forbid_facet 至少要有其一"
                )
        out.append((cid, "fail" if problems else "pass", "；".join(problems)))
    return out


def _validate_episode(name: str, data: object) -> list[tuple[str, str, str]]:
    """情景召回用例集：episodes + queries 两个顶层键。"""
    if not isinstance(data, dict):
        return [(f"{name}::shape", "fail", "用例集必须是 JSON 对象")]
    out: list[tuple[str, str, str]] = []
    episodes = data.get("episodes")
    ok = (
        isinstance(episodes, list)
        and bool(episodes)
        and all(isinstance(text, str) and text.strip() for text in episodes)
    )
    out.append((f"{name}::episodes", "pass" if ok else "fail",
                "" if ok else "episodes 必须是非空字符串数组"))
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        out.append((f"{name}::queries", "fail", "queries 必须是非空数组"))
    else:
        for index, item in enumerate(queries):
            cid = f"{name}::q::{index:02d}"
            problems: list[str] = []
            if not isinstance(item, dict):
                problems.append("条目不是对象")
            else:
                for key in ("id", "query", "expect"):
                    value = item.get(key)
                    if not isinstance(value, str) or not value.strip():
                        problems.append(f"{key} 缺失或不是非空字符串")
            out.append((cid, "fail" if problems else "pass", "；".join(problems)))
    return out


def _validate_gold_graph(name: str, data: object) -> list[tuple[str, str, str]]:
    """图谱金标：file / entities / relations / aliases / max_hops。"""
    if not isinstance(data, dict):
        return [(f"{name}::shape", "fail", "金标文件必须是 JSON 对象")]
    out: list[tuple[str, str, str]] = []
    filename = data.get("file")
    ok = isinstance(filename, str) and bool(filename.strip())
    out.append((f"{name}::file", "pass" if ok else "fail",
                "" if ok else "file 必须是非空字符串"))
    for key in ("entities", "relations"):
        value = data.get(key)
        ok = isinstance(value, list) and bool(value)
        out.append((f"{name}::{key}", "pass" if ok else "fail",
                    "" if ok else f"{key} 必须是非空数组"))
    aliases = data.get("aliases")
    ok = isinstance(aliases, list)
    out.append((f"{name}::aliases", "pass" if ok else "fail",
                "" if ok else "aliases 必须是数组"))
    hops = data.get("max_hops")
    ok = isinstance(hops, int) and hops > 0
    out.append((f"{name}::max_hops", "pass" if ok else "fail",
                "" if ok else "max_hops 必须是正整数"))
    return out


def collect_case_schema(_cfg: dict) -> SuiteResult:
    """用例集 schema 静态校验：纯内存、零 DB、零模型。"""
    suite = SuiteResult("case_schema", tier=0)
    started = time.monotonic()
    validators: dict[str, object] = {
        filename: _validate_query_cases for filename in QUERY_CASE_FILES
    }
    validators[MEMORY_CASE_FILE] = _validate_memory_cases
    validators[EPISODE_CASE_FILE] = _validate_episode
    validators[GOLD_GRAPH_FILE] = _validate_gold_graph

    structural_failures = 0
    for filename, validate in validators.items():
        path = CASES_DIR / filename
        if not path.exists():
            cid = f"{filename}::file"
            suite.cases[cid] = "fail"
            suite.details[cid] = "用例集文件缺失"
            structural_failures += 1
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            cid = f"{filename}::parse"
            suite.cases[cid] = "fail"
            suite.details[cid] = f"解析失败：{exc}"
            structural_failures += 1
            continue
        for cid, status, detail in validate(filename, data):
            suite.cases[cid] = status
            if detail:
                suite.details[cid] = detail
            if status == "fail" and cid.endswith(("::shape", "::file", "::parse")):
                structural_failures += 1
    suite.finish(started)
    suite.message = (
        f"{len(validators)} 份用例集 / {suite.metrics['total']} 个检查项 / "
        f"{suite.metrics['failed']} 项失败"
    )
    if structural_failures:
        # 文件读不到或整体形状坏了，分数没有意义，不能只当一条普通失败
        suite.infra = "invalid_cases"
        suite.message += f"；{structural_failures} 处结构性失效"
    return suite


# ——— T1/T2 外部评测脚本调度 ———


def env_keys() -> set[str]:
    """可用的配置键名（只取键名，绝不读取或记录值）。

    pydantic-settings 既读 os.environ 也读项目根 .env；CI 用 .env 注入，
    所以两边都要看。只判「键是否存在」，不把任何密钥值带进日志或报告。
    """
    keys = set(os.environ)
    env_file = ROOT / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    keys.add(stripped.split("=", 1)[0].strip())
        except (OSError, UnicodeDecodeError):
            pass
    return keys


def collect_external_suite(item: dict, cfg: dict, available: set[str]) -> SuiteResult:
    """跑一个已有评测脚本，把散落的退出码语义归一化成统一状态。

    exit_map 逐条按源码核实后写在 config，不凭猜测：例如
    eval_graph_retrieval 的全量评测恒返回 0，因此只调度它的 --validate。
    stderr 不进报告（可能含连接串），诊断细节看 CI 日志。
    """
    name = item["name"]
    suite = SuiteResult(name, tier=int(item["tier"]))
    started = time.monotonic()
    missing = [key for key in item.get("requires_env", []) if key not in available]
    if missing:
        suite.infra = "skipped"
        suite.message = f"缺少依赖配置：{', '.join(missing)}"
        return suite.finish(started)

    command = item["command"].split()
    result = _run(command, cfg["timeouts_s"].get(name, 300))
    if result.error:
        suite.infra = "env_error"
        suite.message = result.error
        return suite.finish(started)

    mapped = item.get("exit_map", {}).get(str(result.returncode))
    if mapped is None:
        suite.infra = "invalid_cases"
        suite.message = f"未登记的退出码 {result.returncode}；stdout 尾部：{result.stdout[-200:]}"
        return suite.finish(started)

    cid = f"{name}::exit"
    suite.details[f"{name}::stdout"] = result.stdout[-500:]
    if mapped == "pass":
        suite.cases[cid] = "pass"
        suite.message = f"退出码 {result.returncode}；脚本内置阈值通过（{item['policy']}）"
    elif mapped in ("regression", "advisory_fail"):
        suite.cases[cid] = "fail"
        suite.details[cid] = f"退出码 {result.returncode}；脚本内置阈值未通过"
        suite.message = f"退出码 {result.returncode}；阈值未通过（{item['policy']}）"
    else:  # invalid_cases / env_error
        suite.infra = mapped
        suite.message = f"退出码 {result.returncode} 归一化为 {mapped}"
    suite.finish(started)
    return suite


def append_trend(report: dict) -> None:
    """T2 逐次结果追加到 jsonl，只看趋势、不阻断，也不进 system。"""
    advisory = {
        name: suite for name, suite in report["suites"].items()
        if suite.get("tier", 0) >= 2
    }
    if not advisory:
        return
    line = {
        "ts": report["run_id"],
        "git_sha": report["git_sha"],
        "verdict": report["verdict"],
        "suites": {
            name: {
                "infra": suite["infra"],
                "metrics": suite["metrics"],
                "message": suite["message"],
            }
            for name, suite in advisory.items()
        },
    }
    TREND_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TREND_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


# ——— 报告、基线与判定 ———


def git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        return proc.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def fingerprint() -> dict[str, str]:
    """环境指纹：语料/脚本变了，即使分数一样也不可直接比。"""
    return {
        "cases_sha256": _digest(list(CASES_DIR.glob("*.json"))),
        "eval_scripts_sha256": _digest([ROOT / name for name in EVAL_SCRIPTS]),
        "selftests_sha256": _digest([ROOT / name for name in SELFTEST_FILES]),
    }


def build_report(suites: list[SuiteResult], cfg: dict, tiers: list[int]) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha()
    return {
        "run_id": f"{sha}-{stamp}",
        "git_sha": sha,
        "python": sys.version.split()[0],
        "config_version": cfg.get("version"),
        "tiers": tiers,
        "env_fingerprint": fingerprint(),
        "suites": {suite.name: suite.to_dict() for suite in suites},
        "diff": {},
        "verdict": "",
        "exit_code": None,
    }


def diff_against_baseline(report: dict, baseline: dict, cfg: dict) -> dict:
    """逐条 diff。基线 pass→本轮 fail 才算新增回归；存量失败不阻断。"""
    new_failures: list[str] = []
    blocking_new_failures: list[str] = []
    advisory_new_failures: list[str] = []
    still_failing: list[str] = []
    fixed: list[str] = []
    new_cases: list[str] = []
    removed_cases: list[str] = []
    metric_delta: dict[str, float] = {}
    metric_regressions: list[str] = []

    for suite_name, current in report["suites"].items():
        tier = current.get("tier", 0)
        base_suite = baseline.get("suites", {}).get(suite_name, {})
        base_cases = base_suite.get("cases", {})
        current_cases = current["cases"]
        for cid, status in current_cases.items():
            label = f"{suite_name}::{cid}"
            previous = base_cases.get(cid)
            if previous is None:
                new_cases.append(label)
            if status == "fail":
                if previous in (None, "pass"):
                    new_failures.append(label)
                    # T2 调模型、结果有抖动，失败只记趋势，绝不阻断发布
                    if tier >= 2:
                        advisory_new_failures.append(label)
                    else:
                        blocking_new_failures.append(label)
                else:
                    still_failing.append(label)
            elif status == "pass" and previous == "fail":
                fixed.append(label)
        for cid in base_cases:
            if cid not in current_cases:
                removed_cases.append(f"{suite_name}::{cid}")

    # 分母变了（新增/删除检查项）时比率不可比，跳过容差判定，避免误报
    stable = not new_cases and not removed_cases
    tolerance = cfg["diff"]["metric_tolerance"]
    for suite_name, current in report["suites"].items():
        base_metrics = baseline.get("suites", {}).get(suite_name, {}).get("metrics", {})
        for key in cfg["diff"]["metric_keys"]:
            if key in base_metrics and key in current["metrics"]:
                delta = round(current["metrics"][key] - base_metrics[key], 4)
                metric_delta[f"{suite_name}.{key}"] = delta
                if stable and delta < -tolerance and current.get("tier", 0) < 2:
                    metric_regressions.append(
                        f"{suite_name}.{key} {delta:+.4f} 低于 -{tolerance}"
                    )

    # 语料 / 脚本 / 自测文件变了，基线里的 case id 可能已经对不上。
    # 只告警不阻断：用例集正常演进也会改指纹，但必须让人看见。
    base_fp = baseline.get("env_fingerprint", {})
    current_fp = report.get("env_fingerprint", {})
    env_changed = {
        key: {"baseline": base_fp.get(key), "current": current_fp.get(key)}
        for key in ("cases_sha256", "eval_scripts_sha256", "selftests_sha256")
        if base_fp.get(key) != current_fp.get(key)
    }

    return {
        "new_failures": new_failures,
        "blocking_new_failures": blocking_new_failures,
        "advisory_new_failures": advisory_new_failures,
        "still_failing": still_failing,
        "fixed": fixed,
        "new_cases": new_cases,
        "removed_cases": removed_cases,
        "metric_delta": metric_delta,
        "metric_regressions": metric_regressions,
        "env_changed": env_changed,
    }


def decide(report: dict, diff: dict, cfg: dict) -> tuple[str, int]:
    """门禁顺序：结果可信性优先于分数。T2 只做趋势，永不阻断发布。"""
    rules = cfg["diff"]

    def blocking(name: str) -> bool:
        return report["suites"][name].get("tier", 0) in (0, 1)

    env_errors = [
        name for name, suite in report["suites"].items()
        if suite["infra"] == "env_error" and blocking(name)
    ]
    invalid = [
        name for name, suite in report["suites"].items()
        if suite["infra"] == "invalid_cases" and blocking(name)
    ]
    if env_errors and rules["block_on_env_error"]:
        return "env_error", 3
    if invalid and rules["block_on_invalid_cases"]:
        return "invalid_cases", 2
    reasons: list[str] = []
    blocking_failures = diff.get("blocking_new_failures", diff.get("new_failures", []))
    if blocking_failures and rules["block_on_new_failures"]:
        reasons.append(f"新增回归 {len(blocking_failures)} 条")
    if diff["metric_regressions"]:
        reasons.append(f"指标跌破容差 {len(diff['metric_regressions'])} 项")
    if reasons:
        return "regression", 1
    return "pass", 0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} 顶层必须是 JSON 对象")
    return payload


def baseline_payload(report: dict) -> dict:
    return {
        "version": 1,
        "created_at": report["run_id"],
        "git_sha": report["git_sha"],
        "env_fingerprint": report["env_fingerprint"],
        "suites": {
            name: {"cases": suite["cases"], "metrics": suite["metrics"]}
            for name, suite in report["suites"].items()
        },
    }


def print_list(cfg: dict) -> None:
    print("T0（零依赖，默认执行）：")
    for name in ("pytest_eval_selftest", "ruff_eval_scripts", "case_schema"):
        print(f"  T0  {name}")
    print("T1/T2（需 --allow-external；缺依赖配置自动 skip）：")
    for item in cfg.get("suites", []):
        print(f"  T{item['tier']}  {item['name']:<32} {item['policy']}")
        print(f"       {item['command']}")
        print(f"       {item['note']}")


def print_report(report: dict, report_path: Path) -> None:
    print(f"评测 run={report['run_id']}  git={report['git_sha']}  python={report['python']}")
    for name, suite in report["suites"].items():
        metrics = suite["metrics"]
        flag = f"  infra={suite['infra']}" if suite["infra"] != "ok" else ""
        print(
            f"  {name:<24} {metrics['passed']}/{metrics['total']} pass"
            f"  ({suite['duration_s']:.2f}s) {suite['message']}{flag}"
        )
    diff = report.get("diff") or {}
    if diff:
        print(
            f"diff  new_failures={len(diff['new_failures'])}"
            f"  still_failing={len(diff['still_failing'])}"
            f"  fixed={len(diff['fixed'])}"
            f"  new_cases={len(diff['new_cases'])}"
            f"  removed_cases={len(diff['removed_cases'])}"
            f"  advisory={len(diff.get('advisory_new_failures', []))}"
        )
        for label in diff.get("blocking_new_failures", diff["new_failures"])[:10]:
            print(f"  [回归] {label}")
        for label in diff.get("advisory_new_failures", []):
            print(f"  [趋势] T2 未过阈值（不阻断）：{label}")
        for label in diff["removed_cases"][:10]:
            print(f"  [警告] 用例被移除：{label}")
        for label in diff["metric_regressions"]:
            print(f"  [回归] {label}")
        for key in diff.get("env_changed", {}):
            print(f"  [警告] 环境指纹变化：{key}；基线可能需要 review")
    print(f"verdict {report['verdict'].upper()}  退出码 {report['exit_code']}")
    try:
        shown = report_path.relative_to(ROOT)
    except ValueError:
        # --out 允许指向仓库外的绝对路径（CI / 冒烟脚本会用）
        shown = report_path
    print(f"报告 {shown}")


def main() -> int:
    parser = argparse.ArgumentParser(description="统一评测 runner（默认 T0 静态门禁）")
    parser.add_argument("--tier", type=int, default=0, help="单个评测层（默认 0）")
    parser.add_argument("--tiers", default="", help="逗号分隔的层，如 0,1,2（优先于 --tier）")
    parser.add_argument(
        "--allow-external", action="store_true",
        help="显式允许跑 T1/T2（会连 DB / 调模型）",
    )
    parser.add_argument("--no-diff", action="store_true", help="只体检，不做基线回归判定")
    parser.add_argument("--init-baseline", action="store_true", help="把本轮结果写成基线")
    parser.add_argument(
        "--allow-failing-baseline", action="store_true",
        help="允许把含失败项的结果写成基线（默认拒绝，防止掩盖回归）",
    )
    parser.add_argument("--list", action="store_true", help="列出 suite 注册信息后退出")
    parser.add_argument("--out", default="", help="自定义报告路径（相对仓库根目录）")
    args = parser.parse_args()

    try:
        cfg = load_json(CONFIG_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"配置读不到：{exc}", file=sys.stderr)
        return 4

    if args.list:
        print_list(cfg)
        return 0

    if args.tiers:
        try:
            tiers = [int(part.strip()) for part in args.tiers.split(",") if part.strip()]
        except ValueError:
            print("--tiers 必须是逗号分隔的整数，如 0,1,2", file=sys.stderr)
            return 4
    else:
        tiers = [args.tier]
    if not tiers or any(tier not in cfg.get("enabled_tiers", [0]) for tier in tiers):
        print(
            f"请求的层不在 enabled_tiers={cfg.get('enabled_tiers', [0])} 里；用 --list 查看。",
            file=sys.stderr,
        )
        return 4
    if any(tier >= 1 for tier in tiers) and not args.allow_external:
        print(
            "跑 T1/T2 会连 DB / 调模型，必须显式加 --allow-external；"
            "缺依赖配置的 suite 会自动 skip。",
            file=sys.stderr,
        )
        return 4

    suites: list[SuiteResult] = []
    if 0 in tiers:
        suites += [
            collect_pytest_selftest(cfg),
            collect_ruff(cfg),
            collect_case_schema(cfg),
        ]
    if any(tier >= 1 for tier in tiers):
        available = env_keys()
        for item in cfg.get("suites", []):
            if int(item["tier"]) in tiers:
                suites.append(collect_external_suite(item, cfg, available))
    if not suites:
        print("没有可运行的 suite", file=sys.stderr)
        return 4
    report = build_report(suites, cfg, tiers)
    report_path = Path(args.out) if args.out else RUNS_DIR / f"{report['run_id']}.json"
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    if args.init_baseline:
        failed = sum(suite.metrics["failed"] for suite in suites)
        if failed and not args.allow_failing_baseline:
            print(
                f"本轮有 {failed} 个失败检查项，拒绝写成基线；"
                "确认是存量问题后再加 --allow-failing-baseline。",
                file=sys.stderr,
            )
            return 4
        payload = baseline_payload(report)
        try:
            write_json(BASELINE_PATH, payload)
        except OSError as exc:
            print(f"基线写入失败：{exc}", file=sys.stderr)
            return 3
        report["diff"] = {}
        report["verdict"] = "baseline_created"
        report["exit_code"] = 0
        print(f"基线已写入 {BASELINE_PATH.relative_to(ROOT)}，请 review 后提交")
    elif args.no_diff:
        empty_diff = {
            "new_failures": [], "blocking_new_failures": [],
            "advisory_new_failures": [], "still_failing": [], "fixed": [],
            "new_cases": [], "removed_cases": [], "metric_delta": {},
            "metric_regressions": [], "env_changed": {},
        }
        verdict, code = decide(report, empty_diff, cfg)
        report["diff"] = empty_diff
        report["verdict"] = "inspection" if verdict == "pass" else verdict
        report["exit_code"] = code
    else:
        try:
            baseline = load_json(BASELINE_PATH)
        except FileNotFoundError:
            print(
                f"基线不存在：先跑 `uv run python scripts/eval_run.py --init-baseline`"
                f"（{BASELINE_PATH.relative_to(ROOT)} 需提交进 git）",
                file=sys.stderr,
            )
            return 4
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"基线读不到：{exc}", file=sys.stderr)
            return 4
        diff = diff_against_baseline(report, baseline, cfg)
        verdict, code = decide(report, diff, cfg)
        report["diff"] = diff
        report["verdict"] = verdict
        report["exit_code"] = code

    try:
        write_json(report_path, report)
    except OSError as exc:
        print(f"报告写入失败：{exc}", file=sys.stderr)
        return 3
    try:
        append_trend(report)
    except OSError as exc:
        # 趋势是观测数据，写不进去不影响门禁结论
        print(f"[警告] 趋势写入失败：{exc}")
    print_report(report, report_path)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
