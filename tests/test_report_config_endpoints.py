"""研报配置 lint 结果的可观测端点。

契约：``/health`` 的 status 恒为 ok（探针不该因配置降级摘服务），只多一个
状态枚举；完整 issues 只给管理员的 ``/v1/inspect/report-config``。

项目未装 httpx，这里不引入 TestClient 依赖：直接调用端点函数并校验路由已注册，
逻辑契约与挂载位置都能覆盖。
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from ai_agent.api.app import create_app
from ai_agent.api.routers.health import health
from ai_agent.api.routers.inspect import report_config_lint
from ai_agent.core.config import settings
from ai_agent.report.lint import LintReport


def _app_with(report: LintReport | None = None):
    app = create_app()
    if report is not None:
        app.state.report_config_lint = report
    return app


def _request(app) -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
        }
    )


def test_health_is_unknown_before_lifespan():
    assert health(_request(_app_with())) == {"status": "ok", "report_config": "unknown"}


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (LintReport(("indicators.yaml",), ()), "ok"),
        (LintReport((), ("indicators.yaml: 故意损坏",)), "degraded"),
    ],
)
def test_health_reports_config_state_without_failing_probe(report, expected):
    app = _app_with(report)
    assert health(_request(app)) == {"status": "ok", "report_config": expected}


def test_inspect_returns_full_lint_report():
    report = LintReport(("sections.yaml",), ("indicators.yaml: 故意损坏",))
    body = report_config_lint(_request(_app_with(report)), None)
    assert body == {
        "ok": False,
        "strict": settings.report_config_strict,
        "checked": ["sections.yaml"],
        "issues": ["indicators.yaml: 故意损坏"],
    }


def test_inspect_raises_503_when_lint_has_not_run():
    with pytest.raises(HTTPException) as exc:
        report_config_lint(_request(_app_with()), None)
    assert exc.value.status_code == 503


def test_endpoints_are_registered():
    # 新版 FastAPI 把 include_router 包成惰性 _IncludedRouter，routes 上看不到
    # path；用 OpenAPI schema 校验挂载位置，不依赖内部实现。
    paths = set(create_app().openapi()["paths"])
    assert "/health" in paths
    assert "/v1/inspect/report-config" in paths
