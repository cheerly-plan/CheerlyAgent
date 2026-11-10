from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    """存活探针：status 恒为 ok，探针不该因研报配置降级而把服务摘掉。

    额外的 ``report_config`` 只报状态枚举（ok / degraded / unknown），细节在
    ``/v1/inspect/report-config``（管理员），避免把内部路径暴露到公开端点。
    """
    report = getattr(request.app.state, "report_config_lint", None)
    config_state = "unknown" if report is None else ("ok" if report.ok else "degraded")
    return {"status": "ok", "report_config": config_state}
