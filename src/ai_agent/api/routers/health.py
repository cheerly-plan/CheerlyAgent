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


@router.get("/health/lightweight")
def lightweight_health() -> dict[str, dict[str, int]]:
    """轻量链路的降级计数（记忆抽取空结果、摘要退回拼接、标题回退、原生调用失败）。

    单独一个端点而不是塞进 ``/health``：探针要稳定可断言，这些计数是**会变的
    进程状态**，混进去会让存活探针的输出随对话内容漂移。

    计数只增不减（进程生命周期），巡检时看趋势：一直涨说明后台功能在空转，
    比如长期记忆一直是 0 行、标题一直在回退。
    """
    from ai_agent.llm import metrics

    return {"degraded": metrics.snapshot()}
