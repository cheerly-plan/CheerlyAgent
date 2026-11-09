from contextlib import asynccontextmanager
import logging
import os
import threading
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_agent.agents.loop import shutdown_turn_finalizer
from ai_agent.api.middleware.auth import AuthTokenMiddleware
from ai_agent.api.middleware.request_id import RequestIdMiddleware
from ai_agent.api.routers.auth import router as auth_router
from ai_agent.api.routers.chat import router as chat_router
from ai_agent.api.routers.health import router as health_router
from ai_agent.api.routers.inspect import router as inspect_router
from ai_agent.api.routers.kb import router as kb_router
from ai_agent.api.routers.kg import router as kg_router
from ai_agent.api.routers.report import router as report_router
from ai_agent.api.routers.session import router as session_router
from ai_agent.api.routers.user import router as user_router
from ai_agent.core.config import settings
from ai_agent.core.exceptions import AgentError
from ai_agent.core.logging import setup_logging
from ai_agent.kb.pipeline import kb_pipeline
from ai_agent.storage.db import run_migrations
from ai_agent.memory.short_term import short_term_memory
from ai_agent.storage.store import store
from ai_agent.tasks.cleanup import maintenance_cleaner
from ai_agent.tasks.title import shutdown_title_jobs

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 研报配置纯读文件、不依赖 DB，放在最前：坏配置要么立刻拦住（strict），
    # 要么在降级生效前留下一条醒目的 ERROR 日志。结果挂到 app.state，
    # 供 /health 与管理员诊断端点查询，不必翻启动日志。
    from ai_agent.report.lint import lint_report_config

    app.state.report_config_lint = lint_report_config(strict=settings.report_config_strict)
    # schema 迁移走 Alembic（版本化、可回滚）；ensure_schema 保留作回滚锚点
    run_migrations()
    # 补登记历史深度层审核记录：旧清理逻辑曾把 <id>-deep 当磁盘残留删掉，
    # 那些 .deep.pdf 会因「没有审核记录」既不能预览也不能下载。幂等、只补不删。
    try:
        from ai_agent.report.review import repair_deep_records

        repaired = repair_deep_records()
        if repaired:
            _logger.warning("已补登记 %s 条深度层审核记录（历史清理误删）", repaired)
    except Exception:
        _logger.exception("深度层审核记录补登记失败，不影响启动")
    # 重排后端自检：默认 local_ce，未装依赖/无权重时显式告警并退回 LLM
    from ai_agent.kb.retrieval.rerank import probe_reranker

    probe_reranker()
    # 先让 HTTP 对外就绪，再把中断文件退回 uploaded 队列；真正的解析由
    # ingest-worker 抢任务执行，避免 OCR 崩在 API 进程里。
    short_term_memory.start_cleanup()
    maintenance_cleaner.start()

    def _recover_kb() -> None:
        try:
            # 有独立 ingest-worker 时不要把 parsing 打回 uploaded，避免抢跑中的任务被打断
            kb_pipeline.recover_stuck(requeue=settings.kb_ingest_inline)
        except Exception:
            _logger.exception("kb 启动恢复失败")

    threading.Thread(target=_recover_kb, name="kb-recover", daemon=True).start()
    yield
    maintenance_cleaner.stop()
    short_term_memory.stop_cleanup()
    shutdown_title_jobs()
    # 关服务不等在途长任务跑完：热重载/重启后由队列或客户端重跑即可。
    # wait=False 只让 lifespan 立刻返回；ThreadPoolExecutor 的 atexit 仍会
    # join 非 daemon 线程，研报/巨潮可卡数十分钟 → 旧 worker 占 8000 假死。
    # 热重载直接硬退；其它场景靠看门狗兜底。
    shutdown_turn_finalizer(wait=False)
    kb_pipeline.shutdown(wait=False)
    store.close()

    def _exit_watchdog() -> None:
        time.sleep(5)
        _logger.error("关闭后仍有线程未退出，强制结束进程（避免 8000 假死）")
        os._exit(1)

    threading.Thread(target=_exit_watchdog, name="exit-watchdog", daemon=True).start()
    if os.environ.get("ALMOST_AGENT_RELOAD") == "1":
        _logger.info("热重载：跳过线程池 atexit join，硬退 worker")
        os._exit(0)


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="almost-agent", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(AuthTokenMiddleware)
    app.add_middleware(
        CORSMiddleware,
            allow_origins=[
                origin.strip()
                for origin in settings.cors_origins.split(",")
                if origin.strip()
            ],
        allow_methods=["*"],
        allow_headers=["*"],
        # 研报预览面板要读 M2 门禁状态头；跨域下自定义响应头默认对 JS 不可见。
        expose_headers=["X-Report-Download-Allowed", "X-Report-Review-Status"],
    )
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(session_router)
    app.include_router(inspect_router)
    app.include_router(chat_router)
    app.include_router(kb_router)
    app.include_router(kg_router)
    app.include_router(report_router)

    @app.exception_handler(AgentError)
    async def agent_error(_request: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    return app


app = create_app()