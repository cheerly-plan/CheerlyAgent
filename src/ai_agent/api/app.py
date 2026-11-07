from contextlib import asynccontextmanager
import logging
import threading

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 研报配置纯读文件、不依赖 DB，放在最前：坏配置要么立刻拦住（strict），
    # 要么在降级生效前留下一条醒目的 ERROR 日志。结果挂到 app.state，
    # 供 /health 与管理员诊断端点查询，不必翻启动日志。
    from ai_agent.report.lint import lint_report_config

    app.state.report_config_lint = lint_report_config(strict=settings.report_config_strict)
    # schema 迁移走 Alembic（版本化、可回滚）；ensure_schema 保留作回滚锚点
    run_migrations()
    # 重排后端自检：默认 local_ce，未装依赖/无权重时显式告警并退回 LLM
    from ai_agent.kb.retrieval.rerank import probe_reranker

    probe_reranker()
    # 先让 HTTP 对外就绪，再恢复中断的 KB 任务——避免解析/OCR 原生崩溃
    # 把 worker 打死在 "Waiting for application startup"，表现为登录一直转圈。
    short_term_memory.start_cleanup()
    maintenance_cleaner.start()

    def _recover_kb() -> None:
        try:
            kb_pipeline.recover_stuck()
        except Exception:
            logging.getLogger(__name__).exception("kb 启动恢复失败")

    threading.Thread(target=_recover_kb, name="kb-recover", daemon=True).start()
    yield
    maintenance_cleaner.stop()
    short_term_memory.stop_cleanup()
    shutdown_title_jobs()
    # 等在途的长期记忆抽取/摘要跑完再关连接池，重启不丢记忆
    shutdown_turn_finalizer()
    kb_pipeline.shutdown()
    store.close()


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