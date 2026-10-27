from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_agent.agents.loop import shutdown_turn_finalizer
from ai_agent.api.middleware.auth import AuthTokenMiddleware
from ai_agent.api.middleware.request_id import RequestIdMiddleware
from ai_agent.api.routers.auth import router as auth_router
from ai_agent.api.routers.chat import router as chat_router
from ai_agent.api.routers.health import router as health_router
from ai_agent.api.routers.kb import router as kb_router
from ai_agent.api.routers.kg import router as kg_router
from ai_agent.api.routers.session import router as session_router
from ai_agent.api.routers.user import router as user_router
from ai_agent.core.config import settings
from ai_agent.core.exceptions import AgentError
from ai_agent.core.logging import setup_logging
from ai_agent.kb.pipeline import kb_pipeline
from ai_agent.memory.short_term import short_term_memory
from ai_agent.memory.store import store
from ai_agent.tasks.cleanup import maintenance_cleaner
from ai_agent.tasks.title import shutdown_title_jobs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store.ensure_schema()
    # 上次进程异常退出时卡在 parsing/embedding 的知识库文件标 failed
    kb_pipeline.recover_stuck()
    short_term_memory.start_cleanup()
    maintenance_cleaner.start()
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
    app.include_router(chat_router)
    app.include_router(kb_router)
    app.include_router(kg_router)

    @app.exception_handler(AgentError)
    async def agent_error(_request: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    return app


app = create_app()